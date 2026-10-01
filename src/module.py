import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_timestep_embedding(timesteps, embedding_dim):
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=8, num_channels=in_channels, eps=1e-6, affine=True)



class PosEnc1D(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)  # [max_len, d_model]

    def forward(self, x):  # x: [B, T, C]
        return x + self.pe[:x.size(1)].unsqueeze(0)

class Flow_Embed_3D(nn.Module):
    def __init__(self, cond_dim, resolution, depth, out_channels,
                 d_model=128, nhead=4, nlayers=1, rank_r=8):
        super().__init__()
        self.D, self.H, self.W = depth, resolution, resolution
        self.r = rank_r
        self.cond_dim = cond_dim

        # tiny Transformer encoder (shared across 4 wells)
        self.in_proj = nn.Linear(1, d_model)
        self.pos = PosEnc1D(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model,
            dropout=0.1, batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.to_alpha = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, rank_r))

        # separable bases a_d ⊗ a_h ⊗ a_w → [r, D, H, W]
        self.a_d = nn.Parameter(torch.randn(rank_r, depth)      / math.sqrt(depth))
        self.a_h = nn.Parameter(torch.randn(rank_r, resolution)  / math.sqrt(resolution))
        self.a_w = nn.Parameter(torch.randn(rank_r, resolution)  / math.sqrt(resolution))

        # fuse 4 per-well maps → out_channels
        self.fuse = nn.Conv3d(4, out_channels, kernel_size=1)

    def _bases(self, device):
        return (self.a_d[:, :, None, None] *
                self.a_h[:, None, :, None] *
                self.a_w[:, None, None, :]).to(device=device)  # [r,D,H,W]

    def forward(self, cond):  # cond: [B,216] or [B,4,54]
        if cond.dim() == 2:
            B = cond.size(0)
            assert cond.size(1) == self.cond_dim, "cond must be 216 if 2D"
            cond = cond.view(B, 4, 54)
        else:
            assert cond.size(1) == 4 and cond.size(2) == 54, "cond must be [B,4,54]"
            B = cond.size(0)

        # shared per-well Transformer
        x = cond.reshape(B*4, 54, 1)         # [B*4, T, 1]
        tok = self.in_proj(x)                # [B*4, T, d_model]
        tok = self.pos(tok)
        tok = self.encoder(tok)              # [B*4, T, d_model]
        feat = tok.mean(dim=1)               # [B*4, d_model]  (mean over time)

        alpha = self.to_alpha(feat).reshape(B, 4, self.r)  # [B,4,r]
        basis = self._bases(alpha.device)                  # [r,D,H,W]

        # per-well low-rank 3D maps, then fuse wells
        cond_maps = torch.einsum('bmr,rdhw->bmdhw', alpha, basis)  # [B,4,D,H,W]
        cond_emb  = self.fuse(cond_maps)                           # [B,out,D,H,W]
        return F.gelu(cond_emb)



class Upsample3D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = nn.Conv3d(in_channels,
                                  in_channels,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  padding_mode='circular')

    def forward(self, x):
        # x: [B, C, D, W, H]
        x = F.interpolate(x, scale_factor=(1.0, 2.0, 2.0), mode="nearest")  # 支持 3D 上采样
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample3D(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=(1,3,3),
                                        stride=(1, 2, 2),
                                        padding=(0,1,1),
                                        padding_mode='circular')

    def forward(self, x):
        if self.with_conv:
            # pad = (0, 1, 0, 1, 0, 1)
            # pad = (1, 1, 1, 1, 0, 0)
            # x = F.pad(x, pad, mode="circular")
            x = self.conv(x)
        else:
            x = F.avg_pool3d(x, kernel_size=(1,2,2), stride=(1,2,2))
        return x



class ResnetBlock3D(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, temb_channels, conv_shortcut=False, dropout=0.1):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, 1, 1, padding_mode='circular')

        self.temb_proj = nn.Linear(temb_channels, out_channels)

        self.norm2 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, 1, 1, padding_mode='circular')

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = nn.Conv3d(in_channels, out_channels, 3, 1, 1, padding_mode='circular')
            else:
                self.nin_shortcut = nn.Conv3d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x, temb):
        h = self.norm1(x)
        h = nonlinearity(h)
        h = self.conv1(h)
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None, None]
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h

class AttnBlock3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv3d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv3d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)


        b, c, d, h, w = q.shape
        q = q.reshape(b, c, d*h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, d*h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=-1)

        v = v.reshape(b, c, d*h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, d, h, w)

        h_ = self.proj_out(h_)

        return x+h_




class CrossAttnFusion3D(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.q = nn.Conv3d(in_channels, in_channels, 1)
        self.k = nn.Conv3d(in_channels, in_channels, 1)
        self.v = nn.Conv3d(in_channels, in_channels, 1)
        self.proj = nn.Conv3d(in_channels, in_channels, 1)

    def forward(self, x, cond_emb):
        B, C, D, H, W = x.shape
        HW = D * H * W

        q = self.q(x).reshape(B, C, HW).permute(0, 2, 1)   # [B, HW, C]
        k = self.k(cond_emb).reshape(B, C, HW)             # [B, C, HW]
        v = self.v(cond_emb).reshape(B, C, HW).permute(0, 2, 1)  # [B, HW, C]

        attn = torch.softmax(torch.bmm(q, k) / (C ** 0.5), dim=-1)  # [B, HW, HW]
        out = torch.bmm(attn, v)  # [B, HW, C]
        out = out.permute(0, 2, 1).reshape(B, C, D, H, W)

        return self.proj(out) + x

class UNetCond3D(nn.Module):
    def __init__(self, cond_dim=216, base_ch=64):
        super().__init__()
        self.base_ch = base_ch
        self.resolution = 20
        self.depth = 5
        self.temb_ch = self.base_ch * 4
        in_channels = 1
        out_channels = 1
        attn_resolutions = (20,)
        ch_mult = (1, 1, 2)
        resamp_with_conv = True
        self.num_res_blocks = 1
        self.num_resolutions = len(ch_mult)

        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            nn.Linear(self.base_ch,
                      self.temb_ch),
            nn.Linear(self.temb_ch,
                      self.temb_ch),
        ])


        self.flow_embed = Flow_Embed_3D(cond_dim=cond_dim,
                                     resolution=self.resolution,
                                     depth=self.depth,
                                     out_channels=self.base_ch,
                                     d_model=96, nlayers=1, rank_r=4
                                     )

        self.fusion = CrossAttnFusion3D(self.base_ch)
        self.cond_down1 = nn.Conv3d(base_ch, base_ch, 3, (1, 2, 2), 1, padding_mode='circular')  # 20→10
        self.cond_down2 = nn.Sequential(
            nn.Conv3d(base_ch, base_ch * 2, 3, (1, 2, 2), 1, padding_mode='circular'),  # 20→10
            nn.Conv3d(base_ch * 2, base_ch * 2, 3, (1, 2, 2), 1, padding_mode='circular')  # 10→5
        )

        # downsampling
        self.conv_in = torch.nn.Conv3d(in_channels,
                                       self.base_ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1, padding_mode='circular'
                                       )

        self.combine_conv = torch.nn.Conv3d(self.base_ch*2, self.base_ch, kernel_size=1, stride=1, padding=0)

        curr_res = self.resolution
        in_ch_mult = (1,) + ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = base_ch * in_ch_mult[i_level]
            block_out = base_ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock3D(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=0.1))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample3D(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=0.1)
        self.mid.attn_1 = AttnBlock3D(block_in)
        self.mid.block_2 = ResnetBlock3D(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=0.1)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = base_ch * ch_mult[i_level]
            skip_in = base_ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = base_ch * in_ch_mult[i_level]
                block.append(ResnetBlock3D(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=0.1))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock3D(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level !=0:
                up.upsample = Upsample3D(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv3d(block_in,
                                        out_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        padding_mode='circular'
                                        )


    def forward(self, x, t, cond=None):
        assert x.shape[2] == self.depth
        assert x.shape[3] == x.shape[4] == self.resolution

        # time embedding
        temb = get_timestep_embedding(t, self.base_ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        x = self.conv_in(x)
        if cond is not None:
            cond_emb = self.flow_embed(cond)
        else:
            cond_emb = torch.zeros_like(x)
        x = self.fusion(x, cond_emb)

        hs = [x]

        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
                if i_level==0 and cond is not None:
                    hs[-1] = hs[-1] +self.cond_down1(cond_emb)

        # middle
        h = hs[-1]
        if cond is not None:
            h = h + self.cond_down2(cond_emb)
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


