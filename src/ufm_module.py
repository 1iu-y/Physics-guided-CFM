import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(timesteps, embedding_dim):
    if timesteps.ndim != 1:
        raise ValueError(f"timesteps must have shape [B], got {tuple(timesteps.shape)}")
    half_dim = embedding_dim // 2
    frequencies = math.log(10000.0) / (half_dim - 1)
    frequencies = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device)
        * -frequencies
    )
    embedding = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(embedding), torch.cos(embedding)], dim=1)
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1, 0, 0))
    return embedding


def swish(x):
    return x * torch.sigmoid(x)


def group_norm(channels):
    return nn.GroupNorm(num_groups=8, num_channels=channels, eps=1e-6, affine=True)


def fixed_spatial_features(depth, height, width):
    z = torch.linspace(-1.0, 1.0, depth)
    y = torch.linspace(-1.0, 1.0, height)
    x = torch.linspace(-1.0, 1.0, width)
    zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
    coordinates = (zz, yy, xx)
    features = [zz, yy, xx]
    for frequency in (1.0, 2.0):
        for coordinate in coordinates:
            features.append(torch.sin(math.pi * frequency * coordinate))
            features.append(torch.cos(math.pi * frequency * coordinate))
    return torch.stack(features, dim=0).unsqueeze(0)


class AntiAliasedDownsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = F.avg_pool3d(x, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        return self.conv(x)


class SmoothUpsample3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
        )

    def forward(self, x):
        x = F.interpolate(
            x,
            scale_factor=(1.0, 2.0, 2.0),
            mode="trilinear",
            align_corners=False,
        )
        return self.conv(x)


class ResnetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.norm1 = group_norm(in_channels)
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            3,
            1,
            1,
            padding_mode="reflect",
        )
        self.time_proj = nn.Linear(time_channels, out_channels)
        self.norm2 = group_norm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            3,
            1,
            1,
            padding_mode="reflect",
        )
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(in_channels, out_channels, 1)

    def forward(self, x, time_features):
        h = self.conv1(swish(self.norm1(x)))
        h = h + self.time_proj(swish(time_features))[:, :, None, None, None]
        h = self.conv2(self.dropout(swish(self.norm2(h))))
        residual = self.shortcut(x) if self.shortcut is not None else x
        return residual + h


class SelfAttention3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = group_norm(channels)
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        batch, channels, depth, height, width = h.shape
        tokens = depth * height * width
        q = self.q(h).reshape(batch, channels, tokens).transpose(1, 2).contiguous()
        k = self.k(h).reshape(batch, channels, tokens).transpose(1, 2).contiguous()
        v = self.v(h).reshape(batch, channels, tokens).transpose(1, 2).contiguous()
        attended = F.scaled_dot_product_attention(
            q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1), dropout_p=0.0
        ).squeeze(1)
        attended = attended.transpose(1, 2).reshape(
            batch, channels, depth, height, width
        )
        return x + self.proj_out(attended)


class SpatialCFMBackboneUFM(nn.Module):
    """CFM-like U-Net with fixed spatial encoding and no data side-input path."""

    def __init__(
        self,
        base_channels=64,
        resolution=20,
        depth=5,
        dropout=0.1,
    ):
        super().__init__()
        self.base_channels = base_channels
        self.resolution = resolution
        self.depth = depth
        self.time_channels = base_channels * 4
        self.channel_multipliers = (1, 1, 2)
        self.num_res_blocks = 1

        spatial_features = fixed_spatial_features(depth, resolution, resolution)
        self.register_buffer("spatial_features", spatial_features, persistent=True)
        spatial_channels = spatial_features.shape[1]

        self.time_dense = nn.ModuleList(
            [
                nn.Linear(base_channels, self.time_channels),
                nn.Linear(self.time_channels, self.time_channels),
            ]
        )
        self.conv_in = nn.Conv3d(
            1,
            base_channels,
            3,
            1,
            1,
            padding_mode="reflect",
        )

        current_resolution = resolution
        input_multipliers = (1,) + self.channel_multipliers
        self.position_projections = nn.ModuleList()
        self.down = nn.ModuleList()
        block_channels = base_channels
        for level in range(len(self.channel_multipliers)):
            level_module = nn.Module()
            in_channels = base_channels * input_multipliers[level]
            out_channels = base_channels * self.channel_multipliers[level]
            self.position_projections.append(
                nn.Conv3d(spatial_channels, in_channels, kernel_size=1)
            )
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(
                    ResnetBlock3D(
                        in_channels,
                        out_channels,
                        self.time_channels,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels
                if current_resolution == resolution:
                    attentions.append(SelfAttention3D(in_channels))
            level_module.blocks = blocks
            level_module.attentions = attentions
            if level != len(self.channel_multipliers) - 1:
                level_module.downsample = AntiAliasedDownsample3D(in_channels)
                current_resolution //= 2
            self.down.append(level_module)
            block_channels = in_channels

        self.mid_position = nn.Conv3d(spatial_channels, block_channels, 1)
        self.mid_block1 = ResnetBlock3D(
            block_channels,
            block_channels,
            self.time_channels,
            dropout=dropout,
        )
        self.mid_attention = SelfAttention3D(block_channels)
        self.mid_block2 = ResnetBlock3D(
            block_channels,
            block_channels,
            self.time_channels,
            dropout=dropout,
        )

        self.up = nn.ModuleList()
        for level in reversed(range(len(self.channel_multipliers))):
            level_module = nn.Module()
            out_channels = base_channels * self.channel_multipliers[level]
            skip_channels = out_channels
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for block_index in range(self.num_res_blocks + 1):
                if block_index == self.num_res_blocks:
                    skip_channels = base_channels * input_multipliers[level]
                blocks.append(
                    ResnetBlock3D(
                        block_channels + skip_channels,
                        out_channels,
                        self.time_channels,
                        dropout=dropout,
                    )
                )
                block_channels = out_channels
                if current_resolution == resolution:
                    attentions.append(SelfAttention3D(block_channels))
            level_module.blocks = blocks
            level_module.attentions = attentions
            if level != 0:
                level_module.upsample = SmoothUpsample3D(block_channels)
                current_resolution *= 2
            self.up.insert(0, level_module)

        self.conv_out = nn.Conv3d(
            block_channels,
            1,
            3,
            1,
            1,
            padding_mode="reflect",
        )
        self.out_gain = nn.Parameter(torch.tensor(1.0))

    def resized_spatial_features(self, spatial_size, batch_size, dtype):
        features = self.spatial_features.to(dtype=dtype)
        if features.shape[2:] != spatial_size:
            features = F.interpolate(
                features,
                size=spatial_size,
                mode="trilinear",
                align_corners=False,
            )
        return features.expand(batch_size, -1, -1, -1, -1)

    def forward(self, x, t):
        if x.shape[1:] != (1, self.depth, self.resolution, self.resolution):
            raise ValueError(
                "Expected x with shape "
                f"[B, 1, {self.depth}, {self.resolution}, {self.resolution}], "
                f"got {tuple(x.shape)}"
            )
        if t.ndim == 2 and t.shape[1] == 1:
            t = t[:, 0]

        time_features = timestep_embedding(t, self.base_channels)
        time_features = self.time_dense[1](swish(self.time_dense[0](time_features)))

        hidden_states = [self.conv_in(x)]
        for level, level_module in enumerate(self.down):
            position = self.resized_spatial_features(
                hidden_states[-1].shape[2:], x.shape[0], hidden_states[-1].dtype
            )
            hidden_states[-1] = hidden_states[-1] + self.position_projections[level](
                position
            )
            for block_index, block in enumerate(level_module.blocks):
                h = block(hidden_states[-1], time_features)
                if len(level_module.attentions) > 0:
                    h = level_module.attentions[block_index](h)
                hidden_states.append(h)
            if level != len(self.down) - 1:
                hidden_states.append(level_module.downsample(hidden_states[-1]))

        position = self.resized_spatial_features(
            hidden_states[-1].shape[2:], x.shape[0], hidden_states[-1].dtype
        )
        h = hidden_states[-1] + self.mid_position(position)
        h = self.mid_block1(h, time_features)
        h = self.mid_attention(h)
        h = self.mid_block2(h, time_features)

        for level in reversed(range(len(self.up))):
            level_module = self.up[level]
            for block_index, block in enumerate(level_module.blocks):
                h = block(torch.cat([h, hidden_states.pop()], dim=1), time_features)
                if len(level_module.attentions) > 0:
                    h = level_module.attentions[block_index](h)
            if level != 0:
                h = level_module.upsample(h)

        if hidden_states:
            raise RuntimeError("U-Net skip stack was not fully consumed")
        return self.out_gain * self.conv_out(swish(h))
