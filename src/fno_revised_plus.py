import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flow_matching import RealReservoirDataset3D, normalize_and_create_dataloader



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NX, NY, NZ = 20, 20, 5
WELLS = [(0, 0), (19, 0), (0, 19), (19, 19), (9, 9)]
N_WELLS_USED = 4
N_T = 54


class SpectralConv3d(nn.Module):
    def __init__(self, in_c, out_c, m1, m2, m3):
        super().__init__()
        scale = 1 / (in_c * out_c)
        self.in_c = in_c
        self.out_c = out_c
        self.m1 = m1
        self.m2 = m2
        self.m3 = m3

        def w():
            return nn.Parameter(scale * torch.randn(in_c, out_c, m1, m2, m3, dtype=torch.cfloat))

        self.w_pp = w()
        self.w_mp = w()
        self.w_pm = w()
        self.w_mm = w()

    def _mul(self, a, w):
        return torch.einsum("bixyz,ioxyz->boxyz", a, w)

    def forward(self, x):
        B, C, X, Y, Z = x.shape
        Zf = Z // 2 + 1

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        K1, K2, K3 = min(self.m1, X), min(self.m2, Y), min(self.m3, Zf)

        out_ft = torch.zeros(B, self.out_c, X, Y, Zf, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :K1, :K2, :K3]   = self._mul(x_ft[:, :, :K1, :K2, :K3],    self.w_pp[:, :, :K1, :K2, :K3])
        out_ft[:, :, -K1:, :K2, :K3]  = self._mul(x_ft[:, :, -K1:, :K2, :K3],   self.w_mp[:, :, :K1, :K2, :K3])
        out_ft[:, :, :K1, -K2:, :K3]  = self._mul(x_ft[:, :, :K1, -K2:, :K3],   self.w_pm[:, :, :K1, :K2, :K3])
        out_ft[:, :, -K1:, -K2:, :K3] = self._mul(x_ft[:, :, -K1:, -K2:, :K3],  self.w_mm[:, :, :K1, :K2, :K3])

        return torch.fft.irfftn(out_ft, s=(X, Y, Z), dim=(-3, -2, -1))


class FNO3D(nn.Module):
    def __init__(self, c_in=8, width=96, modes=(12, 12, 6), layers=6, p_drop=0.05):
        super().__init__()
        self.fc0 = nn.Conv3d(c_in, width, 1)
        self.mods = nn.ModuleList([SpectralConv3d(width, width, *modes) for _ in range(layers)])
        self.ws   = nn.ModuleList([nn.Conv3d(width, width, 1) for _ in range(layers)])
        self.gn   = nn.ModuleList([
            nn.GroupNorm(num_groups=min(8, max(1, width // 2)), num_channels=width)
            for _ in range(layers)
        ])
        self.drop = nn.Dropout3d(p=p_drop)

    def forward(self, x):
        x = self.fc0(x)
        for sc, w, gn in zip(self.mods, self.ws, self.gn):
            y = sc(x) + w(x)
            y = F.silu(gn(y))
            x = x + self.drop(y)   # residual
        return x



class WellPoolingHeadV2(nn.Module):
    def __init__(self, c_lat=96, n_wells=4, n_t=54, hidden=256, p_drop=0.05):
        super().__init__()
        self.nw = n_wells
        self.n_t = n_t

        self.attn = nn.ModuleList([
            nn.Conv3d(c_lat, 1, kernel_size=1)
            for _ in range(n_wells)
        ])

        self.pre_norm = nn.LayerNorm(2 * c_lat)

        self.base = nn.Sequential(
            nn.Linear(2 * c_lat, hidden),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_t),
        )

        self.adapter = nn.ModuleList([
            nn.Sequential(
                nn.Linear(n_t, n_t),
                nn.SiLU(),
                nn.Linear(n_t, n_t),
            )
            for _ in range(n_wells)
        ])

        
        self.temporal = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.SiLU(),
            nn.Dropout(p_drop),
            nn.Conv1d(16, 1, kernel_size=3, padding=1),
        )

        self.refine_scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, feats, wmask):

        B, C, X, Y, Z = feats.shape
        outs = []

        for w in range(self.nw):
            m = wmask[:, w:w+1]  # [B,1,X,Y,Z]

            logits = self.attn[w](feats)  # [B,1,X,Y,Z]
            logits = logits.masked_fill(m == 0, float("-inf"))
            attn = torch.softmax(logits.view(B, 1, -1), dim=-1).view(B, 1, X, Y, Z)
            pooled_attn = (feats * attn).sum(dim=(2, 3, 4))  # [B,C]

            den = m.sum(dim=(2, 3, 4)).clamp_min(1.0)        # [B,1]
            pooled_mean = (feats * m).sum(dim=(2, 3, 4)) / den  # [B,C]

            pooled = torch.cat([pooled_attn, pooled_mean], dim=1)  # [B,2C]
            pooled = self.pre_norm(pooled)

            y = self.base(pooled)   # [B,T]

            y = y + 0.3 * self.adapter[w](y)

            y1 = y.unsqueeze(1)  # [B,1,T]
            dy = self.temporal(y1).squeeze(1)  # [B,T]
            y = y + self.refine_scale * dy

            outs.append(y)

        return torch.stack(outs, dim=1)  # [B, n_wells, T]



class FNO3D_Rates(nn.Module):
    def __init__(self, width=96, modes=(12, 12, 6), layers=6, p_drop=0.05):
        super().__init__()

        # coords: [3, NX, NY, NZ]
        x = np.linspace(0, 1, NX, dtype=np.float32).reshape(NX, 1, 1)
        y = np.linspace(0, 1, NY, dtype=np.float32).reshape(1, NY, 1)
        z = np.linspace(0, 1, NZ, dtype=np.float32).reshape(1, 1, NZ)
        Xc = np.broadcast_to(x, (NX, NY, NZ))
        Yc = np.broadcast_to(y, (NX, NY, NZ))
        Zc = np.broadcast_to(z, (NX, NY, NZ))
        coords = np.stack([Xc, Yc, Zc], axis=0).astype(np.float32)
        self.register_buffer("coords", torch.from_numpy(coords))

        wmsk_list = []
        for (i, j) in WELLS[:N_WELLS_USED]:
            m = np.zeros((NX, NY, NZ), np.float32)
            m[i, j, :] = 1.0
            wmsk_list.append(m)
        WMSK = np.stack(wmsk_list, axis=0).astype(np.float32)  # [4,NX,NY,NZ]
        self.register_buffer("wmask", torch.from_numpy(WMSK))

        self.trunk = FNO3D(
            c_in=1 + 3 + N_WELLS_USED,
            width=width,
            modes=modes,
            layers=layers,
            p_drop=p_drop
        )

        self.head = WellPoolingHeadV2(
            c_lat=width,
            n_wells=N_WELLS_USED,
            n_t=N_T,
            hidden=256,
            p_drop=p_drop
        )

    def forward(self, perm):
        B = perm.size(0)

        if perm.dim() == 2:
            k = perm.view(B, 1, NZ, NY, NX)
        else:
            k = perm

        k = k.permute(0, 1, 4, 3, 2)  # -> [B,1,NX,NY,NZ]

        coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)   # [B,3,NX,NY,NZ]
        wmask  = self.wmask.unsqueeze(0).expand(B, -1, -1, -1, -1)    # [B,4,NX,NY,NZ]

        feats = torch.cat([k, coords, wmask], dim=1)                  # [B, 1+3+4, NX,NY,NZ]
        z = self.trunk(feats)                                         # [B,width,NX,NY,NZ]
        qhat = self.head(z, wmask)                                    # [B,4,54]
        return qhat



def rel_l2(pred, tgt):
    num = torch.linalg.norm(pred - tgt, dim=(1, 2))
    den = torch.linalg.norm(tgt, dim=(1, 2))
    return (num / (den))

def l2_mse_loss(pred_norm, gt_norm):
    return torch.mean((pred_norm - gt_norm) ** 2)

def sigma_inv_mse_loss(pred_norm, gt_norm, flow_mean, flow_std, eta_noise=0.45):
    device = pred_norm.device
    flow_mean = flow_mean.to(device).view(1, 4, 1)
    flow_std  = flow_std.to(device).view(1, 4, 1)

    pred_phys = pred_norm * flow_std + flow_mean
    gt_phys   = gt_norm   * flow_std + flow_mean

    sigma_std_well = eta_noise * torch.abs(gt_phys[:, :, 0])      # [B,4]
    sigma_inv = 1.0 / (sigma_std_well.unsqueeze(-1) ** 2)

    diff2 = (pred_phys - gt_phys) ** 2                            # [B,4,54]
    return torch.mean(diff2 * sigma_inv)



def evaluate(model, loader):
    model.eval()
    total = 0.0
    total_n = 0
    with torch.no_grad():
        for perm, flow in loader:
            perm = perm.to(DEVICE)
            flow = flow.to(DEVICE)

            pred = model(perm)
            q = flow.view(pred.size(0), 4, 54)

            rel = rel_l2(pred, q)
            total += rel.sum().item()
            total_n += len(rel)
    return total / total_n


@torch.no_grad()
def eval_fno_physical(model, loader, flow_mean, flow_std, device=DEVICE):
    model.eval()

    flow_mean = flow_mean.to(device).view(1, 4, 1)  # [1,4,1]
    flow_std  = flow_std.to(device).view(1, 4, 1)   # [1,4,1]

    total_rel = 0.0
    total_n = 0

    for perm, flow_norm in loader:
        perm = perm.to(device)
        flow_norm = flow_norm.to(device)    # [B,216]

        B = flow_norm.size(0)

        q_norm = flow_norm.view(B, 4, 54)
        qhat_norm = model(perm)             # [B,4,54]
        q_phys = q_norm * flow_std + flow_mean
        qhat_phys = qhat_norm * flow_std + flow_mean
        rel = rel_l2(qhat_phys, q_phys)

        total_rel += rel.sum().item()
        total_n += B

    return total_rel / max(total_n, 1)


def train_fno(model, train_loader, val_loader, epochs=200, lr=2e-3, save_dir="ckpt",
              alpha=0.5, eta_noise=0.4):
    os.makedirs(save_dir, exist_ok=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4, foreach=False, fused=False)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    best_val = float("inf")
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        train_obj_sum = 0.0
        train_n = 0

        for perm, flow in train_loader:
            perm = perm.to(DEVICE)
            flow = flow.to(DEVICE)

            pred = model(perm)                           # [B,4,54]
            q = flow.view(pred.size(0), 4, 54)           # [B,4,54]

            loss_mse   = l2_mse_loss(pred, q)
            loss_sigma = sigma_inv_mse_loss(pred, q, flow_mean, flow_std, eta_noise=eta_noise)
            loss = alpha * loss_mse + (1 - alpha) * loss_sigma

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_obj_sum += loss.item() * perm.size(0)
            train_n += perm.size(0)

        train_obj = train_obj_sum / train_n
        model.eval()
        val_obj_sum = 0.0
        val_n = 0
        val_mse_sum = 0.0
        val_sigma_sum = 0.0

        with torch.no_grad():
            for perm, flow in val_loader:
                perm = perm.to(DEVICE)
                flow = flow.to(DEVICE)

                pred = model(perm)
                q = flow.view(pred.size(0), 4, 54)

                v_mse   = l2_mse_loss(pred, q)
                v_sigma = sigma_inv_mse_loss(pred, q, flow_mean, flow_std, eta_noise=eta_noise)
                v_obj   = alpha * v_mse + (1 - alpha) * v_sigma

                B = pred.size(0)
                val_obj_sum += v_obj.item() * B
                val_mse_sum += v_mse.item() * B
                val_sigma_sum += v_sigma.item() * B
                val_n += B

        val_obj = val_obj_sum / val_n
        val_mse = val_mse_sum / val_n
        val_sigma = val_sigma_sum / val_n

        scheduler.step()
        current_lr = opt.param_groups[0]['lr']

        if val_obj < best_val:
            best_val = val_obj
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save({
                "state_dict": best_state,
                "epoch": ep,
                "best_val": best_val,
                "optimizer": opt.state_dict(),
                "alpha": alpha,
                "eta_noise": eta_noise
            }, os.path.join(save_dir, "fno3d_revised_plus_v2.pt"))
            tag = " <-- BEST"
        else:
            tag = ""

        print(
            f"[Epoch {ep:03d}] lr={current_lr:.2e} | "
            f"Train obj={train_obj:.6f} | Val obj={val_obj:.6f} "
            f"(Val MSE={val_mse:.6f}, Val Sigma={val_sigma:.6f}){tag}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model



def compute_individual_errors(model, loader, flow_mean, flow_std, device=DEVICE):
    model.eval()

    flow_mean = flow_mean.to(device).view(1, 4, 1)
    flow_std  = flow_std.to(device).view(1, 4, 1)

    all_errors = []

    with torch.no_grad():
        for perm, flow_norm in loader:
            perm = perm.to(device)
            flow_norm = flow_norm.to(device)
            B = flow_norm.size(0)

            q_norm_gt = flow_norm.view(B, 4, 54)
            q_norm_pred = model(perm)

            q_phys_gt = q_norm_gt * flow_std + flow_mean
            q_phys_pred = q_norm_pred * flow_std + flow_mean

            eta_noise = 0.065
            sigma_std_well = eta_noise * torch.abs(q_phys_gt[:, :, 0])
            sigma_std = sigma_std_well.unsqueeze(-1)
            sigma_inv = 1.0 / (sigma_std ** 2)

            diff = q_phys_pred - q_phys_gt
            diff_norm = torch.sqrt(torch.sum(diff ** 2 * sigma_inv, dim=(1, 2)))
            gt_norm = torch.sqrt(torch.sum(q_phys_gt ** 2 * sigma_inv, dim=(1, 2)))
            rel_errs = diff_norm / gt_norm

            all_errors.append(rel_errs.cpu().numpy())

    return np.concatenate(all_errors)

def compute_individual_errors_l2(model, loader, flow_mean, flow_std, device=DEVICE):
    model.eval()

    flow_mean = flow_mean.to(device).view(1, 4, 1)
    flow_std  = flow_std.to(device).view(1, 4, 1)

    all_errors = []

    with torch.no_grad():
        for perm, flow_norm in loader:
            perm = perm.to(device)
            flow_norm = flow_norm.to(device)
            B = flow_norm.size(0)

            q_norm_gt   = flow_norm.view(B, 4, 54)
            q_norm_pred = model(perm)                 # [B, 4, 54]

            q_phys_gt   = q_norm_gt   * flow_std + flow_mean
            q_phys_pred = q_norm_pred * flow_std + flow_mean

            diff_norm = torch.norm(q_phys_pred - q_phys_gt, p=2, dim=(1, 2))
            gt_norm   = torch.norm(q_phys_gt, p=2, dim=(1, 2))

            rel_errs = diff_norm / gt_norm            # [B]
            all_errors.append(rel_errs.cpu().numpy())

    return np.concatenate(all_errors)


def get_data_and_predict(target_idx, loader, model, f_mean, f_std):
    current_idx = 0
    found = False
    target_perm = None
    target_flow_norm_gt = None

    for perm_batch, flow_norm_batch in loader:
        batch_size = perm_batch.size(0)
        if current_idx + batch_size > target_idx:
            local_idx = target_idx - current_idx
            target_perm = perm_batch[local_idx].unsqueeze(0).to(DEVICE)
            target_flow_norm_gt = flow_norm_batch[local_idx].unsqueeze(0).to(DEVICE)
            found = True
            break
        current_idx += batch_size

    if not found:
        raise ValueError(f"Sample {target_idx} not found")

    model.eval()
    with torch.no_grad():
        pred_norm = model(target_perm)
        gt_norm = target_flow_norm_gt.view(1, 4, 54)

        fm = f_mean.to(DEVICE).view(1, 4, 1)
        fs = f_std.to(DEVICE).view(1, 4, 1)

        pred_phys = pred_norm * fs + fm
        gt_phys = gt_norm * fs + fm

        return gt_phys.cpu().numpy().reshape(4, 54), pred_phys.cpu().numpy().reshape(4, 54)


if __name__ == "__main__":

    perm_file = ".../raw_data/perms3d_trans.txt"  # shape [N,2000]
    flow_file = ".../raw_data/subqwt.txt"  # shape [N,216]


    perm_np = np.loadtxt(perm_file)  # [N,2000]
    flow_np = np.loadtxt(flow_file)  # [N,216]

    N = perm_np.shape[0]
    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    n_test = N - n_train - n_val

    train_perm_np = perm_np[:n_train]
    train_flow_np = flow_np[:n_train]

    val_perm_np = perm_np[n_train:n_train + n_val]
    val_flow_np = flow_np[n_train:n_train + n_val]

    test_perm_np = perm_np[n_train + n_val:]
    test_flow_np = flow_np[n_train + n_val:]

    os.makedirs(".../raw_data", exist_ok=True)
    np.savetxt(".../raw_data/train_perm_tmp.txt", train_perm_np)
    np.savetxt(".../raw_data/train_flow_tmp.txt", train_flow_np)

    full_train_ds = RealReservoirDataset3D(
        ".../raw_data/train_perm_tmp.txt",
        ".../raw_data/train_flow_tmp.txt",
        device="cpu",
    )
    train_norm_params = full_train_ds.get_normalization_params()
    flow_mean = train_norm_params["flow_mean"]  # [4,1]
    flow_std = train_norm_params["flow_std"]  # [4,1]


    train_loader, _, _ = normalize_and_create_dataloader(
        train_perm_np, train_flow_np, train_norm_params,
        batch_size=32, shuffle=True, device=DEVICE
    )
    val_loader, _, _ = normalize_and_create_dataloader(
        val_perm_np, val_flow_np, train_norm_params,
        batch_size=32, shuffle=False, device=DEVICE
    )
    test_loader, _, _ = normalize_and_create_dataloader(
        test_perm_np, test_flow_np, train_norm_params,
        batch_size=32, shuffle=False, device=DEVICE
    )

    model = FNO3D_Rates().to(DEVICE)

    # model = train_fno(model, train_loader, val_loader, epochs=400)
    # print("FNO training finished.")

    # test_relL2 = evaluate(model, test_loader)

    ckpt_path = ".../saved_model/fno3d_revised_plus.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        state_dict = ckpt["state_dict"]
        msg = model.load_state_dict(state_dict, strict=False)

        print(f"Loaded checkpoint from {ckpt_path}")

    

    errors = compute_individual_errors(
        model, test_loader,
        flow_mean,
        flow_std,
        device=DEVICE
    )

    mean_err = np.mean(errors)








