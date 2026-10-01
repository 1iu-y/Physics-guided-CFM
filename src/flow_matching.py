import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
import os
import sys
src_path = ".../src"
if src_path not in sys.path:
    sys.path.append(src_path)
from fno_revised_plus import FNO3D_Rates
import argparse
import importlib
import inspect


class RealReservoirDataset3D(Dataset):
    def __init__(self, train_perm_file, train_flow_file, device='cuda'):
        super().__init__()
        self.perm_data = np.loadtxt(train_perm_file)  # Expected shape: [num_samples, 2000]
        self.flow_data = np.loadtxt(train_flow_file)  # Expected shape: [num_samples, 216]

        assert self.perm_data.shape[
                   1] == 2000, f"Perm data should have 2000 elements per sample, got {self.perm_data.shape[1]}"
        assert self.flow_data.shape[
                   1] == 216, f"Flow data should have 216 elements per sample, got {self.flow_data.shape[1]}"
        assert len(self.perm_data) == len(self.flow_data), "Number of samples in perm and flow data do not match"

        self.num_samples = len(self.perm_data)
        self.device = device
        self.num_wells = 4
        self.num_timesteps = 54

        self.perm_data = torch.from_numpy(self.perm_data).float().view(-1, 1, 5, 20, 20).to(self.device)
        self.flow_data = torch.from_numpy(self.flow_data).float().to(self.device)

        self.prepare_data()
        self.preprocess_data()

    def prepare_data(self):
        perm_flat = self.perm_data.view(self.num_samples, -1)  # [num_samples, 2000]
        self.perm_mean = perm_flat.mean()
        self.perm_std = perm_flat.std()

        flow_reshaped = self.flow_data.view(self.num_samples, self.num_wells,
                                            self.num_timesteps)  # [num_samples, 4, 54]
        self.flow_mean = flow_reshaped.mean(dim=(0, 2)).view(self.num_wells, 1)  # [4, 1] (mean per well)
        self.flow_std = flow_reshaped.std(dim=(0, 2)).view(self.num_wells, 1)  # [4, 1] (std per well)

    def preprocess_data(self):
        perm_flat = self.perm_data.view(self.num_samples, -1)
        norm_perm_flat = (perm_flat - self.perm_mean) / self.perm_std
        self.perm_data = norm_perm_flat.view(-1, 1, 5, 20, 20)

        flow_reshaped = self.flow_data.view(self.num_samples, self.num_wells, self.num_timesteps)
        norm_flow = (flow_reshaped - self.flow_mean) / self.flow_std
        self.flow_data = norm_flow.view(self.num_samples, -1)  # Flatten back to [num_samples, 216]

    def restore_perm(self, perm_data):
        num_samples = perm_data.shape[0]
        perm_flat = perm_data.view(num_samples, -1)
        restored_perm = (perm_flat * self.perm_std) + self.perm_mean  # Note: moved to CPU if needed externally
        restored_perm = restored_perm.view(-1, 1, 5, 20, 20)
        return restored_perm.cpu()

    def restore_flow(self, flow_data):
        num_samples = flow_data.shape[0]
        flow_reshaped = flow_data.view(num_samples, self.num_wells, self.num_timesteps)
        restored_flow = (flow_reshaped * self.flow_std) + self.flow_mean
        restored_flow = restored_flow.view(num_samples, -1)
        return restored_flow.cpu()

    def get_normalization_params(self):
        return {
            "perm_mean": self.perm_mean.cpu(),
            "perm_std": self.perm_std.cpu(),
            "flow_mean": self.flow_mean.cpu(),
            "flow_std": self.flow_std.cpu()
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.perm_data[idx], self.flow_data[idx]

def make_test_obs_loader_with_std_t0(
    obs_np,          # [N,216]
    q_sim_np,        # [N,216]
    train_params,
    eta_noise=0.065,
    batch_size=16,
    shuffle=False,
    device="cuda",
):
    obs = torch.from_numpy(obs_np).float().view(-1, 4, 54)      # [N,4,54]
    q_sim = torch.from_numpy(q_sim_np).float().view(-1, 4, 54)  # [N,4,54]

    flow_mean = train_params["flow_mean"].float().view(1, 4, 1).to(device)  # [1,4,1]
    flow_std  = train_params["flow_std"].float().view(1, 4, 1).to(device)   # [1,4,1]

    obs = obs.to(device)
    q_sim = q_sim.to(device)

    obs_norm = (obs - flow_mean) / flow_std                     # [N,4,54]
    obs_norm = obs_norm.view(-1, 216)                           # [N,216]

    std_w = eta_noise * q_sim[:, :, 0].abs()                    # [N,4]
    std = std_w[:, :, None].expand(-1, 4, 54)                   # [N,4,54]

    std_norm = (std / flow_std).view(-1, 216)                   # [N,216]

    loader = DataLoader(
        TensorDataset(obs_norm, std_norm),
        batch_size=batch_size,
        shuffle=shuffle
    )
    return loader


def normalize_and_create_dataloader(perm_np, flow_np, train_params,
                                    batch_size=16, shuffle=False, device='cuda'):
    num_samples = perm_np.shape[0]

    perm_data = torch.from_numpy(perm_np).float().view(num_samples, 1, 5, 20, 20)
    flow_data = torch.from_numpy(flow_np).float()  # [N,216]

    perm_mean = train_params["perm_mean"].to(torch.float32)   # scalar
    perm_std  = train_params["perm_std"].to(torch.float32)    # scalar
    flow_mean = train_params["flow_mean"].to(torch.float32)   # [4,1]
    flow_std  = train_params["flow_std"].to(torch.float32)    # [4,1]

    perm_flat = perm_data.view(num_samples, -1)
    norm_perm = ((perm_flat - perm_mean) / perm_std).view(num_samples, 1, 5, 20, 20)

    flow_reshaped = flow_data.view(num_samples, 4, 54)
    norm_flow = (flow_reshaped - flow_mean) / flow_std
    norm_flow = norm_flow.view(num_samples, -1)  # [N,216]

    dataset = TensorDataset(norm_perm.to(device), norm_flow.to(device))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return loader, perm_data, flow_data

class Flow(nn.Module):
    def __init__(self, model_class=None):
        super().__init__()
        
        if model_class is None:
            raise ValueError("please load your model_class (UNetCond3D / UncondFMNet)!")
        
        self.unet = model_class()
        forward_params = inspect.signature(self.unet.forward).parameters
        self.use_cond = 'cond' in forward_params
        print(f"[Flow Init] Model class: {model_class.__name__}")
        print(f"[Flow Init] Detected conditional input support: {self.use_cond}")

    def forward(self, x_t, t, cond=None):
        if len(t.shape) == 2:
            t = t.squeeze(1)
        dev = x_t.device
        
        if self.use_cond:
            if cond is None:
                raise ValueError("Model requires 'cond' but None was provided!")
            return self.unet(x_t, t.to(dev), cond.to(dev))
        else:
            return self.unet(x_t, t.to(dev))

    def step(self, x_t, t_start, t_end, cond):
        dev = x_t.device
        delta_t = (t_end - t_start).to(dev)
        t_start = t_start.view(-1).to(dev)
        t_end = t_end.view(-1).to(dev)
        delta_t = delta_t.view(-1, 1, 1, 1, 1)

        v_0 = self.forward(x_t, t_start, cond)
        x_mid = x_t + 0.5 * delta_t * v_0
        t_mid = ((t_start + t_end) / 2).to(dev)
        v_mid = self.forward(x_mid, t_mid, cond)
        return x_t + delta_t * v_mid



@torch.no_grad()
def sample_with_model(flow_model, cond, num_steps=50, device='cuda', depth=5, H=20, W=20, seed=None):
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        x = torch.randn(cond.shape[0], 1, depth, H, W, device=device, generator=g)
    else:
        x = torch.randn(cond.shape[0], 1, depth, H, W, device=device)

    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device, dtype=x.dtype)
    for i in range(num_steps):
        t0 = t_grid[i]
        t1 = t_grid[i + 1]
        x = flow_model.step(x, t0, t1, cond)
    return x

@torch.no_grad()
def sample_batch_with_guidance(
    flow_model,
    fno_model,
    cond,                 # [B,216] (normalized obs)
    num_steps=50,
    guidance_scale=0.05,
    device='cuda',
    depth=5, H=20, W=20,
    clamp_ratio=0.05,
    sigma_inv=None,       # can be [216] / [B,216] / [4,54] / [B,4,54]
    sigma_std=None,        # can be [216] / [B,216] / [4,54] / [B,4,54]
    use_guidance = True
):
    B = cond.shape[0]
    cond = cond.to(device)
    cond_fno_target = cond.view(B, 4, 54)               # [B,4,54]
    x = torch.randn(B, 1, depth, H, W, device=device)

    t_grid = torch.linspace(0.0, 1.0, num_steps + 1, device=device)
    t_final = torch.ones(B, device=device)

    # --- build sigma_inv in [B,4,54] ---
    if sigma_inv is None:
        if sigma_std is None:
            raise ValueError("Need either sigma_std (std) or sigma_inv (1/var).")
        sigma_inv = 1.0 / (sigma_std.to(device) ** 2)
    else:
        sigma_inv = sigma_inv.to(device)

    if sigma_inv.numel() == 216:
        sigma_inv_b = sigma_inv.view(1, 4, 54).expand(B, 4, 54)     # [B,4,54]
    else:
        sigma_inv_b = sigma_inv.view(B, 4, 54)                      # [B,4,54]

    for i in range(num_steps):
        t_curr = t_grid[i]
        t_next = t_grid[i + 1]
        dt = t_next - t_curr

        t_curr_b = t_curr.expand(B)
        t_next_b = t_next.expand(B)

        x_guided = x

        if use_guidance:
            current_t_val = t_curr.item()
            v0_metric = flow_model(x, t_curr_b, cond)                   # cond stays [B,216]
            flow_norm = torch.norm(v0_metric.reshape(B, -1), dim=1).mean()

            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                x_clean_est = flow_model.step(x_in, t_curr_b, t_final, cond)
                q_pred = fno_model(x_clean_est)                         # [B,4,54]

                r = q_pred - cond_fno_target                            # [B,4,54]
                loss_phy = (r * r * sigma_inv_b).sum(dim=(1, 2)).mean()
                grad = torch.autograd.grad(loss_phy, x_in)[0]

            grad_norm = torch.norm(grad.reshape(B, -1), dim=1).mean()

            if grad_norm > 1e-10:
                grad_scaled = grad * (flow_norm / grad_norm)

                effective_scale = guidance_scale * current_t_val
                correction_vel = effective_scale * grad_scaled

                correction_norm = correction_vel.reshape(B, -1).norm(dim=1).mean()
                perturbation_ratio = correction_norm / flow_norm

                if perturbation_ratio > clamp_ratio:
                    correction_vel = correction_vel * (clamp_ratio / perturbation_ratio)

                x_guided = x - correction_vel * dt

        x = flow_model.step(x_guided, t_curr_b, t_next_b, cond)

    return x





def train_flow(train_loader, val_loader, train_params,
               num_epochs=300, lr=1e-4, device='cuda',
               ckpt_path='flow_matching_revised_best.pt',
               weight_decay=1e-3,
               early_stop_patience=50,
               use_amp=True,
               inner_updates_per_batch=4,
               eta_max = 0.0,
               final_ckpt_path='flow_matching_revised_final.pt',
               model_class = None
               ):

    flow = Flow(model_class=model_class).to(device)
    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    flow_mean = train_params["flow_mean"].to(device).float()  # [4,1]
    flow_std = train_params["flow_std"].to(device).float()  # [4,1]
    mu_over_s = (flow_mean / flow_std).squeeze(-1)  # [4]

    best_val = float('inf')
    bad_epochs = 0

    def add_noise_to_cond(cond_flat):
        B = cond_flat.size(0)
        cond_r = cond_flat.view(B, 4, 54)  # Q_norm
        eta = torch.rand(B, 1, device=device) * eta_max  # U(0, eta_max)
        sigma_norm = eta * (cond_r[:, :, 0] + mu_over_s).abs()  # [B,4]
        cond_r_noisy = cond_r + sigma_norm[:, :, None] * torch.randn_like(cond_r)
        return cond_r_noisy.view(B, 216)

    for epoch in range(num_epochs):
        flow.train()
        train_loss_sum, train_count = 0.0, 0

        for x1, cond in train_loader:
            x1 = x1.to(device, non_blocking=True)
            cond = cond.to(device, non_blocking=True)
            B = x1.size(0)

            for _ in range(inner_updates_per_batch):

                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=device)            # t ~ U(0,1)
                t_ = t.view(-1, 1, 1, 1, 1)

                xt = (1 - t_) * x0 + t_ * x1
                target_v = x1 - x0

                cond_noisy = add_noise_to_cond(cond)

                optimizer.zero_grad(set_to_none=True)

                if use_amp:
                    with torch.amp.autocast(device, dtype=torch.float16):
                        pred_v = flow(xt, t, cond_noisy)
                        loss = F.mse_loss(pred_v, target_v)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred_v = flow(xt, t, cond_noisy)
                    loss = F.mse_loss(pred_v, target_v)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(flow.parameters(), 1.0)
                    optimizer.step()

                train_loss_sum += loss.item() * B
                train_count += B
        train_loss = train_loss_sum / max(train_count, 1)

        # ---------------------------
        # Validation
        # ---------------------------
        flow.eval()
        val_loss_sum,val_base_sum, val_count = 0.0, 0.0, 0

        with torch.no_grad():
            for x1, cond in val_loader:
                x1 = x1.to(device)
                B = x1.size(0)

                x0 = torch.randn_like(x1)
                t = torch.rand(B, device=device)
                t_ = t.view(-1, 1, 1, 1, 1)
                xt = (1 - t_) * x0 + t_ * x1
                target_v = x1 - x0

                cond_noisy = add_noise_to_cond(cond)

                pred_v = flow(xt, t, cond_noisy)
                loss = F.mse_loss(pred_v, target_v)
                baseline_loss = F.mse_loss(torch.zeros_like(target_v), target_v)

                val_loss_sum += loss.item() * B
                val_base_sum += baseline_loss.item() * B
                val_count += B

        val_loss = val_loss_sum / max(val_count, 1)
        val_base = val_base_sum / max(val_count, 1)

        scheduler.step()


        if val_loss < best_val - 1e-4:
            best_val = val_loss
            bad_epochs = 0
            torch.save({'model': flow.state_dict(),
                        'opt': optimizer.state_dict(),
                        'epoch': epoch,
                        'val_loss': best_val}, ckpt_path)
            print(f"[Epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} base={val_base:.6f} <-- BEST")

        else:
            bad_epochs += 1
            print(f"[Epoch {epoch:03d}] train={train_loss:.6f} val={val_loss:.6f} base={val_base:.6f} (no improve {bad_epochs})")

        if bad_epochs >= early_stop_patience:
            print("Early stopping triggered.")
            break

    torch.save({'model': flow.state_dict(),
                'opt': optimizer.state_dict(),
                'epoch': num_epochs,
                'val_loss': val_loss}, final_ckpt_path)

    print(f"Final model saved to {final_ckpt_path}")
    return flow


@torch.no_grad()
def sample_and_save_n_times_txt(
        ckpt_path,
        test_loader,
        fno_model,
        dataset,
        n_times=10,
        save_dir='saved',
        save_txt=True,
        guidance_scale=1.0,
        device='cuda',
        num_steps=50,
        depth=5, H=20, W=20,
        seed=None,
        model_class = None
):
    if save_txt and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flow = Flow(model_class=model_class).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    flow.load_state_dict(ckpt['model'])
    flow.eval()

    fno_model.eval()
    fno_model.to(device)
    for p in fno_model.parameters():
        p.requires_grad = False

    if seed is not None:
        torch.manual_seed(seed)

    global_sample_idx = 0
    total_batches = len(test_loader)
    all_denorm_samples = []

    print(f"Start sampling: {n_times} realizations per sample (Save TXT: {save_txt})...")

    for batch_idx, (cond_batch, std_batch) in enumerate(test_loader):
        B = cond_batch.shape[0]
        cond_batch = cond_batch.to(device)
        sigma_inv = 1.0 / (std_batch ** 2)
        batch_realizations = []
        for i in range(n_times):
            x_gen = sample_batch_with_guidance(
                flow_model=flow,
                fno_model=fno_model,
                cond=cond_batch,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                device=device,
                sigma_inv=sigma_inv,
                depth=depth, H=H, W=W
            )
            batch_realizations.append(x_gen.cpu())

        batch_gen_stacked = torch.stack(batch_realizations, dim=1)  # [B, n, 1, D, H, W]
        B_dim, N_dim, C_dim, D_dim, H_dim, W_dim = batch_gen_stacked.shape
        batch_gen_flat = batch_gen_stacked.view(-1, C_dim, D_dim, H_dim, W_dim)
        batch_denorm_flat = dataset.restore_perm(batch_gen_flat)
        batch_denorm = batch_denorm_flat.view(B_dim, N_dim, C_dim, D_dim, H_dim, W_dim)
        all_denorm_samples.append(batch_denorm)
        if save_txt:
            for b in range(B):
                gen_perms_tensor = batch_denorm[b]
                flat_matrix = gen_perms_tensor.view(n_times, -1).numpy()

                save_path = os.path.join(save_dir, f"sample_{global_sample_idx}.txt")
                np.savetxt(save_path, flat_matrix, fmt='%.6e', delimiter=' ')

                global_sample_idx += 1

            print(f"  Batch {batch_idx + 1}/{total_batches} sampled & saved.")
        else:
            global_sample_idx += B
            print(f"  Batch {batch_idx + 1}/{total_batches} sampled.")

    full_tensor = torch.cat(all_denorm_samples, dim=0)  # [N_total, n_times, 1, 5, 20, 20]
    print(f"\nAll done! Returns shape: {full_tensor.shape}")

    return full_tensor



@torch.no_grad()
def sample_loader_obswise_and_save_txt(
        ckpt_path,
        test_loader,
        fno_model,
        dataset,
        n_times=2000,
        chunk_size=50,
        save_dir='saved_obswise',
        save_txt=True,
        guidance_scale=1.0,
        device='cuda',
        num_steps=50,
        depth=5, H=20, W=20,
        clamp_ratio=0.05,
        use_guidance = True,
        model_class=None,
):
    if save_txt and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    flow = Flow(model_class=model_class).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    flow.load_state_dict(ckpt['model'])
    flow.eval()

    fno_model.eval()
    fno_model.to(device)
    for p in fno_model.parameters():
        p.requires_grad = False

    global_obs_idx = 0
    total_batches = len(test_loader)

    print(f"Start obs-wise sampling: {n_times} per observation (chunk_size={chunk_size})")

    for batch_idx, (cond_batch, std_batch) in enumerate(test_loader):
        B = cond_batch.shape[0]
        cond_batch = cond_batch.to(device)

        print(f"\nBatch {batch_idx+1}/{total_batches} | B={B}")

        for b in range(B):
            cond_one = cond_batch[b:b+1]          # [1, ...]
            std_one  = std_batch[b:b+1].to(device)  # [1, ...]

            sigma_inv_one = 1.0 / (std_one ** 2)  # [1, ...]

            all_gen = []
            remaining = n_times
            while remaining > 0:
                cur = min(chunk_size, remaining)
                cond_rep = cond_one.repeat(cur, *([1] * (cond_one.dim() - 1)))
                sig_rep  = sigma_inv_one.repeat(cur, *([1] * (sigma_inv_one.dim() - 1)))

                x_gen = sample_batch_with_guidance(
                    flow_model=flow,
                    fno_model=fno_model,
                    cond=cond_rep,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    device=device,
                    sigma_inv=sig_rep,
                    depth=depth, H=H, W=W,
                    clamp_ratio=clamp_ratio,
                    use_guidance=use_guidance
                )  # [cur, 1, D, H, W]

                all_gen.append(x_gen.cpu())
                remaining -= cur

            gen_stacked = torch.cat(all_gen, dim=0)          # [n_times, 1, D, H, W]
            denorm = dataset.restore_perm(gen_stacked)       # [n_times, 1, D, H, W]

            if save_txt:
                flat = denorm.view(n_times, -1).numpy()      # [n_times, D*H*W]
                save_path = os.path.join(save_dir, f"sample_{global_obs_idx}.txt")
                np.savetxt(save_path, flat, fmt='%.6e', delimiter=' ')
                print(f"  Saved obs {global_obs_idx}: {save_path}")

            global_obs_idx += 1

    print("\nAll done.")
    return


if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32


    parser = argparse.ArgumentParser()

    parser.add_argument("--flow_test", type=str, 
                        default="/home/tonglinjin/projects/flowmatching/paper_flowmatching/raw_data/subqwt_last200_noisy_6.5percent.txt")
    parser.add_argument("--ckpt_path", type=str, 
                        default="/home/tonglinjin/projects/flowmatching/paper_flowmatching/output_pt/flow_matching_best.pt")
    parser.add_argument("--save_dir", type=str, 
                        default="/home/tonglinjin/projects/flowmatching/paper_flowmatching/robust_files/saved_pgcon")
    

    parser.add_argument("--n_test", type=int, default=3, help="Number of test cases")
    parser.add_argument("--n_times", type=int, default=2000, help="Sampling times per observation")
    parser.add_argument("--chunk_size", type=int, default=50, help="Batch size for sampling")
    parser.add_argument("--eta_noise",type=float, default=0.065, help="Noise level")
    parser.add_argument("--net_module", type=str, default="module", help="Python module name (filename without .py)")
    parser.add_argument("--net_class", type=str, default="UNetCond3D", help="Class name inside the module")
    parser.add_argument("--use_guidance", type=int, default=1, help="1 for True, 0 for False")
    parser.add_argument("--guidance_scale", type=float, default=1.0, help="Scale for guidance")
    parser.add_argument("--clamp_ratio", type=float, default=0.05, help="Clamp ratio")




    args = parser.parse_args()

    print(f"\n[Config] n_test: {args.n_test}, n_times: {args.n_times}, chunk_size: {args.chunk_size}")
    print(f"[Config] Save Dir: {args.save_dir}\n")
    print(f"[Dynamic Import] Loading class '{args.net_class}' from module '{args.net_module}'...")

    try:
        module_lib = importlib.import_module(args.net_module)
    except ModuleNotFoundError:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        module_lib = importlib.import_module(args.net_module)

    NetClass = getattr(module_lib, args.net_class)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 32

    perm_file = "/home/tonglinjin/projects/flowmatching/paper_flowmatching/raw_data/perms3d_trans.txt"
    flow_file = "/home/tonglinjin/projects/flowmatching/paper_flowmatching/raw_data/subqwt.txt"
    flow_test = args.flow_test

    perm_np = np.loadtxt(perm_file)  # [N,2000]
    flow_np = np.loadtxt(flow_file)  # [N,216]

    N = perm_np.shape[0]
    n_train = int(0.8 * N)
    n_val   = int(0.1 * N)
    # n_test  = N - n_train - n_val
    n_test = args.n_test

    train_perm_np = perm_np[:n_train]
    train_flow_np = flow_np[:n_train]

    val_perm_np = perm_np[n_train:n_train + n_val]
    val_flow_np = flow_np[n_train:n_train + n_val]

    test_perm_np = perm_np[n_train + n_val:n_train + n_val + n_test]
    test_flow_np_all = np.loadtxt(flow_test)
    test_flow_obs_np = test_flow_np_all[:n_test]
    test_flow_np = flow_np[n_train + n_val:n_train + n_val + n_test]
    # test_flow_np = flow_np[n_train + n_val:]

    np.savetxt("/home/tonglinjin/projects/flowmatching/paper_flowmatching/raw_data/train_perm_tmp.txt", train_perm_np)
    np.savetxt("/home/tonglinjin/projects/flowmatching/paper_flowmatching/raw_data/train_flow_tmp.txt", train_flow_np)

    full_train_ds = RealReservoirDataset3D(
        "/home/tonglinjin/projects/flowmatching/paper_flowmatching/raw_data/train_perm_tmp.txt",
        "/home/tonglinjin/projects/flowmatching/paper_flowmatching/raw_data/train_flow_tmp.txt",
        device='cpu'
    )
    train_norm_params = full_train_ds.get_normalization_params()


    train_loader = DataLoader(
        full_train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True
    )

    val_loader, val_perm_tensor, val_flow_tensor = normalize_and_create_dataloader(
        val_perm_np, val_flow_np, train_norm_params,
        batch_size=batch_size, shuffle=False, device=device
    )

    test_loader = make_test_obs_loader_with_std_t0(
        obs_np=test_flow_obs_np, q_sim_np= test_flow_np,
        train_params=train_norm_params,eta_noise=args.eta_noise,
        batch_size=batch_size, shuffle=False , device=device
    )

    _, _, test_flow_tensor = normalize_and_create_dataloader(
        test_perm_np, test_flow_np, train_norm_params,
        batch_size=batch_size, shuffle=False, device=device
    )


    # 5) Train flow model
    # model = train_flow(
    #     train_loader,
    #     val_loader,
    #     train_params=train_norm_params,
    #     num_epochs=300,
    #     lr=1e-4,
    #     eta_max=0.2,
    #     device=device,
    #     ckpt_path='cfm_best.pt',
    #     inner_updates_per_batch=20,
    #     final_ckpt_path='cfm_final.pt'
    # )

    
    fno_model = FNO3D_Rates().to(device)
    fno_ckpt_path = "/home/tonglinjin/projects/flowmatching/paper_flowmatching/checkpoints/fno3d_revised_plus.pt"
    fno_ckpt = torch.load(fno_ckpt_path, map_location=device, weights_only=True)
    fno_model.load_state_dict(fno_ckpt['state_dict'])
    fno_model.eval()
    for p in fno_model.parameters():
        p.requires_grad = False


    sample_loader_obswise_and_save_txt(
        ckpt_path=args.ckpt_path,
        test_loader=test_loader,
        fno_model=fno_model,
        dataset=full_train_ds,
        n_times=args.n_times,
        chunk_size=args.chunk_size,
        save_dir=args.save_dir,
        save_txt=True,
        guidance_scale=args.guidance_scale,
        device=device,
        clamp_ratio=args.clamp_ratio,
        model_class=NetClass,
        use_guidance=bool(args.use_guidance)
    )











