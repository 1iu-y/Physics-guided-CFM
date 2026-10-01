import math
import torch
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dtype = torch.float64

def load_txt_to_tensor(path, device=device, dtype=dtype):
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = [float(x) for x in line.split()]
            data.append(row)
    return torch.tensor(data, device=device, dtype=dtype)

def build_grid(nx, ny, nz, dx, dy, dz, device=device):
    ncell = nx*ny*nz
    neighbors = torch.full((ncell,6),-1, dtype=torch.long, device=device)
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                idx = k*nx*ny+ j*nx + i
                if i>0:
                    neighbors[idx,0] = idx-1
                if i<nx-1:
                    neighbors[idx,1] = idx+1
                if j>0:
                    neighbors[idx,2] = idx-nx
                if j<ny-1:
                    neighbors[idx,3] = idx+nx
                if k>0:
                    neighbors[idx,4] = idx-nx*ny
                if k<nz-1:
                    neighbors[idx,5] = idx+nx*ny
    vol = dx*dy*dz
    return neighbors, vol

def compute_trans(neighbors, kx, ky, kz, dx, dy, dz):
    ncell = neighbors.size(0)
    B = kx.size(0)
    trans = torch.zeros((B,ncell,6), dtype=kx.dtype, device=kx.device)
    idx_all = torch.arange(ncell, device=kx.device)
    for j in range(6):
        mask = (neighbors[:,j]>=0)
        ie = idx_all[mask]
        je = neighbors[:,j][mask]
        mt1 = mt2 = 1.0
        if j in [0,1]:
            mt1 *= dy*dz
            mt2 *= dy*dz
            k1 = kx[:,ie]
            k2 = kx[:,je]
            dd1 = dd2 = dx/2.0
        if j in [2,3]:
            mt1 *= dx*dz
            mt2 *= dx*dz
            k1 = ky[:,ie]
            k2 = ky[:,je]
            dd1 = dd2 = dy/2.0
        if j in [4,5]:
            mt1 *= dx*dy
            mt2 *= dx*dy
            k1 = kz[:,ie]
            k2 = kz[:,je]
            dd1 = dd2 = dz/2.0
        t1 = mt1 * k1 / dd1
        t2 = mt2 * k2 / dd2
        tt = 1 / (1 / t1 + 1 / t2)
        trans[:,ie,j] = tt
    return trans

def compute_mobility(sw, Siw, mu_o, mu_w):
    denom = 1.0-Siw
    a = (1.0-sw) / denom
    b = (sw - Siw) / denom
    kro = a*a*(1.0-b*b)
    krw = b**4
    vro = kro/mu_o
    vrw = krw/mu_w
    mobio = vro
    mobiw = vrw
    mobit = mobio+mobiw
    return mobio, mobiw, mobit


def twophase_impes_torch(
        chukvec,
        dx=15.0, dy=15.0, dz=6.0,
        mu_o=1.8e-3, mu_w=1.0e-3,
        poro=0.2, Siw=0.2,
        Cw=4e-6/6894.0, Co=100e-6/6894.0,
        p_init=30.0e6,
        bhp_constant=28e6,
        qw_fixed=40.0/86400.0,
        rw=0.05, SS=3.0,
        nt=1080, dt=20000.0, dtscale=1,
        device=device, dtype=dtype,
        return_all=False
):
    B, C, nz, ny, nx = chukvec.shape
    ncell = nx*ny*nz
    chukvec_flat = chukvec.reshape(B, ncell)


    neighbors, volume = build_grid(nx, ny, nz, dx, dy, dz, device=device)
    markbc = torch.zeros(ncell, dtype = torch.int32, device=device)
    markwell = torch.zeros(ncell, dtype = torch.int32, device=device)

    def cell_index(i,j,k):
        return i + nx*j + k*nx*ny

    for k in range(nz):
        markbc[cell_index(0,0, k)] = -2; markwell[cell_index(0,0, k)] = 1
        markbc[cell_index(nx-1,0, k)] = -2; markwell[cell_index(nx-1,0, k)] = 2
        markbc[cell_index(0, ny-1, k)] = -2; markwell[cell_index(0,ny-1, k)] = 3
        markbc[cell_index(nx-1, ny-1, k)] = -2; markwell[cell_index(nx-1, ny-1, k)] = 4
        markbc[cell_index(nx//2-1, ny//2-1, k)] = -1
        markwell[cell_index(nx//2-1, ny//2-1, k)] = 5

    prod_mask = (markbc == -2)
    inj_mask = (markbc == -1)

    ddx = dx
    re = 0.14*math.sqrt(ddx*ddx + ddx*ddx)
    PIcoef = 2.0 * 3.14 * ddx / (math.log(re/rw) + SS)

    kx = chukvec_flat.clone()
    ky = chukvec_flat.clone()
    kz = chukvec_flat.clone()*0.1

    trans = compute_trans(neighbors, kx, ky, kz, dx, dy, dz)

    press = torch.full((B, ncell), p_init, dtype = dtype, device=device)
    Sw = torch.full((B, ncell), Siw, dtype = dtype, device=device)

    q_total = torch.zeros((B, 4, nt), dtype=dtype, device=device)
    q_oil = torch.zeros((B, 4, nt), dtype=dtype, device=device)
    q_water = torch.zeros((B, 4, nt), dtype=dtype, device=device)
    allpress = torch.zeros((B, nt,ncell), dtype=dtype, device=device) if return_all else None
    allSw = torch.zeros((B, nt,ncell), dtype=dtype, device=device) if return_all else None

    well_group = torch.full((ncell,), -1, dtype=dtype, device=device)
    for wid in range(1,5):
        well_group[markwell==wid] = wid - 1
    well_group_b = well_group.unsqueeze(0).expand(B, -1)
    idx_cells = torch.arange(ncell, device=device)

    dt_p = dt*dtscale
    presslast = press.clone()
    for t in range(nt):
        mobio, mobiw, mobit = compute_mobility(Sw, Siw, mu_o, mu_w)
        if (t%dtscale) ==0:
            presslast = press.clone()
            So = 1.0 - Sw
            Ct = Co * So + Cw * Sw

            A = torch.zeros((B, ncell, ncell), dtype=dtype, device=device)
            RHS = torch.zeros((B, ncell), dtype=dtype, device=device)

            diag = poro * Ct*volume/dt_p
            A[:,idx_cells, idx_cells] = diag
            RHS = diag * press

            if prod_mask.any():
                PI_t = PIcoef * mobit[:, prod_mask] * kx[:, prod_mask]
                qt = PI_t * (press[:, prod_mask] - bhp_constant)
                RHS[:, prod_mask] = RHS[:, prod_mask] - qt
            if inj_mask.any():
                RHS[:, inj_mask] = RHS[:, inj_mask] + qw_fixed

            for j in range(6):
                mask = neighbors[:,j] >= 0
                if not mask.any():
                    continue
                ie = idx_cells[mask]
                je = neighbors[:,j][mask]
                pi = press[:, ie]
                pj = press[:, je]
                mobi = torch.where(pi>pj, mobit[:, ie], torch.where(pi<pj, mobit[:, je], 0.5*(mobit[:, ie]+mobit[:, je])))
                Tij = trans[:,ie,j]
                A[:,ie,je] = -Tij * mobi
                A[:,ie,ie] += Tij*mobi

            press = torch.linalg.solve(A, RHS)
            if return_all:
                allpress[:, t] = press

        RHSw = torch.zeros((B, ncell), dtype=dtype, device=device)

        if prod_mask.any():
            PI_t = PIcoef * mobit[:, prod_mask] * kx[:, prod_mask]
            PI_o = PIcoef * mobio[:, prod_mask] * kx[:, prod_mask]
            PI_w = PIcoef * mobiw[:, prod_mask] * kx[:, prod_mask]

            p_prod = press[:, prod_mask]
            qt = PI_t * (p_prod - bhp_constant)
            qo = PI_o * (p_prod - bhp_constant)
            qw = PI_w * (p_prod - bhp_constant)

            RHSw[:, prod_mask] -= qw

            gidx = well_group_b[:, prod_mask]
            for b in range(B):
                for wid in range(4):
                    m = (gidx[b] == wid)
                    if m.any():
                        q_total[b, wid, t] += qt[b, m].sum()
                        q_oil[b, wid, t] += qo[b, m].sum()
                        q_water[b, wid, t] += qw[b, m].sum()


        if inj_mask.any():
            RHSw[:,inj_mask] += qw_fixed

        for j in range(6):
            mask = neighbors[:,j] >= 0
            if not mask.any():
                continue
            ie = idx_cells[mask]
            je = neighbors[:,j][mask]
            pi = press[:, ie]
            pj = press[:, je]
            mw_i = mobiw[:,ie]
            mw_j = mobiw[:,je]

            mw = torch.where(pi>pj, mw_i,
                             torch.where(pi<pj, mw_j, 0.5*(mw_i+mw_j)))
            Tij = trans[:,ie,j]
            RHSw[:, ie] -= mw * Tij * (pi-pj)

        RHSw = RHSw - (press-presslast)/dt*poro*volume*Sw*Cw

        Sw = Sw + RHSw*dt/(poro*volume)
        if return_all:
            allSw[:,t,:] = Sw

    if return_all:
        return q_total.view(B,-1), q_oil.view(B,-1), q_water.view(B,-1), allpress, allSw
    else:
        return q_total.view(B,-1)


if __name__ == '__main__':
    nx, ny, nz = 20, 20, 5
    dx, dy, dz = 15.0, 15.0, 6.0
    perms_raw = load_txt_to_tensor('raw_data/perms3dme.txt', device=device, dtype=dtype)
    perms_sgsim = perms_raw.T
    n_samples = perms_sgsim.size(0)
    perms = (2.0**perms_sgsim) * 0.1e-15
    nt = 1080
    n_batch = 2
    ds = perms[:n_batch]
    perms_batch = ds.reshape(n_batch, 1, nz, ny, nx)
    q_total = twophase_impes_torch(perms_batch)
    qtrue_raw = load_txt_to_tensor('raw_data/q_total.txt', device=device, dtype=dtype)
    q_true = qtrue_raw[:n_batch]
    num = torch.norm(q_total - q_true,p=2)
    den = torch.norm(q_true,p=2)
    rel_err = (num/den).item()
    print(f"Mean Absolute Error:{torch.mean(num)}")
    print(f"Relative L2 Norm on first {n_batch} samples = {rel_err:.6e}")




























