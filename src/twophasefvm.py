import math
import numpy as np
from scipy.sparse.linalg import cg
import time

class NODE:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.z = 0

class CELL:
    def __init__(self):
        self.vertices = [-1, -1, -1, -1, -1, -1, -1, -1]
        self.neighbors = [-1, -1, -1, -1, -1, -1]
        self.dx = 0
        self.dy = 0
        self.dz = 0
        self.volume = 0
        self.xc = 0
        self.yc = 0
        self.zc = 0
        self.porosity = 0
        self.kx = 0
        self.ky = 0
        self.kz = 0
        self.trans = [0, 0, 0, 0, 0, 0]
        self.transw = [0, 0, 0, 0, 0, 0]
        self.markbc = 0
        self.press = 0
        self.Sw = 0
        self.markwell=0
        self.mobiw=0
        self.mobio=0
        self.mobit=0

def computemobi():
    for ie in range(0, ncell):
        sw=celllist[ie].Sw
        a=(1-sw)/(1-Siw)
        b=(sw-Siw)/(1-Siw)
        kro=a*a*(1-b*b)
        krw=b*b*b*b
        vro=kro/mu_o
        vrw=krw/mu_w
        celllist[ie].mobio=vro
        celllist[ie].mobiw=vrw
        celllist[ie].mobit=vro+vrw

class CallingCounter(object):
    def __init__ (self, func):
        self.func = func
        self.count = 0

    def __call__ (self, *args, **kwargs):
        self.count += 1
        return self.func(*args, **kwargs)

@CallingCounter
def twophase_impes(chukvec): #homogeneous reservoir input k output transient p of bottomleft corner producer
    for i in range(0, ncell):  # set chuk
        celllist[i].kx = chukvec[i]
        celllist[i].ky = chukvec[i]
        celllist[i].kz = chukvec[i]*0.1
    for ie in range(0, ncell):  # compute transmissibility
        dx1 = celllist[ie].dx
        dy1 = celllist[ie].dy
        dz1 = celllist[ie].dz
        for j in range(0, 6):
            je = celllist[ie].neighbors[j]
            if je >= 0:
                dx2 = celllist[je].dx
                dy2 = celllist[je].dy
                dz2 = celllist[je].dz
                mt1 = 1.0
                mt2 = 1.0
                if j == 0 or j == 1:
                    mt1 = mt1 * dy1 * dz1
                    mt2 = mt2 * dy2 * dz2
                    k1 = celllist[ie].kx
                    k2 = celllist[je].kx
                    dd1 = dx1 / 2.
                    dd2 = dx2 / 2.
                elif j == 2 or j == 3:
                    mt1 = mt1 * dx1 * dz1
                    mt2 = mt2 * dx2 * dz2
                    k1 = celllist[ie].ky
                    k2 = celllist[je].ky
                    dd1 = dy1 / 2.
                    dd2 = dy2 / 2.
                else:
                    mt1 = mt1 * dx1 * dy1
                    mt2 = mt2 * dx2 * dy2
                    k1 = celllist[ie].kz
                    k2 = celllist[je].kz
                    dd1 = dz1 / 2.
                    dd2 = dz2 / 2.
                t1 = mt1 * k1 / dd1
                t2 = mt2 * k2 / dd2
                tt = 1 / (1 / t1 + 1 / t2)
                celllist[ie].trans[j] = tt
    # qwt=np.zeros((4,nt))   # record flow rate with time
    q_total = np.zeros((4, nt))
    q_oil = np.zeros((4, nt))
    q_water = np.zeros((4, nt))
    allpress = np.zeros((nt, ncell))
    allSw = np.zeros((nt, ncell))
    for i in range(0, ncell):  # initial condition
        celllist[i].press = p_init
        celllist[i].Sw = Siw
    presslast = np.zeros(ncell)
    for t in range(nt):
        # print('iteration: ', t)
        computemobi()
        # implicit pressure
        if t % dtscale == 0:
            for ie in range(ncell):
                presslast[ie] = celllist[ie].press
            Acoef = np.zeros((ncell, ncell))
            RHSvec = np.zeros(ncell)
            for ie in range(ncell):
                Sw = celllist[ie].Sw
                So = 1.0 - Sw
                Ct = Co * So + Cw * Sw
                p_i = celllist[ie].press
                Acoef[ie, ie] = celllist[ie].porosity * Ct * celllist[ie].volume / dt_p
                RHSvec[ie] = celllist[ie].porosity * Ct * celllist[ie].volume / dt_p * p_i
                if celllist[ie].markbc == -2:  # 生产
                    PI = PIcoef * celllist[ie].mobit * celllist[ie].kx
                    qt = PI * (celllist[ie].press - bhp_constant)
                    RHSvec[ie] = RHSvec[ie] - qt
                elif celllist[ie].markbc == -1:  # 注水
                    RHSvec[ie] = RHSvec[ie] + qw_fixed
                for j in range(6):
                    je = celllist[ie].neighbors[j]
                    if je >= 0:
                        Tij = celllist[ie].trans[j]
                        if celllist[ie].press > celllist[je].press:
                            mobi = celllist[ie].mobit
                        elif celllist[ie].press < celllist[je].press:
                            mobi = celllist[je].mobit
                        else:
                            mobi = (celllist[ie].mobit + celllist[je].mobit) * 0.5
                        Acoef[ie, je] = -Tij * mobi
                        Acoef[ie, ie] = Acoef[ie, ie] + Tij * mobi
            # press=np.dot(np.linalg.inv(Acoef),RHS)
            # press, exit_code=minres(Acoef, RHS, x0=None, shift=0.0, tol=1e-10, maxiter=None, M=None, callback=None, show=False, check=False)
            endtime1 = time.time()
            press, exit_code = cg(Acoef, RHSvec, x0=None, rtol=1e-05)
            endtime = time.time()
            # print('iteration:', t, ' exit code: ', exit_code, ' time cost ', endtime - endtime1)
            for ie in range(ncell):
                celllist[ie].press = press[ie]
                if press[ie] < 0:
                    print('negative press at ', ie, press[ie])
            allpress[t] = press
        # explicit saturation
        for ie in range(ncell):
            RHS = 0
            if celllist[ie].markbc == -2:
                PI_t = PIcoef * celllist[ie].mobit * celllist[ie].kx
                PI_o = PIcoef * celllist[ie].mobio * celllist[ie].kx
                PI_w = PIcoef * celllist[ie].mobiw * celllist[ie].kx

                qt = PI_t * (celllist[ie].press - bhp_constant)
                qo_i = PI_o * (celllist[ie].press - bhp_constant)
                qw_i = PI_w * (celllist[ie].press - bhp_constant)

                wid = celllist[ie].markwell - 1
                q_total[wid, t] += qt
                q_oil[wid, t] += qo_i
                q_water[wid, t] += qw_i

                RHS -= qw_i
            elif celllist[ie].markbc == -1:
                RHS = RHS + qw_fixed
            pi = celllist[ie].press
            for j in range(6):
                je = celllist[ie].neighbors[j]
                if je >= 0:
                    pj = celllist[je].press
                    Tij = celllist[ie].trans[j]
                    if pi > pj:
                        mobiw = celllist[ie].mobiw
                    elif pi < pj:
                        mobiw = celllist[je].mobiw
                    else:
                        mobiw = (celllist[ie].mobiw + celllist[je].mobiw) * 0.5
                    RHS = RHS - mobiw * Tij * (pi - pj)
            RHS = RHS - (pi - presslast[ie]) / dt * poro * celllist[ie].volume * celllist[ie].Sw * Cw
            celllist[ie].Sw = celllist[ie].Sw + RHS * dt / poro / celllist[ie].volume
        for ie in range(ncell):
            allSw[t,ie]=celllist[ie].Sw
    return q_total.reshape(-1), q_oil.reshape(-1), q_water.reshape(-1), allpress, allSw

print("build Grid")
dxvec=[0]
for i in range(0, 20):
    dxvec.append(15)

dyvec=[0]
for i in range(0, 20):
    dyvec.append(15)
dzvec=[0]
for i in range(0, 5):
    dzvec.append(6)

nx=len(dxvec)-1
ny=len(dyvec)-1
nz=len(dzvec)-1
nodelist=[]
llz = 0
for k in range(0, nz+1):
    llz = llz + dzvec[k]
    lly=0
    for j in range(0, ny+1):
        lly = lly + dyvec[j]
        llx = 0
        for i in range(0, nx+1):
            llx = llx + dxvec[i]
            node=NODE()
            node.x=llx
            node.y=lly
            node.z=llz
            nodelist.append(node)

# build connectivity and neighbors
celllist=[]

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):
            id = k * nx * ny + j * nx + i
            nc=id
            cell = CELL()
            if i>0:
                cell.neighbors[0] = nc - 1
            if i<nx-1:
                cell.neighbors[1] = nc + 1
            if j>0:
                cell.neighbors[2] = nc - nx
            if j<ny-1:
                cell.neighbors[3] = nc + nx
            if k>0:
                cell.neighbors[4] = nc - nx*ny
            if k<nz-1:
                cell.neighbors[5] = nc + nx * ny
            i0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i1 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i2 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i3 = k * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            i4 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i
            i5 = (k + 1) * (nx + 1) * (ny + 1) + j * (nx + 1) + i + 1
            i6 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i
            i7 = (k + 1) * (nx + 1) * (ny + 1) + (j + 1) * (nx + 1) + i + 1
            cell.dx = nodelist[i1].x - nodelist[i0].x
            cell.dy = nodelist[i2].y - nodelist[i0].y
            cell.dz = nodelist[i4].z - nodelist[i0].z
            cell.vertices[0] = i0
            cell.vertices[1] = i1
            cell.vertices[2] = i2
            cell.vertices[3] = i3
            cell.vertices[4] = i4
            cell.vertices[5] = i5
            cell.vertices[6] = i6
            cell.vertices[7] = i7
            cell.xc = 0.125 * (nodelist[i0].x+nodelist[i1].x+nodelist[i2].x+nodelist[i3].x+nodelist[i4].x+nodelist[i5].x+nodelist[i6].x+nodelist[i7].x)
            cell.yc = 0.125 * (nodelist[i0].y + nodelist[i1].y + nodelist[i2].y + nodelist[i3].y + nodelist[i4].y + nodelist[i5].y + nodelist[i6].y + nodelist[i7].y)
            cell.zc = 0.125 * (nodelist[i0].z + nodelist[i1].z + nodelist[i2].z + nodelist[i3].z + nodelist[i4].z + nodelist[i5].z + nodelist[i6].z + nodelist[i7].z)
            cell.volume=cell.dx*cell.dy*cell.dz
            celllist.append(cell)

cellvolume=celllist[0].volume

ncell=len(celllist)

print("Set all settings-physical, well and simulation")
mu_o = 1.8e-3
mu_w = 1e-3
poro = 0.2
Siw=0.2
Cw = 4 * 1e-6 / 6894
Co = 100 * 1e-6 / 6894
p_init=30.0*1e6

bhp_constant = 28e6 #*
qw_fixed = 40.0/86400 #*
rw = 0.05
SS = 3
length = 3000
ddx = dxvec[1]-dxvec[0]
re = 0.14*(ddx*ddx + ddx*ddx)**0.5
PIcoef = 2 * 3.14*ddx / (math.log(re / rw) + SS)

nt = 1080
dt=20000
dtscale=1
dt_p=dt*dtscale

for i in range(0, ncell):
    celllist[i].porosity = poro

for k in range(0, nz):
    for j in range(0, ny):
        for i in range(0, nx):
            id = k * nx * ny + j * nx + i
            if i==0 and j==0:
                celllist[id].markwell = 1
                celllist[id].markbc = -2
            elif i==19 and j==0:
                celllist[id].markwell = 2
                celllist[id].markbc = -2
            elif i==0 and j==19:
                celllist[id].markwell = 3
                celllist[id].markbc = -2
            elif i==19 and j==19:
                celllist[id].markwell = 4
                celllist[id].markbc = -2
            elif i==9 and j==9:
                celllist[id].markwell = 5
                celllist[id].markbc = -1



