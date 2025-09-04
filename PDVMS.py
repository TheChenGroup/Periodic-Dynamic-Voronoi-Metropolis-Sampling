import pyscf
import numpy as np
import scipy
from pyscf import gto, scf
import jax
import jax.numpy as jnp
from jax import jit
from itertools import permutations
from scipy.optimize import linear_sum_assignment
import ml_collections
import time
from pyscf.pbc import gto
from jax.config import config as jax_config
from DeepSolid import base_config
from DeepSolid import supercell
from DeepSolid.utils import units
from DeepSolid import train
from DeepSolid import process
from DeepSolid import init_guess
from DeepSolid import hf
from DeepSolid import constants
from DeepSolid import network
from DeepSolid import train
from DeepSolid import pretrain
from DeepSolid import qmc
from DeepSolid import checkpoint

def get_config(input_str):
    X, Y, L_Ang, S, z, basis = input_str.split(',')
    S = np.diag([int(S), int(S), 1])
    cfg = base_config.default()
    L_Ang = float(L_Ang)
    z = float(z)
    L_Bohr = units.angstrom2bohr(L_Ang)
    cell = gto.Cell()
    cell.atom = [[X, [3**(-0.5) * L_Bohr,     0.0,     0.0]],
                 [Y, [2*3**(-0.5) * L_Bohr,   0.0,     0.0]]]
    cell.basis = basis
    cell.a = np.array([[L_Bohr * np.cos(np.pi/6), -L_Bohr * 0.5,   0],
                       [L_Bohr * np.cos(np.pi/6),  L_Bohr * 0.5,   0],
                       [0, 0, z],
                       ])
    cell.unit = "B"
    cell.verbose = 5
    cell.exp_to_discard = 0.1
    cell.build()
    simulation_cell = supercell.get_supercell(cell, S)
    cfg.system.pyscf_cell = simulation_cell

    return cfg
input_str = f"{'C'},{'C'},{2.46124},{1},{50},{'ccpvdz'}"
cfg=get_config(input_str)

simulation_cell = cfg.system.pyscf_cell
cfg.system.internal_cell = init_guess.pyscf_to_cell(cell=simulation_cell)
hartree_fock = hf.SCF(cell=simulation_cell, twist=jnp.array(cfg.network.twist))
hartree_fock.init_scf()
if cfg.debug.deterministic:
        seed = 666
else:
    seed = int(1e6 * time.time())
key = jax.random.PRNGKey(seed)
key = jax.random.fold_in(key, 0)

system_dict = {
    'klist': hartree_fock.klist,
    'simulation_cell': simulation_cell,
}
system_dict.update(cfg.network.detnet)
slater_logdet = network.make_solid_fermi_net(**system_dict, method_name='eval_slogdet')
batch_slater_logdet = jax.vmap(slater_logdet.apply, in_axes=(None, 0), out_axes=0)
local_batch_size = cfg.batch_size // 8
ckpt = np.load('qmcjax_ckpt_200477.npz', allow_pickle=True)
params=ckpt['params']
params = jax.tree_map(lambda x: x[0], params.tolist())        
wf = lambda data: slater_logdet.apply(params, data)
wf_jit=jit(wf)

def add_core(coord):
    coord = np.append(core_alpha, coord)
    coord = np.append(coord, core_beta)
    return coord

def psi(coord):
    coord = add_core(coord)
    coord = jnp.asarray(coord)
    a = wf_jit(coord)
    return a

def dist(coord, site):
    l = len(coord)
    s = 0
    for i in range(0, l):
        s = s + (coord[i] - site[i]) ** 2
    return s

def return_to_original_cell(coord):
    to_orthogonal = np.array([[np.sqrt(6)/6, -np.sqrt(2)/2], [np.sqrt(6)/6, np.sqrt(2)/2]])
    to_absolute = np.array([[np.sqrt(6)/2, np.sqrt(6)/2], [-np.sqrt(2)/2, np.sqrt(2)/2]])
    coord = coord.reshape(8, 3)
    coord_o = np.empty_like(coord)
    imax = np.sqrt(3) / np.sqrt(2)*r
    jmax = np.sqrt(3)/np.sqrt(2)*r
    for i in range(0,8):
        coord_o[i,:2] = to_orthogonal @ coord[i,:2]
        coord_o[i, 2] = coord[i, 2]
    for i in range(0,8):
        if(coord_o[i][0]<0 or coord_o[i][0]>imax):
            coord_o[i][0]=coord_o[i][0]-np.floor(coord_o[i][0]/imax)*imax
        if(coord_o[i][1]<0 or coord_o[i][1]>jmax):
            coord_o[i][1]=coord_o[i][1]-np.floor(coord_o[i][1]/jmax)*jmax
    for i in range(0,8):
        coord[i,:2] = to_absolute @ coord_o[i,:2]
        coord[i, 2] = coord_o[i, 2]
    return coord.reshape(-1)

def periodic_dist(site, coord):
    dist_cs = dist(coord, site)
    for i in range(-1, 2):
        for j in range(-1, 2):
            icoord = coord + i * a1 + j * a2
            idist_cs = dist(icoord, site)
            if(idist_cs < dist_cs):
                dist_cs = idist_cs
                label = [i, j]
    return dist_cs

def periodic_voronoi(coord, site):
    coord = return_to_original_cell(coord)
    coord = coord.reshape(8, 3)
    site = site.reshape(8, 3)
    cost_alpha = np.zeros(shape = (alpha, alpha))
    cost_beta = np.zeros(shape = (beta, beta))
    for i in range(0, alpha):
        for j in range(0, alpha):
            cost_alpha[i][j]=periodic_dist(site[i], coord[j]) 
    for i in range(0,beta):
        for j in range(0,beta):
            cost_beta[i][j] =periodic_dist(site[i+alpha], coord[j+alpha]) 
    match1 = linear_sum_assignment(cost_alpha)
    match2 = linear_sum_assignment(cost_beta)
    coord[:alpha] = np.take(coord[:alpha], match1[1], axis=0)
    coord[alpha:] = np.take(coord[alpha:], match2[1], axis=0)
    coord = coord.reshape(-1)
    site = site.reshape(-1) 
    coord = return_to_original_cell(coord)
    coord = coord.reshape(-1)
    return coord

def periodic_COM_calculation(walkers):
    l = len(walkers)
    r=2.68529624
    to_orthogonal = np.array([[np.sqrt(6)/6,-np.sqrt(2)/2],[np.sqrt(6)/6,np.sqrt(2)/2]])
    to_absolute = np.array([[np.sqrt(6)/2,np.sqrt(6)/2],[-np.sqrt(2)/2,np.sqrt(2)/2]])
    B = walkers
    B = walkers.reshape(l, 8, 3)
    C = np.empty_like(B) 
    for i in range(B.shape[0]): 
        for j in range(B.shape[1]): 
            C[i, j, :2] =  to_orthogonal @ B[i, j, :2] 
            C[i, j, 2] = B[i, j, 2]
    imax=np.sqrt(3)/np.sqrt(2)*r
    jmax=np.sqrt(3)/np.sqrt(2)*r
    ri=imax/(2*np.pi)
    rj=jmax/(2*np.pi)
    mean=np.array([])
    for i in range(B.shape[1]): 
        x1=0
        z1=0
        y2=0
        z2=0
        z=0
        for j in range(B.shape[0]): 
            x1=x1+ri*np.cos(C[j][i][0]/imax*2*np.pi)
            z1=z1+ri*np.sin(C[j][i][0]/imax*2*np.pi)
            y2=y2+rj*np.cos(C[j][i][1]/jmax*2*np.pi)
            z2=z2+rj*np.sin(C[j][i][1]/jmax*2*np.pi)
            z=z+C[j][i][2]
        xmean=ri*(np.arctan2(-z1 / l, -x1 / l) + np.pi)
        ymean=rj*(np.arctan2(-z2 / l, -y2 / l) + np.pi)
        mean=np.append(mean,np.array([xmean, ymean, z / l]))
    mean=mean.reshape(8,3)
    mean_absolute=np.empty_like(mean)
    for i in range(0,8):
        mean_absolute[i,:2] = to_absolute @ mean[i,:2]
        mean_absolute[i, 2] = mean[i, 2]
    return mean_absolute.reshape(-1)

walker0 = np.array([
    6.05073231089043, 1.1322720529297547, -0.004617132417008949, 
    3.981128184888181, 0.04153212172948586, 0, 
    2.68529624, 0, 1, 
    2.0185448095538883, 1.15980474446267, -0.014795022617116, 
    4.357592024454734, -0.037833383266765316, 0, 
    5.37059249, 0, -1, 
    6.016034524979097, -1.1678953262550378, 0.12647857531741064, 
    1.97945976419909, -1.1042241176210292, 0.0016884352012948474])

electrons = 12 
Outer_shell = electrons - 4
r = 2.68529624
C_core=np.array([
    r, 0.0, 0.0,        
    2*r, 0.0, 0.0])
core1 = C_core + np.random.rand(6) * 0.1
core2 = C_core + np.random.rand(6) * 0.1
core_alpha = core1
core_beta = core2
core_electrons=4
a1 = np.array([1.5 * r, 1.5 * 3**(-0.5) * r, 0])
a2 = np.array([1.5 * r, -1.5 * 3**(-0.5) * r, 0])
T = 500000
N = 500
alpha = 4
beta = 4
x = []
for i in range(0, N):
    x.append(walker0)
x = np.asarray(x)
mean_site = walker0
for t in range(0, T):
    for num in range(0, N):
        coord_t = x[num]
        wf1 = psi(coord_t)
        coord_tt = np.array([])
        for i in range(0, Outer_shell * 3):
            coord_tt = np.append(coord_tt, np.random.normal(coord_t[i], 0.3, 1))
        coord_tt = return_to_original_cell(coord_tt)
        coord_tt = periodic_voronoi(coord_tt, mean_site)
        coord_tt = return_to_original_cell(coord_tt)
        wf2 = psi(coord_tt)
        if((wf2 - wf1) >= np.log(np.sqrt(np.random.rand()))):
            x[num] = coord_tt
    mean_site = periodic_COM_calculation(x)
    print(mean_site.tolist())
