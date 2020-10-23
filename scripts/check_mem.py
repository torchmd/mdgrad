import os
import numpy as np
import matplotlib.pyplot as plt
import sys 
import mdtraj

ODE_PATH = '/home/wwj/Repo/projects/torchdiffeq/'

sys.path.insert(0, ODE_PATH)
sys.path.insert(0, '../..')

import torch
from torch.optim import Adam
from torchmd.ode import ODE
from torchmd.hamiltoinians import PairPot, MLP, LennardJones, Buck, LennardJones69
from torchmd.observable import DiffRDF
from torchmd.utils import dump_mov
from torchmd.sovlers import odeint_adjoint
from torchdiffeq._impl.odeint import odeint
from torchdiffeq import odeint_adjoint
from nglview import show_ase, show_file, show_mdtraj

from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing
from ase import Atoms
from math import sqrt



class NHCHAIN_ODE(torch.nn.Module):

    def __init__(self, model, mass, T, num_chains=2, Q=1.0,  device=0, dim=3, adjoint=True):
        super().__init__()
        self.model = model  
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        self.T = T
        self.N_dof = mass.shape[0] * dim
        self.target_ke = (0.5 * self.N_dof * T )
        
        self.num_chains = num_chains
        self.Q = np.array([Q,
                   *[Q/mass.shape[0]]*(num_chains-1)])
        self.Q = torch.Tensor(self.Q).to(device)
        self.dim = dim
        self.adjoint = adjoint
        
    def forward(self, t, state):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):        
            print("step")

            v = state[0]
            q = state[1]
            p_v = state[2]
            
            if self.adjoint:
                q.requires_grad = True
            
            N = self.N_dof
            
            p = v.reshape(-1, self.dim) * self.mass[:, None]
            q = q.reshape(-1, self.dim)
            
            sys_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum() 
            
            u = self.model(q)
            
            dqdt = (p / self.mass[:, None]).reshape(-1)
            dvdt = -compute_grad(inputs=q, output=u.sum(-1)).reshape(-1) - p_v[0] * p.reshape(-1) / self.Q[0]

            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1]/ self.Q[1]
            dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:]*p_v[1:-1]/ self.Q[2:]
            dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T
            
        return (dvdt, v, torch.cat((dpvdt_0[None], dpvdt_mid, dpvdt_last[None])))
    
class Harmonic1D(torch.nn.Module):
    def __init__(self):
        super(Harmonic1D, self).__init__()

    def forward(self, x):
        return 0.5 *  x ** 2


tau = 100
device = 1

adjoint = True
#adjoint = False

num_chains = 5

dt = 1.0
t_len = int( tau / dt )
t = torch.Tensor([dt * i for i in range(t_len)]).to(device)
t_np = t.detach().cpu().numpy()

harm1d = Harmonic1D().to(device)
mass = np.array([1.])


# sample position and momentum, p_v is virtual momenta 
p_v = torch.Tensor([0.0] * num_chains).to(device)
q = torch.Tensor([1.0]).to(device) 
v = torch.Tensor([0.0]).to(device)


if adjoint:
    q.requires_grad = True
    v.requires_grad = True
    p_v.requires_grad = True
    t.requires_grad = True

    f_x = NHCHAIN_ODE(harm1d, 
        mass, 
        Q=100000.0, 
        T=1.0,
        num_chains=num_chains, 
        device=device,
        dim=1, 
        adjoint=True).to(device)

    v_v, q_v, pv_v = odeint_adjoint(f_x, (v, q, p_v), t, method='rk4')
    q_v[-1].backward()
    print( "{} Mb".format(torch.cuda.memory_allocated(device) / 1024**2) )

else:
    q.requires_grad = True
    v.requires_grad = True
    p_v.requires_grad = True
    t.requires_grad = True

    f_x = NHCHAIN_ODE(harm1d, 
        mass, 
        Q=100000.0, 
        T=1.0,
        num_chains=num_chains, 
        device=device,
        dim=1, 
        adjoint=False).to(device)

    v_v, q_v, pv_v = odeint(f_x, (v, q, p_v), t, method='rk4')
    q_v.sum().backward()
    print( "{} Mb".format(torch.cuda.memory_allocated(device) / 1024**2) )
