import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
from nglview import show_ase, show_file, show_mdtraj
import torch

from ase.lattice.cubic import FaceCenteredCubic
from ase import units

from torchmd.system import GNNPotentials,PairPotentials,System, Stack
from torchmd.potentials import LennardJones
from nff.train import get_model

size = 4
L = 4.0
device = 1

def get_mem():
    return torch.cuda.max_memory_allocated(device) * 1e-6

print(get_mem())

atoms = FaceCenteredCubic(directions=[[1, 0, 0], 
                                    [0, 1, 0], 
                                    [0, 0, 1]],
                          symbol='H',
                          size=(size, size, size),
                          latticeconstant=L,
                          pbc=True)

from torchmd.system import System

system = System(atoms, device=device)
system.set_temperature(298.0)

# Potential 
lj_params = {'epsilon': 0.05, 
             'sigma': 2.5}

from torchmd.system import PairPotentials, GNNPotentials, Stack

pair = PairPotentials(LennardJones, lj_params,
                cell=torch.Tensor(system.get_cell_len()), 
                device=device,
                cutoff=8.0,
                ).to(device)

# parameter for SchNet 
params = {
    'n_atom_basis': 128,
    'n_filters': 128,
    'n_gaussians': 128,
    'n_convolutions': 4,
    'cutoff': 5.0,
    'trainable_gauss': False
}

model = get_model(params)
# GNN = GNNPotentials(model, 
#                     system.get_batch(), 
#                     system.get_cell_len(), 
#                     cutoff=params['cutoff'], 
#                     device=system.device)
model = Stack({
                #'gnn': GNN, 
                'pair': pair})

from torchmd.md import NoseHooverChain 
from torchmd.md import Simulations

# Set up simulations 
# Simulate 

mem_list = []
t_list = [1000, 2000]
for t in t_list:
    all_mem = []
    for i in range(5):

        diffeq =NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=298.0 * units.kB,
            num_chains=5, 
            adjoint=True).to(device)
        
        sim = Simulations(system, diffeq)

        v_t, q_t, pv_t = sim.simulate(steps=t, frequency=t, dt=0.25 *units.fs)
        q_t.sum().backward()
        memory = get_mem()
        all_mem.append(memory)

    mem_list.append(np.array(all_mem).mean())

print(mem_list)
print(t_list)