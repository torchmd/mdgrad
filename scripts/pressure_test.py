import torch
import numpy as np

from ase import Atom, Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase import units

from torchmd.interface import GNNPotentials,PairPotentials, Stack
from torchmd.system import System
from torchmd.potentials import ExcludedVolume, LennardJones

from torchmd.system import System
from torchmd.md import NoseHooverChain 
from torchmd.potentials import ExcludedVolume
from nff.train import get_model

from torchmd.md import Simulations
from time import time
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-device", type=int, default=0)
    parser.add_argument("--force", action='store_true', default=False)

    params = vars(parser.parse_args())
    device = params['device']

    atoms = FaceCenteredCubic(symbol='H',
                              size=(5, 5, 5),
                              latticeconstant=1.679,
                              pbc=True)

    system = System(atoms, device=device)
    system.set_temperature(1.0 /units.kB)

    lj_params = {'epsilon': 1.0, 
                 'sigma': 1.0}


    pair = PairPotentials(system, LennardJones(**lj_params),
                    cutoff=2.5,
                    ).to(device)

    model = Stack({
                'pair': pair
    })

    diffeq =NoseHooverChain(model, 
                system,
                Q=50.0, 
                T=1.0,
                num_chains=5, 
                adjoint=False).to(device)

    sim = Simulations(system, diffeq)

    traj = []
    for i in range(20):
        v_t, q_t, pv_t = sim.simulate(steps=50, frequency=50, dt=0.01)
        traj.append(q_t[-1].detach().cpu())

    from nff.utils.scatter import compute_grad

    start = time()

    for xyz in traj:
        xyz = traj[0].to(device)
        xyz.requires_grad = True 

        start = time()
        u = pair(xyz)
        u.backward()


    print( "computed for {} times".format(len(traj)) )
    print( "time cost {} microsecond".format( (time() - start) / len(traj) * 10**6 ) )
    print( "Memory cost {} Mb".format(torch.cuda.memory_allocated(device) / 1024**2) )

