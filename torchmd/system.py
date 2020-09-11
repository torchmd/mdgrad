import torch
from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict
from ase import Atoms 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution 
from ase import units
import numpy as np 

def generate_nbr_list(xyz, cutoff, cell):
    
    # todo to make it compatible for non-cubic cells
    device = xyz.device 
    dis_mat = xyz[..., None, :, :] - xyz[..., :, None, :]
    offsets = -dis_mat.ge(0.5 *  cell).to(torch.float).to(device) + \
                    dis_mat.lt(-0.5 *  cell).to(torch.float).to(device)
    dis_mat = dis_mat + offsets * cell
    dis_sq = dis_mat.pow(2).sum(-1)
    mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
    nbr_list = torch.triu(mask.to(torch.long)).nonzero()
    return nbr_list

class System(Atoms):
    def __init__(
        self,
        *args,
        device,
        props={},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.props = props
        self.device = device 
        
    def get_nxyz(self):
        """Gets the atomic number and the positions of the atoms
            inside the unit cell of the system.
        Returns:
            nxyz (np.array): atomic numbers + cartesian coordinates
                of the atoms.
        """
        nxyz = np.concatenate([
            self.get_atomic_numbers().reshape(-1, 1),
            self.get_positions().reshape(-1, 3)
        ], axis=1)

        return nxyz
    
    def get_cell_len(self):
        return np.diag( self.get_cell() )

    def get_batch(self):
        batch = {
         'nxyz': torch.Tensor(self.get_nxyz()), 
         'num_atoms': torch.LongTensor([self.get_number_of_atoms()]),
         'energy': 0.0}
        
        return batch
        
    def set_temperature(self, T):
        MaxwellBoltzmannDistribution(self, T * units.kB )
    
    def initial_conditions(self):
        # This should be in the integrator, the initialization should be integrator specific
        
        if hasattr(self, 'traj'):            
            return tuple([torch.Tensor(var).to(self.device) for var in self.traj[-1]])

        else:            
            self.traj = [[self.get_velocities(), self.get_positions(), [0.0] * 5]]

            return tuple([torch.Tensor(var).to(self.device) for var in self.traj[-1]])
    
    def update_traj(self, states):
        assert len(states) == 3
        assert all([type(state) == torch.Tensor for state in states])        
        if states[0].device != 'cpu':
            self.traj.append([var.detach().cpu().numpy() for var in states])
        else:
            self.traj.append([var.detach().cpu().numpy() for var in states])

        # TODO: update system states for velocity and momenta
        
        

class GNNPotentials(torch.nn.Module):
    def __init__(self, module, inputs, cell, cutoff, device):
        super().__init__()
        self.module = module
        self.cutoff = cutoff
        self.inputs = batch_to(inputs, device)
        self.cell = torch.Tensor(cell).to(device)
        self.to(device)

    def forward(self, q): 
        self.inputs['nbr_list'] = generate_nbr_list(q, self.cutoff, self.cell)
        results = self.module(self.inputs, q)
        return results['energy']

    
class Stack(torch.nn.Module):
    def __init__(self, model_dict, mode='sum'):
        super().__init__()
        self.models = ModuleDict(model_dict)
        
    def forward(self, x):
        for i, key in enumerate(self.models.keys()):
            if i == 0:
                result = self.models[key](x).reshape(-1)
            else:
                new_result = self.models[key](x)
                result += new_result.sum().reshape(-1)
        
        return result


if __name__ == "__main__":
    from ase.lattice.cubic import FaceCenteredCubic
    from potentials import PairPot, ExcludedVolume
    from nff.train import get_model

    params = {
        'n_atom_basis': 32,
        'n_filters': 32,
        'n_gaussians': 32,
        'n_convolutions': 32,
        'cutoff': 5.0,
        'trainable_gauss': False
    }

    # Define prior potential 
    lj_params = {'epsilon': 0.05, 
                 'sigma': 3.0, 
                 'power': 12}

    size = 4
    L = 19.73 / size

    device = 'cpu'
    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol='H',
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)

    system = System(atoms, device=device)

    pair = PairPot(ExcludedVolume, lj_params,
                    cell=torch.Tensor(system.get_cell_len()), 
                    device=device,
                    cutoff=L/2,
                    ).to(device)

    model = get_model(params)
    PES = GNNPotentials(model, system.get_batch(), system.get_cell_len(), cutoff=5.0, device=system.device)

    system.set_temperature(298.0)