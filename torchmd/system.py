import torch
from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict
from ase import Atoms 
from ase import units
import numpy as np 

from torchmd.topology import generate_nbr_list, get_offsets, generate_angle_list

def check_system(object):
    import torchmd
    if object.__class__ != torchmd.system.System:
        raise TypeError("input should be a torchmd.system.System")

class System(Atoms):

    """Object that contains system information. Inherited from ase.Atoms
    
    Attributes:
        device (int or str): torch device "cpu" or an integer
        dim (int): dimension of the system (if n_dim < 3, the first n_dim columns are used for position calculation)
        props (dict{}): additional properties 
    """

    def __init__(
        self,
        *args,
        device,
        dim=3,
        props={},
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.props = props
        self.device = device
        self.dim = dim
        
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
        from ase.md.velocitydistribution import MaxwellBoltzmannDistribution 
        MaxwellBoltzmannDistribution(self, T)
        if self.dim < 3:
            vel =  self.get_velocities()
            vel[:, -1] = 0.0
            self.set_velocities(vel)
    

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


    # Todo test Pair pot with fixed atom index  



