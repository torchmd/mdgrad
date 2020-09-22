import torch
import torchmd
from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict
from ase import Atoms 
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution 
from ase.geometry import wrap_positions
from ase import units
import numpy as np 

def check_system(object):
    if object.__class__ != torchmd.system.System:
        raise TypeError("input should be a torchmd.system.System")


def generate_pair_index(N, idx1, idx2, ex_pairs=None):

    import itertools

    mask_sel = torch.zeros(N, N)

    pair_mask = torch.LongTensor( [list(items) for items in itertools.product(idx1, 
                                                                              idx2)]) 

    #todo: imporse index convention
    mask_sel[pair_mask[:, 0], pair_mask[:, 1]] = 1
    mask_sel[pair_mask[:, 1], pair_mask[:, 0]] = 1

    if ex_pairs is not None:
        mask_sel[ex_pairs[:, 0], ex_pairs[:, 1]] = 0
        mask_sel[ex_pairs[:, 1], ex_pairs[:, 0]] = 0
    
    return mask_sel


def generate_nbr_list(xyz, cutoff, cell, index_tuple=None, ex_pairs=None, get_dis=False):
    
    # todo: make it compatible for non-cubic cells
    # todo: topology should be a class to handle some initialization 
    device = xyz.device

    dis_mat = (xyz[..., None, :, :] - xyz[..., :, None, :])

    if index_tuple is not None:
        N = xyz.shape[-2] # the 2nd dim is the atoms dim

        mask_sel = generate_pair_index(N, index_tuple[0], index_tuple[1], ex_pairs).to(device)
        # todo handle this calculation like a sparse tensor 
        dis_mat =  dis_mat * mask_sel[..., None]

    offsets = -dis_mat.ge(0.5 *  cell).to(torch.float).to(device) + \
                    dis_mat.lt(-0.5 *  cell).to(torch.float).to(device)
    dis_mat = dis_mat + offsets * cell
    dis_sq = torch.triu( dis_mat.pow(2).sum(-1) )
    mask = (dis_sq < cutoff ** 2) & (dis_sq != 0)
    nbr_list = torch.triu(mask.to(torch.long)).nonzero()

    if get_dis:
        return nbr_list, dis_sq[mask].sqrt() 
    else:
        return nbr_list

def get_offsets(vecs, cell, device):
    
    offsets = -vecs.ge(0.5 *  cell).to(torch.float).to(device) + \
                vecs.lt(-0.5 *  cell).to(torch.float).to(device)
    
    return offsets


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
        self.dim = self.get_cell().shape[0]
        
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
    
        
        
class GNNPotentials(torch.nn.Module):
    def __init__(self, module, inputs, cell, cutoff, device):
        super().__init__()
        self.module = module
        self.cutoff = cutoff
        # initialize the dictionary for model inputs 
        self.inputs = batch_to(inputs, device)
        self.cell = torch.Tensor(cell).to(device)
        self.to(device)

    def forward(self, xyz): 
        self.inputs['nbr_list'] = generate_nbr_list(xyz, self.cutoff, self.cell)
        results = self.module(self.inputs, xyz)
        return results['energy']


class PairPotentials(torch.nn.Module):

    def __init__(self, pair_model, model_arg, cell, device=0, cutoff=2.5, index_tuple=None, ex_pairs=None):
        super().__init__()
        self.model = pair_model(**model_arg)
        self.cell = torch.Tensor(cell).to(device)
        self.device = device
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        self.ex_pairs = ex_pairs

    def forward(self, xyz):

        nbr_list, pair_dis = generate_nbr_list(xyz, 
                                               self.cutoff, 
                                               self.cell, 
                                               index_tuple=self.index_tuple, 
                                               ex_pairs=self.ex_pairs, 
                                               get_dis=True)

        # compute pair energy 
        energy = self.model(pair_dis[..., None]).sum()

        return energy
   

class Stack(torch.nn.Module):
    def __init__(self, model_dict, mode='sum'):
        super().__init__()
        self.models = ModuleDict(model_dict)
        
    def forward(self, x):
        for i, key in enumerate(self.models.keys()):
            if i == 0:
                result = self.models[key](x).sum().reshape(-1)
            else:
                new_result = self.models[key](x).sum().reshape(-1)
                result += new_result
        
        return result


class BondPotentials(torch.nn.Module):
    def __init__(self, system, top, k, ro):
        super().__init__()
        self.device = system.device
        self.cell = torch.Tensor( system.get_cell() )
        # transform into a diagonal 
        self.cell = self.cell.diag().to(self.device)
        self.k = k 
        self.ro = ro 
        self.top = top.to(self.device)
        
    def forward(self, xyz):
        bond_vec = xyz[self.top[:,0]] - xyz[self.top[:, 1]]
        offsets = get_offsets(bond_vec, self.cell, self.device)
        bond_vec = bond_vec + offsets * self.cell
        bond = bond_vec.pow(2).sum(-1)
        
        energy = 0.5 * self.k * (bond - self.ro).pow(2).sum(-1)
        
        return energy
    
class AnglePotentials(torch.nn.Module):
    def __init__(self, system, top, k, thetao):
        super().__init__()
        self.device = system.device
        self.cell = torch.Tensor( system.get_cell() )
        # transform into a diagonal 
        self.cell = self.cell.diag().to(self.device)
        self.k = k 
        self.thetao = thetao 
        self.top = top.to(self.device)
        
    def forward(self, xyz):
        
        bond_vec1 = xyz[self.top[:,0]] - xyz[self.top[:, 1]]
        bond_vec2 = xyz[self.top[:,2]] - xyz[self.top[:, 1]]
        bond_vec1 = bond_vec1 + get_offsets(bond_vec1, self.cell, self.device) * self.cell
        bond_vec2 = bond_vec2 + get_offsets(bond_vec2, self.cell, self.device) * self.cell
        
        angle_dot = (bond_vec1 * bond_vec2).sum(-1)
        norm = ( bond_vec1.pow(2).sum(-1) * bond_vec2.pow(2).sum(-1) ).sqrt()

        cos = angle_dot / norm
        
        angle = torch.acos(cos)

        energy = 0.5 * self.k * (angle - self.thetao).pow(2).sum(-1)

        return energy


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



