import torch
from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict
from ase import Atoms 
from ase import units
import numpy as np 
from nff.utils.scatter import compute_grad
from torchmd.topology import generate_nbr_list, get_offsets, generate_angle_list

'''
    basic ideas for parallelization 

    tooplogy should keep system index, for example in nbr_list (it is true for angle_list now), it should look like:
        [[0, 1, 2], 
         [0, 2, 3],
         [1, 1, 3],
            ....  ]

    A function that concatnates different systems and concatate all the toplogies 

    A function should exist to aggregate interaction for each and return in the interface 
    
    The bath variable in md.py should also be able to initalize bath variables for each system copy 
    
    Topology should a indivudal class that handles indexing?

    A Trajectory class to stores system information, interface with system 
'''

class GeneralInteraction(torch.nn.Module):
    def __init__(self, system):
        super(GeneralInteraction, self).__init__()
        self.system = system
        self.cell = torch.Tensor(system.get_cell()).to(system.device)
        self.cell_diag = self.cell.diag()
        self.device = system.device

    def _reset_system(self):
        pass

    def _reset_topology(self, xyz):
        pass

    def forward(self):
        pass

class SpecificInteraction(torch.nn.Module):
    def __init__(self, system):
        super(SpecificInteraction, self).__init__()
        self.cell = torch.Tensor(system.get_cell()).to(system.device)
        self.cell_diag = self.cell.diag()
        self.device = device
        self.cutoff = cutoff
        self.topology = topology 

    def _reset_system(self):
        pass

    def forward(self):
        pass


class GNNPotentialsTrain(GeneralInteraction):
    def __init__(self, system, gnn_module, prior_module):
        '''
            only works for batch size 1 
        '''
        super().__init__(system)
        self.gnn_module = gnn_module
        self.prior = prior_module
        # initialize the dictionary for model inputs 
        self.to(self.device)

    def forward(self, batch): 
        
        xyz = batch['nxyz'][:, 1:]
        xyz.requires_grad = True
        
        results = self.gnn_module(batch, xyz)

        prior_energy = self.prior(xyz)
        prior_grad = compute_grad(xyz, prior_energy)
        
        results['energy'] += prior_energy
        results['energy_grad'] += prior_grad
        
        return results

class GNNPotentials(GeneralInteraction):
    def __init__(self, system, gnn, cutoff, ex_pairs=None):
        super().__init__(system)
        self.gnn = gnn
        self.cutoff = cutoff
        # initialize the dictionary for model inputs 
        self.inputs = batch_to(self.system.get_batch(), self.device)
        self.ex_pairs = ex_pairs
        self.to(self.device)

    def _reset_topology(self, xyz):
        self.inputs['nbr_list'], offsets = generate_nbr_list(xyz, self.cutoff, self.cell_diag, ex_pairs=self.ex_pairs)
        offsets = offsets[self.inputs['nbr_list'][:,0], self.inputs['nbr_list'][:,1], :]
        self.inputs['offsets'] = offsets

    def forward(self, xyz): 
        self._reset_topology(xyz)
        results = self.gnn(self.inputs, xyz)
        return results['energy']


class PairPotentials(GeneralInteraction):

    def __init__(self, system, pair_model, cutoff=2.5, index_tuple=None, ex_pairs=None):
        super().__init__(system)
        self.model = pair_model
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        self.ex_pairs = ex_pairs

    def _reset_topology(self, xyz):
        nbr_list, pair_dis, _ = generate_nbr_list(xyz, 
                                               self.cutoff, 
                                               self.cell_diag, 
                                               index_tuple=self.index_tuple, 
                                               ex_pairs=self.ex_pairs, 
                                               get_dis=True)

        return nbr_list, pair_dis, _

    def forward(self, xyz):
        nbr_list, pair_dis, _ = self._reset_topology(xyz)

        # compute pair energy 
        energy = self.model(pair_dis[..., None]).sum()
        return energy


class Electrostatics(torch.nn.Module):
    def __init__(self, charges, cell, device=0, cutoff=2.5, index_tuple=None, ex_pairs=None):
        super(Electrostatics, self).__init__()
        self.charges = charges.to(device)
        from ase import units 
        k_e = 8.987551787e9
        EV_TO_J = 1.60210e-19
        self.conversion = k_e * units.C**-2 * (1/EV_TO_J)  * (units.m) 
        
        self.cell = torch.Tensor(cell).to(device)
        self.device = device
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        self.ex_pairs = ex_pairs
        
    def forward(self, x):
        nbr_list, pair_dis, _ = generate_nbr_list(x, 
                                               self.cutoff, 
                                               self.cell, 
                                               index_tuple=self.index_tuple, 
                                               ex_pairs=self.ex_pairs, 
                                               get_dis=True)
        
        q1 = self.charges[nbr_list[:,0]]
        q1 = self.charges[nbr_list[:,1]]
        U = -self.conversion * (q1 * q1 /pair_dis)#.sum()

        return U.sum()
   

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


class topology:
    def __init__():
        self.top = top

    def _mutate(self, xyz, boundary):
        '''update topology'''
        pass

    def _get_topology(): 
        '''return toplogy'''
        pass 

    def _stack(self):
        '''stack topology for parrallelized calculations'''
        pass
