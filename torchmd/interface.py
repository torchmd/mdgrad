"""Summary
"""
import torch
from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict
from ase import Atoms 
from ase import units
import numpy as np 
from nff.utils.scatter import compute_grad
from torchmd.topology import generate_nbr_list, get_offsets, generate_angle_list, compute_dis

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

    """Interface for energy calculator that requires dynamic update of interaction topology 
    
        example:
            Graph neural networks (GNNPotentials) need to update its topology given coordinates update during the simulation 
            Pair potentials (PairPotentials) need to search for interaction neighbor list 
    
    Attributes:
        cell (torch.Tensor): N x N tensor that specifies the basis of the simulation box 
        device (int or str): integer if for GPU; "cpu" for cpu  
        system (torchmd.system.System): Object to contain simulation information 
    """
    
    def __init__(self, system):
        """Summary
        
        Args:
            system (TYPE): Description
        """
        super(GeneralInteraction, self).__init__()
        self.system = system
        self.cell = torch.Tensor(system.get_cell()).to(system.device)
        self.cell.requires_grad = True 
        self.device = system.device

class SpecificInteraction(torch.nn.Module):

    """Summary
    
    Attributes:
        cell (TYPE): Description
        cell_diag (TYPE): Description
        cutoff (TYPE): Description
        device (TYPE): Description
        system (TYPE): Description
        topology (TYPE): Description
    """
    
    def __init__(self, system, topology):
        """Summary
        
        Args:
            system (TYPE): Description
        """
        super(SpecificInteraction, self).__init__()
        self.system = system 
        self.cell = torch.Tensor(system.get_cell()).to(system.device)
        self.cell.requires_grad = True
        self.device = system.device
        self.topology = topology 


class GNNPotentials(GeneralInteraction):

    """Summary
    
    Attributes:
        cutoff (TYPE): Description
        ex_pairs (TYPE): Description
        gnn (TYPE): Description
        inputs (TYPE): Description
    """
    
    def __init__(self, system, gnn, cutoff, ex_pairs=None):
        """Summary
        
        Args:
            system (TYPE): Description
            gnn (TYPE): Description
            cutoff (TYPE): Description
            ex_pairs (None, optional): Description
        """
        super().__init__(system)
        self.gnn = gnn
        self.cutoff = cutoff
        # initialize the dictionary for model inputs 
        self.inputs = batch_to(self.system.get_batch(), self.device)
        self.ex_pairs = ex_pairs
        self.to(self.device)

        self._reset_topology(torch.Tensor(system.get_positions()).to(system.device))

    def _reset_topology(self, xyz):
        """Summary
        
        Args:
            xyz (TYPE): Description
        """
        self.inputs['nbr_list'], offsets = generate_nbr_list(xyz, self.cutoff, self.cell, ex_pairs=self.ex_pairs)
        self.inputs['offsets'] = offsets

    def forward(self, xyz): 
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        #self._reset_topology(xyz)
        results = self.gnn(self.inputs, xyz)
        return results['energy']


class TPairPotentials(GeneralInteraction):

    """Summary
    
    Attributes:
        cutoff (TYPE): Description
        ex_pairs (TYPE): Description
        index_tuple (TYPE): Description
        model (TYPE): Description
    """
    
    def __init__(self, system, pair_model, T, cutoff=2.5, index_tuple=None, ex_pairs=None, nbr_list_device=None):
        """Summary
        
        Args:
            system (TYPE): Description
            pair_model (TYPE): Description
            cutoff (float, optional): Description
            index_tuple (None, optional): Description
            ex_pairs (None, optional): Description
        """
        super().__init__(system)

        if nbr_list_device == None:
            self.nbr_list_device = system.device
        else:
            self.nbr_list_device = nbr_list_device

        self.model = pair_model
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        self.ex_pairs = ex_pairs
        self.T = T 

        nbr_list, offsets = generate_nbr_list(
                                       torch.Tensor(
                                            system.get_positions()
                                                ).to(self.nbr_list_device), 
                                       self.cutoff, 
                                       self.cell.to(self.nbr_list_device), 
                                       index_tuple=self.index_tuple, 
                                       ex_pairs=self.ex_pairs)

        self.nbr_list = nbr_list.detach().to('cpu')
        self.offsets = offsets.detach().to(system.device)


    def _reset_topology(self, xyz):
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        nbr_list, pair_dis, offsets = generate_nbr_list(xyz.to(self.nbr_list_device), 
                                               self.cutoff, 
                                               self.cell.to(self.nbr_list_device), 
                                               index_tuple=self.index_tuple, 
                                               ex_pairs=self.ex_pairs, 
                                               get_dis=True)

        self.nbr_list = nbr_list.detach().to('cpu')
        self.offsets = offsets.detach().to(xyz.device)

        return nbr_list, pair_dis, offsets

    def forward(self, xyz):

        pair_dis = compute_dis(xyz, self.nbr_list, self.offsets, self.cell)

        # the energy takes temperatures here
        # construct temperature, distance input 
        #input = torch.hstack([ torch.ones_like(pair_dis) * self.T * units.kB, pair_dis])
        energy = self.model(pair_dis, units.kB* self.T).sum()
        return energy

class PairPotentials(GeneralInteraction):

    """Summary
    
    Attributes:
        cutoff (TYPE): Description
        ex_pairs (TYPE): Description
        index_tuple (TYPE): Description
        model (TYPE): Description
    """
    
    def __init__(self, system, pair_model, cutoff=2.5, index_tuple=None, ex_pairs=None, nbr_list_device=None):
        """Summary
        
        Args:
            system (TYPE): Description
            pair_model (TYPE): Description
            cutoff (float, optional): Description
            index_tuple (None, optional): Description
            ex_pairs (None, optional): Description
        """
        super().__init__(system)

        if nbr_list_device == None:
            self.nbr_list_device = system.device
        else:
            self.nbr_list_device = nbr_list_device

        self.model = pair_model
        self.cutoff = cutoff
        self.index_tuple = index_tuple
        self.ex_pairs = ex_pairs

        nbr_list, offsets = generate_nbr_list(
                                       torch.Tensor(
                                            system.get_positions()
                                                ).to(self.nbr_list_device), 
                                       self.cutoff, 
                                       self.cell.to(self.nbr_list_device), 
                                       index_tuple=self.index_tuple, 
                                       ex_pairs=self.ex_pairs)

        self.nbr_list = nbr_list.detach().to('cpu')
        self.offsets = offsets.detach().to(system.device)


    def _reset_topology(self, xyz):
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        nbr_list, pair_dis, offsets = generate_nbr_list(xyz.to(self.nbr_list_device), 
                                               self.cutoff, 
                                               self.cell.to(self.nbr_list_device), 
                                               index_tuple=self.index_tuple, 
                                               ex_pairs=self.ex_pairs, 
                                               get_dis=True)

        self.nbr_list = nbr_list.detach().to('cpu')
        self.offsets = offsets.detach().to(xyz.device)

        return nbr_list, pair_dis, offsets

    def forward(self, xyz):
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        #nbr_list, pair_dis, _ = self._reset_topology(xyz)

        # directly compute distances 
        # compute pair energy 

        pair_dis = compute_dis(xyz, self.nbr_list, self.offsets, self.cell)
        energy = self.model(pair_dis).sum()
        return energy


class Electrostatics(torch.nn.Module):

    """Summary
    
    Attributes:
        cell (TYPE): Description
        charges (TYPE): Description
        conversion (TYPE): Description
        cutoff (TYPE): Description
        device (TYPE): Description
        ex_pairs (TYPE): Description
        index_tuple (TYPE): Description
    """
    
    def __init__(self, charges, cell, device=0, cutoff=2.5, index_tuple=None, ex_pairs=None):
        """Summary
        
        Args:
            charges (TYPE): Description
            cell (TYPE): Description
            device (int, optional): Description
            cutoff (float, optional): Description
            index_tuple (None, optional): Description
            ex_pairs (None, optional): Description
        """
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
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
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

    """Summary
    
    Attributes:
        models (TYPE): Description
    """
    
    def __init__(self, model_dict, mode='sum'):
        """Summary
        
        Args:
            model_dict (TYPE): Description
            mode (str, optional): Description
        """
        super().__init__()
        self.models = ModuleDict(model_dict)

    def _reset_topology(self, x):

        for i, key in enumerate(self.models.keys()):
            self.models[key]._reset_topology(x)
        
    def forward(self, x):
        """Summary
        
        Args:
            x (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        for i, key in enumerate(self.models.keys()):
            if i == 0:
                result = self.models[key](x).sum().reshape(-1)
            else:
                new_result = self.models[key](x).sum().reshape(-1)
                result += new_result
        
        return result


class BondPotentials(torch.nn.Module):

    """Summary
    
    Attributes:
        cell (TYPE): Description
        device (TYPE): Description
        k (TYPE): Description
        ro (TYPE): Description
        top (TYPE): Description
    """
    
    def __init__(self, system, top, k, ro):
        """Summary
        
        Args:
            system (TYPE): Description
            top (TYPE): Description
            k (TYPE): Description
            ro (TYPE): Description
        """
        super().__init__()
        self.device = system.device
        self.cell = torch.Tensor( system.get_cell() )
        # transform into a diagonal 
        self.cell = self.cell.diag().to(self.device)
        self.k = k 
        self.ro = ro 
        self.top = top.to(self.device)

    def _reset_topology(self, xyz):
        pass 
        
    def forward(self, xyz):
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
        bond_vec = xyz[self.top[:,0]] - xyz[self.top[:, 1]]
        offsets = get_offsets(bond_vec, self.cell, self.device)
        bond_vec = bond_vec + offsets * self.cell
        bond = bond_vec.pow(2).sum(-1)
        
        energy = 0.5 * self.k * (bond - self.ro).pow(2).sum(-1)
        
        return energy
    
class AnglePotentials(torch.nn.Module):

    """Summary
    
    Attributes:
        cell (TYPE): Description
        device (TYPE): Description
        k (TYPE): Description
        thetao (TYPE): Description
        top (TYPE): Description
    """
    
    def __init__(self, system, top, k, thetao):
        """Summary
        
        Args:
            system (TYPE): Description
            top (TYPE): Description
            k (TYPE): Description
            thetao (TYPE): Description
        """
        super().__init__()
        self.device = system.device
        self.cell = torch.Tensor( system.get_cell() )
        # transform into a diagonal 
        self.cell = self.cell.diag().to(self.device)
        self.k = k 
        self.thetao = thetao 
        self.top = top.to(self.device)
        
    def forward(self, xyz):
        """Summary
        
        Args:
            xyz (TYPE): Description
        
        Returns:
            TYPE: Description
        """
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

    """Summary
    
    Attributes:
        top (TYPE): Description
    """
    
    def __init__():
        """Summary
        """
        self.top = top

    def _mutate(self, xyz, boundary):
        '''update topology
        
        Args:
            xyz (TYPE): Description
            boundary (TYPE): Description
        '''
        pass

    def _get_topology(): 
        '''return toplogy
        '''
        pass 

    def _stack(self):
        '''stack topology for parrallelized calculations
        '''
        pass
