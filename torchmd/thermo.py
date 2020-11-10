import torch
import torchmd
import numpy as np
from nff.utils.scatter import compute_grad
from torchmd.observable import Observable
from torchmd.system import check_system



class BulkObservable(torch.nn.Module):
    def __init__(self, system):
        super(BulkObservable, self).__init__()
        check_system(system)
        self.device = system.device
        self.system = system 

class Pressure(BulkObservable):
    def __init__(self, system, model):
        '''
        This paper is a nice read: 
            https://doi.org/10.1016/j.cplett.2006.01.087
        '''
        super(Pressure, self).__init__(system)
        self.model = model
        self.mass = torch.Tensor(system.get_masses()).to(system.device)
        
    def forward(self, q, v):
        
        # use automatic differentiation to compute Virial 
        # u = self.model(q)
        #f = -compute_grad(inputs=q, output=u.sum(-1))
        
        nbr, dis, offsets = self.model._reset_topology(q)
        cell = self.model.cell_diag
        cell.requires_grad = True 
        dis = x[nbr[:,0]] - x[nbr[:,1]] - offsets[nbr[:,0], nbr[:,1]] * cell
        
        N_dof = self.mass.shape[0] * self.system.dim
        
        dis_norm = dis.pow(2).sum(-1).sqrt()
        u = pair.model(dis_norm).sum()
        
        # compute temperature 
        p = v * self.mass[:, None]         
        ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum()  
        Temperature = ke / (N_dof * 0.5)  
        
        Pideal =  self.system.get_number_of_atoms() * Temperature / self.system.get_volume()
        
        Pvirial = compute_grad(cell, u) * (1 / (cell[0] * cell[1]))
        
        Pressure = Pideal - Pvirial 

        return Pressure 


class Temperature(BulkObservable):
    def __init__(self, system):
        super(Temperature, self).__init__(system)
        self.mass = torch.Tensor(system.get_masses()).to(system.device)
        
    def forward(self, v):
        N_dof = self.mass.shape[0] * self.system.dim
        p = v * self.mass[:, None]         
        ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum()  
        Temperature = ke / (N_dof * 0.5)  
        return Temperature # temperature is in energy units 