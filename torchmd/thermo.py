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
        super(Pressure, self).__init__(system)
        self.model = model
        self.mass = torch.Tensor(system.get_masses()).to(system.device)
        
    def forward(self, q, v):

        u = self.model(q)
        f = -compute_grad(inputs=q, output=u.sum(-1))
        
        N_dof = self.mass.shape[0] * self.system.dim
        
        # compute temperature 
        p = v * self.mass[:, None]         
        ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum()  
        Temperature = ke / (N_dof * 0.5)  
        
        Pideal =  self.system.get_number_of_atoms() * Temperature / self.system.get_volume()
        Pressure = Pideal + (f * q).sum() / (self.system.get_volume() * self.system.dim)
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