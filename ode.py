import torch
from nff.utils.scatter import compute_grad
import math 


class ODE(torch.nn.Module):

    def __init__(self, model, mass, device=0):
        super().__init__()
        self.model = model  # declarce model that outputs a dictionary with key ['energy']
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        
    def forward(self, t, pq):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):
            pq.requires_grad = True
            N = int(pq.shape[0]//2)
            
            p = pq[:N]
            q = pq[N:]
            
            q = q.reshape(-1, 3)
            
            u = self.model(q)
            
            v = (p.reshape(-1, 3) / self.mass[:, None]).reshape(-1)
            f = -compute_grad(inputs=q, output=u).reshape(-1)
        return torch.cat((f, v))


class NHODE(torch.nn.Module):

    def __init__(self, model, mass, target_momentum=4.0, ttime = 100.0, device=0):
        super().__init__()
        self.model = model  
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        self.target_momentum = target_momentum
        self.ttime = ttime 
        N = mass.shape[0]
        self.target_ke = (0.5 * 3*  N * self.target_momentum **2 )
        self.Q = (0.5 * 3 *  N * self.target_momentum **2  * ttime ** 2 )
        
    def forward(self, t, pq):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):
            pq.requires_grad = True
            N = int((pq.shape[0] - 1)//2)
            
            p = pq[:N]
            q = pq[N:2* N]
            z = pq[-1]
            
            q = q.reshape(-1, 3)
            
            u = self.model(q)
            
            v = (p.reshape(-1, 3) / self.mass[:, None]).reshape(-1)
            f = -compute_grad(inputs=q, output=u).reshape(-1) -  z * p
            
            sys_ke = (0.5 * (p.reshape(-1, 3).pow(2) / self.mass[:, None]).sum() )
            dzdt = (1/self.Q )* ( sys_ke - self.target_ke )
            
            print(z.item())
        
            print("KE {} Target KE{} ".format( sys_ke.item(), self.target_ke)) 
        return torch.cat((f, v, dzdt[None]))