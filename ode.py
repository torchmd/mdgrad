import torch
from nff.utils.scatter import compute_grad
import numpy as np 
import math 


class ODE(torch.nn.Module):

    def __init__(self, model, mass, dim=3, device=0):
        super().__init__()
        self.model = model  # declarce model that outputs a dictionary with key ['energy']
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        self.dim = dim
        
    def forward(self, t, pq):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):
            pq.requires_grad = True
            N = int(pq.shape[0]//2)
            
            p = pq[:N]
            q = pq[N:]
            
            q = q.reshape(-1, self.dim)
            
            u = self.model(q)
            
            v = (p.reshape(-1, self.dim) / self.mass[:, None]).reshape(-1)
            f = -compute_grad(inputs=q, output=u).reshape(-1)
            #print(f.shape, f)
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


class NHCHAIN_ODE(torch.nn.Module):

    def __init__(self, model, mass, target_momentum=3.0, num_chains=2, Q=1.0, dt= 0.005, device=0, dim=3):
        super().__init__()
        self.model = model  
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        self.target_momentum = target_momentum
        self.N_dof = mass.shape[0] * dim
        self.target_ke = (0.5 * self.N_dof * self.target_momentum **2 )
        
        self.T = self.target_momentum **2
        self.num_chains = num_chains
#         self.Q = np.array([self.N_dof * self.T * (ttime * dt)**2,
#                    *[self.T * (self.ttime * dt)**2]*(num_chains-1)])
        self.Q = np.array([Q,
                   *[Q/mass.shape[0]]*(num_chains-1)])
        self.Q = torch.Tensor(self.Q).to(device)
        self.dim = dim
        
    def forward(self, t, pq):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):
            pq.requires_grad = True            
            
            N = self.N_dof
            
            p = pq[:N]
            q = pq[N:2* N].reshape(-1, self.dim)
            
            sys_ke = 0.5 * (p.reshape(-1, self.dim).pow(2) \
                 / self.mass[:, None]).sum() 
            
            # definite all the virtual momentums 
            p_v = pq[-self.num_chains:]      
            u = self.model(q)
            
            dqdt = (p.reshape(-1, self.dim) / self.mass[:, None]).reshape(-1)
            dpdt = -compute_grad(inputs=q, output=u.sum(-1)).reshape(-1) - p_v[0] * p / self.Q[0]
        
            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1]/ self.Q[1]
            dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:]*p_v[1:-1]/ self.Q[2:]
            dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T
            
        return torch.cat((dpdt, dqdt, dpvdt_0[None], dpvdt_mid, dpvdt_last[None]))