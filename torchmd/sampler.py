import torch
import numpy as np 
from nff.utils.scatter import compute_grad
import matplotlib.pyplot as plt


class NH_sampler(torch.nn.Module):

    def __init__(self, model, mass, target_momentum=4.0, num_chains=2, ttime = 10.0, dt= 0.005, device=0, dim=3,
                 time_dependent=True):
        super().__init__()
        self.model = model  
        self.mass = torch.Tensor(mass).to(device)
        self.time_dependent = time_dependent
    
        self.device = device 
        self.target_momentum = target_momentum
        self.ttime = ttime 
        self.N_dof = mass.shape[0] * dim
        self.target_ke = (0.5 * self.N_dof * self.target_momentum **2 )
        
        self.T = torch.Tensor( [self.target_momentum **2] ).to(device)
        self.num_chains = num_chains
        self.Q = np.array([1,
                   *[1]*(num_chains-1)]) 
        self.Q = torch.Tensor(self.Q).to(device)
        self.dim = dim
        
    def forward(self, t, state):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):
            
            N = self.N_dof
            
            if self.time_dependent:
                pq = state[0]
            else:
                pq = state 
            
            B = pq.shape[0]
            
            t.requires_grad = True
            pq.requires_grad = True
            t = t.expand(B, 1)
            
            p = pq[:, :N]
            q = pq[:, N:2* N].reshape(-1, self.dim)
            
            sys_ke = 0.5 * (p.reshape(-1, self.dim).pow(2) / self.mass[:, None]).sum(-1) 
            
            # definite all the virtual momentums 
            p_v = pq[:, -self.num_chains:]

            u = self.model(q, t)
            
            dqdt = (p.reshape(-1, self.dim) / self.mass[:, None])#.reshape(-1)
            dpdt = -compute_grad(inputs=q, output=u).reshape(-1, self.dim) - p_v[:, [0]] * p / self.Q[0]
            
            u_sum = u.squeeze()

            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[:, 0] * p_v[:, 1]/ self.Q[1]
            dpvdt_mid = (p_v[:, :-2].pow(2) / self.Q[:-2] - self.T) - p_v[:, 2:]*p_v[:, 1:-1]/ self.Q[2:]
            dpvdt_last = p_v[:, -2].pow(2) / self.Q[-2] - self.T

            f = torch.cat((dpdt, dqdt, dpvdt_0[:, None], dpvdt_mid, dpvdt_last[:, None]), dim=1)

            if self.time_dependent:
                dWdt = compute_grad(inputs=t, output=u_sum) 
                dQdt = (-dqdt * dpdt).sum(-1).reshape(-1, 1)
                return (f, dWdt, dQdt)
            else:
                return f


# construct a NN non-equilibrium flow model 
from torch import nn
import torch

nlr = nn.ReLU()
# initial model: a NN 
class Noneq_NN(torch.nn.Module):
    def __init__(self, device, tau=100, k_0=1.0):
        super(Noneq_NN, self).__init__()
        self.tau = tau
        self.k_0 = k_0
        self.device = device
        self.net = nn.Sequential(
                                nn.Linear(3, 32), 
                                nlr,
                                nn.Linear(32, 64),
                                nlr,  
                                nn.Linear(64, 64),
                                nlr,
                                nn.Linear(64, 32),
                                nlr,
                                nn.Linear(32, 16),
                                nlr,
                                nn.Linear(16, 1),
                                #nn.Tanh()
                               )
        
    def E(self, x, t):
        #print(t.shape, x.shape)
        t = (1 /self.tau) * t 
        inp = torch.cat((x, t.reshape(-1, 1)), dim=1)
        #print(t.shape, inp.shape)
        return self.net(inp) #- self.net(torch.Tensor([0.0]).to(self.device))
    
    def control(self, t): 
        return torch.exp(-(8/self.tau) * t ) 
    
    def forward(self, x, t):

        H1 = (0.5 * self.k_0  * (x ** 2).sum(-1)[:, None])  * self.control(t).reshape(-1, 1)
        #print(H1.shape, self.control(t).shape, (x ** 2).sum(-1)[:, None].shape) 
        H2 = ((1 - self.control(t)).reshape(-1, 1) * self.E(x, t) )#.reshape(-1)

        H = H1 + H2 
        return H
    

class eq_NN(torch.nn.Module):
    def __init__(self, device, tau=100, k_0=1.0):
        super(eq_NN, self).__init__()
        self.tau = tau
        self.k_0 = k_0
        self.device = device
        self.net = nn.Sequential(
                                nn.Linear(2, 32), 
                                nlr,
                                nn.Linear(32, 64),
                                nlr,  
                                nn.Linear(64, 64),
                                nlr,
                                nn.Linear(64, 32),
                                nlr,
                                nn.Linear(32, 16),
                                nlr,
                                nn.Linear(16, 1),
                               )
        
    def E(self, x, t):
        #print(t.shape, inp.shape)
        return self.net(x) #- self.net(torch.Tensor([0.0]).to(self.device))
    
    def forward(self, x, t):
        H = self.E(x, t) 

        return H