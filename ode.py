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