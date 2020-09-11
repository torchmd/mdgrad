import torch
from nff.utils.scatter import compute_grad
import numpy as np 
import math 
from ase import units
from torchmd.sovlers import odeint_adjoint, odeint

class Simulations():
    
    def __init__(self,
                 system,
                  intergrator,
                  adjoint=True,
                  method="NH_verlet"):
        self.system = system 
        self.device = system.device
        self.intergrator = intergrator
        self.adjoint = adjoint
        self.solvemethod = method
        
    def simulate(self, steps=1, dt=1.0 * units.fs, frequency=1):
        
        states = self.system.initial_conditions()
        sim_epochs = int(steps//frequency)
        t = torch.Tensor([dt * i for i in range(frequency)]).to(self.device)

        for epoch in range(sim_epochs):

            if self.adjoint:
                trajs = odeint_adjoint(self.intergrator, states, t, method=self.solvemethod)
            else:
                trajs = odeint(self.intergrator, states, t, method=self.solvemethod)

            self.system.update_traj(tuple([var[-1] for var in trajs]))

        return trajs
    
def _check_T(T):
    if T >= units.kB * 1000:
        print("The input temperature {} K, It seems too high.".format(T / units.kB) )

class NVE(torch.nn.Module):

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

        return torch.cat((f, v))

class NoseHoover(torch.nn.Module):

    def __init__(self, model, mass, target_momentum=4.0, ttime = 100.0, device=0):
        super().__init__()
        self.model = model  
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        self.target_momentum = target_momentum
        self.ttime = ttime 
        N = mass.shape[0]
        self.target_ke = (0.5 * 3 *  N * self.target_momentum **2 )
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
        
            print("KE {} Target KE{} ".format( sys_ke.item(), self.target_ke)) 
        return torch.cat((f, v, dzdt[None]))


class NoseHooverChain(torch.nn.Module):

    def __init__(self, model, mass, T, num_chains=2, Q=1.0,  device=0, dim=3, adjoint=True):
        super().__init__()
        self.model = model  
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        self.T = T
        self.N_dof = mass.shape[0] * dim
        self.target_ke = (0.5 * self.N_dof * T )
        
        self.num_chains = num_chains
        self.Q = np.array([Q,
                   *[Q/mass.shape[0]]*(num_chains-1)])
        self.Q = torch.Tensor(self.Q).to(device)
        self.dim = dim
        self.adjoint = adjoint

        # check temperature
        _check_T(T)
        
    def forward(self, t, state):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            p_v = state[2]
            
            if self.adjoint:
                q.requires_grad = True
            
            N = self.N_dof
            
            p = v.reshape(-1, self.dim) * self.mass[:, None]
            q = q.reshape(-1, self.dim)

            #print(self.mass.max())

            sys_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum() 
            
            u = self.model(q)

            dqdt = (p / self.mass[:, None]).reshape(-1) # mass is just a scalar?
            
            f = -compute_grad(inputs=q, output=u.sum(-1))

            coupled_forces = (p_v[0] * p.reshape(-1) / self.Q[0]).reshape(-1, 3)

            dvdt = f - coupled_forces

            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1]/ self.Q[1]
            dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:]*p_v[1:-1]/ self.Q[2:]
            dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T
            
        return (dvdt, v, torch.cat((dpvdt_0[None], dpvdt_mid, dpvdt_last[None])))

class Isomerization(torch.nn.Module):

    """ODE class for model isomerization"""

    def __init__(self, dipole, e_field, ham, max_e_t, device=0):
        super().__init__()

        self.device = device
        self.dipole = dipole.to(self.device)
        # hamiltonian
        self.ham = ham.to(self.device)
        # hilbert space dimension
        self.dim = len(ham)
        # optimizable electric field
        self.e_field = torch.nn.Parameter(e_field)
        # max time the electric field can be on
        self.max_e_t = max_e_t

    
    def forward(self, t, psi):
        with torch.set_grad_enabled(True):
            psi.requires_grad = True      
            
            # real and imaginary parts of psi
            psi_R = psi[:self.dim]
            psi_I =  psi[self.dim:]

            if t < self.max_e_t:
                # find the value of E at the time closest
                # to now
                t_index = torch.argmin(abs(self.e_field[:, 0] - t))
                e_now = self.e_field[t_index][-1]
            else:
                e_now = 0

            # total Hamiltonian =  H - mu \dot E
            H_eff = self.ham - self.dipole * e_now
            
            # d/dt of real and imaginary parts of psi
            dpsi_R = torch.matmul(H_eff, psi_I)
            dpsi_I = -torch.matmul(H_eff, psi_R)
            
            d_psi_dt = torch.cat((dpsi_R, dpsi_I))

        return d_psi_dt

