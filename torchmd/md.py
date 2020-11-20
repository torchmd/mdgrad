import torch
from nff.utils.scatter import compute_grad
import numpy as np 
import math 
from ase import units
from torchmd.sovlers import odeint_adjoint, odeint
from ase.geometry import wrap_positions

def _check_T(T):
    if T >= units.kB * 1000:
        print("The input temperature is {} K, it seems too high.".format(T / units.kB) )

class Simulations():
    
    def __init__(self,
                 system,
                  diffeq,
                  wrap=True,
                  method="NH_verlet"):
        self.system = system 
        self.device = system.device
        self.integrator = diffeq
        self.adjoint = diffeq.adjoint
        self.solvemethod = method
        # flat for printing out simulation status
        self.verbose = True
        self.wrap = wrap
        self.keys = self.integrator.state_keys
        self.initialize_log()

    def initialize_log(self):
        self.log = {}
        for key in self.keys:
            self.log[key] = []

    def update_log(self, trajs):
        for i, key in enumerate( self.keys ):

            if trajs[i][0].device != 'cpu':
                self.log[key].append(trajs[i][-1].detach().cpu().numpy()) 
            else:
                self.log[key].append(trajs[i][-1].detach().numpy()) 

    def get_check_point(self):

        if hasattr(self, 'log'):
            states = [torch.Tensor(self.log[key][-1]).to(self.device) for key in self.log]

            if self.wrap:
                wrapped_xyz = wrap_positions(self.log['positions'][-1], self.system.get_cell())
                states[1] = torch.Tensor(wrapped_xyz).to(self.device)

            return states 
        else:
            raise ValueError("No log available")
        
    def simulate(self, steps=1, dt=1.0 * units.fs, frequency=1):

        if self.log['positions'] == []:
            states = self.integrator.get_inital_states(self.wrap)
        else:
            states = self.get_check_point()

        sim_epochs = int(steps//frequency)
        t = torch.Tensor([dt * i for i in range(frequency)]).to(self.device)

        for epoch in range(sim_epochs):

            if self.adjoint:
                trajs = odeint_adjoint(self.integrator, states, t, method=self.solvemethod)
            else:
                for var in states:
                    var.requires_grad = True 
                trajs = odeint(self.integrator, tuple(states), t, method=self.solvemethod)
                # check for NaN
            #self.integrator.update_traj(tuple([var[-1] for var in trajs]))
            self.update_log(trajs)
            states = self.get_check_point()

        return trajs

class NVE(torch.nn.Module):

    def __init__(self, model, mass, dim=3, device=0):
        super().__init__()
        self.model = model  # declarce model that outputs a dictionary with key ['energy']
        self.mass = torch.Tensor(mass).to(device)
        self.device = device 
        self.dim = dim
        self.state_keys = ['velocities', 'positions']
        
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

class NoseHooverChain(torch.nn.Module):

    def __init__(self, potentials, system, T, num_chains=2, Q=1.0, adjoint=True):
        super().__init__()
        self.model = potentials 
        self.system = system
        self.device = system.device # should just use system.device throughout
        self.mass = torch.Tensor(system.get_masses()).to(self.device)
        self.T = T # in energy unit(eV)
        self.N_dof = self.mass.shape[0] * system.dim
        self.target_ke = (0.5 * self.N_dof * T )
        
        self.num_chains = num_chains
        self.Q = np.array([Q,
                   *[Q/self.system.get_number_of_atoms()]*(num_chains-1)])
        self.Q = torch.Tensor(self.Q).to(self.device)
        self.dim = system.dim
        self.adjoint = adjoint
        self.num_vars = 3 
        self.state_keys = ['velocities', 'positions', 'baths']

        # check temperature
        _check_T(T)

    def update_T(self, T):
        self.T = T 
        
    def forward(self, t, state):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            p_v = state[2]
            
            if self.adjoint:
                q.requires_grad = True
            
            N = self.N_dof
            
            p = v * self.mass[:, None]
            #q = q.reshape(-1, self.dim)

            sys_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum() 
            
            u = self.model(q)
            f = -compute_grad(inputs=q, output=u.sum(-1))

            coupled_forces = (p_v[0] * p.reshape(-1) / self.Q[0]).reshape(-1, 3)

            dvdt = f - coupled_forces

            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1]/ self.Q[1]
            dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:]*p_v[1:-1]/ self.Q[2:]
            dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T
            
        return (dvdt, v, torch.cat((dpvdt_0[None], dpvdt_mid, dpvdt_last[None])))

    def get_inital_states(self, wrap=True):
        states = [
                self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap), 
                [0.0] * self.num_chains]

        states = [torch.Tensor(var).to(self.system.device) for var in states]

        self.traj = []
        return states


class NeuralNVT(torch.nn.Module):

    def __init__(self, potentials, bathnn, system, T, num_chains=2, Q=1.0, adjoint=True):
        super().__init__()
        self.model = potentials 
        self.system = system
        self.device = system.device
        self.mass = torch.Tensor(system.get_masses()).to(self.device)
        self.T = T # in energy unit(eV)
        self.N_dof = self.mass.shape[0] * system.dim
        self.target_ke = (0.5 * self.N_dof * T )
        self.dim = system.dim
        self.adjoint = adjoint
        
        self.bath = bath
        # check temperature
        _check_T(T)

    def initial_conditions():
        pass
        
    def forward(self, t, state):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            
            if self.adjoint:
                q.requires_grad = True
                p.requires_grad = True
            
            N = self.N_dof
            
            p = v.reshape(-1, self.dim) * self.mass[:, None]
            q = q.reshape(-1, self.dim)
            
            u = self.model(q)
            f = -compute_grad(inputs=q, output=u.sum(-1))

            atomwise_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum(-1)
            sys_ke = atomwise_ke.sum()
            bath_input = atomwise_ke - self.T * 3 * 0.5
            bath_u = self.bath(bath_input)
            
            coupled_forces = -compute_grad(inputs=p, output=u.sum(-1))
            
            dvdt = f - coupled_forces
            
            bath_input = atomwise_ke - self.T * 3 * 0.5
            u_bath = self.bath(bath_input)
            
        return (dvdt, v)


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

