import torch
from nff.utils.scatter import compute_grad
import numpy as np 
import math 
from ase import units
from torchmd.sovlers import odeint_adjoint, odeint
from ase.geometry import wrap_positions

'''
    Here contains object for simulation and computing the equation of state
'''


class Simulations():

    """Simulation object for handing runnindg MD and logging
    
    Attributes:
        device (str or int): int for GPU, "cpu" for cpu
        integrator (nn.module): function that updates force and velocity n
        keys (list): name of the state variables e.g. "velocities" and "positions "
        log (dict): save state vaiables in numpy arrays 
        solvemethod (str): integration method, current options are 4th order Runge-Kutta (rk4) and Verlet 
        system (torch.System): System object to contain state of molecular systems 
        wrap (bool): if True, wrap the coordinates based on system.cell 
    """
    
    def __init__(self,
                 system,
                  integrator,
                  wrap=True,
                  method="NH_verlet"):

        self.system = system 
        self.device = system.device
        self.integrator = integrator
        self.solvemethod = method
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

    def update_states(self):
        if "positions" in self.log.keys():
            self.system.set_positions(self.log['positions'][-1])
        if "velocities" in self.log.keys():
            self.system.set_velocities(self.log['velocities'][-1])

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

            if self.integrator.adjoint:
                trajs = odeint_adjoint(self.integrator, states, t, method=self.solvemethod)
            else:
                for var in states:
                    var.requires_grad = True 
                trajs = odeint(self.integrator, tuple(states), t, method=self.solvemethod)
            self.update_log(trajs)
            self.update_states()

            states = self.get_check_point()

        return trajs

class NVE(torch.nn.Module):

    """Equation of state for constant energy integrator (NVE ensemble)
    
    Attributes:
        adjoint (str): if True using adjoint sensitivity 
        dim (int): system dimensions
        mass (torch.Tensor): masses of each particle
        model (nn.module): energy functions that takes in coordinates 
        N_dof (int): total number of degree of freedoms
        state_keys (list): keys of state variables "positions", "velocity" etc. 
        system (torchmd.System): system object
    """
    
    def __init__(self, potentials, system, adjoint=True, topology_update_freq=1):
        super().__init__()
        self.model = potentials 
        self.system = system
        self.mass = torch.Tensor(system.get_masses()).to(self.system.device)
        self.N_dof = self.mass.shape[0] * system.dim
        self.dim = system.dim
        self.adjoint = adjoint
        self.state_keys = ['velocities', 'positions']

        self.topology_update_freq = topology_update_freq
        self.update_count = 0

    def update_topology(self, q):

        if self.update_count % self.topology_update_freq == 0:
            self.model._reset_topology(q)
        self.update_count += 1
        
    def forward(self, t, state):
        # pq are the canonical momentum and position variables
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            
            if self.adjoint:
                q.requires_grad = True
            
            p = v * self.mass[:, None]

            self.update_topology(q)
            u = self.model(q)
            f = -compute_grad(inputs=q, output=u.sum(-1))
            dvdt = f

        return (dvdt, v)

    def get_inital_states(self, wrap=True):
        states = [
                self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap)]

        states = [torch.Tensor(var).to(self.system.device) for var in states]

        return states

class NoseHooverChain(torch.nn.Module):

    """Equation of state for NVT integrator using Nose Hoover Chain 

    Nosé, S. A unified formulation of the constant temperature molecular dynamics methods. The Journal of Chemical Physics 81, 511–519 (1984).
    
    Attributes:
        adjoint (str): if True using adjoint sensitivity 
        dim (int): system dimensions
        mass (torch.Tensor): masses of each particle
        model (nn.module): energy functions that takes in coordinates 
        N_dof (int): total number of degree of freedoms
        state_keys (list): keys of state variables "positions", "velocity" etc. 
        system (torchmd.System): system object
        num_chains (int): number of chains 
        Q (float): Heat bath mass
        T (float): Temperature
        target_ke (float): target Kinetic energy 
    """
    
    def __init__(self, potentials, system, T, num_chains=2, Q=1.0, adjoint=True
                ,topology_update_freq=1):
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
        self.state_keys = ['velocities', 'positions', 'baths']
        self.topology_update_freq = topology_update_freq
        self.update_count = 0

    def update_topology(self, q):

        if self.update_count % self.topology_update_freq == 0:
            self.model._reset_topology(q)
        self.update_count += 1


    def update_T(self, T):
        self.T = T 
        
    def forward(self, t, state):
        with torch.set_grad_enabled(True):        
            
            v = state[0]
            q = state[1]
            p_v = state[2]
            
            if self.adjoint:
                q.requires_grad = True
            
            N = self.N_dof
            p = v * self.mass[:, None]

            sys_ke = 0.5 * (p.pow(2) / self.mass[:, None]).sum() 
            
            self.update_topology(q)           
            
            u = self.model(q)
            f = -compute_grad(inputs=q, output=u.sum(-1))

            coupled_forces = (p_v[0] * p.reshape(-1) / self.Q[0]).reshape(-1, 3)

            dpdt = f - coupled_forces

            dpvdt_0 = 2 * (sys_ke - self.T * self.N_dof * 0.5) - p_v[0] * p_v[1]/ self.Q[1]
            dpvdt_mid = (p_v[:-2].pow(2) / self.Q[:-2] - self.T) - p_v[2:]*p_v[1:-1]/ self.Q[2:]
            dpvdt_last = p_v[-2].pow(2) / self.Q[-2] - self.T

            dvdt = dpdt / self.mass[:, None]

        return (dvdt, v, torch.cat((dpvdt_0[None], dpvdt_mid, dpvdt_last[None])))

    def get_inital_states(self, wrap=True):
        states = [
                self.system.get_velocities(), 
                self.system.get_positions(wrap=wrap), 
                [0.0] * self.num_chains]

        states = [torch.Tensor(var).to(self.system.device) for var in states]
        return states


class Isomerization(torch.nn.Module):

    """Quantum isomerization equation of state. 

    The hamiltonian is precomputed in the new basis obtained by orthogonalizing
         the original tensor product space of vibrational and rotational coordinates 
    
    Attributes:
        device (int or str): device
        dim (int): the size of wave function 
        dipole (torch.nn.Parameter): dipole operator
        e_field (torch.nn.Parameter): electric field 
        ham (torch.nn.Parameter): hamiltonian
        max_e_t (int): max time the electric field can be on
    """

    def __init__(self, dipole, e_field, ham, max_e_t, device=0):
        super().__init__()

        self.device = device
        self.dipole = dipole.to(self.device)
        self.ham = ham.to(self.device)
        self.dim = len(ham)
        self.e_field = torch.nn.Parameter(e_field)
        self.max_e_t = max_e_t

    
    def forward(self, t, psi):
        with torch.set_grad_enabled(True):
            psi.requires_grad = True
            # real and imaginary parts of psi
            psi_R = psi[:self.dim]
            psi_I =  psi[self.dim:]

            if t < self.max_e_t.to(t.device):
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

