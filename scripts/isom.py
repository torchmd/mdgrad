"""Optimize the quantum yield of a retinal model (Hahn, Susanne, and Gerhard Stock.
"Quantum-mechanical modeling of the femtosecond isomerization in rhodopsin."
The Journal of Physical Chemistry B 104.6 (2000): 1146-1149.)"""


import torch
import numpy as np
import os
import torch
import random
import pdb
from math import pi
import sys
import json

from torchmd.md import Isomerization
from torchmd.sovlers import odeint_adjoint as odeint
import argparse

    
# time conversion
FS_TO_EV = 41.341 / 27.2 
# time step
DT = 2 * pi / 2.8 / 60
# max time
TMAX = 1500 * FS_TO_EV
# TMAX = 1 * FS_TO_EV

NUM_EPOCHS = 60
# NUM_EPOCHS = 5

# pulse duration
TAU =  10 * FS_TO_EV 
# center pulse frequency
W0 = 2.4
# pulse time
TP = 3 * TAU

def make_quants():


    """Load matrices for the retinal model.
    Args:
        None
    Returns:
        dic (dict): dictionary of matrices (operators)
    """

    # hamiltonian
    ham = torch.tensor(np.load('../data/isom/hamiltonian.npy')).float().to(DEVICE)
    # dipole operator
    dipole =  torch.tensor(np.load('../data/isom/unitless_mu.npy')).float().to(DEVICE)
    # product projection operator
    prod_op = torch.tensor(np.load('../data/isom/Pt_11.npy')).float().to(DEVICE)
    # reactant projection operator
    reac_op =  torch.tensor(np.load('../data/isom/Pc_00.npy')).float().to(DEVICE)
    # initial wave function is in the ground state
    # note that we double its size: the first half of the elements 
    # are the real part, and the second half is the imaginary part
    psi_0 = torch.zeros(int(2*len(ham))).to(DEVICE)
    psi_0[0] = 1


    dic = {"ham": ham, "dipole": dipole, "prod_op": prod_op,
           "reac_op": reac_op, "psi_0": psi_0}

    return dic


def initialize_Et(dt=DT, tmax=TMAX, w0=W0, tau=TAU, tp=TP):

    """Initialize a reasonable guess for the electric field
    and create the time grid.
    Args:
        dt (float): time step
        tmax (float): max time
        w0 (float): pulse center frequency
        tau (float): pulse duration
        tp (float): pulse arrival time
    Returns:
        combined (torch.Tensor): a tensor with elements
            (t, E(t))
        t_grid (torch.Tensor): time grid for the simulation
        t_grid_0 (torch.Tensor): coarse time grid for the
            first half of the simulation for E(t)
    """

    # number of total steps
    num_steps = int(tmax/dt)
    # number of steps during which the electric field
    # can be non-zero. Take larger time steps so that
    # it can't vary more quickly than we can resolve
    first_num_steps = int(tmax/dt/5)

    # time grid for the electric field part
    t_grid_0 = torch.linspace(0, tmax/2, first_num_steps)
    # time grid for the rest
    t_grid_1 = torch.linspace(t_grid_0[-1] + dt, tmax, int(num_steps/2))

    # time grid for the rest of the simulation (more steps, finer grid)
    t_grid = torch.linspace(0, tmax, num_steps)


    # electric field amplitude: this number maximizes the population that
    # will make it to the excited state
    e0 = pi**0.5 / tau 
    # E(t)
    e_t = e0 * np.cos(w0 * (t_grid_0-tp)) * np.exp(-(t_grid_0-tp)**2 / tau**2)
    # [t, E(t)]
    combined = torch.stack((t_grid_0, e_t), dim=-1)

    return combined, t_grid, t_grid_0

def calc_yield(psi_t, prod_op, reac_op):
    """ Calculate the quantum yield.
    
    Args:
        psi_t (torch.Tensor): wave function as a function of time
        prod_op (torch.Tensor): product operator
        reac_op (torch.Tensor): reactant operator
    Returns:
        expec_t (list): quantum yield as a function of time
    
    """

    # dimension of the Hilbert space
    dim = int(len(psi_t[-1])/2)

    y1_t = []
    y2_t = []
    y3_t = []

    # loop over times
    for i in range(len(psi_t)):

        # real and imaginary parts of psi
        psi_r = psi_t[i, :dim]
        psi_i = psi_t[i, dim:]

        # expression for expectation values from the real and imaginary parts.
        # Valid for real-valued operators (which is the case here)

        # <product>
        expec_r = (psi_r * (torch.matmul(prod_op, psi_r))).sum().reshape(-1)
        expec_i = (psi_i * (torch.matmul(prod_op, psi_i))).sum().reshape(-1)

        # <reactant>
        expec_rC = (psi_r * (torch.matmul(reac_op, psi_r))).sum().reshape(-1)
        expec_iC = (psi_i * (torch.matmul(reac_op, psi_i))).sum().reshape(-1)

        # subtract the part that remained in the ground state
        pg = psi_r[0]**2 + psi_i[0] **2
        y1 = (expec_r + expec_i) / ((expec_r + expec_i) + (expec_rC + expec_iC) - pg)


        # def1
        pC_g = pg + 2 * ((reac_op[0, 1:].reshape(-1) * psi_r[1:]).sum() * psi_r[0] + \
                        (reac_op[0, 1:].reshape(-1) * psi_i[1:]).sum())
        y2 = (expec_r + expec_i) / ((expec_r + expec_i) + (expec_rC + expec_iC) - pC_g)

        # def2 
        y3 = (expec_r + expec_i) / (1 - pg)

        y1_t.append(y1)
        y2_t.append(y2)
        y3_t.append(y3)

    return y1_t, y2_t, y3_t


def objective(expec_t, look_back=20000):

    """
    Creates the objective function to minimize.
    Args:
        expec_t (list): time-dependent quantum yield
        look_back (int): number of previous time steps
            over which to average the yield
    Returns:
        obj (torch.Tensor): objective function
    Note:
        20,000 time steps = 1 ps, since dt = 0.05 fs,
        so this defaults to averaging the QY over 1 ps.
    """

    # want to maximize quantum yield (i.e. minimize its negative)
    obj =  -torch.mean(torch.cat(expec_t)[-look_back:])

    return obj


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-lr", type=float)
    parser.add_argument("-device", type=int)
    parser.add_argument("-nepochs", type=int, default=40)
    parser.add_argument("--adam", action='store_true', default=False)
    params = vars(parser.parse_args())
    print(params)

    DEVICE = params['device']

    # files for saving
    YIELD_FILE = '{}/q_yields.json'.format(params['logdir'])
    FIELD_FILE = '{}/e_fields.json'.format(params['logdir'])
    Y1_FILE = '{}/t_dep_yields1.json'.format(params['logdir'])
    Y2_FILE = '{}/t_dep_yields2.json'.format(params['logdir'])
    Y3_FILE = '{}/t_dep_yields3.json'.format(params['logdir'])

    if os.path.exists(params['logdir']) == False:
        os.makedirs(params['logdir'])


    def main():

        quant_dic = make_quants()
        e_field, t, t_grid_et = initialize_Et()
        max_e_t = max(t_grid_et)

        # initialize the ode
        ode = Isomerization(dipole=quant_dic["dipole"], e_field=e_field, ham=quant_dic["ham"],
                        max_e_t=max_e_t, device=DEVICE).to(DEVICE)

        # define optimizer 
        trainable_params = filter(lambda p: p.requires_grad, ode.parameters())

        if params['adam']:
            optimizer = torch.optim.Adam(trainable_params, lr=params['lr'])
        else:
            optimizer = torch.optim.SGD(trainable_params, lr=params['lr'])

        q_yields = []
        e_fields = []
        y1_traj = []
        y2_traj = []
        y3_traj = []

        for i in range(params['nepochs']):

            print("simulation epoch {}".format(i))
            psi_0 = quant_dic['psi_0'].to(DEVICE)
            psi_t = odeint(ode, psi_0, t, method='rk4')
            y1_t, y2_t, y3_t = calc_yield(psi_t, quant_dic["prod_op"].to(DEVICE), quant_dic["reac_op"].to(DEVICE))

            loss = objective(y1_t)

            loss.backward()
            print("Average quantum yield is {}".format(-loss.item()))

            q_yields.append(-loss.item())
            e_fields.append(ode.e_field.cpu().detach().numpy().tolist())

            # save different yields 
            y1_traj.append([item.item() for item in y1_t])
            y2_traj.append([item.item() for item in y2_t])
            y3_traj.append([item.item() for item in y3_t])

            with open(YIELD_FILE, 'w') as f:
                json.dump(q_yields, f)

            with open(Y1_FILE, 'w') as f:
                json.dump(y1_traj, f)
            with open(Y2_FILE, 'w') as f:
                json.dump(y2_traj, f)
            with open(Y3_FILE, 'w') as f:
                json.dump(y3_traj, f)

            with open(FIELD_FILE, 'w') as f:
                json.dump(e_fields, f)


            optimizer.step()
            optimizer.zero_grad()

    main()
