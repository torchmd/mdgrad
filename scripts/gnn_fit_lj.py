import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import sys 
import mdtraj
from nglview import show_ase, show_file, show_mdtraj
import torch

import matplotlib
from scipy import interpolate

from ase.lattice.cubic import FaceCenteredCubic
from ase import units

from torchmd.interface import GNNPotentials,PairPotentials, Stack
from torchmd.system import System
from torchmd.potentials import ExcludedVolume, LennardJones
from nff.train import get_model

from torchmd.potentials import ExcludedVolume
from nff.train import get_model

matplotlib.rcParams.update({'font.size': 25})
matplotlib.rc('lines', linewidth=3, color='g')
matplotlib.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams["xtick.major.size"] = 6
matplotlib.rcParams["ytick.major.size"] = 6
matplotlib.rcParams["ytick.major.width"] = 2
matplotlib.rcParams["xtick.major.width"] = 2
matplotlib.rcParams['text.usetex'] = False


width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}


def plot_vacf(vacf_sim, vacf_target, fn, path, dt=0.01):

    t_range = np.linspace(0.0,  vacf_sim.shape[0], vacf_sim.shape[0]) * dt 

    plt.plot(t_range, vacf_sim.detach().cpu().numpy(), label='simulation', linewidth=4, alpha=0.6, )
    plt.plot(t_range, vacf_target.detach().cpu().numpy(), label='target', linewidth=2,linestyle='--', c='black' )

    plt.legend()
    plt.show()
    plt.savefig(path + '/vacf_{}.jpg'.format(fn), bbox_inches='tight')
    plt.close()

def plot_rdf( g_sim, rdf_target, fn, path, start, nbins):

    bins = np.linspace(start, 2.5, nbins)

    plt.plot(bins, g_sim.detach().cpu().numpy() , label='simulation', linewidth=4, alpha=0.6)
    plt.plot(bins, rdf_target.detach().cpu().numpy() , label='target', linewidth=2,linestyle='--', c='black')
    
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")

    plt.show()
    plt.savefig(path + '/rdf_{}.jpg'.format(fn), bbox_inches='tight')
    plt.close()

def get_exp_rdf(data, nbins, r_range, obs):
    # load RDF data 
    f = interpolate.interp1d(data[0], data[1])
    start = r_range[0]
    end = r_range[1]
    xnew = np.linspace(start, end, nbins)
    g_obs = torch.Tensor(f(xnew)).to(obs.device)
    
    return g_obs
    

def JS_rdf(g_obs, g):
    e0 = 1e-5
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).mean()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).mean()

    return loss_js



def fit_lj(assignments, suggestion_id, device, sys_params, project_name):

    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 

    cutoff = assignments['cutoff']
    nbins = assignments['nbins']
    tau = assignments['opt_freq']

    rdf_start = assignments['rdf_start']

    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    print("Training for {} epochs".format(n_epochs))


    gnn_params = {
        'n_atom_basis': width_dict[assignments['n_atom_basis']],
        'n_filters': width_dict[assignments['n_filters']],
        'n_gaussians': int(assignments['cutoff']//assignments['gaussian_width']),
        'n_convolutions': assignments['n_convolutions'],
        'cutoff': assignments['cutoff'],
        'trainable_gauss': False
    }

    size = 3
    L = 1.679 

    atoms = FaceCenteredCubic(directions=[[1, 0, 0], 
                                        [0, 1, 0], 
                                        [0, 0, 1]],
                              symbol='H',
                              size=(size, size, size),
                              latticeconstant=L,
                              pbc=True)

    from torchmd.system import System
    system = System(atoms, device=device)
    system.set_temperature(1.0 /units.kB)

    # Define prior potential 
    lj_params = {'epsilon': 0.1, 
                 'sigma': 0.9,
                "power": 12}

    pair = PairPotentials(system, ExcludedVolume, lj_params,
                    cutoff=cutoff,
                    ).to(device)

    model = get_model(gnn_params)
    GNN = GNNPotentials(system, model,  cutoff=cutoff)
    model = Stack({
                    'gnn': GNN, 
                    'pair': pair
    })

    from torchmd.md import NoseHooverChain 
    diffeq =NoseHooverChain(model, 
                system,
                Q=50.0, 
                T=1.0,
                num_chains=5, 
                adjoint=False).to(device)

    from torchmd.observable import rdf, vacf

    # Set up observable 
    obs = rdf(system, nbins=nbins, r_range=(rdf_start, 2.5))
    vacf_obs = vacf(system, t_range=40)

    nbins = assignments['nbins']

    vacf_target = np.loadtxt('../data/LJ_data/vacf_rho0.884_T1.0_dt0.01.csv', delimiter=',')
    rdf_target = np.loadtxt('../data/LJ_data/rdf_rho0.884_T1.0_dt0.01.csv', delimiter=',')
    rdf_target = get_exp_rdf(rdf_target, nbins, (rdf_start, 2.5), obs)

    vacf_target = torch.Tensor(vacf_target).to(device)
    rdf_target = rdf_target#.to(device)

    from torchmd.md import Simulations
    optimizer = torch.optim.Adam(list(GNN.parameters()), lr=assignments['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  'min', 
                                                  min_lr=1e-8, 
                                                  verbose=True, factor = 0.5, patience= 20,
                                                  threshold=5e-5)

    # Set up simulations 
    sim = Simulations(system, diffeq)

    loss_log = []

    for i in range(sys_params['n_epochs']):

        # Simulate 
        v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=0.01)

        if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
            return 5 - (i / n_epochs) * 5

        # compute observable 
        _, _, g_sim = obs(q_t)

        vacf_sim = vacf_obs(v_t)
        vacf_sim = vacf_sim # / vacf_sim[0]


        loss_vacf = (vacf_sim - vacf_target[:40]).pow(2).mean()
        loss_rdf = (g_sim - rdf_target).pow(2).mean() + JS_rdf(g_sim, rdf_target)

        loss = assignments['rdf_weight'] * loss_rdf + assignments['vacf_weight'] * loss_vacf
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        print(loss_vacf.item(), loss_rdf.item())
        
        scheduler.step(loss)
        
        loss_log.append([loss_vacf.item(), loss_rdf.item() ])
        
        if i % 5 ==0 :
            plot_vacf(vacf_sim, vacf_target[:40], fn=i, path=model_path)

            plot_rdf(g_sim, rdf_target, fn=i, path=model_path, start=rdf_start, nbins=nbins)

        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr <= 5.0e-8:
            print("training converged")
            break

    
    # save loss curve 
    plt.plot(np.array( loss_log)[:, 0])
    plt.plot(np.array( loss_log)[:, 1])
    
    plt.savefig(model_path + '/loss_{}.jpg'.format(i), bbox_inches='tight')
    plt.show()
    plt.close()


    return np.array(loss_log[-10:]).mean(0).sum()
