import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import sys 
import torch

from scipy import interpolate
from ase import units, Atoms

from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.system import System
from torchmd.potentials import ExcludedVolume, LennardJones, pairMLP
from nff.train import get_model

from torchmd.md import NoseHooverChain 
from torchmd.observable import rdf, vacf
from torchmd.md import Simulations
from data import pair_data_dict
from plot import *

import json 
import matplotlib

# matplotlib.rcParams.update({'font.size': 25})
# matplotlib.rc('lines', linewidth=3, color='g')
# matplotlib.rcParams['axes.linewidth'] = 2.0
# matplotlib.rcParams['axes.linewidth'] = 2.0
# matplotlib.rcParams["xtick.major.size"] = 6
# matplotlib.rcParams["ytick.major.size"] = 6
# matplotlib.rcParams["ytick.major.width"] = 2
# matplotlib.rcParams["xtick.major.width"] = 2
# matplotlib.rcParams['text.usetex'] = False


width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}

from data import get_exp_rdf

def plot_vacf(vacf_sim, vacf_target, fn, path, dt=0.01, save_data=False):

    t_range = np.linspace(0.0,  vacf_sim.shape[0], vacf_sim.shape[0]) * dt 

    plt.plot(t_range, vacf_sim, label='simulation', linewidth=4, alpha=0.6, )

    if vacf_target:
        plt.plot(t_range, vacf_target, label='target', linewidth=2,linestyle='--', c='black' )

    plt.legend()
    plt.show()

    if save_data:
         np.savetxt(path + '/vacf_{}.txt'.format(fn), np.stack((t_range, vacf_sim)), delimiter=',' )

    plt.savefig(path + '/vacf_{}.pdf'.format(fn), bbox_inches='tight')
    plt.close()

def plot_rdf( g_sim, rdf_target, fn, path, start, nbins, save_data=False, end=2.5):

    bins = np.linspace(start, 2.5, nbins)

    plt.plot(bins, g_sim , label='simulation', linewidth=4, alpha=0.6)
    plt.plot(bins, rdf_target , label='target', linewidth=2,linestyle='--', c='black')
    
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")

    if save_data:
        np.savetxt(path + '/rdf_{}.txt'.format(fn), np.stack((bins, g_sim)), delimiter=',' )

    plt.show()
    plt.savefig(path + '/rdf_{}.pdf'.format(fn), bbox_inches='tight')
    plt.close()

def JS_rdf(g_obs, g):
    e0 = 1e-5
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).mean()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).mean()

    return loss_js

def get_unit_len(rho, N_unitcell):
 
    L = (N_unitcell / rho) ** (1/3)
    
    return L 


def get_system(data_str, device, size):


    rho = pair_data_dict[data_str]['rho']
    T = pair_data_dict[data_str]['T']

    dim = pair_data_dict[data_str].get("dim", 3)

    if dim == 3:
        # initialize states with ASE 
        cell_module = pair_data_dict[data_str]['cell']
        N_unitcell = pair_data_dict[data_str]['N_unitcell']

        L = get_unit_len(rho, N_unitcell)

        print("lattice param:", L)

        atoms = cell_module(symbol=pair_data_dict[data_str]['element'],
                                  size=(size, size, size),
                                  latticeconstant= L,
                                  pbc=True)
        system = System(atoms, device=device)
        system.set_temperature(T)

    elif dim == 2:
        positions, cell = lattice_2d(rho, pair_data_dict[data_str]['size'])
        
        print("2D system")
        print("Number of atoms:{}".format(positions.shape[0]))
        print("Cell dim: {}".format(cell))

        atoms = Atoms(numbers=[1.] * positions.shape[0], positions=positions, cell=cell, pbc=True)
        system = System(atoms, device=device, dim=2)
        system.set_temperature(T)

    return system 

def lattice_2d(rho, size):

    L = np.sqrt(size ** 2 / rho)/size # compute cell dim
    v1 = np.array([0,  1., 0.]) * L
    v2 = np.array([1., 0., 0.]) * L

    positions = []
    for i in range(size):
        for j in range(size):
            xyz = np.array(v1) * i + np.array(v2) * j
            positions.append(xyz)
            
    cell_len = L * size        
    cell = np.diag([cell_len] * 3)
    positions = np.array( positions )

    return positions, cell

def get_observer(system, data_str, nbins, t_range):

    rdf_data_path = pair_data_dict[data_str]['rdf_fn']
    rdf_data = np.loadtxt(rdf_data_path, delimiter=',')

    if pair_data_dict[data_str].get("vacf_fn", None):
        vacf_data_path = pair_data_dict[data_str]['vacf_fn']
        vacf_target = np.loadtxt(vacf_data_path, delimiter=',')[:t_range]
        vacf_target = torch.Tensor(vacf_target).to(system.device)
    else:
        vacf_target = None

    # define the equation of motion to propagate 
    rdf_start = pair_data_dict[data_str]['start']
    rdf_end = pair_data_dict[data_str]['end']

    xnew = np.linspace(rdf_start , rdf_end, nbins)
        # initialize observable function 
    obs = rdf(system, nbins, (rdf_start , rdf_end) )

    # get experimental rdf 
    dim = pair_data_dict[data_str].get("dim", 3) 
    _, rdf_target = get_exp_rdf(rdf_data, nbins, (rdf_start, rdf_end), obs, dim=dim)

    vacf_obs = vacf(system, t_range=t_range) 

    return xnew, rdf_target, obs, vacf_target, vacf_obs

def get_sim(system, model, data_str):

    T = pair_data_dict[data_str]['T']

    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=T,
            num_chains=5, 
            adjoint=True).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    return sim

def plot_pair(fn, path, model, prior, device, end=2.5): 

    pair_true = LennardJones(1.0, 1.0).to(device)
    x = torch.linspace(0.1, end, 50)[:, None].to(device)
    
    u_fit = (model(x) + prior(x)).detach().cpu().numpy()
    u_fit = u_fit = u_fit - u_fit[-1] 

    plt.plot( x.detach().cpu().numpy(), 
              u_fit, 
              label='fit', linewidth=4, alpha=0.6)
    
    # plt.plot( x.detach().cpu().numpy(), 
    #           pair_true(x).detach().cpu().numpy(),
    #            label='truth', 
    #            linewidth=2,linestyle='--', c='black')

    plt.ylim(-4, 6)
    plt.legend()      
    plt.show()
    plt.savefig(path + '/potential_{}.jpg'.format(fn), bbox_inches='tight')
    plt.close()

    return u_fit

def fit_lj(assignments, suggestion_id, device, sys_params, project_name):

    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    size = sys_params['size']

    nbins = assignments['nbins']
    tau = assignments['opt_freq']

    cutoff = sys_params['cutoff']
    t_range = sys_params['t_range']

    rdf_start = 0.75

    data_str_list = sys_params['data']

    if sys_params['val']:
        val_str_list = sys_params['val']
    else:
        val_str_list = []

    print(json.dumps(assignments, indent=1))

    model_path = '{}/{}'.format(project_name, suggestion_id)

    os.makedirs(model_path)

    print("Training for {} epochs".format(n_epochs))

    system_list = []
    for data_str in data_str_list + val_str_list:
        system = get_system(data_str, device, size) 
        system_list.append(system)

    # Define prior potential
    mlp_parmas = {'n_gauss': int(cutoff//assignments['gaussian_width']), 
              'r_start': 0.0,
              'r_end': cutoff, 
              'n_width': assignments['n_width'],
              'n_layers': assignments['n_layers'],
              'nonlinear': assignments['nonlinear']}

    lj_params = {'epsilon': assignments['epsilon'], 
         'sigma': assignments['sigma'],
        "power": assignments['power']}

    NN = pairMLP(**mlp_parmas)
    pair = ExcludedVolume(**lj_params)

    model_list = []
    for i, data_str in enumerate(data_str_list + val_str_list):

        pairNN = PairPotentials(system_list[i], NN,
                    cutoff=cutoff,
                    ).to(device)
        prior = PairPotentials(system_list[i], pair,
                        cutoff=2.5,
                        ).to(device)

        model = Stack({'pairnn': pairNN, 'pair': prior})
        model_list.append(model)


    sim_list = [get_sim(system_list[i], 
                        model_list[i], 
                        data_str) for i, data_str in enumerate(data_str_list + val_str_list)]

    from torchmd.observable import rdf, vacf

    nbins = assignments['nbins']

    rdf_obs_list = []
    vacf_obs_list = []

    rdf_target_list = []
    vacf_target_list = []
    rdf_bins_list = []

    for i, data_str in enumerate(data_str_list + val_str_list):
        x, rdf_target, rdf_obs, vacf_target, vacf_obs = get_observer(system_list[i], data_str, nbins, t_range=t_range)
        rdf_bins_list.append(x)

        rdf_obs_list.append(rdf_obs)
        rdf_target_list.append(rdf_target)
        vacf_obs_list.append(vacf_obs)
        vacf_target_list.append(vacf_target)

    from torchmd.md import Simulations
    optimizer = torch.optim.Adam(list(NN.parameters()), lr=assignments['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  'min', 
                                                  min_lr=5e-6, 
                                                  verbose=True, factor = 0.5, patience= 20,
                                                  threshold=5e-5)

    # Set up simulations 
    loss_log = []

    # 
    obs_log = dict()

    for i, data_str in enumerate(data_str_list + val_str_list):
        obs_log[data_str] = {}
        obs_log[data_str]['rdf'] = []
        obs_log[data_str]['vacf'] = []


    for i in range(sys_params['n_epochs']):

        loss_rdf = torch.Tensor([0.0]).to(device)
        loss_vacf = torch.Tensor([0.0]).to(device)

        # temperature annealing 
        for j, sim in enumerate(sim_list):

            data_str = (data_str_list + val_str_list)[j]

            # Simulate 
            v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=0.01)

            if data_str in val_str_list:
                v_t = v_t.detach()
                q_t = q_t.detach()
                pv_t = pv_t.detach()

            if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                return 5 - (i / n_epochs) * 5

            _, _, g_sim = rdf_obs_list[j](q_t)
            vacf_sim = vacf_obs_list[j](v_t)

            if data_str in data_str_list:
                if vacf_target_list[j]:
                    loss_vacf += (vacf_sim - vacf_target_list[j][:t_range]).pow(2).mean()
                else:
                    loss_vacf += 0.0
                loss_rdf += (g_sim - rdf_target_list[j]).pow(2).mean() + JS_rdf(g_sim, rdf_target_list[j])

            obs_log[data_str]['rdf'].append(g_sim.detach().cpu().numpy())
            obs_log[data_str]['vacf'].append(vacf_sim.detach().cpu().numpy())

            if i % 20 ==0 :
                if vacf_target_list[j]:
                    vacf_target = vacf_target_list[j][:t_range].detach().cpu().numpy()
                else:
                    vacf_target = None
                rdf_target = rdf_target_list[j].detach().cpu().numpy()

                plot_vacf(vacf_sim.detach().cpu().numpy(), vacf_target, 
                    fn=data_str + "_{}".format(i), 
                    path=model_path)

                plot_rdf(g_sim.detach().cpu().numpy(), rdf_target, 
                    fn=data_str + "_{}".format(i),
                     path=model_path, 
                     start=rdf_start, 
                     nbins=nbins,
                     end=cutoff)

            if i % 20 ==0 :
                potential = plot_pair( path=model_path,
                             fn=str(i),
                              model=sim.integrator.model.models['pairnn'].model, 
                              prior=sim.integrator.model.models['pair'].model, 
                              device=device,
                              end=cutoff)

        if assignments['train_vacf'] == "True":
            loss = assignments['rdf_weight'] * loss_rdf + assignments['vacf_weight'] * loss_vacf
        else:
            loss = assignments['rdf_weight'] * loss_rdf

        # save potential file
        if np.array(loss_log[-10:]).mean(0).sum() <=  0.01: 
            np.savetxt(model_path + '/potential.txt',  potential, delimiter=',')

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        print(loss_vacf.item(), loss_rdf.item())
        
        scheduler.step(loss)
        
        loss_log.append([loss_vacf.item(), loss_rdf.item() ])

        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr <= 5.0e-6:
            print("training converged")
            break

    for j, sim in enumerate(sim_list):

        data_str = (data_str_list + val_str_list)[j]

        if vacf_target_list[j]:
            vacf_target = vacf_target_list[j][:t_range].detach().cpu().numpy()
        else:
            vacf_target = None

        rdf_target = rdf_target_list[j].detach().cpu().numpy()

        plot_vacf(np.array(obs_log[data_str]['vacf'])[-10:].mean(0), vacf_target, 
            fn=data_str + "_{}".format("final"), 
            path=model_path,
            save_data=True)

        plot_rdf(np.array(obs_log[data_str]['rdf'])[-10:].mean(0), rdf_target, 
            fn=data_str + "_{}".format("final"),
             path=model_path, 
             start=rdf_start, 
             nbins=nbins,
             save_data=True,
             end=cutoff)

        # save rdf data 

    # save loss curve 
    plt.plot(np.array( loss_log)[:, 0], label='vacf', alpha=0.7)
    plt.plot(np.array( loss_log)[:, 1], label='rdf', alpha=0.7)
    plt.yscale("log")
    plt.legend()
    plt.savefig(model_path + '/loss.pdf', bbox_inches='tight')
    plt.show()
    plt.close()


    return np.array(loss_log[-10:]).mean(0).sum()
