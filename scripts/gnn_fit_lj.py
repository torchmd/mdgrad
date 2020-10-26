import os
import sys 
import numpy as np
import matplotlib.pyplot as plt
import sys 
#import mdtraj
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

from torchmd.md import NoseHooverChain 
from torchmd.observable import rdf, vacf
from torchmd.md import Simulations

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


data_dict = {
    'lj_0.845_1.5': { 
                      'rdf_fn': '../data/LJ_data/rdf_rho0.845_T1.5_dt0.01.csv' ,
                      'vacf_fn': '../data/LJ_data/vacf_rho0.845_T1.5_dt0.01.csv',
                       'rho': 0.845,
                        'T': 1.5, 
                        'start': 0.75, 
                        'end': 3.3,
                        'element': "H",
                        'mass': 1.0,
                        "N_unitcell": 4,
                        "cell": FaceCenteredCubic
                        },

    'lj_0.845_1.0': {
                    'rdf_fn': '../data/LJ_data/rdf_rho0.845_T1.0_dt0.01.csv' ,
                    'vacf_fn': '../data/LJ_data/vacf_rho0.845_T1.0_dt0.01.csv' ,
                   'rho': 0.845,
                    'T': 1.0, 
                    'start': 0.75, 
                    'end': 3.3,
                    'element': "H",
                    'mass': 1.0,
                    "N_unitcell": 4,
                    "cell": FaceCenteredCubic
                    },

    'lj_0.845_0.75': {
                    'rdf_fn': '../data/LJ_data/rdf_rho0.845_T0.75_dt0.01.csv' ,
                    'vacf_fn': '../data/LJ_data/vacf_rho0.845_T0.75_dt0.01.csv' ,
                   'rho': 0.845,
                    'T': 0.75, 
                    'start': 0.75, 
                    'end': 3.3,
                    'element': "H",
                    'mass': 1.0,
                    "N_unitcell": 4,
                    "cell": FaceCenteredCubic
                    }
                }

from nff.nn.layers import GaussianSmearing
from torch import nn

nlr = nn.ReLU()

class MLP(nn.Module):
    def __init__(self, n_gauss, r_start, r_end):
        super(MLP, self).__init__()
        
        self.smear = GaussianSmearing(
            start=r_start,
            stop=r_end,
            n_gaussians=n_gauss,
            trainable=False
        )
        
        self.layers = nn.Sequential(
            nn.Linear(n_gauss, n_gauss),
            nlr,
            nn.Linear(n_gauss, 64),
            nlr,
            nn.Linear(64, 32),
            nlr,
            nn.Linear(32, 1)
        )
        
    def forward(self, r):
        r = self.smear(r)
        r = self.layers(r)
        return r


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

def get_unit_len(rho, N_unitcell):
 
    L = (N_unitcell / rho) ** (1/3)
    
    return L 


def get_system(data_str, device, size):

    rho = data_dict[data_str]['rho']
    T = data_dict[data_str]['T']

    # initialize states with ASE 
    cell_module = data_dict[data_str]['cell']
    N_unitcell = data_dict[data_str]['N_unitcell']

    L = get_unit_len(rho, N_unitcell)

    print("lattice param:", L)

    atoms = cell_module(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol=data_dict[data_str]['element'],
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)
    system = System(atoms, device=device)
    system.set_temperature(T / units.kB)

    return system 

def get_observer(system, data_str, nbins, t_range):

    rdf_data_path = data_dict[data_str]['rdf_fn']
    rdf_data = np.loadtxt(rdf_data_path, delimiter=',')

    vacf_data_path = data_dict[data_str]['vacf_fn']
    vacf_target = np.loadtxt(vacf_data_path, delimiter=',')[:t_range]
    vacf_target = torch.Tensor(vacf_target).to(system.device)

    # define the equation of motion to propagate 
    rdf_start = data_dict[data_str]['start']
    rdf_end = data_dict[data_str]['end']

    xnew = np.linspace(rdf_start , rdf_end, nbins)
        # initialize observable function 
    obs = rdf(system, nbins, (rdf_start , rdf_end) )

    # get experimental rdf 
    rdf_target = get_exp_rdf(rdf_data, nbins, (rdf_start, rdf_end), obs)

    vacf_obs = vacf(system, t_range=t_range) 

    return xnew, rdf_target, obs, vacf_target, vacf_obs

def get_sim(system, model, data_str):

    T = data_dict[data_str]['T']

    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=T,
            num_chains=5, 
            adjoint=True).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    return sim

def fit_lj(assignments, suggestion_id, device, sys_params, project_name):

    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    size = sys_params['size']

    #cutoff = assignments['cutoff']
    nbins = assignments['nbins']
    tau = assignments['opt_freq']

    cutoff = 2.5
    t_range = sys_params['t_range']

    rdf_start = 0.75#assignments['rdf_start']

    data_str_list = sys_params['data']

    if sys_params['val']:
        val_str_list = sys_params['val']
    else:
        val_str_list = []

    print(assignments)

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
              'r_end': 2.5}

    lj_params = {'epsilon': assignments['epsilon'], 
         'sigma': assignments['sigma'],
        "power": assignments['power']}

    NN = MLP(**mlp_parmas)

    pair = ExcludedVolume(**lj_params)

    model_list = []
    for i, data_str in enumerate(data_str_list + val_str_list):

        pairNN = PairPotentials(system_list[i], NN,
                    cutoff=2.5,
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
                                                  min_lr=1e-8, 
                                                  verbose=True, factor = 0.5, patience= 20,
                                                  threshold=5e-5)

    # Set up simulations 
    loss_log = []

    for i in range(sys_params['n_epochs']):

        loss_rdf = torch.Tensor([0.0]).to(device)
        loss_vacf = torch.Tensor([0.0]).to(device)

        # temperature annealing 
        for j, sim in enumerate(sim_list):

            data_str = (data_str_list + val_str_list)[j]

            # Simulate 
            v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=tau, dt=0.01)

            if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                return 5 - (i / n_epochs) * 5

            _, _, g_sim = rdf_obs_list[j](q_t)

            vacf_sim = vacf_obs_list[j](v_t)

            loss_vacf += (vacf_sim - vacf_target_list[j][:t_range]).pow(2).mean()
            loss_rdf += (g_sim - rdf_target_list[j]).pow(2).mean() + JS_rdf(g_sim, rdf_target)

            if i % 25 ==0 :
                plot_vacf(vacf_sim, vacf_target_list[j][:t_range], 
                    fn=data_str + "_{}".format(i), 
                    path=model_path)
                plot_rdf(g_sim, rdf_target_list[j], 
                    fn=data_str + "_{}".format(i),
                     path=model_path, 
                     start=rdf_start, 
                     nbins=nbins)


                def plot_pair(fn, path): 

                    pair_true = LennardJones(1.0, 1.0).to(device)
                    x = torch.linspace(0.95, 2.5, 50)[:, None].to(device)

                    plt.plot( pairNN(x).detach().cpu().numpy() + pair(x).detach().cpu().numpy() + 0.3, 
                              label='fit')
                    plt.plot( pair_true(x).detach().cpu().numpy(), label='truth')
                    plt.legend()      
                    plt.show()
                    plt.savefig(path + '/potential_{}.jpg'.format(fn), bbox_inches='tight')
                    plt.close()

                plot_pair( path=model_path, fn=i)

        loss = assignments['rdf_weight'] * loss_rdf + assignments['vacf_weight'] * loss_vacf
        
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
        
        print(loss_vacf.item(), loss_rdf.item())
        
        scheduler.step(loss)
        
        loss_log.append([loss_vacf.item(), loss_rdf.item() ])

        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr <= 5.0e-8:
            print("training converged")
            break

    # save loss curve 
    plt.plot(np.array( loss_log)[:, 0])
    plt.plot(np.array( loss_log)[:, 1])
    
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.show()
    plt.close()


    return np.array(loss_log[-10:]).mean(0).sum()
