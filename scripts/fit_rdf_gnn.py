import torchmd
from nff.train import get_model
from torchmd.system import System
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.md import Simulations
from torchmd.observable import angle_distribution
from torchmd.potentials import pairMLP
from ase import units
from data import exp_rdf_data_dict, get_exp_rdf, get_unit_len
from plot import *
from scripts import * 
import os
import json

import math 

width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}


def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 

def JS_rdf(g_obs, g):
    e0 = 1e-4
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).mean()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).mean()

    return loss_js

def plot_rdfs(bins, target_g, simulated_g, fname, path, pname=None):
    plt.title("epoch {}".format(pname))
    plt.plot(bins, simulated_g.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    plt.plot(bins, target_g.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(path + '/{}.jpg'.format(fname), bbox_inches='tight')
    plt.show()
    plt.close()


def get_system(data_tag, device, size):

    rho = exp_rdf_data_dict[data_tag]['rho']
    mass = exp_rdf_data_dict[data_tag]['mass']
    T = exp_rdf_data_dict[data_tag]['T']

    # initialize states with ASE 
    cell_module = exp_rdf_data_dict[data_tag]['cell']
    N_unitcell = exp_rdf_data_dict[data_tag]['N_unitcell']

    L = get_unit_len(rho, mass, N_unitcell)

    print("lattice param: {:.3f} Angstroms".format(L))

    atoms = cell_module(symbol=exp_rdf_data_dict[data_tag]['element'],
                          size=(size, size, size),
                          latticeconstant= L,
                          pbc=True)
    system = System(atoms, device=device)
    system.set_temperature(T * ase.units.kB)

    return system 

def get_sim(system, model, data_tag):

    T = exp_rdf_data_dict[data_tag]['T']

    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=T * units.kB,
            num_chains=5, 
            adjoint=True).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    return sim

def get_observer(system, data_tag, nbins):

    data_path = exp_rdf_data_dict[data_tag]['fn']
    data = np.loadtxt(data_path, delimiter=',')

    # define the equation of motion to propagate 
    start = exp_rdf_data_dict[data_tag]['start']
    end = exp_rdf_data_dict[data_tag]['end']

    xnew = np.linspace(start, end, nbins)
    # initialize observable function 
    obs = rdf(system, nbins, (start, end) )

    # get experimental rdf 
    count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs.device)

    return xnew, g_obs, obs

def get_temp(T_start, T_equil, n_epochs, i, anneal_rate):
    return (T_start - T_equil) * np.exp( - i * (1/n_epochs) * anneal_rate) + T_equil


def get_gnn_potential(assignments,  sys_params):
    lj_params = {'epsilon': assignments['epsilon'], 
                 'sigma': assignments['sigma'], 
                 'power': 12}

    gnn_params = {
        'n_atom_basis': width_dict[assignments['n_atom_basis']],
        'n_filters': width_dict[assignments['n_filters']],
        'n_gaussians': int(assignments['cutoff']//assignments['gaussian_width']),
        'n_convolutions': assignments['n_convolutions'],
        'cutoff': assignments['cutoff'],
        'trainable_gauss': False
    }

    net = get_model(gnn_params)
    prior = ExcludedVolume(**lj_params)
    return net, prior

def get_pair_potential(assignments, sys_params):

    mlp_parmas = {'n_gauss': int(cutoff//assignments['gaussian_width']), 
              'r_start': 0.0,
              'r_end': cutoff, 
              'n_width': assignments['n_width'],
              'n_layers': assignments['n_layers'],
              'nonlinear': assignments['nonlinear']}

    lj_params = {'epsilon': assignments['epsilon'], 
         'sigma': assignments['sigma'],
        "power": assignments['power']}

    net = pairMLP(**mlp_parmas)
    prior = ExcludedVolume(**lj_params)

    return net, prior

def build_simulators(data_list, system_list, net, prior, cutoff, pair_flag): 
    model_list = []
    for i, data_tag in enumerate(data_list):
        prior = PairPotentials(system_list[i], prior,
                        cutoff=cutoff,
                        ).to(system_list[i].device)

        if pair_flag:
            NN = PairPotentials(system_list[i], net,
                cutoff=cutoff,
                ).to(device)
        else:
            NN = GNNPotentials(system_list[i], 
                            net, 
                            cutoff=cutoff)

        model = Stack({'nn': NN, 'pair': prior})
        model_list.append(model)

    sim_list = [get_sim(system_list[i], 
                        model_list[i], 
                        data_tag) for i, data_tag in enumerate(data_list)]

    return sim_list


def fit_rdf(assignments, i, suggestion_id, device, sys_params, project_name):
    # parse params 
    size = sys_params['size']
    dt = sys_params['dt']
    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    cutoff = assignments['cutoff']
    nbins = assignments['nbins']
    tau = assignments['opt_freq'] 


    print(json.dumps( assignments, indent=1) )

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    print("Training for {} epochs".format(n_epochs))

    # get cell parameter and data 
    train_list = sys_params['data']

    if sys_params['val']:
        all_sys = train_list + val_str_list
    else:
        all_sys = train_list

    system_list = []
    for data_tag in all_sys:
        system = get_system(data_tag, device, size) 
        if sys_params['anneal_flag'] == 'True':
            system.set_temperature(assignments['start_T'] * units.kB)
        system_list.append(system)

    # Initialize potentials, one model that simulate all 
    if sys_params["pair_flag"]:
        net, prior = get_pair_potential(assignments, sys_params)

    else:
        net, prior = get_gnn_potential(assignments, sys_params)

    sim_list = build_simulators(all_sys, system_list, net, prior, 
                                cutoff=cutoff, pair_flag=sys_params["pair_flag"])

    g_target_list = []
    obs_list = []
    bins_list = []

    for i, data_tag in enumerate(all_sys):
        x, g_obs, obs = get_observer(system_list[i], data_tag, nbins)
        bins_list.append(x)
        g_target_list.append(g_obs)
        obs_list.append(obs)

    # define optimizer 
    optimizer = torch.optim.Adam(list(net.parameters()), lr=assignments['lr'])

    loss_log = []

    solver_method = 'NH_verlet'
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  'min', 
                                                  min_lr=0.9e-7, 
                                                  verbose=True, factor = 0.5, patience=15,
                                                  threshold=5e-5)

    for i in range(0, n_epochs):

        loss_js = torch.Tensor([0.0]).to(device)
        loss_mse = torch.Tensor([0.0]).to(device)

        # temperature annealing 
        for j, sim in enumerate(sim_list):
            data_tag = all_sys[j]

            if sys_params['anneal_flag'] == 'True' and i % assignments['anneal_freq'] == 0:

                T_equil = exp_rdf_data_dict[data_tag]['T']
                T_start = assignments['start_T']
                new_T  = get_temp(T_start, T_equil, n_epochs, i, assignments['anneal_rate'])
                sim.integrator.update_T(new_T * units.kB)

                print("update T: {:.2f}".format(new_T))

            v_t, q_t, pv_t = sim.simulate(steps=tau, frequency=int(tau))


            if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                return 5 - (i / n_epochs) * 5

            _, bins, g = obs_list[j](q_t[::5])
            
            # only optimize on data that needs training 
            if data_tag in train_list:
                loss_js += JS_rdf(g_target_list[j], g)
                loss_mse += assignments['mse_weight'] * (g- g_target_list[j]).pow(2).mean() 

            if i % 20 == 0:
                plot_rdfs(bins_list[j], g_target_list[j], g, "{}_{}".format(data_tag, i),
                             model_path, pname=i)

                if sys_params['pair_flag']:
                    potential = plot_pair( path=model_path,
                                 fn=str(i),
                                  model=net, 
                                  prior=pair, 
                                  device=device)

        loss = loss_js + loss_mse 
        loss.backward()
        
        print("epoch {} | loss: {:.5f}".format(i, loss.item()) ) 
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step(loss)

        loss_log.append(loss_js.item() )

        if optimizer.param_groups[0]["lr"] <= 1.0e-7:
            print("training converged")
            break

    plt.plot(loss_log)
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.close()

    total_loss = 0.0
    for j, sim in enumerate(sim_list):    

        data_tag = all_sys[j]

        train_traj = sim.log['positions']

        if all_sys[j] in train_list:
            save_traj(system_list[j], train_traj, model_path + '/{}_train.xyz'.format(data_tag), skip=10)
        else:
            save_traj(system_list[j], train_traj, model_path + '/{}_val.xyz'.format(data_tag), skip=10)

        # Inference 
        sim_trajs = []

        for i in range(n_sim):
            _, q_t, _ = sim.simulate(steps=100, frequency=25)
            
        sim_trajs = torch.Tensor( np.stack( sim.log['positions'])).to(system.device)

        # compute equilibrate rdf with finer bins 
        test_nbins = 128

        x, g_obs, obs = get_observer(system_list[j], data_tag, test_nbins)

        _, bins, g = obs(sim_trajs[::20]) # compute simulated rdf

        # compute equilibrated rdf 
        loss_js = JS_rdf(g_obs, g)

        save_traj(system_list[j], sim_trajs.detach().cpu().numpy(),  
            model_path + '/{}_sim.xyz'.format(data_tag), skip=1)

        plot_rdfs(x, g_obs, g, "{}_final".format(data_tag), model_path, pname='final')

        total_loss += loss_js.item()

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))

    return loss_js.item() 