#from settings import *
import sys

import torchmd
from scripts import * 
from nff.train import get_model
from torchmd.system import System
from torchmd.interface import GNNPotentials, PairPotentials, Stack
from torchmd.md import Simulations
from torchmd.observable import angle_distribution
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.lattice.cubic import FaceCenteredCubic, Diamond
from ase import units

from gnn_fit_lj import pairMLP

import math 

width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}


angle_data_dict = {
   "water":
        {
        2.7: '../data/water_angle_deepcg_2.7.csv',
        3.7: '../data/water_angle_deepcg_3.7.csv', 
        }
}

rdf_data_dict = {
    'Si_2.293_100K': { 'fn': '../data/a-Si/100K_2.293.csv' ,
                       'rho': 2.293,
                        'T': 100.0, 
                        'start': 1.8, 
                        'end': 7.9,
                        'element': "H",
                        'mass': 28.0855,
                        "N_unitcell": 8,
                        "cell": Diamond
                        },
                        
    'Si_2.287_83K': { 'fn': '../data/a-Si/83K_2.287_exp.csv' ,
                       'rho': 2.287,
                        'T': 83.0, 
                        'start': 1.8, 
                        'end': 10.0,
                        'element': "H",
                        'mass': 28.0855,
                        "N_unitcell": 8,
                        "cell": Diamond
                        },

    'Si_2.327_102K_cry': { 'fn': '../data/a-Si/102K_2.327_exp.csv' ,
                       'rho': 2.3267,
                        'T': 102.0, 
                        'start': 1.8, 
                        'end': 8.0,
                        'element': "H",
                        'mass': 28.0855,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        'anneal_flag': True
                        },

    'H20_0.997_298K': { 'fn': "../data/water_exp/water_exp_pccp.csv",
                        'rho': 0.997,
                        'T': 298.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond, #FaceCenteredCubic
                        "pressure": 1.0 # MPa
                        },

    'H20_0.978_342K': { 'fn': "../data/water_exp/water_exp_skinner_342K_0.978.csv",
                       'rho': 0.978,
                        'T': 342.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1,  #MPa
                        "ref": "https://doi.org/10.1063/1.4902412"
                        },

    'H20_0.921_423K_soper': { 'fn': "../data/water_exp/water_exp_Soper_423K_0.9213.csv",
                       'rho': 0.9213,
                        'T': 423.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 10.0, # MPa
                        "ref": "https://doi.org/10.1016/S0301-0104(00)00179-8"
                        },

    'H20_0.999_423K_soper': { 'fn': "../data/water_exp/water_exp_Soper_423K_0.999.csv",
                       'rho': 0.999,
                        'T': 423.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 190, 
                        "ref": "https://doi.org/10.1016/S0301-0104(00)00179-8"
                        },

    'H20_298K_redd': { 'fn': "../data/water_exp/water_exp_298K_redd.csv",
                       'rho': 0.99749,
                        'T': 298.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H20_308K_redd': { 'fn': "../data/water_exp/water_exp_308K_redd.csv",
                       'rho': 0.99448,
                        'T': 308.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H20_338K_redd': { 'fn': "../data/water_exp/water_exp_338K_redd.csv",
                       'rho': 0.98103,
                        'T': 338.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H20_368K_redd': { 'fn': "../data/water_exp/water_exp_368K_redd.csv",
                       'rho': 0.96241,
                        'T': 368.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H2O_long_correlation' : {
                        'ref': 'https://aip.scitation.org/doi/pdf/10.1063/1.4961404'
    },

    'H2O_soper': {
                        'ref': 'https://doi.org/10.1016/S0301-0104(00)00179-8'
    },

    'Argon_1.417_298k': { 'fn': "../data/argon_exp/argon_exp.csv",
                       'rho': 1.417,
                        'T': 298.0, 
                        'start': 2.0, 
                        'end': 9.0,
                        'element': "H",
                        'mass': 39.948,
                        "N_unitcell": 4,
                        "cell": FaceCenteredCubic
                        }
}


def warmup(net, pair, device, size, cutoff):

    print("Warming up ")    
     
    system = get_system('Si_2.327_102K_cry', device, 4)

    GNN = GNNPotentials(system , net, cutoff=cutoff)
    pair = PairPotentials(system , pair,
                    cutoff=8.0,
                    )#.to(device)

    model = Stack({'gnn': GNN, 'pair': pair})

    #import ipdb; ipdb.set_trace()

    optimizer = torch.optim.Adam(list(model.models['gnn'].parameters() ), lr=0.0005)
    
    for epoch in range(200):
        q = torch.Tensor( system.get_positions() ).to(system.device)
        q.requires_grad = True
        u = model(q)

        f = compute_grad(q, u)
        loss = f.pow(2).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print("average forces", f.abs().mean().item())
        

def plot_pair(fn, path, model, prior, device): 

    x = torch.linspace(0.95, 2.5, 50)[:, None].to(device)
    
    u_fit = (model(x) + prior(x)).detach().cpu().numpy()
    u_fit = u_fit = u_fit - u_fit[-1] 

    plt.plot( x.detach().cpu().numpy(), 
              u_fit, 
              label='fit', linewidth=4, alpha=0.6)

    #plt.ylabel("g(r)")
    plt.legend()      
    plt.show()
    plt.savefig(path + '/potential_{}.jpg'.format(fn), bbox_inches='tight')
    plt.close()

    return u_fit

def get_unit_len(rho, mass, N_unitcell):
    
    Na = 6.02214086 * 10**23 # avogadro number 

    N = (rho * 10**6 / mass) * Na  # number of molecules in 1m^3 of water 

    rho = N / (10 ** 30) # number density in 1 A^3
 
    L = (N_unitcell / rho) ** (1/3)
    
    return L 

def get_exp_rdf(data, nbins, r_range, obs):
    # load RDF data 
    f = interpolate.interp1d(data[:,0], data[:,1])
    start = r_range[0]
    end = r_range[1]
    xnew = np.linspace(start, end, nbins)

    # make sure the rdf data is normalized
    V = (4/3)* np.pi * (end ** 3 - start ** 3)
    g_obs = torch.Tensor(f(xnew)).to(obs.device)
    g_obs_norm = ((g_obs.detach() * obs.vol_bins).sum()).item()
    g_obs = g_obs * (V/g_obs_norm)
    count_obs = g_obs * obs.vol_bins / V

    return count_obs, g_obs

def exp_angle_data(nbins, angle_range, fn='../data/water_angle_pccp.csv'):
    angle_data = np.loadtxt(fn, delimiter=',')
    # convert angle to cos(phi)
    cos = angle_data[:, 0] * np.pi / 180
    density = angle_data[:, 1]
    f = interpolate.interp1d(cos, density)
    start = angle_range[0]
    end = angle_range[1]
    xnew = np.linspace(start, end, nbins)
    density = f(xnew)
    density /= density.sum()
    
    return density

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


def get_system(data_str, device, size):

    rho = rdf_data_dict[data_str]['rho']
    mass = rdf_data_dict[data_str]['mass']
    T = rdf_data_dict[data_str]['T']

    # initialize states with ASE 
    cell_module = rdf_data_dict[data_str]['cell']
    N_unitcell = rdf_data_dict[data_str]['N_unitcell']

    L = get_unit_len(rho, mass, N_unitcell)

    print("lattice param:", L)

    atoms = cell_module(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol=rdf_data_dict[data_str]['element'],
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)
    system = System(atoms, device=device)
    system.set_temperature(T)

    return system 

def get_sim(system, model, data_str):

    T = rdf_data_dict[data_str]['T']

    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=T * units.kB,
            num_chains=5, 
            adjoint=True).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    return sim

def get_observer(system, data_str, nbins):

    data_path = rdf_data_dict[data_str]['fn']
    data = np.loadtxt(data_path, delimiter=',')

    # define the equation of motion to propagate 
    start = rdf_data_dict[data_str]['start']
    end = rdf_data_dict[data_str]['end']

    xnew = np.linspace(start, end, nbins)
        # initialize observable function 
    obs = rdf(system, nbins, (start, end) )

    # get experimental rdf 
    count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs)

    return xnew, g_obs, obs


def fit_rdf(assignments, i, suggestion_id, device, sys_params, project_name):
    # parse params 
    size = sys_params['size']
    tmax = sys_params['tmax']
    dt = sys_params['dt']
    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    cutoff = assignments['cutoff']
    frameskip =  2 

    print('frames skip: ', frameskip)

    nbins = assignments['nbins']
    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    tau = assignments['opt_freq'] 
    print("Training for {} epochs".format(n_epochs))

    # get cell parameter and data 
    data_str_list = sys_params['data']

    if 'validate' in sys_params.keys():
        val_str_list = sys_params['validate']
    else:
        val_str_list = []

    system_list = []
    for data_str in data_str_list + val_str_list:
        system = get_system(data_str, device, size) 
        if sys_params['anneal_flag'] == 'True':
            system.set_temperature(assignments['start_T'] * units.kB)
        system_list.append(system)

    # Initialize potentials, one model that simulate all 

    if sys_params["pair_flag"]:

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
        pair = ExcludedVolume(**lj_params)

    else:
        # Define prior potential 
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
        pair = ExcludedVolume(**lj_params)

    # build GNN_list 
    model_list = []
    for i, data_str in enumerate(data_str_list + val_str_list):
        
        if sys_params["pair_flag"]:
            NN = PairPotentials(system_list[i], net,
                        cutoff=assignments['cutoff'],
                        ).to(device)
        else:
            NN = GNNPotentials(system_list[i], 
                                net, 
                                cutoff=cutoff)

        prior = PairPotentials(system_list[i], pair,
                        cutoff=cutoff,
                        ).to(device)

        model = Stack({'nn': NN, 'pair': prior})
        model_list.append(model)

    sim_list = [get_sim(system_list[i], 
                        model_list[i], 
                        data_str) for i, data_str in enumerate(data_str_list + val_str_list)]

    g_obs_list = []
    obs_list = []
    bins_list = []

    for i, data_str in enumerate(data_str_list + val_str_list):
        x, g_obs, obs = get_observer(system_list[i], data_str, nbins)
        bins_list.append(x)
        g_obs_list.append(g_obs)
        obs_list.append(obs)

    # define optimizer 
    optimizer = torch.optim.Adam(list(net.parameters()), lr=assignments['lr'])

    loss_log = []
    test_loss_log = []
    loss_js_log = []
    traj = []

    solver_method = 'NH_verlet'

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  'min', 
                                                  min_lr=0.9e-7, 
                                                  verbose=True, factor = 0.5, patience=15,
                                                  threshold=5e-5)

    for i in range(0, n_epochs):

        # if i % assignments['minimize_freq']:
        #     warmup(net, pair, device, size, cutoff)
        loss_js = torch.Tensor([0.0]).to(device)
        loss_mse = torch.Tensor([0.0]).to(device)

        # temperature annealing 
        for j, sim in enumerate(sim_list):
            data_str = (data_str_list + val_str_list)[j]

            if sys_params['anneal_flag'] == 'True':

                if i % assignments['anneal_freq'] == 0:
                    def get_temp(T_start, T_equil, n_epochs, i, anneal_rate):
                        return (T_start - T_equil) * np.exp( - i * (1/n_epochs) * anneal_rate) + T_equil

                    anneal_rate = assignments['anneal_rate']

                    T_equil = rdf_data_dict[data_str]['T']
                    T_start = assignments['start_T']
                    new_T  = get_temp(T_start, T_equil, n_epochs, i, anneal_rate)
                    sim.intergrator.update_T(new_T * units.kB)

                    print("update T:", new_T)

            current_time = datetime.now() 
            trajs = sim.simulate(steps=tau, frequency=int(tau))
            v_t, q_t, pv_t = trajs 

            if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                return 5 - (i / n_epochs) * 5

            _, bins, g = obs_list[j](q_t[::frameskip])
            
            # only optimize on data that needs training 
            if data_str in data_str_list:
                # this shoud be wrapped in some way 
                loss_js += JS_rdf(g_obs_list[j], g)
                loss_mse += assignments['mse_weight'] * (g- g_obs_list[j]).pow(2).mean() 

            if i % 20 == 0:
                plot_rdfs(bins_list[j], g_obs_list[j], g, "{}_{}".format(data_str, i),
                             model_path, pname=i)

                if sys_params['pair_flag']:
                    potential = plot_pair( path=model_path,
                                 fn=str(i),
                                  model=net, 
                                  prior=pair, 
                                  device=device)

        loss = loss_js + loss_mse 
        loss.backward()
        
        print(loss_js.item(),  loss.item())

        duration = (datetime.now() - current_time)
        
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step(loss)

        if torch.isnan(loss):
            plt.plot(loss_log)
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            if len(test_loss_log) != 0:   
                return np.array(loss_log[-5:-1]).mean() + 1.0
            else:
                return 2 - (i / n_epochs) * 2

        else:
            loss_log.append(loss_js.item() )

        # # check for loss convergence
        # min_idx = np.array(loss_log).argmin()

        # if i >= assignments['angle_train_start'] + 101:
        #     if i - min_idx >= 100:
        #         print("converged")
        #        break
        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr <= 1.0e-7:
            print("training converged")
            break

    plt.plot(loss_log)
    plt.yscale("log")
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.close()

    total_loss = 0.0
    for j, sim in enumerate(sim_list):    

        data_str = (data_str_list + val_str_list)[j]

        train_traj = [var[1] for var in sim.intergrator.traj]

        if (data_str_list + val_str_list)[j] in data_str_list:
            save_traj(system_list[j], train_traj, model_path + '/{}_train.xyz'.format(data_str), skip=10)
        else:
            save_traj(system_list[j], train_traj, model_path + '/{}_val.xyz'.format(data_str), skip=10)

        # Inference 
        sim_trajs = []

        for i in range(n_sim):
            _, q_t, _ = sim.simulate(steps=100, frequency=25)

            if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                return 5.0

            sim_trajs.append(q_t[-1].detach().cpu().numpy())

        sim_trajs = torch.Tensor(np.array(sim_trajs)).to(device)
        sim_trajs.requires_grad = False # no gradient required 

        # compute equilibrate rdf with finer bins 
        test_nbins = 128

        x, g_obs, obs = get_observer(system_list[j], data_str, test_nbins)

        _, bins, g = obs(sim_trajs[::5]) # compute simulated rdf

        # compute equilibrated rdf 
        loss_js = JS_rdf(g_obs, g)

        save_traj(system_list[j], sim_trajs.detach().cpu().numpy(),  
            model_path + '/{}_sim.xyz'.format(data_str), skip=1)

        plot_rdfs(x, g_obs, g, "{}_final".format(data_str), model_path, pname='final')

        total_loss += loss_js.item()

    # ANGLE_FACTOR = (angle_end - angle_start)/nbins_angle_test

    # for i, traj in enumerate(sim_trajs):
    #     print(traj.shape)
    #     bins, sim_angle_density, angle_sim = angle_obs_test(traj)

    #     sim_angle_density =  sim_angle_density / ANGLE_FACTOR
    #     #angle_exp_test = angle_exp_test / ANGLE_FACTOR

    #     if i == 0:
    #         all_angle_desnity = sim_angle_density
    #     else:
    #         all_angle_desnity += sim_angle_density

    # all_angle_desnity /= sim_trajs.shape[0]
    # angle_exp_test = angle_exp_test/ANGLE_FACTOR

    # plot_angle(all_angle_desnity, angle_exp_test, angle_start, angle_end, "angle_final", path=model_path, nbins_angle=nbins_angle_test)

    # loss_angle = JS_rdf(all_angle_desnity,  angle_exp_test)

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))

    if torch.isnan(loss_js):
        return np.array(test_loss_log[-2:-1]).mean()+ 1.0
    else:
        print(loss_js.item())
        return loss_js.item() 

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 
