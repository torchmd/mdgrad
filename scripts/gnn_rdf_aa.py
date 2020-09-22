#from settings import *
import sys

import torchmd
from scripts import * 
from nff.train import get_model
from torchmd.system import GNNPotentials, PairPotentials, System, Stack, AnglePotentials, BondPotentials
from torchmd.md import Simulations
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torchmd.potentials import ExcludedVolume, LennardJones
from nff.train import get_model
from ase import units
from ase.calculators.tip3p import rOH, angleHOH


width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}

KCAL_TO_EV = 4.3363e-2

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

def JS_rdf(g_obs, g):
    e0 = 1e-4
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).sum()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).sum()

    return loss_js

def MSE_rdf(target, simulated, weight):
    return weight * (target - simulated).pow(2).sum()

def plot_rdfs(bins, target_g, simulated_g, fname, path):
    plt.title("epoch {}".format(fname))
    plt.plot(bins, simulated_g.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    plt.plot(bins, target_g.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(path + '/{}.jpg'.format(fname), bbox_inches='tight')
    plt.show()
    plt.close()

def fit_rdf_aa(assignments, i, suggestion_id, device, sys_params, project_name):
    # parse params 

    tmax = sys_params['tmax']
    dt = sys_params['dt']
    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    cutoff = assignments['cutoff']
    nbins = assignments['nbins']

    esp_scale = assignments['epsilon_scale']
    sigma_scale = assignments['sigma_scale']

    oo_start = 2.25
    oo_end = 5.75
    oh_start = 1.25
    oh_end = 5.75
    hh_start = 1.0
    hh_end = 5.75

    size = 3

    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    tau = assignments['opt_freq'] 
    print("Training for {} epochs".format(n_epochs))

    # initialize systems 
    atoms = ase.io.read("../data/water_init_64.xyz")
    system = System(atoms, device=device)
    system.set_temperature(298.0)

    # Initialize topologies 
    bond_top = [ [3 * i, 3 * i + j + 1] for i in range(size**3) for j in range(2)  ]
    bond_top = torch.LongTensor(bond_top)

    angle_top = [[ 3* i +1, 3 * i, 3 * i + 2] for i in range(size ** 3) ]
    angle_top = torch.LongTensor(angle_top)

    hh_tuple = [[ 3* i +1, 3 * i + 2] for i in range(size ** 3) ]
    hh_tuple = torch.LongTensor(hh_tuple)

    # get atom type index
    o_index = [i * 3 for i in range(size ** 3)]
    h_index = [i * 3 + j + 1 for i in range(size ** 3) for j in range(2)]


    gnn_params = {
        'n_atom_basis': width_dict[assignments['n_atom_basis']],
        'n_filters': width_dict[assignments['n_filters']],
        'n_gaussians': int(assignments['cutoff']//assignments['gaussian_width']),
        'n_convolutions': assignments['n_convolutions'],
        'cutoff': assignments['cutoff'],
        'trainable_gauss': False
    }


    print(system.get_temperature())

    params = {
        'n_atom_basis': 128,
        'n_filters': 128,
        'n_gaussians': 25,
        'n_convolutions': 2,
        'cutoff': 4.5,
        'trainable_gauss': False
    }

    epsilon_scale = 1.0

    pair_oo = PairPotentials(LennardJones, {'epsilon': epsilon_scale * 0.1521 * KCAL_TO_EV, 'sigma': 3.15 * sigma_scale},
                    cell=torch.Tensor(system.get_cell_len()), 
                    device=device,
                    index_tuple=(o_index, o_index),
                    cutoff=5.5,
                    ).to(device)

    pair_oh = PairPotentials(LennardJones, {'epsilon': epsilon_scale * 0.086 * KCAL_TO_EV, 'sigma': 1.77 * sigma_scale},
                    cell=torch.Tensor(system.get_cell_len()), 
                    device=device,
                    index_tuple=(o_index, h_index),
                    ex_pairs=bond_top,
                    cutoff=5.5,
                    ).to(device)

    pair_hh = PairPotentials(LennardJones, {'epsilon': epsilon_scale * 0.046 * KCAL_TO_EV, 'sigma': 0.4 * sigma_scale},
                    cell=torch.Tensor(system.get_cell_len()), 
                    device=device,
                    index_tuple=(h_index, h_index),
                    ex_pairs=hh_tuple,
                    cutoff=5.5,
                    ).to(device)

    # define classical potentials 
    k_bond =  450 * KCAL_TO_EV 
    k_angle = 55 * KCAL_TO_EV 

    bondenergy = BondPotentials(system, bond_top, k_bond, rOH)
    angleenergy = AnglePotentials(system, angle_top, k_angle, angleHOH * np.pi / 180 )

    model = get_model(params)
    GNN = GNNPotentials(model, system.get_batch(), system.get_cell_len(), cutoff=5.5, device=system.device)
    model = Stack({'gnn': GNN, 
                   'pair_oo': pair_oo,
                   'pair_oh': pair_oh, 
                   'pair_hh': pair_hh, 
                   'angle': angleenergy, 
                   'bond': bondenergy})

    # define the equation of motion to propagate 
    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=298.0 * units.kB,
            num_chains=5, 
            adjoint=True).to(device)

    # define simulator with 
    sim = Simulations(system, diffeq)

    # Set up observable 
    obs_oo = rdf(system, nbins=nbins, r_range=(oo_start, oo_end), index_tuple=(o_index, o_index))
    obs_oh = rdf(system, nbins=nbins, r_range=(oh_start, oh_end), index_tuple=(o_index, h_index))
    obs_hh = rdf(system, nbins=nbins, r_range=(hh_start, hh_end), index_tuple=(h_index, h_index))

    # initialize observable function 
    data_oo = np.load("../data/water_exp_pccp.npy")
    data_oh = np.load("../data/water_exp_jcp_oh.npy")
    data_hh = np.load("../data/water_exp_jcp_hh.npy")

    count_obs, g_oo_data = get_exp_rdf(data_oo, nbins, (oo_start, oo_end), obs_oo)
    count_obs, g_oh_data = get_exp_rdf(data_oh, nbins, (oh_start, oh_end), obs_oh)
    count_obs, g_hh_data = get_exp_rdf(data_hh, nbins, (hh_start, hh_end), obs_hh)

    # define optimizer 
    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=assignments['lr'])

    loss_log = []
    loss_js_log = []
    traj = []

    for i in range(0, n_epochs):
        
        current_time = datetime.now() 
        trajs = sim.simulate(steps=tau, frequency=int(tau), dt=0.5 * units.fs)
        v_t, q_t, pv_t = trajs 

        _, bins, g_oo =  obs_oo(q_t[::2])
        _, bins, g_oh =  obs_oh(q_t[::2])
        _, bins, g_hh =  obs_hh(q_t[::2])
        
        loss_oo = JS_rdf(g_oo, g_oo_data) + MSE_rdf(g_oo, g_oo_data, 1.0)
        loss_oh = JS_rdf(g_oh, g_oh_data) + MSE_rdf(g_oh, g_oh_data, 1.0)
        loss_hh = JS_rdf(g_hh, g_hh_data) + MSE_rdf(g_hh, g_hh_data, 1.0)
        loss = loss_oo + loss_oh + loss_hh 

        print(loss_oo.item(), loss_oh.item(), loss_hh.item(), loss.item())
        loss.backward()
        # duration = (datetime.now() - current_time)
        optimizer.step()
        optimizer.zero_grad()

        if i % 25 == 0:
            xnew = np.linspace(oo_start, oo_end, nbins)
            plot_rdfs(xnew, g_oo_data, g_oo, '{} O-O'.format(i), model_path)
            xnew = np.linspace(oh_start, oh_end, nbins)
            plot_rdfs(xnew, g_oh_data, g_oh, '{} O-H'.format(i), model_path)
            xnew = np.linspace(hh_start, hh_end, nbins)
            plot_rdfs(xnew, g_hh_data, g_hh, '{} H-H'.format(i), model_path)

        if torch.isnan(loss):
            plt.plot(loss_log)
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            return np.array(loss_log[-16:-2]).mean()
        else:
            loss_log.append(loss.item())

        loss_log.append(loss.item())

        # check for loss convergence
        min_idx = np.array(loss_log).argmin()

        if i - min_idx >= 125:
            print("converged")
            break

    plt.plot(loss_log)
    plt.yscale("log")
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.close()

    train_traj = [var[1] for var in diffeq.traj]
    save_traj(system, train_traj, model_path + '/train.xyz', skip=10)

    # Inference 
    sim_trajs = []
    del diffeq.traj
    for i in range(n_sim):
        _, q_t, _ = sim.simulate(steps=100, frequency=25)
        sim_trajs.append(q_t[-1].detach().cpu().numpy())

    sim_trajs = torch.Tensor(np.array(sim_trajs)).to(device)
    sim_trajs.requires_grad = False # no gradient required 

    # compute equilibrate rdf with finer bins 
    test_nbins = 128

    obs_oo = rdf(system, nbins=test_nbins, r_range=(oo_start, oo_end), index_tuple=(o_index, o_index))
    obs_oh = rdf(system, nbins=test_nbins, r_range=(oh_start, oh_end), index_tuple=(o_index, h_index))
    obs_hh = rdf(system, nbins=test_nbins, r_range=(hh_start, hh_end), index_tuple=(h_index, h_index))

    count_obs, g_oo_data = get_exp_rdf(data_oo, test_nbins, (oo_start, oo_end), obs_oo)
    count_obs, g_oh_data = get_exp_rdf(data_oh, test_nbins, (oh_start, oh_end), obs_oh)
    count_obs, g_hh_data = get_exp_rdf(data_hh, test_nbins, (hh_start, hh_end), obs_hh)

    _, bins, g_oo =  obs_oo(q_t[::2])
    _, bins, g_oh =  obs_oh(q_t[::2])
    _, bins, g_hh =  obs_hh(q_t[::2])

    # compute equilibrated rdf         
    loss_oo = JS_rdf(g_oo, g_oo_data) 
    loss_oh = JS_rdf(g_oh, g_oh_data)
    loss_hh = JS_rdf(g_hh, g_hh_data)
    loss = loss_oo + loss_oh + loss_hh 

    save_traj(system, sim_trajs.detach().cpu().numpy(),  model_path + '/sim.xyz', skip=1)

    xnew = np.linspace(oo_start, oo_end, test_nbins)
    plot_rdfs(xnew, g_oo_data, g_oo, 'final O-O', model_path)
    xnew = np.linspace(oh_start, oh_end, test_nbins)
    plot_rdfs(xnew, g_oh_data, g_oh, 'final O-H', model_path)
    xnew = np.linspace(hh_start, hh_end, test_nbins)
    plot_rdfs(xnew, g_hh_data, g_hh, 'final H-H', model_path)

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))

    if torch.isnan(loss):
        return np.array(loss_log[-16:-2]).mean()
    else:
        return loss.item()

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 

