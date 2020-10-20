#from settings import *
import sys
import torchmd
from scripts import * 
from nff.train import get_model
from torchmd.system import System
from torchmd.interface import GNNPotentials, GNNPotentialsTrain, PairPotentials, Stack, AnglePotentials, BondPotentials, Electrostatics
from torchmd.md import Simulations
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from torchmd.potentials import ExcludedVolume, LennardJones
from nff.train import get_model
from ase import units
from ase.calculators.tip3p import rOH, angleHOH
from pretrain import * 

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
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).mean()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).mean()

    return loss_js

def MSE_rdf(target, simulated, weight):
    return weight * (target - simulated).pow(2).mean()

def plot_rdfs(bins, target_g, simulated_g, fname, path):
    plt.title("epoch {}".format(fname))
    plt.plot(bins, simulated_g.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    plt.plot(bins, target_g.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(path + '/{}.jpg'.format(fname), bbox_inches='tight')
    plt.show()
    plt.close()

def plot_all(g_oo, g_oo_data, oo_range, g_oh, g_oh_data, oh_range, g_hh, g_hh_data, hh_range, nbins, path, fname):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    fig.set_size_inches(15,4)

    xnew = np.linspace(oo_range[0], oo_range[1], nbins)
    ax1.set_title('O-O')
    ax1.plot(xnew, g_oo.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    ax1.plot(xnew, g_oo_data.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    ax1.set_ylabel("g(r)")
    ax1.set_xlabel("$\AA$")

    ax2.set_title("O-H")
    xnew = np.linspace(oh_range[0], oh_range[1], nbins)
    ax2.plot(xnew, g_oh.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    ax2.plot(xnew, g_oh_data.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    ax2.set_xlabel("$\AA$")

    ax3.set_title('H-H')
    xnew = np.linspace(hh_range[0], hh_range[1], nbins)
    ax3.plot(xnew, g_hh.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    ax3.plot(xnew, g_hh_data.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    ax3.set_xlabel("$\AA$")
    ax3.legend()

    plt.savefig(path + '/{}.jpg'.format(fname), bbox_inches='tight')
    plt.show()
    plt.close()

def fit_rdf_aa(assignments, i, suggestion_id, device, sys_params, project_name):
    # parse params 

    dt = sys_params['dt']
    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    n_train = assignments['n_train']
    cutoff = assignments['cutoff']
    nbins = assignments['nbins']


    sigma_oo = assignments['sigma_oo']
    epsilon_oo = assignments['epsilon_oo']
    sigma_oh = assignments['sigma_oh']
    epsilon_oh = assignments['epsilon_oh']
    sigma_hh = assignments['sigma_hh']
    epsilon_hh = assignments['epsilon_hh']

    oo_start = 2.25
    oo_end = 5.75
    oh_start = 1.25
    oh_end = 5.75
    hh_start = 1.0
    hh_end = 5.75

    # make the starting position of rdf as a param
    oo_start_train = assignments['rdf_start_oo']
    oh_start_train = assignments['rdf_start_oh']
    hh_start_train = assignments['rdf_start_hh']

    skip = assignments['frameskip']
    rdf_smear_width = assignments['rdf_smear_width']

    size = 4

    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    tau = assignments['opt_freq'] 
    print("Training for {} epochs".format(n_epochs))

    # initialize systems 
    size = 4 

    n_mols = 4 ** 3

    from ase.geometry import wrap_positions

    box = np.load('../data/water_aimd/box_0.npy').reshape(-1,  3, 3)
    xyz = np.load('../data/water_aimd/coord_0.npy').reshape(-1, n_mols * 3, 3)

    atoms = Atoms(positions=xyz[0], cell=box[0])
    z =  [8] * 64 + [1] * 64 * 2 

    positions = wrap_positions(xyz[0], box[0])
    atoms.set_positions(positions)
    atoms.set_atomic_numbers(z)

    from torchmd.system import System
    system = System(atoms, device=device)
    system.set_temperature(298.0)

    print("device: ", device)

    # Initialize topologies 
    bond_top = [ [ i,  i + 64 * j] for i in range(64) for j in range(1, 3)]
    bond_top = torch.LongTensor(bond_top)

    angle_top = [[i + 64, i,  i + 64 * 2] for i in range(64)]
    angle_top = torch.LongTensor(angle_top)

    hh_tuple = [[i + 64,  i + 64 * 2] for i in range(64)]
    hh_tuple = torch.LongTensor(hh_tuple)

        # get atom type index
    o_index = [i for i in range(size ** 3)]
    h_index = [i + size ** 3 for i in range(size ** 3 * 2)]


    intra_pairs = torch.cat((hh_tuple, bond_top), dim=0)

    gnn_params = {
        'n_atom_basis': width_dict[assignments['n_atom_basis']],
        'n_filters': width_dict[assignments['n_filters']],
        'n_gaussians': int(assignments['cutoff']//assignments['gaussian_width']),
        'n_convolutions': assignments['n_convolutions'],
        'cutoff': assignments['cutoff'],
        'trainable_gauss': False
    }

    print(system.get_temperature())
    pair_oo = PairPotentials(system, LennardJones, {'epsilon': epsilon_oo, 'sigma': sigma_oo},
                    index_tuple=(o_index, o_index),
                    cutoff=6.0,
                    ).to(device)

    pair_oh = PairPotentials(system, LennardJones, {'epsilon': epsilon_oh, 'sigma': sigma_oh},
                    index_tuple=(o_index, h_index),
                    ex_pairs=bond_top,
                    cutoff=6.0,
                    ).to(device)

    pair_hh = PairPotentials(system, LennardJones, {'epsilon': epsilon_hh, 'sigma': sigma_hh},
                    index_tuple=(h_index, h_index),
                    ex_pairs=hh_tuple,
                    cutoff=6.0,
                    ).to(device)

    # define classical potentials 
    k_bond =  450 * KCAL_TO_EV 
    k_angle = 55 * KCAL_TO_EV 

    bondenergy = BondPotentials(system, bond_top, k_bond, rOH)
    angleenergy = AnglePotentials(system, angle_top, k_angle, angleHOH * np.pi / 180 )

    #print(torch.cat((hh_tuple, bond_top)))

    # initialize coulomb charges 
    charges = torch.Tensor( [-0.834] * (size ** 3) + [0.417] * 2 * (size ** 3) ) * assignments['charge_scale']
    coulomb = Electrostatics(charges, system.get_cell_len(), device=device,
                                cutoff=6, index_tuple=None, ex_pairs=intra_pairs)
    schnet = get_model(gnn_params)

    # pre-training with aimd data 

    # build model wrapper 
    prior = Stack({
                   'pair_oo': pair_oo,
                   'pair_oh': pair_oh, 
                   'pair_hh': pair_hh, 
                   'angle': angleenergy, 
                   'bond': bondenergy,
                   'coulomb': coulomb
                    })

    model = GNNPotentialsTrain(system, schnet, prior)

    if assignments['n_train'] > 0:
        # Train a NN 
        model = pretrain_aimd(model, system, device, gnn_params['cutoff'], model_path, n_train)

    # Build GNN for simulation 
    GNN = GNNPotentials(system, 
                        model.gnn_module, 
                         cutoff=cutoff, 
                         )

    # define simulator with 
    FF = Stack({
           'gnn': GNN,
           'pair_oo': pair_oo,
           'pair_oh': pair_oh, 
           'pair_hh': pair_hh, 
           'angle': angleenergy, 
           'bond': bondenergy,
           'coulomb': coulomb
        })

    diffeq = NoseHooverChain(FF, 
            system,
            Q=50.0, 
            T=298.0 * units.kB,
            num_chains=5, 
            adjoint=True).to(device)

    sim = Simulations(system, diffeq, wrap=True)

    # Set up observable 
    obs_oo = rdf(system, nbins=nbins, r_range=(oo_start_train, oo_end), index_tuple=(o_index, o_index), width=rdf_smear_width)
    obs_oh = rdf(system, nbins=nbins, r_range=(oh_start_train, oh_end), index_tuple=(o_index, h_index), width=rdf_smear_width)
    obs_hh = rdf(system, nbins=nbins, r_range=(hh_start_train, hh_end), index_tuple=(h_index, h_index), width=rdf_smear_width)

    # initialize observable function 
    data_oo = np.load("../data/water_exp_pccp.npy")
    data_oh = np.load("../data/water_exp_jcp_oh.npy")
    data_hh = np.load("../data/water_exp_jcp_hh.npy")

    count_obs, g_oo_data = get_exp_rdf(data_oo, nbins, (oo_start_train, oo_end), obs_oo)
    count_obs, g_oh_data = get_exp_rdf(data_oh, nbins, (oh_start_train, oh_end), obs_oh)
    count_obs, g_hh_data = get_exp_rdf(data_hh, nbins, (hh_start_train, hh_end), obs_hh)


    # Set up test functions 
    test_nbins = 64

    test_obs_oo = rdf(system, nbins=test_nbins, r_range=(oo_start, oo_end), index_tuple=(o_index, o_index))
    test_obs_oh = rdf(system, nbins=test_nbins, r_range=(oh_start, oh_end), index_tuple=(o_index, h_index))
    test_obs_hh = rdf(system, nbins=test_nbins, r_range=(hh_start, hh_end), index_tuple=(h_index, h_index))

    count_obs, test_g_oo_data = get_exp_rdf(data_oo, test_nbins, (oo_start, oo_end), test_obs_oo)
    count_obs, test_g_oh_data = get_exp_rdf(data_oh, test_nbins, (oh_start, oh_end), test_obs_oh)
    count_obs, test_g_hh_data = get_exp_rdf(data_hh, test_nbins, (hh_start, hh_end), test_obs_hh)


    def get_test_loss(traj):

        traj = traj.detach()

        _, bins, test_g_oo =  test_obs_oo(traj[::5])
        _, bins, test_g_oh =  test_obs_oh(traj[::5])
        _, bins, test_g_hh =  test_obs_hh(traj[::5])

        # compute equilibrated rdf         
        loss_oo = JS_rdf(test_g_oo, test_g_oo_data).item()
        loss_oh = JS_rdf(test_g_oh, test_g_oh_data).item()
        loss_hh = JS_rdf(test_g_hh, test_g_hh_data).item()

        return loss_oo + loss_oh + loss_hh

    # define optimizer, only optimize GNN params 
    #optimizer = torch.optim.Adam(list(diffeq.model.models['gnn'].parameters() ), lr=assignments['lr'])
    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=assignments['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                  'min', 
                                                  min_lr=5.0e-8, 
                                                  verbose=True, factor = 0.5, patience= 30,
                                                  threshold=5e-5)

    loss_log = []
    test_loss_log = []
    loss_js_log = []
    traj = []

    for i in range(0, n_epochs):
        
        current_time = datetime.now() 
        trajs = sim.simulate(steps=tau, frequency=int(tau), dt=dt * units.fs)
        v_t, q_t, pv_t = trajs 

        _, bins, g_oo =  obs_oo(q_t[::skip])
        _, bins, g_oh =  obs_oh(q_t[::skip])
        _, bins, g_hh =  obs_hh(q_t[::skip])
        
        loss_oo = JS_rdf(g_oo, g_oo_data) + MSE_rdf(g_oo, g_oo_data, assignments['mse_weight_oo'])
        loss_oh = JS_rdf(g_oh, g_oh_data) + MSE_rdf(g_oh, g_oh_data, assignments['mse_weight_oh'])
        loss_hh = JS_rdf(g_hh, g_hh_data) + MSE_rdf(g_hh, g_hh_data, assignments['mse_weight_hh'])
        loss = loss_oo + loss_oh + loss_hh 

        print(loss_oo.item(), loss_oh.item(), loss_hh.item(), loss.item())
        loss.backward()
        # duration = (datetime.now() - current_time)
        optimizer.step()
        optimizer.zero_grad()

        scheduler.step(loss)

        if i % 25 == 0:
            plot_all(g_oo, g_oo_data, (oo_start_train, oo_end),
                     g_oh, g_oh_data, (oh_start_train, oh_end),
                     g_hh, g_hh_data, (hh_start_train, hh_end),
                     nbins, 
                     model_path, fname="{}".format(i))
            if not torch.isnan(loss):
                test_loss = get_test_loss(q_t.detach())
                test_loss_log.append(test_loss)

            print("getting test loss", np.array(test_loss_log).mean())

        if torch.isnan(loss):
            plt.plot(test_loss_log)
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()

            return np.array(test_loss_log).mean() + (1 - (i / n_epochs)) * 5
        else:
            loss_log.append(loss.item())

        current_lr = optimizer.param_groups[0]["lr"]

        if current_lr <= 1.0e-7:
            print("training converged")
            break

    plt.plot(test_loss_log)
    plt.yscale("log")
    plt.savefig(model_path + '/test_loss.jpg', bbox_inches='tight')
    plt.close()

    plt.plot(loss_log)
    plt.yscale("log")
    plt.savefig(model_path + '/train_loss.jpg', bbox_inches='tight')
    plt.close()

    train_traj = [var[1] for var in diffeq.traj]
    save_traj(system, train_traj, model_path + '/train.xyz', skip=10)

    # Inference 
    sim_trajs = []
    del diffeq.traj
    for i in range(n_sim):
        _, q_t, _ = sim.simulate(steps=100, frequency=25, dt=dt * units.fs)
        sim_trajs.append(q_t[-1].detach().cpu().numpy())

    sim_trajs = torch.Tensor(np.array(sim_trajs)).to(device)
    test_loss = get_test_loss(sim_trajs)

    save_traj(system, sim_trajs.detach().cpu().numpy(),  model_path + '/sim.xyz', skip=1)

    # save final plots 
    _, _, test_g_oo =  test_obs_oo(sim_trajs[::2])
    _, _, test_g_oh =  test_obs_oh(sim_trajs[::2])
    _, _, test_g_hh =  test_obs_hh(sim_trajs[::2])

    plot_all(test_g_oo, test_g_oo_data, (oo_start, oo_end),
             test_g_oh, test_g_oh_data, (oh_start, oh_end),
             test_g_hh, test_g_hh_data, (hh_start, hh_end),
             test_nbins, 
             model_path, 
             fname="final" )

    np.savetxt(model_path + '/loss.csv', np.array(test_loss_log))

    if torch.isnan(torch.Tensor([test_loss])):
        return np.array(test_loss_log[-5:-1]).mean() + 0.25
    else:
        return test_loss

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 

