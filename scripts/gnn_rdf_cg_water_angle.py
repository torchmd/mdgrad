#from settings import *
import sys

import torchmd
from scripts import * 
from nff.train import get_model
from torchmd.system import GNNPotentials, PairPotentials, System, Stack
from torchmd.md import Simulations
from torchmd.observable import angle_distribution
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units
import math 



width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}

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

def plot_rdfs(bins, target_g, simulated_g, fname, path):
    plt.title("epoch {}".format(fname))
    plt.plot(bins, simulated_g.detach().cpu().numpy() , linewidth=4, alpha=0.6, label='sim.' )
    plt.plot(bins, target_g.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black', label='exp.')
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(path + '/{}.jpg'.format(fname), bbox_inches='tight')
    plt.show()
    plt.close()

def fit_rdf(assignments, i, suggestion_id, device, sys_params, project_name):
    # parse params 
    data = sys_params['data']
    size = sys_params['size']
    L = sys_params['L']
    end = sys_params['end']
    tmax = sys_params['tmax']
    dt = sys_params['dt']
    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 
    cutoff = assignments['cutoff']
    frameskip = math.ceil(assignments['frameskip_ratio'] * assignments['opt_freq'])

    print('frames skip: ', frameskip)

    nbins = assignments['nbins']
    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    tau = assignments['opt_freq'] 
    print("Training for {} epochs".format(n_epochs))

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

    # initialize states with ASE 
    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol='H',
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)
    system = System(atoms, device=device)
    system.set_temperature(298.0)

    print(system.get_temperature())

    # Initialize potentials 
    model = get_model(gnn_params)
    GNN = GNNPotentials(model, system.get_batch(), system.get_cell_len(), cutoff=cutoff, device=system.device)
    pair = PairPotentials(ExcludedVolume, lj_params,
                    cell=torch.Tensor(system.get_cell_len()), 
                    device=device,
                    cutoff=8.0,
                    ).to(device)

    model = Stack({'gnn': GNN, 'pair': pair})

    # define the equation of motion to propagate 
    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=298.0 * units.kB,
            num_chains=5, 
            adjoint=True).to(device)

    # define simulator with 
    sim = Simulations(system, diffeq)
    start = 2.0

    # initialize observable function 
    obs = rdf(system, nbins, (start, end) )
    vacf_obs = vacf(system, t_range=int(tau//2))

    # angle observation function 
    cos_start = 0.45
    cos_end = 3.1
    nbins_angle_test = 64
    nbins_angle_train = assignments['nbins_angle_train']
    angle_obs_train = angle_distribution(system, nbins_angle_train, (cos_start, cos_end), cutoff=3.75) # 3.25 is from the PCCP paper
    angle_obs_test = angle_distribution(system, nbins_angle_test, (cos_start, cos_end), cutoff=3.75) 

    # get experimental angle distribution 
    cos_exp_train = exp_angle_data(nbins_angle_train, (cos_start, cos_end), fn='../data/water_angle_deepcg_3.7.csv')
    cos_exp_train = torch.Tensor(cos_exp_train).to(device)
    cos_exp_test = exp_angle_data(nbins_angle_test, (cos_start, cos_end), fn='../data/water_angle_deepcg_3.7.csv')
    cos_exp_test = torch.Tensor(cos_exp_test).to(device)

    ANGLE_FACTOR = (cos_end - cos_start)/nbins_angle_train

    xnew = np.linspace(start, end, nbins)
    # get experimental rdf 
    count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs)
    # define optimizer 
    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=assignments['lr'])

    loss_log = []
    test_loss_log = []
    loss_js_log = []
    traj = []

    solver_method = 'NH_verlet'

    cos_exp_train = cos_exp_train / ANGLE_FACTOR

    for i in range(0, n_epochs):
        
        current_time = datetime.now() 
        trajs = sim.simulate(steps=tau, frequency=int(tau))
        v_t, q_t, pv_t = trajs 

        if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
            return 100.0

        if i >= assignments['angle_train_start']:

            bins, sim_angle_density, cos = angle_obs_train(q_t[::frameskip])

            sim_angle_density =  sim_angle_density / ANGLE_FACTOR

            loss_angle = JS_rdf(sim_angle_density, cos_exp_train)  * assignments['angle_JS_weight'] + \
                         (sim_angle_density - cos_exp_train).pow(2).mean() * assignments['angle_MSE_weight']

        else:
            loss_angle = torch.Tensor([0.0]).to(device)

        _, bins, g = obs(q_t[::frameskip])
        
        # this shoud be wrapped in some way 
        loss_js = JS_rdf(g_obs, g)
        loss = loss_js + assignments['mse_weight'] * (g- g_obs).pow(2).mean() + loss_angle

        print(loss_js.item(), loss_angle.item(), loss.item())
        loss.backward()
        
        duration = (datetime.now() - current_time)
        
        optimizer.step()
        optimizer.zero_grad()

        if i % 20 == 0:
           plot_rdfs(xnew, g_obs, g, i, model_path)

           # plotting angles 
           def plot_angle(sim_angle, exp_angle, cos_start, cos_angle, fname, path, nbins_angle):
                bins = np.linspace(cos_start, cos_end, nbins_angle)
                plt.plot(bins * 180/np.pi, sim_angle.detach().cpu(), linewidth=4, alpha=0.6, label='sim.' )
                plt.plot(bins * 180/np.pi, exp_angle.detach().cpu(), linewidth=2,linestyle='--', c='black', label='exp.')
                plt.show()
                plt.savefig(path + '/angle_{}.jpg'.format(fname), bbox_inches='tight')
                plt.close()
           # measure angle distirbutions 
           bins, sim_angle_density, cos_sim = angle_obs_test(q_t[::5].detach())
           test_angle_loss = (sim_angle_density - cos_exp_test).abs().mean() 

           # This is computed incorrectly, the loss_js should be computed on a test calculations. 
           test_loss_log.append(test_angle_loss.item() + loss_js.item())
           plot_angle(sim_angle_density, cos_exp_test, cos_start, cos_end, i, path=model_path, nbins_angle=nbins_angle_test)

        if torch.isnan(loss):
            plt.plot(loss_log)
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            if len(test_loss_log) != 0:   
                return np.array(test_loss_log[-5:-1]).mean()
            else:
                return 100.0

        else:
            loss_log.append(loss_js.item() + loss_angle.item())

        # check for loss convergence
        min_idx = np.array(loss_log).argmin()

        if i >= assignments['angle_train_start'] + 101:
            if i - min_idx >= 100:
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
    for i in range(n_sim):
        _, q_t, _ = sim.simulate(steps=100, frequency=25)

        if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
            return 100.0

        sim_trajs.append(q_t[-1].detach().cpu().numpy())

    sim_trajs = torch.Tensor(np.array(sim_trajs)).to(device)
    sim_trajs.requires_grad = False # no gradient required 

    # compute equilibrate rdf with finer bins 
    test_nbins = 128
    obs = rdf(system, test_nbins,  (start, end))
    xnew = np.linspace(start, end, test_nbins)
    count_obs, g_obs = get_exp_rdf(data, test_nbins, (start, end), obs) # recompute exp. rdf
    _, bins, g = obs(sim_trajs) # compute simulated rdf

    # compute equilibrated rdf 
    loss_js = JS_rdf(g_obs, g)

    save_traj(system, sim_trajs.detach().cpu().numpy(),  
        model_path + '/sim.xyz', skip=1)

    plot_rdfs(xnew, g_obs, g, "final", model_path)

    ANGLE_FACTOR = (cos_end - cos_start)/nbins_angle_test

    for i, traj in enumerate(sim_trajs):
        print(traj.shape)
        bins, sim_angle_density, cos_sim = angle_obs_test(traj)

        sim_angle_density =  sim_angle_density / ANGLE_FACTOR
        #cos_exp_test = cos_exp_test / ANGLE_FACTOR

        if i == 0:
            all_angle_desnity = sim_angle_density
        else:
            all_angle_desnity += sim_angle_density

    all_angle_desnity /= sim_trajs.shape[0]
    cos_exp_test = cos_exp_test/ANGLE_FACTOR

    plot_angle(all_angle_desnity, cos_exp_test, cos_start, cos_end, "angle_final", path=model_path, nbins_angle=nbins_angle_test)

    loss_angle = (all_angle_desnity - cos_exp_test).abs().mean()

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))

    if torch.isnan(loss_js):
        return np.array(test_loss_log[-2:-1]).mean()
    else:
        print(loss_js.item(), loss_angle.item())
        return loss_js.item() + loss_angle.item()

def save_traj(system, traj, fname, skip=10):
    atoms_list = []
    for i, frame in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=frame, numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 
