from settings import *

from nff.train import get_model
from torchmd.system import GNNPotentials, System, Stack
from torchmd.md import Simulations
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units


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
        'n_gaussians': gaussian_dict[assignments['n_gaussians']],
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
    pair = PairPot(ExcludedVolume, lj_params,
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

    xnew = np.linspace(start, end, nbins)
    # get experimental rdf 
    count_obs, g_obs = get_exp_rdf(data, nbins, (start, end), obs)
    # define optimizer 
    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=assignments['lr'])

    loss_log = []
    loss_js_log = []
    traj = []

    solver_method = 'NH_verlet'

    for i in range(0, n_epochs):
        
        current_time = datetime.now() 
        trajs = sim.simulate(steps=tau, frequency=int(tau//2))
        v_t, q_t, pv_t = trajs 
        #import ipdb; ipdb.set_trace()
        # vacf_sim = vacf_obs(v_t.detach())
        # plt.plot(vacf_sim.detach().cpu().numpy())
        # plt.savefig(model_path + '/{}.jpg'.format("vacf"), bbox_inches='tight')
        # plt.show()
        # plt.close()

        _, bins, g = obs(q_t)
        
        if i % 25 == 0:
           plot_rdfs(xnew, g_obs, g, i, model_path)
        
        # this shoud be wrapped in some way 
        loss_js = JS_rdf(g_obs, g)
        loss = loss_js + assignments['mse_weight'] * (g- g_obs).pow(2).sum()
                
        print(loss_js.item(), loss.item())
        loss.backward()
        
        duration = (datetime.now() - current_time)
        
        optimizer.step()
        optimizer.zero_grad()

        if torch.isnan(loss):
            plt.plot(loss_log)
            plt.yscale("log")
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            return np.array(loss_log[-16:-2]).mean()
        else:
            loss_log.append(loss_js.item())

        # check for loss convergence
        min_idx = np.array(loss_log).argmin()

        if i - min_idx >= 125:
            print("converged")
            break

    plt.plot(loss_log)
    plt.yscale("log")
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.close()
    save_traj(system, model_path + '/train.xyz', skip=10)

    # Inference 
    sim_trajs = []
    for i in range(n_sim):
        _, q_t, _ = sim.simulate(steps=100, frequency=25)
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

    save_traj(system, model_path + '/sim.xyz', 
                        skip=1)

    plot_rdfs(xnew, g_obs, g, "final", model_path)

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))

    return loss_js.item()

def save_traj(system, fname, skip=10):
    atoms_list = []
    for i, states in enumerate(system.traj):
        if i % skip == 0: 
            frame = Atoms(positions=states[1], numbers=system.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 

