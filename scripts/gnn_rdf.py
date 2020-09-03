from settings import *

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
    xnew = np.linspace(0.0, r_range, nbins)

    V = (4/3)* np.pi * (r_range) ** 3
    g_obs = torch.Tensor(f(xnew)).to(obs.device)
    g_obs_norm = ((g_obs.detach() * obs.vol_bins).sum()).item()
    g_obs = g_obs * (V/g_obs_norm)
    count_obs = g_obs * obs.vol_bins / V

    return count_obs, g_obs

def evaluate_model(assignments, i, suggestion_id, device, sys_params, project_name):

    data = sys_params['data']
    size = sys_params['size']
    L = sys_params['L']
    r_range = sys_params['r_range']
    tmax = sys_params['tmax']
    dt = sys_params['dt']
    n_epochs = sys_params['n_epochs'] 
    n_sim = sys_params['n_sim'] 

    nbins = assignments['nbins']
    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    tau = assignments['opt_freq'] 
    print("Training for {} epochs".format(n_epochs))

    # set up lattices to have the same density as water at 300K 1atm
    CUTOFF = assignments['cutoff']

    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol='H',
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)

    N = atoms.get_number_of_atoms()
    mass = atoms.get_masses()

    print(np.array(atoms.get_cell()))

    # construct graphs 
    edge_from, edge_to, offsets = neighbor_list('ijS', atoms, CUTOFF)
    nbr_list = torch.LongTensor(np.stack([edge_from, edge_to], axis=1))
    offsets = torch.Tensor(offsets)[nbr_list[:, 1] > nbr_list[:, 0]].detach().cpu().numpy()
    nbr_list = nbr_list[nbr_list[:, 1] > nbr_list[:,0]]

    # contruct NN model 
    atoms = AtomsBatch(atoms)
    batch = {'nxyz': torch.Tensor(atoms.get_nxyz()), 
             'nrb_list': nbr_list, 
             'num_atoms': torch.LongTensor([atoms.get_number_of_atoms()]),
             'energy': torch.Tensor([0.]) }

    # Define prior potential 
    lj_params = {'epsilon': assignments['epsilon'], 
                 'sigma': assignments['sigma'], 
                 'power': 12}

    pair = PairPot(ExcludedVolume, lj_params,
                    cell=torch.Tensor(atoms.get_cell()).diag(), 
                    device=device,
                    cutoff=9.0,
                    ).to(device)

    params = {
        'n_atom_basis': width_dict[assignments['n_atom_basis']],
        'n_filters': width_dict[assignments['n_filters']],
        'n_gaussians': width_dict[assignments['n_gaussians']],
        'n_convolutions': assignments['n_convolutions'],
        'cutoff': assignments['cutoff'],
        'trainable_gauss': False
    }

    model = get_model(params)
    batch = batch_to(batch, device)
    wrap = schwrap(model=model, batch=batch, device=device , cell=np.diag(atoms.get_cell())) 

    model_dict = {'gcn': wrap,
                 'prior': pair}

    stack = Stack(model_dict)

    T= 298.0 * units.kB

    # declare position and momentum as initial values
    xyz = torch.Tensor(atoms.get_positions())
    xyz = xyz.reshape(-1)

    # generate random velocity 
    MaxwellBoltzmannDistribution(atoms, T)

    p = torch.Tensor( atoms.get_velocities().reshape(-1)) #.to(device)
    p_v = torch.Tensor([0.0] * 5)
    pq = torch.cat((p, xyz, p_v)).to(device)
    pq.requires_grad= True

    # define the equation of motion to propagate 
    f_x = NHCHAIN_ODE(stack, 
            mass, 
            Q=50.0, 
            T=T,
            num_chains=5, 
            device=device).to(device)

    # initialize observable function 
    obs = rdf(atoms, nbins, device, r_range)

    xnew = np.linspace(0.0, r_range, nbins)
    count_obs, g_obs = get_exp_rdf(data, nbins, r_range, obs)

    # compute target observable 
    # define optimizer 
    optimizer = torch.optim.Adam(list( f_x.parameters() ), lr=assignments['lr'])

    e0 = 1e-4

    loss_log = []
    loss_js_log = []
    traj = []

    for i in range(0, n_epochs):
        
        current_time = datetime.now() 
        
        if i == 0:
            xyz = torch.Tensor(atoms.get_positions() 
                              # + np.random.rand(N, 3) * 0.1
                              ).reshape(-1) 
            # generate random velocity 
            p = torch.Tensor( atoms.get_velocities().reshape(-1))
            p_v = torch.Tensor([0.0] * 5)
            t = torch.Tensor([dt * units.fs * i for i in range(tau)]).to(device)
            
        else:
            xyz = frames[-1].detach().cpu()#.reshape(-1)
            xyz = torch.Tensor( wrap_positions( xyz.numpy(), atoms.get_cell()) ).reshape(-1)
            p = x[-1, :N * 3 ].detach().cpu().reshape(-1)
            p_v = x[-1, N*3*2: ].detach().cpu().reshape(-1)
            t = torch.Tensor([dt * units.fs * i for i in range(tau)]).to(device)

            traj.append(xyz.detach().cpu().numpy().reshape(-N, 3))

        pq = torch.cat((p, xyz, p_v)).to(device)
        pq.requires_grad= True
        x = odeint(f_x, pq, t, method='rk4')
        
        frames = x[::3, N*3: N*3*2].reshape(-1, N, 3)
        _, bins, g = obs(frames)
        
        if i % 20 == 0:
            plt.title("epoch {}".format(i))
            plt.plot(xnew, g.detach().cpu().numpy() , linewidth=4, alpha=0.6,)
            plt.plot(xnew, g_obs.detach().cpu().numpy(), linewidth=2,linestyle='--', c='black')
            plt.xlabel("$\AA$")
            plt.ylabel("g(r)")
            plt.savefig(model_path + '/{}.jpg'.format(i), bbox_inches='tight')
            plt.show()
            plt.close()
        
        g_m = 0.5 * (g_obs + g)
        loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).sum()
        loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).sum()
        loss = loss_js + assignments['mse_weight'] * (g- g_obs).pow(2).sum()
                
        print(loss_js.item(), loss.item())
        loss.backward()
        
        duration = (datetime.now() - current_time)
        #print( "{} seconds".format(duration.total_seconds())) 
        
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

        if i - min_idx >= 50:
            print("converged")
            break

    plt.plot(loss_log)
    plt.yscale("log")
    plt.savefig(model_path + '/loss.jpg', bbox_inches='tight')
    plt.close()

    save_traj(traj, atoms, model_path + '/train.xyz', skip=10)

    # Inference 
    sim_trajs = []
    for i in range(n_sim):
        # --------------- simulate with trained FF ---------------
        xyz = frames[-1].detach().cpu()#.reshape(-1)
        xyz = torch.Tensor( wrap_positions( xyz.numpy(), atoms.get_cell()) ).reshape(-1)
        p = x[-1, :N * 3 ].detach().cpu().reshape(-1)
        p_v = x[-1, N*3*2: ].detach().cpu().reshape(-1)
        t = torch.Tensor([dt* units.fs * i for i in range(100)]).to(device)
        pq = torch.cat((p, xyz, p_v)).to(device)
        pq.requires_grad= True
        x = odeint(f_x, pq, t, method='rk4')
        frames = x[::25, N*3: N*3*2].reshape(-1, N, 3).detach()
        print(frames.shape)
        sim_trajs.append(frames.detach().cpu().numpy())

    #print(len(sim_trajs))
    sim_trajs = torch.Tensor(np.array(sim_trajs)).to(device).reshape(-1, N, 3)

    #print(sim_trajs.shape)
    # compute equilibrate rdf with finer bins 
    test_nbins = 128
    obs = rdf(atoms, test_nbins, device, r_range)
    xnew = np.linspace(0.0, r_range, test_nbins)
    count_obs, g_obs = get_exp_rdf(data, test_nbins, r_range, obs) # recompute exp. rdf
    _, bins, g = obs(sim_trajs) # compute simulated rdf

    # compute equilibrated rdf 
    g_m = 0.5 * (g_obs + g)
    loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).sum()
    loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).sum()

    save_traj(sim_trajs.detach().cpu().numpy(), atoms, 
                        model_path + '/sim.xyz', 
                        skip=1)

    plt.plot(xnew, g.detach().cpu().numpy(), linewidth=4, alpha=0.6, label='sim')
    plt.plot(xnew, g_obs.detach().cpu().numpy() , linewidth=2, linestyle='--', c='black', label='exp')
    plt.legend()
    plt.xlabel("$\AA$")
    plt.ylabel("g(r)")
    plt.savefig(model_path + '/final.jpg', bbox_inches='tight')
    plt.show()
    plt.close()

    np.savetxt(model_path + '/loss.csv', np.array(loss_log))

    return loss_js.item()

def save_traj(traj, atoms, fname, skip=10):
    atoms_list = []
    for i, xyz in enumerate(traj):
        if i % skip == 0: 
            frame = Atoms(positions=xyz, numbers=atoms.get_atomic_numbers())
            atoms_list.append(frame)
    ase.io.write(fname, atoms_list) 

