from settings import *

width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}


T_MAX = 25000 #100000

def evaluate_model(assignments, i, suggestion_id, device, sys_params, project_name):


    data = sys_params['data']
    size = sys_params['size']
    L = sys_params['L']
    r_range = sys_params['r_range']
    nbins = sys_params['nbins']

    print(assignments)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    tau =  assignments['opt_freq'] 
    num_epochs = T_MAX // tau

    print(num_epochs)
    # load RDF data 
    f = interpolate.interp1d(data[:,0], data[:,1])
    xnew = np.linspace(0.0, r_range, nbins)

    # set up lattices to have the same density as water at 300K 1atm
    CUTOFF = assignments['cutoff']

    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol='H',
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)

    N = atoms.get_number_of_atoms()
    mass = atoms.get_masses()

    print(atoms.get_cell())

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


    T= 300.0 * units.kB

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

    # target observable 
    V = (4/3)* np.pi * (r_range) ** 3

    g_obs = torch.Tensor(f(xnew)).to(device)
    g_obs_norm = ((g_obs.detach() * obs.vol_bins).sum()).item()
    g_obs = g_obs * (V/g_obs_norm)
    count_obs = g_obs * obs.vol_bins / V

    # define optimizer 
    optimizer = torch.optim.Adam(list( f_x.parameters() ), lr=assignments['lr'])

    e0 = 1e-4

    loss_log = []

    for i in range(0, num_epochs):
        
        current_time = datetime.now() 
        
        if i == 0:
            xyz = torch.Tensor(atoms.get_positions() 
                              # + np.random.rand(N, 3) * 0.1
                              ).reshape(-1) 
            # generate random velocity 
            p = torch.Tensor( atoms.get_velocities().reshape(-1))
            p_v = torch.Tensor([0.0] * 5)
            t = torch.Tensor([0.25 * units.fs * i for i in range(tau)]).to(device)
            
        else:
            xyz = frames[-1].detach().cpu()#.reshape(-1)
            xyz = torch.Tensor( wrap_positions( xyz.numpy(), atoms.get_cell()) ).reshape(-1)
            p = x[-1, :N * 3 ].detach().cpu().reshape(-1)
            p_v = x[-1, N*3*2: ].detach().cpu().reshape(-1)
            t = torch.Tensor([0.25 * units.fs * i for i in range(tau)]).to(device)
            
        pq = torch.cat((p, xyz, p_v)).to(device)
        pq.requires_grad= True
        x = odeint(f_x, pq, t, method='rk4')
        
        frames = x[::, N*3: N*3*2].reshape(-1, N, 3)
        _, bins, g = obs(frames)
        
        plt.title("epoch {}".format(i))
        plt.plot(xnew, g.detach().cpu().numpy())
        plt.plot(xnew, g_obs.detach().cpu().numpy())
        plt.savefig(model_path + '/{}.jpg'.format(i))
        plt.show()
        plt.close()
        
        g_m = 0.5 * (g_obs + g)
        loss_js =  ( -(g_obs + e0 ) * (torch.log(g_m + e0 ) - torch.log(g_obs +  e0)) ).sum()
        loss_js += ( -(g + e0 ) * (torch.log(g_m + e0 ) - torch.log(g + e0) ) ).sum()
        loss = loss_js + assignments['mse_weight'] * (g- g_obs).pow(2).sum()
                
        print(loss.item())
        loss.backward()
        
        dt = (datetime.now() - current_time)
        #print( "{} seconds".format(dt.total_seconds())) 
        
        if i !=0:
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()


        if torch.isnan(loss):
            plt.plot(loss_log)
            plt.savefig(model_path + '/loss.jpg')
            plt.close()
            return loss_log[-1]
        else:
            loss_log.append(loss_js.item())

    plt.plot(loss_log)
    plt.savefig(model_path + '/loss.jpg')
    plt.close()

    return loss_js.item()

