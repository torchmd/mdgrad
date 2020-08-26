from settings import *

width_dict = {'tiny': 64,
               'low': 128,
               'mid': 256, 
               'high': 512}

gaussian_dict = {'tiny': 16,
               'low': 32,
               'mid': 64, 
               'high': 128}


T_MAX = 100000

def evaluate_model(assignments, i, suggestion_id, device):

    print(assignments)

    model_path = '{}/{}'.format(logdir, suggestion_id)
    os.makedirs(model_path)

    tau =  assignments['opt_freq'] 
    num_epochs = T_MAX // tau

    print(num_epochs)
    # load RDF data 
    data = np.load("../experiments/rdf_exp.npy")
    f = interpolate.interp1d(data[:,0], data[:,1])
    xnew = np.linspace(0.0, 7, 50)
    plt.plot(xnew, f(xnew))


    # set up lattices to have the same density as water at 300K 1atm
    device=1
    CUTOFF = assignments['cutoff']
    size = 3
    L = 14.798/ 3

    atoms = FaceCenteredCubic(directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                              symbol='O',
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)

    N = atoms.get_number_of_atoms()
    mass = atoms.get_masses()

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
                 'power': assignments['power']}

    pair = PairPot(ExcludedVolume, lj_params,
                    cell=torch.Tensor(atoms.get_cell()).diag(), 
                    device=device,
                    cutoff=CUTOFF,
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
            Q=10.0, 
            T=T,
            num_chains=5, 
            device=device).to(device)

    # initialize observable function 
    obs = rdf(atoms, 50, device, 7.0)

    # target observable 
    g_obs = torch.Tensor(f(xnew)).to(device)

    # define optimizer 
    optimizer = torch.optim.SGD(list( f_x.parameters() ), lr=assignments['lr'])

    e0 = 1e-4

    for i in range(0, num_epochs):
        
        current_time = datetime.now() 
        
        if i == 0:
            xyz = torch.Tensor(atoms.get_positions() 
                              # + np.random.rand(N, 3) * 0.1
                              ).reshape(-1) 
            # generate random velocity 
            p = torch.Tensor( atoms.get_velocities().reshape(-1))
            p_v = torch.Tensor([0.0] * 5)
            t = torch.Tensor([0.5 * units.fs * i for i in range(tau)]).to(device)
            
        else:
            xyz = frames[-1].detach().cpu()#.reshape(-1)
            xyz = torch.Tensor( wrap_positions( xyz.numpy(), atoms.get_cell()) ).reshape(-1)
            p = x[-1, :N * 3 ].detach().cpu().reshape(-1)
            p_v = x[-1, N*3*2: ].detach().cpu().reshape(-1)
            t = torch.Tensor([0.5 * units.fs * i for i in range(tau)]).to(device)
            
        pq = torch.cat((p, xyz, p_v)).to(device)
        pq.requires_grad= True
        x = odeint(f_x, pq, t, method='rk4')
        
        frames = x[::5, N*3: N*3*2].reshape(-1, N, 3)
        bins, g = obs(frames)
        
        plt.title("epoch {}".format(i))
        plt.plot(xnew, g.detach().cpu().numpy())
        plt.plot(xnew, g_obs.detach().cpu().numpy())
        plt.savefig(model_path + '/{}.jpg'.format(i))
        plt.show()
        plt.close()
        
        loss = ( -(g + e0 ) * (torch.log(g_obs + e0 ) - torch.log(g + e0) ) ).sum()
        loss +=  ( -(g_obs + e0 ) * (torch.log(g + e0 ) - torch.log(g_obs +  e0)) ).sum()
        #loss += (g- g_obs).pow(2).sum()
                
        print(loss.item())
        loss.backward()
        
        dt = (datetime.now() - current_time)
        #print( "{} seconds".format(dt.total_seconds())) 
        
        if i !=0:
            optimizer.step()
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()
        
        # if i % 10 == 0:
        #     plt.plot(x[:, 500].detach().cpu().numpy())
        #     plt.show()

        if torch.isnan(loss):
            return 100.0

    return loss.item()

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
    n_obs = 2
else:
    token = 'JQJLZYNHOWKBUXWMYBZFKRKHURZAZRIQWERJSBKWZUBODXEQ'
    n_obs = 200

logdir = params['logdir']


#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=logdir,
        metrics=[dict(name='stability', objective='minimize')],
        parameters=[
            dict(name='n_atom_basis', type='categorical',categorical_values=["low", "mid"]),
            dict(name='n_filters', type='categorical', categorical_values=["tiny", "low", "mid"]),
            dict(name='n_gaussians', type='categorical', categorical_values= ["tiny", "low", "mid"]),
            dict(name='n_convolutions', type='int', bounds=dict(min=2, max=5)),
            dict(name='power', type='int', bounds=dict(min=10, max=12)),
            dict(name='sigma', type='double', bounds=dict(min=1.5, max=2.5)),
            dict(name='epsilon', type='double', bounds=dict(min=0.01, max=0.15)),
            dict(name='opt_freq', type='int', bounds=dict(min=25, max=200)),
            dict(name='lr', type='double', bounds=dict(min=1e-4, max=1e-2)),
            dict(name='cutoff', type='double', bounds=dict(min=2.0, max=6.0))
        ],
        observation_budget = n_obs, # how many iterations to run for the optimization
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()


i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()


    value = evaluate_model(assignments=suggestion.assignments, i=i, suggestion_id=suggestion.id, device=params['device'])

    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=value,
    )

    experiment = conn.experiments(experiment.id).fetch()