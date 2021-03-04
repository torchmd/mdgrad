
from ase import Atoms
import torch
import numpy as np
import math
import os 

def gen_helix(n_spirals, n_atoms, a, dz):

    t = np.linspace(0, 3.1415926 * n_spirals, n_atoms)
    I = np.array([1,0,0])
    J = np.array([0,1,0])
    K = np.array([0,0,1])

    # a = 1.5
    # dz = 0.25

    z = 0.0 
    positions = np.array([ [np.cos(dt) * a *  I + np.sin(dt) * a * J +  i * dz * K] for i, dt in enumerate(t)] ).reshape(-1,3)

    return positions

def compute_angle(xyz, angles):
    assert len(xyz.shape) == 3 
    n_frames = xyz.shape[0]
    N = xyz.shape[1]
    D = xyz[:, :, None, :].expand(n_frames, N,N,3)-xyz[:, None, :, :].expand(n_frames, N,N,3)#.transpose(1,2)
    angle_vec1 = D[:, angles[:,0], angles[:,1], :]
    angle_vec2 = D[:, angles[:,1], angles[:,2], :]
    dot_unnorm = (-angle_vec1*angle_vec2).sum(-1)
    norm = torch.sqrt((angle_vec1.pow(2)).sum(-1)*(angle_vec2.pow(2)).sum(-1))
    cos_theta = (dot_unnorm/norm)
    return cos_theta

def compute_dihe(xyz, dihes): 
    assert len(xyz.shape) == 3 
    n_frames = xyz.shape[0]
    N = xyz.shape[1]
    D = xyz[:, :, None, :].expand(n_frames, N,N,3)-xyz[:, None, :, :].expand(n_frames, N,N,3)#.transpose(1,2)
    vec1 = D[:, dihes[:,1], dihes[:,0]]
    vec2 = D[:, dihes[:,1], dihes[:,2]]
    vec3 = D[:, dihes[:,2], dihes[:,1]]
    vec4 = D[:, dihes[:,2], dihes[:,3]]
    cross1 = torch.cross(vec1, vec2)
    cross2 = torch.cross(vec3, vec4)

    norm = (cross1.pow(2).sum(-1)*cross2.pow(2).sum(-1)).sqrt()
    cos_phi = ((cross1*cross2).sum(-1)/norm)
    
    return cos_phi 

def compute_bond(xyz, bonds):
    assert len(xyz.shape) == 3 
    bonds = (xyz[:, bonds[:,0], :] - xyz[:, bonds[:,1], :]).pow(2).sum(-1).sqrt()
    return bonds

def compute_intcoord(xyz):
    # compute internal coordinates for a chain  
    vec = xyz[: ,:-1] - xyz[:, 1:]

    u_norm = vec.pow(2).sum(-1).sqrt()
    u_i = vec/u_norm[..., None]

    a = (u_i[:, :-1] * u_i[:, 1:]).sum(-1).clamp(-0.99, 0.99).acos()

    n_unorm = u_i[:, :-1].cross(u_i[:, 1:]) # u_{i-1} x u_{i}
    n_norm = n_unorm.pow(2).sum(-1).sqrt()  

    n_i = n_unorm / n_norm[..., None]
    d_i = (n_i[:, :-1] * n_i[:, 1:]).sum(-1).clamp(-0.99, 0.99).acos() * (u_i[:, :-2] * n_i[:, 1:]).sum(-1).sign()

    return u_norm, a, d_i

def train(params, suggestion_id, project_name, device, n_epochs):

    print(params)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    n_atoms = params['n_atoms']
    n_spirals = params['n_spiral']
    dz = params['dz_spiral']
    a = params['a_spiral']
    loss_cutoff = params['loss_cutoff']

    xyz = torch.Tensor( gen_helix(n_spirals, n_atoms, a, dz) )[None, ...]

    bond_index = [[i, i+1]  for i in range(n_atoms) if max([i, i+1]) <= n_atoms-1]
    bond_top = torch.LongTensor(bond_index)

    end2end = torch.LongTensor([[0, n_atoms-1]])
    dis_end2end_targ = compute_bond(xyz, end2end)

    def get_dis_list(xyz, cutoff=5.0):

        n_atoms = xyz.shape[1]
        adj = torch.ones(n_atoms, n_atoms) 

        atom_idx = torch.LongTensor([[i, i] for i in range(n_atoms)] )
        adj[atom_idx[:, 0], atom_idx[:, 1]] = 0.0

        adj = adj.nonzero(as_tuple=False)
        adj = adj[(compute_bond(xyz, adj).squeeze() < cutoff), :]
        targ_dis = compute_bond(xyz, adj)        

        return targ_dis, adj 

    dis_targ, adj = get_dis_list(xyz, loss_cutoff)
    b_targ, a_targ, d_targ = compute_intcoord(xyz)

    # compute bond distances 
    bond_len = b_targ[0, 0].item()


    # define system objects
    chain = Atoms(numbers=[1.] * n_atoms, 
                  positions=[np.array([50., 50., 50.]) + np.array([bond_len, 0., 0.]) * i for i in range(n_atoms)],
                  cell=[100.0, 100.0, 100.0])


    from torchmd.system import System
    from torchmd.potentials import LennardJones, ExcludedVolume
    from ase import units 

    system = System(chain, device=device)
    system.set_temperature(params['T'])

    from torchmd.interface import BondPotentials, GNNPotentials, Stack, PairPotentials
    bondenergy = BondPotentials(system, bond_top, params['k0'], bond_len)

    from nff.train import get_model

    gnnparams = {
        'n_atom_basis': params['n_atom_basis'],
        'n_filters': params['n_filters'],
        'n_gaussians': params['n_gaussians'],
        'n_convolutions': params['n_convolutions'],
        'cutoff': params['cutoff']
    }

    schnet = get_model(gnnparams)

    GNN = GNNPotentials(system, 
                        schnet, 
                         cutoff=gnnparams['cutoff'], 
                         )

    pair = PairPotentials(system, ExcludedVolume(**{'epsilon': params['epsilon'], 
                                                     'sigma': params['sigma'],
                                                      'power':10}),
                        cutoff=2.5,
                        ex_pairs=bond_top
                        ).to(system.device)

    FF = Stack({
           'gnn': GNN,
           'prior': bondenergy,
           'pair': pair
            }) 

    from torchmd.md import NoseHooverChain, NVE, Simulations


    if params['method'] == 'NH_verlet' or params['method'] == 'rk4':
        diffeq = NoseHooverChain(FF, 
                system,
                Q=50.0, 
                T=params['T'],
                num_chains=5, 
                adjoint=True).to(device)
    else:
        diffeq = NVE(FF, 
                system,
                adjoint=True).to(device)

    tau = params['tau']
    sim = Simulations(system, diffeq, wrap=False, method=params['method'])

    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=params['lr'])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
    #                                           'min', 
    #                                           min_lr=5e-5, 
    #                                           verbose=True, factor=0.75, patience=60,
    #                                           threshold=5e-5)

    loss_log = []

    for i in range(0, n_epochs):

        trajs = sim.simulate(steps=tau , frequency=int(tau), dt=params['dt'])

        if params['method'] == 'NH_verlet' or params['method'] == 'rk4':
            v_t, q_t, pv_t = trajs 
        elif params['method'] == 'verlet':
            v_t, q_t = trajs
        
        if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
            return 55.0 

        # angle1 = compute_angle(q_t, angle1_top.to(device))
        # dihe1 = compute_dihe(q_t, dihe1_top.to(device))
        # angle2 = compute_angle(q_t, angle2_top.to(device))
        # dihe2 = compute_dihe(q_t, dihe2_top.to(device))

        # bonds = compute_bond(q_t, bond_top.to(device))
        # bonds13 = compute_bond(q_t, bond13_top.to(device))
        # bonds14 = compute_bond(q_t, bond14_top.to(device))
        # bonds15 = compute_bond(q_t, bond15_top.to(device))
        # bonds16 = compute_bond(q_t, bond16_top.to(device))
        # bonds17 = compute_bond(q_t, bond17_top.to(device))
        # bonds18 = compute_bond(q_t, bond18_top.to(device))

        dis_end2end = compute_bond(q_t, end2end.to(device))

        if i > 0:
            # if params['lastframe'] == 'True':
            #     traj_train = q_t[[-1]]
            # else:
            traj_train = q_t

            b, a, d = compute_intcoord(traj_train) 
            dis = compute_bond(traj_train, adj.to(device))
            
            loss_b = (b - b_targ.to(device).squeeze()).pow(2).mean()
            loss_a = (a - a_targ.to(device).squeeze()).pow(2).mean()
            loss_d = (d - d_targ.to(device).squeeze()).pow(2).mean()
            #loss_end2end = (dis_end2end - dis_end2end_targ.to(device).squeeze()).pow(2).mean()

            dis_diff = dis - dis_targ.to(dis.device)

            # focused distance loss
            #focus = (dis_diff.abs() * (1/params['focus_temp'])).softmax(-1)
            #print(dis.mean().item())
            #loss_dis = (focus * dis_diff.pow(2)).mean()

            loss_dis = dis_diff.pow(2).mean()

            loss = params['l_b'] * loss_b + \
                    params['l_a'] * loss_a + \
                     params['l_d'] * loss_d + \
                     params['l_dis'] * loss_dis + \
                     # params['l_end2end'] * loss_end2end

            loss_record = loss_b + loss_a + loss_d #+ dis_diff.pow(2).mean()

            #print(loss_b, loss_a, loss_d, dis_diff.pow(2).mean())

            loss.backward()
            # duration = (datetime.now() - current_time)
            optimizer.step()
            optimizer.zero_grad()

            #scheduler.step(loss)
            
            print(i, loss.item())
            if math.isnan(loss_record.item()):
                print("NaN encountered")
                return 55.0 

            loss_log.append(loss_record.item())

    from utils import to_mdtraj 
    traj = to_mdtraj(system, sim.log)
    traj.center_coordinates()
    traj.save_xyz("{}/train.xyz".format(model_path))

    np.savetxt("{}/loss.csv".format(model_path), np.array(loss_log))

    return np.array( loss_log[-10:] ).mean()


if __name__ == '__main__':

    import argparse
    from sigopt import Connection

    parser = argparse.ArgumentParser()
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-device", type=int, default=0)
    parser.add_argument("-id", type=int, default=None)
    parser.add_argument("--dry_run", action='store_true', default=False)
    params = vars(parser.parse_args())

    if params['dry_run']:
        token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
        n_obs = 2
        n_epochs = 5
    else:
        token = 'RXGPHWIUAMLHCDJCDBXEWRAUGGNEFECMOFITCRHCEOBRMGJU'
        n_obs = 1000
        n_epochs = 1000

    logdir = params['logdir']
    #Intiailize connections 
    conn = Connection(client_token=token)

    if params['id'] == None:
        experiment = conn.experiments().create(
            name=logdir,
            metrics=[dict(name='loss', objective='minimize')],
            parameters=[
                dict(name='sigma', type='double', bounds=dict(min=0.7, max=1.3)),
                dict(name='epsilon', type='double', bounds=dict(min=0.01, max=0.2)),
                dict(name='tau', type='int', bounds=dict(min=10, max=60)),
                dict(name='lr', type='double', bounds=dict(min=1e-6, max=1e-3)),
                dict(name='dt', type='double', bounds=dict(min=0.005, max=0.05)),
                dict(name='method', type='categorical', categorical_values=["verlet", "NH_verlet", "rk4"]),
                dict(name='l_b', type='double', bounds=dict(min=0.01, max=1.0)),
                dict(name='l_a', type='double', bounds=dict(min=0.01, max=1.0)),
                dict(name='l_d', type='double', bounds=dict(min=0.01, max=1.0)),
                dict(name='l_dis', type='double', bounds=dict(min=0.01, max=1.0)),
                #dict(name='l_end2end', type='double', bounds=dict(min=0.0, max=0.1)),
                # spiral hyperparam
                dict(name='a_spiral', type='double', bounds=dict(min=0.8, max=2.0)),
                dict(name='dz_spiral', type='double', bounds=dict(min=0.1, max=0.5)),
                dict(name='n_spiral', type='int', bounds=dict(min=5, max=12)),
                dict(name='n_atoms', type='int', bounds=dict(min=20, max=50)),
                #dict(name='focus_temp', type='double', bounds=dict(min=0.01, max=1.0)),
                dict(name='k0', type='double', bounds=dict(min=0.2, max=5.0)),
                dict(name='cutoff', type='double', bounds=dict(min=1.5, max=5.0)),
                dict(name='loss_cutoff', type='double', bounds=dict(min=1.5, max=5.0)),
                dict(name='n_convolutions', type='int', bounds=dict(min=2, max=5)),
                dict(name='T', type='double', bounds=dict(min=0.001, max=0.15)),
            ],
            observation_budget = n_obs, # how many iterations to run for the optimization
            parallel_bandwidth=10,
        )

    elif type(params['id']) == int:
        experiment = conn.experiments(params['id']).fetch()

    i = 0
    while experiment.progress.observation_count < experiment.observation_budget:

        suggestion = conn.experiments(experiment.id).suggestions().create()
        trainparams = suggestion.assignments

        gnn_params = {"n_atom_basis": 128,
                  "n_filters": 128,
                  "n_gaussians": 32}

        trainparams = {**gnn_params, **trainparams}

        value = train(params=trainparams, 
                    suggestion_id=suggestion.id, 
                    device=params['device'],
                    project_name=logdir,
                    n_epochs=n_epochs)

        print(value)

        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          value=value,
        )

        experiment = conn.experiments(experiment.id).fetch()
