
from ase import Atoms
import torch
import numpy as np
import math
import os 

def gen_helix(n_spirals, n_atoms):

    t = np.linspace(0, 3.14 * n_spirals, n_atoms)
    I = np.array([1,0,0])
    J = np.array([0,1,0])
    K = np.array([0,0,1])

    a = 1.0
    z = 0.0 ; dz = 0.5
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
    cos_phi = 1.0*((cross1*cross2).sum(-1)/norm)
    
    return cos_phi 

def compute_bond(xyz, bonds):
    assert len(xyz.shape) == 3 
    
    bonds = (xyz[:, bonds[:,0], :] - xyz[:, bonds[:,1], :]).pow(2).sum(-1).sqrt()
    
    return bonds



def train(params, suggestion_id, project_name, device, n_epochs):

    print(params)

    model_path = '{}/{}'.format(project_name, suggestion_id)
    os.makedirs(model_path)

    n_atoms = 50 

    xyz = torch.Tensor( gen_helix(15, n_atoms) )[None, ...]

    dihe_index = [[i, i+1, i+2, i+3]  for i in range(n_atoms) if max([i, i+1, i+2, i+3]) <= n_atoms-1]
    dihe_top = torch.LongTensor(dihe_index)
    angle_index = [[i, i+1, i+2]  for i in range(n_atoms) if max([i, i+1, i+2]) <= n_atoms-1]
    angle_top = torch.LongTensor(angle_index)
    bond_index = [[i, i+1]  for i in range(n_atoms) if max([i, i+1]) <= n_atoms-1]
    bond_top = torch.LongTensor(bond_index)

    targ_dihe = compute_dihe(xyz, dihe_top)
    targ_angle = compute_angle(xyz, angle_top)
    targ_bond = compute_bond(xyz, bond_top)

    bond_len = targ_bond[0, 0].item()

    # define system objects
    chain = Atoms(numbers=[1.] * n_atoms, 
                  positions=[np.array([50., 50., 50.]) + np.array([bond_len, 0., 0.]) * i for i in range(n_atoms)],
                  cell=[100.0, 100.0, 100.0])


    from torchmd.system import System
    from ase import units 

    system = System(chain, device=device)
    system.set_temperature(params['T']/units.kB)

    from torchmd.interface import BondPotentials, GNNPotentials, Stack
    bondenergy = BondPotentials(system, bond_top, 1.0, bond_len)

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
                         cutoff=params['cutoff'], 
                         )

    FF = Stack({
           'gnn': GNN,
           'prior': bondenergy
            }) 

    from torchmd.md import NoseHooverChain, Simulations

    diffeq = NoseHooverChain(FF, 
            system,
            Q=50.0, 
            T=0.1,
            num_chains=5, 
            adjoint=True).to(device)

    tau = params['tau']
    sim = Simulations(system, diffeq, wrap=False)

    optimizer = torch.optim.Adam(list(diffeq.parameters() ), lr=params['lr'])

    loss_log = []

    for i in range(0, n_epochs):
        trajs = sim.simulate(steps=tau , frequency=int(tau), dt=0.1)
        
        v_t, q_t, pv_t = trajs 
        
        if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
            return 5.0 

        angles = compute_angle(q_t, angle_top.to(device))
        dihes = compute_dihe(q_t, dihe_top.to(device))
        bonds = compute_bond(q_t, bond_top.to(device))

        if i > 0:
            loss = (angles - targ_angle.to(device).squeeze()).pow(2).mean()
            loss += (dihes - targ_dihe.to(device).squeeze()).pow(2).mean()
            loss += (bonds - targ_bond.to(device).squeeze()).pow(2).mean()
            
            loss.backward()
            # duration = (datetime.now() - current_time)
            optimizer.step()
            optimizer.zero_grad()
            
            print(loss.item())
            if math.isnan(loss.item()):
                return 5.0 

            loss_log.append(loss.item())

    from utils import to_mdtraj 
    mdtraj = to_mdtraj(system, diffeq.traj[::1])
    mdtraj.save_xyz("{}/train.xyz".format(model_path))

    np.savetxt("{}/loss.csv".format(model_path), np.array(loss_log))

    return np.array( loss_log[-5:] ).mean()

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
    n_epochs = 20000

logdir = params['logdir']
#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=logdir,
        metrics=[dict(name='loss', objective='minimize')],
        parameters=[
            dict(name='n_atom_basis', type='int', bounds=dict(min=16, max=64)),
            dict(name='n_filters', type='int', bounds=dict(min=16, max=64)),
            dict(name='n_gaussians', type='int', bounds=dict(min=16, max=64)),
            dict(name='n_convolutions', type='int', bounds=dict(min=2, max=5)),
            dict(name='cutoff', type='double', bounds=dict(min=1.5, max=5.0)),
            dict(name='tau', type='int', bounds=dict(min=10, max=80)),
            dict(name='lr', type='double', bounds=dict(min=1e-6, max=5e-4)),
            dict(name='T', type='double', bounds=dict(min=0.01, max=0.25)),
            dict(name='dt', type='double', bounds=dict(min=0.005, max=0.1)),
        ],
        observation_budget = n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()

i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    value = train(params=suggestion.assignments, 
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
