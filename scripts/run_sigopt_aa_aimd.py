
import argparse
from sigopt import Connection
from gnn_rdf_aa_aimd import *

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
    n_obs = 2
    n_epochs = 21
    n_sim = 5
    max_n_epochs = 10
   #n_train = 1
else:
    token = 'RXGPHWIUAMLHCDJCDBXEWRAUGGNEFECMOFITCRHCEOBRMGJU'
    n_obs = 1000
    n_epochs = 1500
    n_sim = 120
    max_n_epochs = 80
    #n_train = 100

logdir = params['logdir']

#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=logdir,
        metrics=[dict(name='loss', objective='minimize')],
        parameters=[
            dict(name='n_atom_basis', type='categorical',categorical_values=["tiny", "low", "mid", "high"]),
            dict(name='n_filters', type='categorical', categorical_values=["tiny", "low", "mid", "high"]),
            dict(name='gaussian_width', type='double', bounds=dict(min=0.025, max=0.25)),
            dict(name='n_convolutions', type='int', bounds=dict(min=1, max=4)),

            dict(name='sigma_oo', type='double', bounds=dict(min=2.2, max=3.75)),
            dict(name='epsilon_oo', type='double', bounds=dict(min=0.00, max=0.0001)),
            dict(name='sigma_oh', type='double', bounds=dict(min=0.8, max=2.0)),
            dict(name='epsilon_oh', type='double', bounds=dict(min=0.00, max=0.0001)),
            dict(name='sigma_hh', type='double', bounds=dict(min=0.3, max=1.5)),
            dict(name='epsilon_hh', type='double', bounds=dict(min=0.00, max=0.0001)),
            dict(name='charge_scale', type='double', bounds=dict(min=0.0, max=0.01)),

            dict(name='opt_freq', type='int', bounds=dict(min=5, max=max_n_epochs)),
            dict(name='lr', type='double', bounds=dict(min=2.5e-7, max=2e-4)),
            dict(name='n_train', type='int', bounds=dict(min=0, max=30)),

            dict(name='cutoff', type='double', bounds=dict(min=3.0, max=7.0)),
            dict(name='mse_weight_oo', type='double', bounds=dict(min=0.0, max=1.0)),
            dict(name='mse_weight_oh', type='double', bounds=dict(min=0.0, max=1.0)),
            dict(name='mse_weight_hh', type='double', bounds=dict(min=0.0, max=1.0)),
            dict(name='nbins', type='int', bounds=dict(min=16, max=128)),

            dict(name='rdf_start_oo', type='double', bounds=dict(min=1.0, max=2.4)),
            dict(name='rdf_start_oh', type='double', bounds=dict(min=1.13, max=1.5)),
            dict(name='rdf_start_hh', type='double', bounds=dict(min=0.2, max=1.25)),

            dict(name='frameskip', type='int', bounds=dict(min=1, max=5)),
            dict(name='rdf_smear_width', type='double', bounds=dict(min=0.01, max=0.2)),
            dict(name='dt', type='double', bounds=dict(min=0.4, max=1.0)),
        ],
        observation_budget = n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=10,
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()


i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    sys_params = {
    'dt': suggestion.assignments['dt'],
    'n_epochs': n_epochs,
    'n_sim': n_sim,
    }

    value = fit_rdf_aa(assignments=suggestion.assignments, 
                            i=i, 
                            suggestion_id=suggestion.id, 
                            device=params['device'],
                            sys_params=sys_params,
                            project_name=logdir)

    print(value)

    conn.experiments(experiment.id).observations().create(
      suggestion=suggestion.id,
      value=value,
    )

    experiment = conn.experiments(experiment.id).fetch()