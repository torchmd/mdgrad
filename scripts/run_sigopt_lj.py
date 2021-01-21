
import argparse
from sigopt import Connection
from fit_rdf_pair import *

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-val", type=str, nargs='+')
parser.add_argument("-id", type=int, default=None)
parser.add_argument("-cutoff", type=float)
parser.add_argument("-dt", type=float)
parser.add_argument("-update_freq", type=int)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
    n_obs = 2
    n_epochs = 10
    n_sim = 2
else:
    token = 'RXGPHWIUAMLHCDJCDBXEWRAUGGNEFECMOFITCRHCEOBRMGJU'
    n_obs = 1000
    n_epochs = 1000
    n_sim = 50

logdir = params['logdir']

#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    experiment = conn.experiments().create(
        name=logdir,
        metrics=[dict(name='loss', objective='minimize')],
        parameters=[
            dict(name='gaussian_width', type='double', bounds=dict(min=0.025, max=0.25)),
            dict(name='sigma', type='double', bounds=dict(min=0.8, max=1.1)),
            dict(name='epsilon', type='double', bounds=dict(min=0.05, max=0.5)),
            dict(name='power', type='int', bounds=dict(min=9, max=12)),
            dict(name='opt_freq', type='int', bounds=dict(min=15, max=200)),
            dict(name='lr', type='double', bounds=dict(min=1.0e-4, max=5e-3)),
            dict(name='rdf_start', type='double', bounds=dict(min=0.5, max=0.7)),
            dict(name='rdf_weight', type='double', bounds=dict(min=0.1, max=1.0)),
            dict(name='vacf_weight', type='double', bounds=dict(min=0.1, max=1.0)),
            dict(name='nbins', type='int', bounds=dict(min=64, max=128)),
            dict(name='train_vacf', type='categorical', categorical_values=["True", "False"]),
            dict(name='n_width', type='int', bounds=dict(min=64, max=128)),
            dict(name='n_layers', type='int', bounds=dict(min=2, max=4)),
            dict(name='nonlinear', type='categorical', categorical_values=['ReLU', 'ELU', 'Tanh', 'LeakyReLU', 'ReLU6', 'SELU', 'CELU', 'Tanhshrink']),
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
    'dt': params['dt'],
    'size': 4,
    'n_epochs': n_epochs,
    'n_sim': n_sim,
    'data': params['data'],
    'val': params['val'],
    't_range': 50,
    'cutoff': params['cutoff'],
    'skip': 5,
    'topology_update_freq': params['update_freq'],
    'nbr_list_device': "cpu" #params['device']
    }

    print(sys_params)

    value = fit_lj(assignments=suggestion.assignments, 
                            #i=i, 
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