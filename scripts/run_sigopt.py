
import argparse
from sigopt import Connection
from fit_rdf_gnn import *

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-nepochs", type=int, default=700)
parser.add_argument("-nsim", type=int, default=20)
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--pair", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_obs = 2
    tmax = 200
    n_epochs = 4
    n_sim = 2
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_obs = 1000
    tmax = 25000
    n_epochs = 1000
    n_sim = 50

logdir = params['logdir']


#Intiailize connections 
conn = Connection(client_token=token)

if params['id'] == None:
    if params['pair'] == False:
        experiment = conn.experiments().create(
            name=logdir,
            metrics=[dict(name='loss', objective='minimize')],
            parameters=[
                dict(name='n_atom_basis', type='categorical',categorical_values=["tiny", "low", "mid", "high"]),
                dict(name='n_filters', type='categorical', categorical_values=["tiny", "low", "mid", "high"]),
                #dict(name='n_gaussians', type='categorical', categorical_values= ["tiny", "low", "mid"]),
                dict(name='gaussian_width', type='double', bounds=dict(min=0.05, max=0.25)),
                dict(name='n_convolutions', type='int', bounds=dict(min=1, max=3)),
                dict(name='sigma', type='double', bounds=dict(min=2.25, max=3.0)),
                dict(name='epsilon', type='double', bounds=dict(min=0.005, max=0.025)),
                dict(name='opt_freq', type='int', bounds=dict(min=10, max=100)),
                dict(name='lr', type='double', bounds=dict(min=1e-6, max=1e-4)),
                dict(name='cutoff', type='double', bounds=dict(min=4.0, max=8.0)),
                dict(name='mse_weight', type='double', bounds=dict(min=0.0, max=20.0)),
                dict(name='nbins', type='int', bounds=dict(min=32, max=128)),
            ],
            observation_budget = n_obs, # how many iterations to run for the optimization
            parallel_bandwidth=10,
        )
    else:
        experiment = conn.experiments().create(
            name=logdir,
            metrics=[dict(name='loss', objective='minimize')],
            parameters=[
                dict(name='gaussian_width', type='double', bounds=dict(min=0.025, max=0.25)),
                dict(name='sigma', type='double', bounds=dict(min=1.5, max=2.2)),
                dict(name='epsilon', type='double', bounds=dict(min=0.05, max=2.0)),
                dict(name='power', type='int', bounds=dict(min=9, max=12)),
                dict(name='opt_freq', type='int', bounds=dict(min=60, max=200)),
                dict(name='lr', type='double', bounds=dict(min=1.0e-4, max=5e-3)),
                dict(name='cutoff', type='double', bounds=dict(min=4.0, max=8.0)),
                dict(name='mse_weight', type='double', bounds=dict(min=0.1, max=1.0)),
                dict(name='nbins', type='int', bounds=dict(min=64, max=128)),
                dict(name='n_width', type='int', bounds=dict(min=64, max=128)),
                dict(name='n_layers', type='int', bounds=dict(min=2, max=5)),
                dict(name='nonlinear', type='categorical', categorical_values=['ReLU', 'ELU', 'Tanh', 'LeakyReLU', 'ReLU6', 'SELU', 'CELU', 'Tanhshrink']),
            ],
            observation_budget = n_obs, # how many iterations to run for the optimization
            parallel_bandwidth=10,
        )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()

# if params['data'] == 'water':
#     data = np.load("../data/water_exp_pccp.npy")
#     size = 4
#     L = 19.73 / size
#     end = 7.5

# elif params['data'] == 'argon':
#     data = np.load("../data/argon_exp.npy")
#     size = 4
#     L = 22.884 / size
#     end = 9.0 

data = params['data']

i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    sys_params = {
            'dt': 0.5,
            'n_epochs': n_epochs,
            'n_sim': params['nsim'],
            'data': params['data'],
            'val': None,
            'size': 4,
            'anneal_flag': 'False',
            'pair_flag': params['pair'],
            'topology_update_freq': 1
            }

    value = fit_rdf(assignments=suggestion.assignments, 
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