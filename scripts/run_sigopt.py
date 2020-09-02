 
#from settings import *
import argparse
from sigopt import Connection
from gnn_rdf import *

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("-data", type=str, default='water')
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
    n_obs = 2
    tmax = 200
else:
    token = 'RXGPHWIUAMLHCDJCDBXEWRAUGGNEFECMOFITCRHCEOBRMGJU'
    n_obs = 500
    tmax = 25000

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
            dict(name='n_gaussians', type='categorical', categorical_values= ["tiny", "low"]),
            dict(name='n_convolutions', type='int', bounds=dict(min=1, max=3)),
            dict(name='sigma', type='double', bounds=dict(min=2.25, max=3.0)),
            dict(name='epsilon', type='double', bounds=dict(min=0.005, max=0.025)),
            dict(name='opt_freq', type='int', bounds=dict(min=10, max=100)),
            dict(name='lr', type='double', bounds=dict(min=1e-6, max=2e-4)),
            dict(name='cutoff', type='double', bounds=dict(min=4.0, max=7.0)),
            dict(name='mse_weight', type='double', bounds=dict(min=0.0, max=10.0))
        ],
        observation_budget = n_obs, # how many iterations to run for the optimization
        parallel_bandwidth=25,
    )

elif type(params['id']) == int:
    experiment = conn.experiments(params['id']).fetch()

if params['data'] == 'water':
    data = np.load("../data/water_exp_pccp.npy")
    size = 4
    L = 19.73 / size
    r_range = 7.5
    nbins = 100

elif params['data'] == 'argon':
    data = np.load("../data/argon_exp.npy")
    size = 4
    L = 22.884 / size
    r_range = 9.0 
    nbins = 100

i = 0
while experiment.progress.observation_count < experiment.observation_budget:

    suggestion = conn.experiments(experiment.id).suggestions().create()

    sys_params = {
    'data': data, 
    'size': size,
    'L': L, 
    'r_range': r_range,
    'nbins': nbins,
    'tmax': tmax,
    'dt': 1.0
    }

    value = evaluate_model(assignments=suggestion.assignments, 
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