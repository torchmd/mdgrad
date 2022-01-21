

import argparse
from sigopt import Connection
from fit_2_comp import *

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-nepochs", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'GMBSZWXFWPHHUCXSDYLLCBBCBTKZUBVCBQRMCMXEFNEYGCFY'
    n_obs = 2
else:
    token = 'JGTKFUYDJMOKBMDFXICMGNEFBXOOSIPAVSGUWPSMJCVDWYMA'
    n_obs = 1000

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
            dict(name='lr', type='double', bounds=dict(min=1.0e-4, max=5e-3)),
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

    sys_params = {'size': 4, 'T': 1.2, 'rho':0.8, 'x':0.5, 'n_sim': 800}

    exp_params = {**params, **suggestion.assignments, **sys_params} 
    exp_params['subjob'] = suggestion.id

    if params['dry_run']:
        exp_params['nepochs'] = 1
        exp_params['n_sim'] = 1 

    value, failed = run(exp_params)
    print(value)

    if failed:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          failed=failed
        )
    else:
        conn.experiments(experiment.id).observations().create(
          suggestion=suggestion.id,
          value=value,
        )

    experiment = conn.experiments(experiment.id).fetch()