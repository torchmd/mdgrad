import argparse
from sigopt import Connection
from fit_mix import *



parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
#parser.add_argument("-subjob", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-id", type=int, default=None)
parser.add_argument("-nepochs", type=int, default=30)
parser.add_argument("-nsim", type=int, default=40)
parser.add_argument("-trainx", type=float, nargs='+')
parser.add_argument("-valx", type=float, nargs='+')
parser.add_argument("-cutoff", type=float, default=2.5)
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

    runparams = {**params, **suggestion.assignments}
    runparams['subjob'] = suggestion.id

    if params['dry_run']:
        runparams['nepochs'] = 1
        runparams['n_sim'] = 1 

    value, failed = run_mix(runparams)
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
