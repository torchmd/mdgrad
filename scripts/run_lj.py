import argparse
from sigopt import Connection
from fit_rdf_pair import *
from datetime import datetime
from datetime import date
import random 

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-nruns", type=int, default=1)

parser.add_argument("-paramset", type=str, default='None')

# options to overide default param set 
parser.add_argument("-sigma", type=float)
parser.add_argument("-lr", type=float)
parser.add_argument("-cutoff", type=float)
parser.add_argument("-vacf_weight", type=float)

parser.add_argument("-name", type=str)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-val", type=str, nargs='+')
parser.add_argument("--dry_run", action='store_true', default=False)
parser.add_argument("--trainvacf", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    n_obs = 2
    n_epochs = 4
    n_sim = 2
else:
    n_obs = 1000
    n_epochs = 1000
    n_sim = 50

logdir = params['logdir']


if params['paramset'] != 'None':
    import json 
    assignments = json.load(open(params['paramset']))

else:
  # use default assignments 
  assignments = {
    "epsilon": 0.4,
    "gaussian_width": 0.10,
    "n_layers": 2,
    "n_width": 87,
    "nbins": 100,
    "nonlinear": "ELU",
    "opt_freq": 60,
    "power": 10,
    "rdf_weight": 0.95,
    "train_vacf": "True",
    "lr": 0.002,
    "sigma": 0.9,
    "vacf_weight": 0.41,
    "cutoff": 2.5
  }

if assignments['train_vacf'] == 'False':
    assignments['vacf_weight'] = 0.0

# param overwrite 
if params['trainvacf']:
    assignments['train_vacf'] = 'True'

if params['sigma']:
    print("chaging default sigma from {} to {}".format(assignments['sigma'], params['sigma']))
    assignments['sigma'] = params['sigma']

if params['lr']:
    print("chaging default lr from {} to {}".format(assignments['lr'], params['lr']))
    assignments['lr'] = params['lr']

if params['cutoff']:
    print("chaging default cutoff from {} to {}".format(assignments['cutoff'], params['cutoff']))
    assignments['cutoff'] = params['cutoff']

if params['vacf_weight']:
    print("chaging default vacf_weight from {} to {}".format(assignments['vacf_weight'], params['vacf_weight']))
    assignments['vacf_weight'] = params['vacf_weight']

sys_params = {
'dt': 0.01,
'size': 4,
'n_epochs': n_epochs,
'n_sim': n_sim,
'data': params['data'],
'val': params['val'],
't_range': 50, # should I make this a variable
}

print(assignments)

for i in range(params['nruns']):

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))

    value = fit_lj(assignments=assignments, 
                            suggestion_id=params['name'] + dt_string, 
                            device=params['device'],
                            sys_params=sys_params,
                            project_name=logdir)

print(value)