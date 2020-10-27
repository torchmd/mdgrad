
import argparse
from sigopt import Connection
from gnn_fit_lj import *

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-name", type=str)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-val", type=str, nargs='+')
parser.add_argument("--train_vacf", action='store_true', default=False)
parser.add_argument("--dry_run", action='store_true', default=False)
params = vars(parser.parse_args())

if params['dry_run']:
    token = 'FSDXBSGDUZUQEDGDCYPCXFTRXFNYBVXVACKZQUWNSOKGKGFN'
    n_obs = 2
    n_epochs = 4
    n_sim = 2
else:
    token = 'RXGPHWIUAMLHCDJCDBXEWRAUGGNEFECMOFITCRHCEOBRMGJU'
    n_obs = 1000
    n_epochs = 1000
    n_sim = 50

logdir = params['logdir']


assignments = {
  "epsilon": 0.473500357318721,
  "gaussian_width": 0.15039612844414205,
  "lr": 5e-3, #0.004010476824304749,
  "nbins": 88,
  "opt_freq": 59,
  "power": 10,
  "rdf_weight": 0.941665949654944,
  "sigma": 1.0004267435304715,
  "vacf_weight": 0.9226457751885145
}

if not params['train_vacf']:
    assignments['vacf_weight'] = 0.0

sys_params = {
'dt': 0.01,
'size': 4,
'n_epochs': n_epochs,
'n_sim': n_sim,
'data': params['data'],
'val': params['val'],
't_range': 50
}

value = fit_lj(assignments=assignments, 
                        suggestion_id=params['name'], 
                        device=params['device'],
                        sys_params=sys_params,
                        project_name=logdir)

print(value)