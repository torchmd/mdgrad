
import argparse
from sigopt import Connection
from gnn_fit_lj import *
from datetime import datetime
parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-nruns", type=int, default=1)
parser.add_argument("-sigma", type=float, default=0.9)
parser.add_argument("-lr", type=float, default=0.002)
parser.add_argument("-vacf_weight", type=float, default=0.41)
parser.add_argument("-name", type=str)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-val", type=str, nargs='+')
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
  "epsilon": 0.3948606926382243,
  "gaussian_width": 0.10881955784962147,
  "lr": params['lr'],
  "n_layers": 2,
  "n_width": 87,
  "nbins": 100,
  "nonlinear": "ELU",
  "opt_freq": 60,
  "power": 10,
  "rdf_weight": 0.9484437969901747,
  "sigma": params['sigma'],
  "train_vacf": "True",
  "vacf_weight": params['vacf_weight']
}

# assignments = {
#   "epsilon": 0.4,
#   "gaussian_width": 0.1,
#   "lr": 0.002,
#   "n_layers": 2,
#   "n_width": 87,
#   "nbins": 100,
#   "nonlinear": "ELU",
#   "opt_freq": 60,
#   "power": 10,
#   "rdf_weight": 1.0,
#   "sigma": params['sigma'],
#   "train_vacf": "True",
#   "vacf_weight": 0.4
# }

if assignments['train_vacf'] == 'False':
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


for i in range(params['nruns']):

    from datetime import date
    import random 

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))

    value = fit_lj(assignments=assignments, 
                            suggestion_id=params['name'] + dt_string, 
                            device=params['device'],
                            sys_params=sys_params,
                            project_name=logdir)

print(value)