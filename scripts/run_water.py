import argparse
from fit_rdf_gnn import *
from datetime import datetime
import random 

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-val", type=str, nargs='+')
parser.add_argument("-name", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-nepochs", type=int, default=700)
parser.add_argument("-nsim", type=int, default=20)
parser.add_argument("--pair", action='store_true', default=False)
parser.add_argument("-nruns", type=int, default=1)
params = vars(parser.parse_args())


# assignments = {
#   "cutoff": 6.0,
#   "epsilon": 0.010,
#   "gaussian_width": 0.5, #0.1,
#   "lr": 0.0001,
#   "mse_weight": 4.0,
#   "n_atom_basis": "tiny",
#   "n_convolutions": 2,
#   "n_filters": "high",
#   "nbins": 64, #119,
#   "opt_freq": 56,
#   "sigma": 2.6
# }

# assignments = {
#   "cutoff": 5.847718540914188,
#   "epsilon": 0.015028200854741066,
#   "gaussian_width": 0.19711403980320677,
#   "lr": 0.0001,
#   "mse_weight": 3.199742565956084,
#   "n_atom_basis": "low",
#   "n_convolutions": 2,
#   "n_filters": "low",
#   "nbins": 109,
#   "opt_freq": 52,
#   "sigma": 2.61227614490785
# }


assignments = {
    "mse_weight": 3.199742565956084,
    "epsilon": 0.4,
    "gaussian_width": 0.10,
    "n_layers": 3,
    "n_width": 128,
    "nbins": 100,
    "nonlinear": "ELU",
    "opt_freq": 60, # 60
    "power": 10,
    "rdf_weight": 0.95,
    "train_vacf": "True",
    "lr": 0.002,
    "sigma": 2.0,
    "vacf_weight": 0.0, 
    "cutoff": 8.0
  }

sys_params = {
'dt': 0.5,
'n_epochs': params['nepochs'],
'n_sim': params['nsim'],
'data': params['data'],
'val': params['val'],
'size': 4,
'anneal_flag': 'False',
'pair_flag': params['pair'],
'topology_update_freq': 1
}

print(sys_params)

for i in range(params['nruns']):

  now = datetime.now()
  dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))

  value = fit_rdf(assignments=assignments, 
                          i=i, 
                          suggestion_id=params['name'] + dt_string, 
                          device=params['device'],
                          sys_params=sys_params,
                          project_name=params['logdir'])
