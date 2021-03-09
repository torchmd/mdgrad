import argparse
from fit_rdf_gnn import *


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

assignments = {
  "cutoff": 5.847718540914188,
  "epsilon": 0.015028200854741066,
  "gaussian_width": 0.19711403980320677,
  "lr": 0.0001,
  "mse_weight": 3.199742565956084,
  "n_atom_basis": "low",
  "n_convolutions": 2,
  "n_filters": "low",
  "nbins": 109,
  "opt_freq": 52,
  "sigma": 2.61227614490785
}

sys_params = {
'dt': 1.0,
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

  value = fit_rdf(assignments=assignments, 
                          i=i, 
                          suggestion_id=params['name'] + '_{}'.format(i), 
                          device=params['device'],
                          sys_params=sys_params,
                          project_name=params['logdir'])
