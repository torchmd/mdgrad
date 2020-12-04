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
params = vars(parser.parse_args())

i = 0

assignments = {
  "cutoff": 7.5,
  "epsilon": 0.010,
  "gaussian_width": 0.1,
  "lr": 0.0001,
  "mse_weight": 4.0,
  "n_atom_basis": "tiny",
  "n_convolutions": 2,
  "n_filters": "high",
  "nbins": 119,
  "opt_freq": 56,
  "sigma": 2.6
}

assignments = {
  "cutoff": 7.495446445532104,
  "epsilon": 0.010419107274105301,
  "gaussian_width": 0.09428836476711633,
  "lr": 0.0001,
  "mse_weight": 4.06718828005474,
  "n_atom_basis": "tiny",
  "n_convolutions": 2,
  "n_filters": "high",
  "nbins": 119,
  "opt_freq": 56,
  "sigma": 2.5888283843240973
}



sys_params = {
'dt': 1.0,
'n_epochs': params['nepochs'],
'n_sim': params['nsim'],
'data': params['data'],
'val': params['val'],
'size': 3,
'anneal_flag': 'False',
'pair_flag': params['pair']
}

value = fit_rdf(assignments=assignments, 
                        i=i, 
                        suggestion_id=params['name'], 
                        device=params['device'],
                        sys_params=sys_params,
                        project_name=params['logdir'])
