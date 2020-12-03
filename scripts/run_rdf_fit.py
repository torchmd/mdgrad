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
  "anneal_freq": 17,
  "anneal_rate": 8.123308705790908,
  "cutoff": 4.282285152799321,
  "epsilon": 0.022418593690728312,
  "gaussian_width": 0.125,
  "lr": 0.00002788550012566861,
  "mse_weight": 0,
  "n_atom_basis": "mid",
  "n_convolutions": 3,
  "n_filters": "high",
  "nbins": 120,
  "opt_freq": 20,
  "sigma": 1.9581522427812932,
  "start_T": 379.19160932066393
}

sys_params = {
'dt': 1.0,
'n_epochs': params['nepochs'],
'n_sim': params['nsim'],
'data': params['data'],
'val': params['val'],
'size': 4,
'anneal_flag': "True",
'pair_flag': params['pair']
}

value = fit_rdf(assignments=assignments, 
                        i=i, 
                        suggestion_id=params['name'], 
                        device=params['device'],
                        sys_params=sys_params,
                        project_name=params['logdir'])
