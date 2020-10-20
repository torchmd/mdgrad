
import argparse
from gnn_rdf_amorphous import *


parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-data", type=str, nargs='+')
parser.add_argument("-validate", type=str, nargs='+')
parser.add_argument("-name", type=str)
parser.add_argument("-device", type=int, default=0)
params = vars(parser.parse_args())

n_obs = 1000
tmax = 25000
n_epochs = 700
n_sim = 50

i = 0


# assignments = {
#   "anneal_freq": 17,
#   "anneal_rate": 8.123308705790908,
#   "cutoff": 4.282285152799321,
#   "epsilon": 0.022418593690728312,
#   "gaussian_width": 0.13258126090781497,
#   "lr": 0.00002788550012566861,
#   "mse_weight": 0,
#   "n_atom_basis": "mid",
#   "n_convolutions": 3,
#   "n_filters": "high",
#   "nbins": 115,
#   "opt_freq": 20,
#   "sigma": 1.9581522427812932,
#   "start_T": 379.19160932066393
# }

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
'tmax': tmax,
'dt': 5.0,
'n_epochs': n_epochs,
'n_sim': n_sim,
'data': params['data'],
'validate': params['validate'],
'size': 4,
'anneal_flag': False
}

value = fit_rdf(assignments=assignments, 
                        i=i, 
                        suggestion_id=params['name'], 
                        device=params['device'],
                        sys_params=sys_params,
                        project_name=params['logdir'])
