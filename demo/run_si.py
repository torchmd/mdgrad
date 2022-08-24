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
parser.add_argument("-nepochs", type=int, default=1000)
parser.add_argument("-nsim", type=int, default=20)
parser.add_argument("--pair", action='store_true', default=False)
params = vars(parser.parse_args())

i = 0

assignments = {
  "anneal_freq": 7,
  "anneal_rate": 5.2,
  "cutoff": 4.9,
  "epsilon": 0.015,
  "gaussian_width": 0.145,
  "lr": 0.000025,
  "mse_weight": 0.4,
  "n_atom_basis": "high",
  "n_convolutions": 3,
  "n_filters": "mid",
  "nbins": 90,
  "opt_freq": 26,
  "sigma": 1.9,
  "start_T": 200
}

sys_params = {
'dt': 1.0,
'n_epochs': params['nepochs'],
'n_sim': params['nsim'],
'data': params['data'],
'val': params['val'],
'size': 4,
'anneal_flag': 'True',
'pair_flag': params['pair'],
'topology_update_freq': 1
}


now = datetime.now()
dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))

value = fit_rdf(assignments=assignments, 
                        i=i, 
                        suggestion_id=params['name'] + dt_string, 
                        device=params['device'],
                        sys_params=sys_params,
                        project_name=params['logdir'])
