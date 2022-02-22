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
parser.add_argument("--tpair", action='store_true', default=False)
parser.add_argument("-nruns", type=int, default=1)
parser.add_argument('-opt_freq', type=int, default=84)
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


# assignments = {"cutoff":6.16,"epsilon":1.5858519218705223,"gaussian_width":0.20182823407126177,
#                 "lr":0.00048351883733428714,"mse_weight":0.12409285934040738,"n_layers":5,
#                 "n_width":128,"nbins":110,"nonlinear":"LeakyReLU","opt_freq":params['opt_freq'],"power":10,
#                 "sigma":1.68, 'res': False}

assignments =  {"cutoff":5.03357321594869,"epsilon":1.8245160642515632,"gaussian_width":0.1548726489982903,"lr":0.0006548601438181719,
"mse_weight":0.34529615069857633,"n_layers":3,"n_width":115,"nbins":125,"nonlinear":"ELU","opt_freq":192,"power":12,"sigma":1.68191635809129}

sys_params = {
'dt': 0.5,
'n_epochs': params['nepochs'],
'n_sim': params['nsim'],
'data': params['data'],
'val': params['val'],
'size': 4,
'anneal_flag': 'False',
'pair_flag': params['pair'],
'tpair_flag': params['tpair'],
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
