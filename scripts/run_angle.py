
import argparse
from gnn_rdf_cg_water_angle import *


parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-name", type=str)
parser.add_argument("-device", type=int, default=0)
params = vars(parser.parse_args())

n_obs = 1000
tmax = 25000
n_epochs = 1000
n_sim = 50

data = np.load("../data/water_exp_pccp.npy")
size = 3
L = 6.2148165251414
end = 7.5
i = 0


assignments = {
  "angle_JS_weight": 0.5174990198213657,
  "angle_MSE_weight": 0.33829901232081416,
  "angle_cutoff": 3.25,
  "angle_train_start": 10,
  "cutoff": 5.356750236048232,
  "epsilon": 0.01048872370783196,
  "frameskip_ratio": 0.2765443257921782,
  "gaussian_width": 0.21180434292221162,
  "lr": 0.00006792254995167143,
  "mse_weight": 0.28935857429983436,
  "n_atom_basis": "mid",
  "n_convolutions": 4,
  "n_filters": "mid",
  "nbins": 132,
  "nbins_angle_train": 70,
  "opt_freq": 55,
  "sigma": 1.3,
  "angle_start_train": 0.7,
  'anneal_rate': 5,
  'start_T': 400.0,
  'anneal_freq': 5
}


sys_params = {
'data': data, 
'size': size,
'L': L, 
'end': end,
'tmax': tmax,
'dt': 1.0,
'n_epochs': n_epochs,
'n_sim': n_sim,
'anneal_flag': 'True'
}

value = fit_rdf(assignments=assignments, 
                        i=i, 
                        suggestion_id=params['name'], 
                        device=params['device'],
                        sys_params=sys_params,
                        project_name=params['logdir'])
