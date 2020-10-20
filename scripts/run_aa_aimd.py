
import argparse
from sigopt import Connection
from gnn_rdf_aa_aimd import *

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-name", type=str)
parser.add_argument("-device", type=int, default=0)
params = vars(parser.parse_args())

n_obs = 1000
n_epochs = 1500
n_sim = 150
max_n_epochs = 80


i = 0
assignments ={
  "charge_scale": 0.003900205867209021,
  "cutoff": 4.210582618907898,
  "epsilon_hh": 0.002989186345851903,
  "epsilon_oh": 0.00729139831506259,
  "epsilon_oo": 0.009259258786038749,
  "frameskip": 2,
  "gaussian_width": 0.05061846767217373,
  "lr": 0.000005360211537981676,
  "mse_weight_hh": 0.6324572857833789,
  "mse_weight_oh": 0.9652937150722376,
  "mse_weight_oo": 0.4076961229313364,
  "n_atom_basis": "high",
  "n_convolutions": 2,
  "n_filters": "high",
  "n_train": 23,
  "nbins": 46,
  "opt_freq": 100,
  "rdf_smear_width": 0.056840302285244344,
  "rdf_start_hh": 0.8467116781350758,
  "rdf_start_oh": 1.3482504908676147,
  "rdf_start_oo": 2.076217603394575,
  "sigma_hh": 0.3379393887243994,
  "sigma_oh": 1.0087492538227876,
  "sigma_oo": 3.2073245354688096
}

sys_params = {
'dt': 0.5,
'n_epochs': n_epochs,
'n_sim': n_sim,
}

value = fit_rdf_aa(assignments=assignments, 
                        i=i, 
                        suggestion_id=params['name'], 
                        device=params['device'],
                        sys_params=sys_params,
                        project_name=params['logdir'])

print(value)
