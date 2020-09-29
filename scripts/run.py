from gnn_rdf_cg_water_angle import *

data = np.load("../data/water_exp_pccp.npy")
size = 4
L = 19.73 / size
end = 7.5
nbins = 100
n_obs = 1000
tmax = 25000
n_epochs = 1000
n_sim = 50

sys_params = {
    'data': data, 
    'size': size,
    'L': L, 
    'end': end,
    'tmax': tmax,
    'dt': 1.0,
    'n_epochs': n_epochs,
    'n_sim': n_sim
    }

assignments = {
  "cutoff": 7.153,
  "epsilon": 0.0133,
  "gaussian_width": 0.196,
  "lr": 0.000082,
  "mse_weight": 6.7,
  "n_atom_basis": "tiny",
  "n_convolutions": 3,
  "n_filters": "mid",
  "nbins": 115,
  "opt_freq": 63,
  "sigma": 2.68
}

# get time 
from datetime import datetime
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


logdir = 'junk'
suggestion_id = 'test_{}'.format(current_time)
device = 3
i = 0

print(suggestion_id)

fit_rdf(assignments, i, suggestion_id, device, sys_params, logdir)