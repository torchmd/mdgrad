from gnn_rdf import *

data = np.load("../experiments/water_exp_pccp.npy")
size = 4
L = 19.73 / size
r_range = 7.5
nbins = 100
tmax = 45000 #100000


sys_params = {
'data': data, 
'size': size,
'L': L, 
'r_range': r_range,
'nbins': nbins, 
'tmax': tmax
}

assignments = {
  "cutoff": 6.012129873307664,
  "epsilon": 0.021137576470382537,
  "lr": 1e-5,
  "mse_weight": 4.2908588596170585,
  "n_atom_basis": "high",
  "n_convolutions": 2,
  "n_filters": "high",
  "n_gaussians": "low",
  "opt_freq": 30,
  "sigma": 2.5232758974537326,
}


logdir = 'water_rdf_298k'
suggestion_id = 1
device = 3
i = 0 

evaluate_model(assignments, i, suggestion_id, device, sys_params, logdir)