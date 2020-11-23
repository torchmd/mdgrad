from fold import train 


params = {
  "T": 0.0075,
  "cutoff": 3.1559881833397347,
  "dt": 0.034251112820653726,
  "k0": 1.978841425420648,
  "l_angle1": 0.2727264843535817,
  "l_bond": 0.6648074692624293,
  "l_bond13": 1,
  "l_bond14": 0.14802716271046765,
  "l_bond15": 0.2920580429389884,
  "l_bond16": 0.6028453335703119,
  "l_dihe1": 0.25553783649798717,
  "l_end2end": 0.01,
  "lr": 0.00009466977730217914,
  "method": "NH_verlet",
  "n_atom_basis": 16,
  "n_convolutions": 3,
  "n_filters": 59,
  "n_gaussians": 42,
  "tau": 52,
  'epsilon' : 0.01,
  'sigma': 1.0
}

params['device'] = 0 

logdir = 'test1'
name = 'end2end'
n_epochs = 500

train(params=params, 
        suggestion_id=name, 
        device=params['device'],
        project_name=logdir,
        n_epochs=n_epochs)