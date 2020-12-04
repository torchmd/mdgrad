from fold import train 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-name", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-nepochs", type=int, default=1000)
params = vars(parser.parse_args())

hyperparams = {
  "T": 0.005,
  "cutoff": 3.15,
  "dt": 0.034,
  "k0": 2.0,
  "l_angle1": 0.27,
  "l_bond": 0.6648074692624293,
  "l_bond13": 1,
  "l_bond14": 0.14802716271046765,
  "l_bond15": 0.2920580429389884,
  "l_bond16": 0.6028453335703119,
  "l_dihe1": 0.25553783649798717,
  "l_end2end": 0.0025,
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


train(params=hyperparams, 
        suggestion_id=params['name'],
        device=params['device'],
        project_name=params['logdir'],
        n_epochs=params['nepochs'])