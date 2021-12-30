from fold import train 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-logdir", type=str)
parser.add_argument("-name", type=str)
parser.add_argument("-device", type=int, default=0)
parser.add_argument("-nepochs", type=int, default=1500)
params = vars(parser.parse_args())

# hyperparams = {
#   "T": 0.005,
#   "cutoff": 3.15,
#   "dt": 0.034,
#   "k0": 2.0,
#   "l_angle1": 0.27,
#   "l_bond": 0.6648074692624293,
#   "l_bond13": 1,
#   "l_bond14": 0.14802716271046765,
#   "l_bond15": 0.2920580429389884,
#   "l_bond16": 0.6028453335703119,
#   "l_dihe1": 0.25553783649798717,
#   "l_end2end": 0.0025,
#   "lr": 0.00009466977730217914,
#   "method": "NH_verlet",
#   "n_atom_basis": 16,
#   "n_convolutions": 3,
#   "n_filters": 59,
#   "n_gaussians": 42,
#   "tau": 52,
#   'epsilon' : 0.01,
#   'sigma': 1.0, 
#   'n_atoms': 
# }



# hyperparams = {"T":0.027237479852570856,
# "a_spiral":0.9698497829001933,
# "cutoff":2.5345516183162085,
# "dt":0.029608400270764428,
# "dz_spiral":0.16647481667794975,
# "epsilon":0.14115561497526943,
# "k0":0.6563094659770681,
# "l_a": 0.1, #0.7767943984337666,
# "l_b":2.6537840216273112,
# "l_d": 0.1, # 0.7240882203029186,
# "l_dis":2.6732394921685241,
# "loss_cutoff":3.9957892626785347,
# "lr":0.0015,
# "method":"NH_verlet",
# "n_atoms":25,
# "n_convolutions":2,
# "n_spiral":5,"sigma":1.2905084721268403,"tau":33, "n_atom_basis": 128,
#                   "n_filters": 128,
#                   "n_gaussians": 32 }


hyperparams = {"T":0.027237479852570856,"a_spiral":0.9698497829001933,"cutoff":2.5345516183162085,"dt":0.029608400270764428,
"dz_spiral":0.16647481667794975,"epsilon":0.14115561497526943,"k0":0.6563094659770681,
"l_a":0.7767943984337666,"l_b":0.6537840216273112,"l_d":0.7240882203029186,
"l_dis":0.6732394921685241,"loss_cutoff":3.9957892626785347,"lr":0.0001349921735606441,
"method":"rk4","n_atoms":22,"n_convolutions":3,"n_spiral":5,"sigma":1.2905084721268403,"tau":33,"n_atom_basis": 128,
                  "n_filters": 128,
                  "n_gaussians": 32}

train(params=hyperparams, 
        suggestion_id=params['name'],
        device=params['device'],
        project_name=params['logdir'],
        n_epochs=params['nepochs'])