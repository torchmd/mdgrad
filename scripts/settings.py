from sigopt import Connection
from sigopt.examples import franke_function

import os
import numpy as np
import matplotlib.pyplot as plt
import sys 

ODE_PATH = '/home/wwj/Repo/projects/torchdiffeq/'

sys.path.insert(0, ODE_PATH)
sys.path.insert(0, '../..')
sys.path.insert(0, '../')

import torch
from torch.optim import Adam
from torchmd.md import NHCHAIN_ODE
from torchmd.observable import rdf
from torchmd.potentials import PairPot, MLP, LennardJones, ExcludedVolume, Buck, LennardJones69
from torchmd.utils import *
from nff.io.ase import AtomsBatch

from ase import Atoms
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing

import ase
from ase.neighborlist import neighbor_list
from ase.lattice.cubic import FaceCenteredCubic, SimpleCubic
from ase.geometry import wrap_positions
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase import units

from datetime import datetime

from torchmd.io import Stack, schwrap
from torchmd.sovlers import odeint_adjoint, odeint

from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict
from nff.train import Trainer, get_trainer, get_model

matplotlib.rcParams.update({'font.size': 25})
matplotlib.rc('lines', linewidth=3, color='g')
matplotlib.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams['axes.linewidth'] = 2.0
matplotlib.rcParams["xtick.major.size"] = 6
matplotlib.rcParams["ytick.major.size"] = 6
matplotlib.rcParams["ytick.major.width"] = 2
matplotlib.rcParams["xtick.major.width"] = 2
matplotlib.rcParams['text.usetex'] = False

from scipy import interpolate