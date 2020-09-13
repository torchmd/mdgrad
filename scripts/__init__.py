from sigopt import Connection

import os
import numpy as np
import matplotlib.pyplot as plt
import sys 

import torch
from torch.optim import Adam
from torchmd.md import NoseHooverChain
from torchmd.observable import rdf, vacf
from torchmd.potentials import ExcludedVolume

from ase import Atoms
from nff.utils.scatter import compute_grad
from nff.nn.layers import GaussianSmearing

import ase
from ase.neighborlist import neighbor_list
from ase.lattice.cubic import FaceCenteredCubic
from ase.geometry import wrap_positions
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

from datetime import datetime

from torchmd.system import Stack, System, GNNPotentials, PairPot
from torchmd.sovlers import odeint_adjoint, odeint

from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict

import matplotlib
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