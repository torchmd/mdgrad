
from ase import io 
from ase import Atoms 
import numpy as np 
import torch
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 15})
matplotlib.rc('lines', linewidth=4, color='g')
matplotlib.rcParams['axes.linewidth'] = 3.0

def dump_mov(movie, ref_atoms, fname="./LJ.pdb"):
	atoms_list = []
	for frame in movie:
	    sys = Atoms(ref_atoms.get_atomic_numbers(),
	           positions=frame.reshape(-1, 3), pbc=True, cell=ref_atoms.get_cell()) 
	    sys.set_positions(sys.get_positions(wrap=True))
	    atoms_list.append(sys) 

	io.write(fname, atoms_list)


def plot_lesp(model, traj=None, res=50, start=[3.5, 0.8], end=[0.8, 3.5], fname=None):
    xlist = np.linspace(0.5, 5.0, res)
    ylist = np.linspace(0.5, 5.0, res)
    X, Y = np.meshgrid(xlist, ylist)

    data = torch.Tensor(np.concatenate((X[:,:, None],Y[:,:,None]), axis=2).reshape(-1,2))
    E = model(data).detach().cpu().numpy().reshape(res, res)
    
    plt.figure(figsize=(7,7))
    cp = plt.contourf(X, Y, E, 40, cmap='GnBu', alpha=0.4)
    plt.colorbar(cp)
    
    if traj is not None:
        traj = traj.detach().cpu().numpy()
        colors = cm.rainbow(np.linspace(0, 1, traj.shape[0]))

        for i,c in enumerate(colors):

            plt.scatter(traj[i, 2], traj[i, 3], color=c, s=1)
        
    plt.scatter(start[0], start[1], c='red')
    plt.scatter(end[0], end[1], c='red')
    
    plt.xlim((0.5, 5.0))
    plt.ylim((0.5, 5.0))
    
    if fname is not None:
        plt.savefig(fname)
    
    plt.show()