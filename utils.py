
from ase import io 
from ase import Atoms 

def dump_mov(movie, ref_atoms, fname="./LJ.pdb"):
	atoms_list = []
	for frame in movie:
	    sys = Atoms(ref_atoms.get_atomic_numbers(),
	           positions=frame.reshape(-1, 3), pbc=True, cell=ref_atoms.get_cell()) 
	    sys.set_positions(sys.get_positions(wrap=True))
	    atoms_list.append(sys) 

	io.write(fname, atoms_list)