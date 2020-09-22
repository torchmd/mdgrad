"""Summary
"""
import torch
import torchmd
from nff.nn.layers import GaussianSmearing
import numpy as np
from torchmd.system import generate_nbr_list, check_system

class Observable(torch.nn.Module):
    def __init__(self, system):
        super(Observable, self).__init__()
        check_system(system)
        self.device = system.device
        self.volume = system.get_volume()
        self.cell = torch.Tensor( system.get_cell()).diag().to(self.device)
        self.natoms = system.get_number_of_atoms()

class rdf(Observable):
    def __init__(self, system, nbins, r_range, index_tuple=None):
        super(rdf, self).__init__(system)
        PI = np.pi

        start = r_range[0]
        end = r_range[1]

        self.device = system.device
        self.bins = torch.linspace(start, end, nbins + 1).to(self.device)
        self.smear = GaussianSmearing(
            start=start,
            stop=self.bins[-1],
            n_gaussians=nbins,
            trainable=False
        ).to(self.device)
        self.cutoff = end
        # compute volume differential 
        self.vol_bins = 4 * PI /3*(self.bins[1:]**3 - self.bins[:-1]**3).to(self.device)
        self.nbins = nbins
        self.cutoff_boundary = self.cutoff + 5e-1
        self.index_tuple = index_tuple
        
    def forward(self, xyz):
        
        # Compute RDF         
        # dis_mat = xyz[:, None, :, :] - xyz[:, :, None, :]
        # offsets = -dis_mat.ge(0.5 * self.cell).to(torch.float).to(self.device) + \
        #                 dis_mat.lt(-0.5 * self.cell).to(torch.float).to(self.device)
        # dis_mat = dis_mat + offsets * self.cell
        # dis_sq = dis_mat.pow(2).sum(-1)
        # mask = (dis_sq < (self.cutoff_boundary) ** 2) & (dis_sq != 0)
        # pair_dis = dis_sq[mask].sqrt()

        nbr_list, pair_dis = generate_nbr_list(xyz, 
                                               self.cutoff, 
                                               self.cell, 
                                               index_tuple=self.index_tuple, 
                                               get_dis=True)

        count = self.smear(pair_dis.reshape(-1).squeeze()[..., None]).sum(0) 
        norm = count.sum()   # normalization factor for histogram 
        count = count / norm   # normalize 
        count = count
                         
        V = (4/3)* np.pi * (self.cutoff) ** 3
        rdf =  count / (self.vol_bins / V )  
        
        return count, self.bins, rdf 


class vacf(Observable):
    def __init__(self, system, t_range):
        super(vacf, self).__init__(system)
        self.t_window = [i for i in range(1, t_range, 1)]

    def forward(self, vel):
        vacf = [(vel * vel).mean()[None]]
        # can be implemented in parrallel
        vacf += [ (vel[t:] * vel[:-t]).mean()[None] for t in self.t_window]

        return torch.stack(vacf).reshape(-1)

def plot_ke(v, target_mometum):
    target = 0.5 * Natoms * 3 * (target_mometum **2)
    particle_ke = 0.5 * (v.reshape(-1, Natoms, 3).pow(2) / f_x.mass[:, None])
    sys_ke = particle_ke.sum(-1).sum(-1)
    plt.plot(sys_ke.detach().cpu().numpy())
    plt.plot([i for i in range(sys_ke.shape[0])], [target for i in range(sys_ke.shape[0])] )

def compute_virial(q, model):
    u = model(q)
    f = -compute_grad(inputs=q, output=u)
    virial = (f * q).sum(-1).sum(-1)
    
    return virial 

def compute_bond(xyz, bonds):
    assert len(xyz.shape) == 3 
    bonds = (xyz[:, bonds[:,0], :] - xyz[:, bonds[:,1], :]).pow(2).sum(-1).sqrt()
    return bonds

def compute_angle(xyz, angles):
    assert len(xyz.shape) == 3 
    n_frames = xyz.shape[0]
    xyz = xyz[:, :, None, :]
    N = xyz.shape[1]
    D = xyz.expand(n_frames, N,N,3)-xyz.expand(n_frames, N,N,3).transpose(1,2)
    angle_vec1 = D[:, angles[:,0], angles[:,1], :]
    angle_vec2 = D[:, angles[:,1], angles[:,2], :]
    dot_unnorm = (-angle_vec1*angle_vec2).sum(-1)
    norm = torch.sqrt((angle_vec1.pow(2)).sum(-1)*(angle_vec2.pow(2)).sum(-1))
    cos_theta = (dot_unnorm/norm)
    return cos_theta

def compute_dihe(xyz, dihes): 
    assert len(xyz.shape) == 3
    n_frames = xyz.shape[0]
    N = xyz.shape[1]
    xyz = xyz[:, :, None, :]
    D = xyz.expand(n_frames, N,N,3)-xyz.expand(n_frames, N,N,3).transpose(1,2)
    vec1 = D[:, dihes[:,1], dihes[:,0]]
    vec2 = D[:, dihes[:,1], dihes[:,2]]
    vec3 = D[:, dihes[:,2], dihes[:,1]]
    vec4 = D[:, dihes[:,2], dihes[:,3]]
    cross1 = torch.cross(vec1, vec2)
    cross2 = torch.cross(vec3, vec4)

    norm = (cross1.pow(2).sum(-1)*cross2.pow(2).sum(-1)).sqrt()
    cos_phi = 1.0*((cross1*cross2).sum(-1)/norm)
    
    return cos_phi 

def var_K(N_atoms, avg_momentum):
    """compute variances of kinetic energy 
    
    Args:
        N_atoms (TYPE): Description
        avg_momentum (TYPE): Description
    
    Returns:
        TYPE: Description
    """
    return (2 * ((0.5 * 3 * N_atoms * avg_momentum **2 ) ** 2)/(3 * N_atoms) ) ** (1/2)