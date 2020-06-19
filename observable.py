"""Summary
"""
import torch
from nff.nn.layers import GaussianSmearing
import numpy as np


class Observable(torch.nn.Module):
    def __init__(self):
        super(Observable, self).__init__()
    

class rdf(Observable):
    def __init__(self, atoms, nbins, device, cutoff):
        super(rdf, self).__init__()
        PI = np.pi
        self.bins = torch.linspace(0, cutoff, nbins + 1).to(device)
        
        self.smear = GaussianSmearing(
                    start=0.0,
                    stop=self.bins[-1],
                    n_gaussians=nbins,
                    trainable=False
                ).to(device)
        self.volume = atoms.get_volume()
        self.cutoff = cutoff
        self.cell = torch.Tensor( atoms.get_cell()).diag().to(device)
        self.vol_bins = 4 * PI /3*(self.bins[1:]**3 - self.bins[:-1]**3).to(device)
        self.device = device 
        self.natoms = atoms.get_number_of_atoms()
        self.nbins = nbins
        
        # scale cutoff to adjust smearing error of the last bin 
        self.cutoff += (self.bins[1] - self.bins[0]) * 2
        
        
    def forward(self, xyz):
        
        if len(list( xyz.shape )) != 3 and xyz.shape[-1] != 3:
            # Get positions 
            xyz = xyz[:, self.natoms * 3:].reshape(-1, self.natoms, 3)

        # Compute RDF         
        dis_mat = xyz[:, None, :, :] - xyz[:, :, None, :]
        offsets = -dis_mat.ge(0.5 * self.cell).to(torch.float).to(self.device) + \
                        dis_mat.lt(-0.5 * self.cell).to(torch.float).to(self.device)
        dis_mat = dis_mat + offsets * self.cell

        dis_sq = dis_mat.pow(2).sum(-1)
        mask = (dis_sq < (self.cutoff) ** 2) & (dis_sq != 0)

        pair_dis = dis_sq[mask].sqrt()

        N_count = mask.sum()
        count = self.smear(pair_dis.squeeze()[..., None]).sum(0) 
        norm = count.sum() # normalization factor for histogram 
        count = count / norm # normalize to get probability distributions 
        rdf =  count / (2 * self.vol_bins / ( (2 * self.cutoff) ** 3)) # interactions are considered twice 
        
        return self.bins, rdf 

def plot_ke(v, target_mometum):
    target = 0.5 * Natoms * 3 * (target_mometum **2)
    particle_ke = 0.5 * (v.reshape(-1, Natoms, 3).pow(2) / f_x.mass[:, None])
    sys_ke = particle_ke.sum(-1).sum(-1)
    plt.plot(sys_ke.detach().cpu().numpy())
    plt.plot([i for i in range(sys_ke.shape[0])], [target for i in range(sys_ke.shape[0])] )

def VACF(vel, t_stop=30):
    
    t_list= [i for i in range(1, t_stop, 1)]
    vacf = [(vel * vel).mean()[None]]
    vacf += [ (vel[t:] * vel[:-t]).mean()[None] for t in t_list]
    
    return torch.cat(vacf)

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

def DiffRDF(xyz, 
            cell, 
            box_volume,
            dis_bin, 
            smear_model,
            cutoff, 
            vol, 
            device,
            skip=10):  
    """Summary
    
    Args:
        xyz (TYPE): Description
        cell (TYPE): Description
        box_volume (TYPE): Description
        dis_bin (TYPE): Description
        smear_model (TYPE): Description
        cutoff (TYPE): Description
        vol (TYPE): Description
        device (TYPE): Description
        skip (int, optional): Description
    
    Returns:
        TYPE: Description
    """
    assert skip <= xyz.shape[0]

    PI = torch.Tensor([np.pi]).to(device)
    
    dis_mat = xyz[skip:, None, :, :] - xyz[skip:, :, None, :]

    offsets = -dis_mat.ge(0.5 *  cell).to(torch.float).to(device) + \
                    dis_mat.lt(-0.5 * cell).to(torch.float).to(device)
    dis_mat = dis_mat + offsets * cell

    dis_sq = dis_mat.pow(2).sum(-1)
    mask = (dis_sq < cutoff ** 2 * 1.1) & (dis_sq != 0)

    pair_dis = dis_sq[mask].sqrt()
    
    N_count = mask.sum()
    
    count = smear_model(pair_dis.squeeze()[..., None]).sum(0)
    norm = count.sum()
    
#     count = count * mask.sum() / norm
#     rdf =  count / (2 * vol *  mask.sum() / box_volume) # interactions are considered twice 

    count = count / norm
    rdf =  count / (2 * vol / box_volume) # interactions are considered twice 
     
    return rdf 
