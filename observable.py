"""Summary
"""
import torch
from nff.nn.layers import GaussianSmearing
import numpy as np

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
