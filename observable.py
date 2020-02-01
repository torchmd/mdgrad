"""Summary
"""
import torch
from nff.nn.layers import GaussianSmearing
import numpy as np

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
