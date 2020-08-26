import torch
from nff.utils.scatter import compute_grad
from nff.utils import batch_to
from torch.nn import ModuleDict

class schwrap(torch.nn.Module):
    def __init__(self, model, batch, cell, device, cutoff = 5.0):
        super().__init__()
        self.model = model 
        self.cutoff = cutoff
        self.batch = batch_to(batch, device)
        self.cell = torch.Tensor(cell).to(device)
        self.device = model.device 
        
    def generate_nbr_list(self, xyz, cutoff=5.0):
        dis_mat = xyz[..., None, :, :] - xyz[..., :, None, :]
        offsets = -dis_mat.ge(0.5 *  self.cell).to(torch.float).to(self.device) + \
                        dis_mat.lt(-0.5 *  self.cell).to(torch.float).to(self.device)
        dis_mat = dis_mat + offsets * self.cell
        dis_sq = dis_mat.pow(2).sum(-1)
        mask = (dis_sq < self.cutoff ** 2) & (dis_sq != 0)
        nbr_list = torch.triu(mask.to(torch.long)).nonzero()
        return nbr_list
        
    def forward(self, q):   
        xyz = q.reshape(-1, 3)
        self.batch['nbr_list'] = self.generate_nbr_list(xyz).to(self.model.device)
        results = self.model(self.batch, xyz)
        return results['energy']


class Stack(torch.nn.Module):
    def __init__(self, model_dict, mode='sum'):
        super().__init__()
        self.models = ModuleDict(model_dict)
        
    def forward(self, x):
        for i, key in enumerate(self.models.keys()):
            if i == 0:
                result = self.models[key](x).reshape(-1)
            else:
                new_result = self.models[key](x)
                result += new_result.reshape(-1)
        
        return result