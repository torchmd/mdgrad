import torch

LJPARAMS = {'epsilon': 1.0, 
             'sigma': 1.0}

MLPPARAMS = {'D_in': 1,
              'H': 128, 
              'num_layers': 3,
              'act': 'relu',
              'D_out': 1}

class leps(torch.nn.Module):
    def __init__(self):
        super(leps, self).__init__()
        
    def Q(self, d, r ):
        alpha = 1.942
        r0 = 0.742
        return d*( 3*torch.exp(-2*alpha*(r-r0))/2 - torch.exp(-alpha*(r-r0)) )/2
               
    def J(self, d, r ):
        alpha = 1.942
        r0 = 0.742
        return d*( torch.exp(-2*alpha*(r-r0)) - 6*torch.exp(-alpha*(r-r0)) )/4
        
    def getEnergy(self, r):  
        x=r[:, 0]
        y=r[:, 1]
        
        a = 0.05
        b = 0.3
        c = 0.05
        dAB = 4.746
        dBC = 4.746
        dAC = 3.445

        rAB = x
        rBC = y
        rAC = rAB + rBC

        JABred = self.J(dAB, rAB)/(1+a)
        JBCred = self.J(dBC, rBC)/(1+b)
        JACred = self.J(dAC, rAC)/(1+c)
                              
        return self.Q(dAB, rAB)/(1+a) + \
               self.Q(dBC, rBC)/(1+b) + \
               self.Q(dAC, rAC)/(1+c) - \
               torch.sqrt( JABred*JABred + \
                           JBCred*JBCred + \
                           JACred*JACred - \
                           JABred*JBCred - \
                           JBCred*JACred - \
                           JABred*JACred )
    def forward(self, xyz):
        
        if len( xyz.shape) == 1:
            xyz = xyz[None, ...]
        return self.getEnergy(xyz)

class MLP2d(torch.nn.Module):
    def __init__(self, D_in=2, H=128, D_out=1, num_layers=3, act='relu', excluded_vol=True):
        super(MLP2d, self).__init__()
        
        act_dict = {'relu': torch.nn.ReLU()}
        
        self.NN = torch.nn.ModuleList([])
        self.NN.append(torch.nn.Linear(D_in, H))
        self.NN.append(act_dict['relu'])
        for i in range(num_layers):
            self.NN.append(torch.nn.Linear(H, H))
            self.NN.append(act_dict['relu'])
        self.NN.append(torch.nn.Linear(H, 1))
        
    def forward(self, x):
        for layer in self.NN:
            x = layer(x)
        return x.squeeze()


class LennardJones(torch.nn.Module):
    def __init__(self, sigma=1.0, epsilon=1.0):
        super(LennardJones, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))

    def LJ(self, r, sigma, epsilon):
        return 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)

    def forward(self, x):
        return self.LJ(x, self.sigma, self.epsilon)

class LennardJones69(torch.nn.Module):
    def __init__(self, sigma=1.0, epsilon=1.0):
        super(LennardJones69, self).__init__()
        self.sigma = torch.nn.Parameter(torch.Tensor([sigma]))
        self.epsilon = torch.nn.Parameter(torch.Tensor([epsilon]))

    def LJ(self, r, sigma, epsilon):
        return 4 * epsilon * ((sigma/r)**9 - (sigma/r)**6)

    def forward(self, x):
        return self.LJ(x, self.sigma, self.epsilon)

class Buck(torch.nn.Module):
    def __init__(self, A=1.0, B=1.0, C=1.0):
        super(Buck, self).__init__()
        self.A = torch.nn.Parameter(torch.Tensor([A]))
        self.B = torch.nn.Parameter(torch.Tensor([B]))
        self.C = torch.nn.Parameter(torch.Tensor([C]))

    def Buckingham(self, r, A, B, C):
        return A * torch.exp(- B * r) - C / r**6

    def forward(self, x):
        return self.Buckingham(x, self.A, self.B, self.C)


class MLP(torch.nn.Module):
    def __init__(self, D_in=1, H=128, D_out=1, num_layers=3, act='relu', excluded_vol=True):
        super(MLP, self).__init__()
        
        act_dict = {'relu': torch.nn.ReLU()}
        
        self.NN = torch.nn.ModuleList([])
        self.NN.append(torch.nn.Linear(D_in, H))
        self.NN.append(act_dict['relu'])
        for i in range(num_layers):
            self.NN.append(torch.nn.Linear(H, H))
            self.NN.append(act_dict['relu'])
        self.NN.append(torch.nn.Linear(H, 1))
        self.excluded_vol = excluded_vol
        
    def forward(self, x):
        if self.excluded_vol:
            u_ex =  (0.6/x) ** 12
        else:
            u_ex = 0.0 
        for layer in self.NN:
            x = layer(x)
        u_ex += x
        return u_ex

class PairPot(torch.nn.Module):

    def __init__(self, pair_model, model_arg, cell, device=0, cutoff=2.5):
        super().__init__()
        self.model = pair_model(**model_arg)
        self.cell = torch.Tensor(cell).to(device)
        self.device = device
        self.cutoff = cutoff

    def forward(self, xyz):
        
        # get_nbr_list 
        dis_mat = xyz[..., None, :, :] - xyz[..., :, None, :]

        offsets = -dis_mat.ge(0.5 *  self.cell).to(torch.float).to(self.device) + \
                        dis_mat.lt(-0.5 *  self.cell).to(torch.float).to(self.device)
        dis_mat = dis_mat + offsets * self.cell

        dis_sq = dis_mat.pow(2).sum(-1)
        mask = (dis_sq < self.cutoff ** 2) & (dis_sq != 0)

        pair_dis = dis_sq[mask].sqrt()

        # compute pair energy 
        energy = self.model(pair_dis[..., None])

        return energy

### tests 

def test(pair_dis):
    linspace = torch.linspace(0, 2.5, 100)
    pair = MLP(**MLPPARAMS).to(3)
    pair(pair_dis[..., None])