
from torchmd.system import System
from fit_rdf_pair import * 
from data import pair_data_dict
from ase.lattice.cubic import FaceCenteredCubic
from torchmd.potentials import ExcludedVolume, LennardJones, LJFamily,  pairMLP
from datetime import datetime
from datetime import date
import random
from random import shuffle 



def mix_system(system, type1_composition=0.5):
    
    assert type1_composition >= 0 
    assert type1_composition <= 1
    
    '''return: system, type1_index, type2_index '''
    # generate type1 index 
    n = system.get_number_of_atoms()

    n1 = int(n * type1_composition)

    all_idx = list(range(0, n))
    shuffle(all_idx )
    idx1 = all_idx[:n1]
    idx2 = all_idx[n1:]
    z = system.get_atomic_numbers()
    z[idx2] = 2 

    system.set_atomic_numbers(z)

    return system, idx1, idx2

def collect_equilibrium_rdf(trajs, rdf_func): 
    all_g_sim = []
    for i in range(len(trajs)):
        _, xrange, g_sim = rdf_func(trajs[[i]])
        all_g_sim.append(g_sim.detach().cpu().numpy())
        
    return xrange, np.array(all_g_sim).mean(0)

def save_rdf(rdf, rdf_range, fn):
    bins = np.linspace(*rdf_range, rdf.shape[0])
    rdf_data = np.stack( (bins, rdf) )
    np.savetxt(fn, rdf_data, delimiter=',')
    
    
# plot and save potentials and got to bed 

def save_rdf(rdf, rrange, fn):
    nbins = rdf.shape[-1]
    xbins = np.linspace(*rrange, nbins)
    pack_rdf = np.stack((xbins, rdf))
    np.savetxt("{}_equi_rdf.txt".format(fn), pack_rdf, delimiter=',')
    return pack_rdf

def plot_pairs(sim, pair11, pair12, pair22, fn, save=False):

    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))

    device = sim.device

    rrange = torch.linspace(0.5, 2.5, 100).to(device)

    prior = sim.integrator.model.models['prior'].model(rrange[..., None]) 
    u11_fit = sim.integrator.model.models['mlppot11'].model(rrange[..., None])+ prior 
    u12_fit = sim.integrator.model.models['mlppot12'].model(rrange[..., None])+ prior 
    u22_fit = sim.integrator.model.models['mlppot22'].model(rrange[..., None])+ prior 
    u11_target = pair11(rrange[..., None])
    u12_target = pair12(rrange[..., None])
    u22_target = pair22(rrange[..., None])

    u11_fit = u11_fit - u11_fit[-1].item()
    u22_fit = u22_fit - u22_fit[-1].item()
    u12_fit = u12_fit - u12_fit[-1].item()

    axes[0].plot(rrange.cpu(), u11_fit.detach().cpu())
    axes[0].plot(rrange.cpu(), u11_target.detach().cpu())
    axes[1].plot(rrange.cpu(), u22_fit.detach().cpu())
    axes[1].plot(rrange.cpu(), u22_target.detach().cpu())
    axes[2].plot(rrange.cpu(), u12_fit.detach().cpu())
    axes[2].plot(rrange.cpu(), u12_target.detach().cpu())

    axes[0].set_ylim(-4, 5)
    axes[1].set_ylim(-4, 5)
    axes[2].set_ylim(-4, 5)

    plt.tight_layout()
    plt.savefig("{}_pot.pdf".format(fn))
    plt.show()
    plt.close()
    
    if save: 
        np.savetxt('{}_pot11.txt', u11_fit.detach().cpu().numpy(), delimiter=',')
        np.savetxt('{}_pot22.txt', u22_fit.detach().cpu().numpy(), delimiter=',')
        np.savetxt('{}_pot12.txt', u12_fit.detach().cpu().numpy(), delimiter=',')
        
    
def plot_sim_rdfs(sim_rdf11, sim_rdf12, sim_rdf22, target_rdf11, target_rdf12, target_rdf22, rdf_range, fn):
    fig, axes = plt.subplots(ncols=3, figsize=(10, 3))

    r_range = np.linspace(*rdf_range, sim_rdf11.shape[-1])

    axes[0].plot(r_range, sim_rdf11)
    axes[1].plot(r_range, sim_rdf22)
    axes[2].plot(r_range, sim_rdf12)

    axes[0].plot(r_range, target_rdf11)
    axes[1].plot(r_range, target_rdf22)
    axes[2].plot(r_range, target_rdf12)

    plt.tight_layout()
    plt.savefig("{}_rdf.pdf".format(fn))
    plt.show()
    plt.close()
# potential = plot_pair( path="./",
#              fn=str(i).zfill(3),
#               model=sim.integrator.model.models['mlppot22'].model, 
#               prior=sim.integrator.model.models['prior'].model, 
#               device=device,
#               target_pot=LennardJones(epsilon=1.0, sigma=1.375),
#               end=cutoff)

def run(params):
    
    logdir = params['logdir']
    subjob = params['subjob']
    device = params['device']
    size = params['size']
    T = params['T']
    rho = params['rho']
    x = params['x']
    gaussian_width = params['gaussian_width'] 
    n_width = params['n_width']
    n_layers = params['n_layers']
    nonlinear = params['nonlinear']
    sigma = params['sigma']
    n_sim = params['n_sim']
    cutoff = 2.5 

    failed = False

    model_path = '{}/{}'.format(logdir, subjob)
    os.makedirs(model_path)

    L = get_unit_len(rho=rho, N_unitcell=4)
    atoms = FaceCenteredCubic(symbol="H",
                              size=(size, size, size),
                              latticeconstant= L,
                              pbc=True)
    system = System(atoms, device=device)
    system.set_temperature(T)

    system, atom1_index, atom2_index = mix_system(system, x)
    atom1_index = torch.LongTensor(atom1_index)
    atom2_index = torch.LongTensor(atom2_index)
    
    # Define potentials for the ground truth 
    pair11 = LennardJones(epsilon=1.0, sigma=0.9)
    pair22 = LennardJones(epsilon=1.0, sigma=1.1)
    pair12 = LennardJones(epsilon=1.0, sigma=1.0)
    
    atom1_index = torch.LongTensor(list(range(0, 128)))
    atom2_index = torch.LongTensor(list(range(128,256)))
    
    pot_11 = PairPotentials(system, pair11, cutoff=2.5, 
                     nbr_list_device=device, 
                     index_tuple=(atom1_index, atom1_index)).to(device)

    pot_12 = PairPotentials(system, pair12, cutoff=2.5, 
                         nbr_list_device=device, 
                         index_tuple=(atom1_index, atom2_index)).to(device)

    pot_22 = PairPotentials(system, pair22, cutoff=2.5, 
                         nbr_list_device=device, 
                         index_tuple=(atom2_index, atom2_index)).to(device)
    
    target_model = Stack({'pot11': pot_11, 'pot22': pot_22, 'pot12': pot_12})
    
    rdf_range = (0.6, 3.3)
    
    # define 
    diffeq = NoseHooverChain(target_model, 
            system,
            Q=50.0, 
            T=1.2,
            num_chains=5, 
            adjoint=True,
            topology_update_freq=10).to(system.device)

    # define simulator with 
    sim = Simulations(system, diffeq)
    rdf11 = rdf(system, nbins=100, r_range=rdf_range, index_tuple=(atom1_index, atom1_index))
    rdf22 = rdf(system, nbins=100, r_range=rdf_range, index_tuple=(atom2_index, atom2_index))
    rdf12 = rdf(system, nbins=100, r_range=rdf_range, index_tuple=(atom1_index, atom2_index))

    print('simulating ground truth ')
    for i in range(n_sim):
        print(f"simulation epoch {i}")
        v_t, q_t, pv_t = sim.simulate(steps=50, dt=0.005, frequency=50)

    # loop over to compute observables 
    trajs = torch.Tensor( np.stack( sim.log['positions'])).to(system.device).detach()


    if trajs.shape[0] > 10:
        skip = trajs.shape[0] // 3
    else:  
        skip = 0 

    xrange, target_rdf11 = collect_equilibrium_rdf(trajs[skip:], rdf11)
    xrange, target_rdf12 = collect_equilibrium_rdf(trajs[skip:], rdf12)
    xrange, target_rdf22 = collect_equilibrium_rdf(trajs[skip:], rdf22)

    # combine save rdf 
    save_rdf(target_rdf11, rdf_range, f"{model_path}/rdf11")
    save_rdf(target_rdf12, rdf_range, f"{model_path}/rdf12")
    save_rdf(target_rdf22, rdf_range, f"{model_path}/rdf22")

    mlp_parmas = {'n_gauss': int(cutoff//gaussian_width), 
              'r_start': 0.0,
              'r_end': cutoff, 
              'n_width': n_width,
              'n_layers': n_layers,
              'nonlinear': nonlinear}

    # # Define prior potential
    pairmlp11 = pairMLP(**mlp_parmas)
    pairmlp22 = pairMLP(**mlp_parmas)
    pairmlp12 = pairMLP(**mlp_parmas)
    pair = LJFamily(epsilon=2.0, sigma=sigma, rep_pow=6, attr_pow=3) 

    mlp11 = PairPotentials(system, pairmlp11, cutoff=2.5, 
                         nbr_list_device=device, 
                         index_tuple=(atom1_index, atom1_index)).to(device)

    mlp12 = PairPotentials(system, pairmlp12, cutoff=2.5, 
                         nbr_list_device=device, 
                         index_tuple=(atom1_index, atom2_index)).to(device)

    mlp22 = PairPotentials(system, pairmlp22, cutoff=2.5, 
                         nbr_list_device=device, 
                         index_tuple=(atom2_index, atom2_index)).to(device)

    prior = PairPotentials(system, pair, cutoff=2.5, 
                           nbr_list_device=device).to(device) # prior over all patricles 

    model = Stack({'mlppot11': mlp11, 'mlppot22': mlp22, 'mlppot12': mlp12, 'prior': prior})


    # define 
    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=1.2,
            num_chains=5, 
            adjoint=True,
            topology_update_freq=10).to(system.device)

    # reinsitialize system
    # system = System(atoms, device=device)
    # system.set_temperature(T)
    sim = Simulations(system, diffeq)

    # try simulating 
    optimizer = torch.optim.Adam(list(pairmlp11.parameters()) + list(pairmlp22.parameters()) + \
                                 list(pairmlp12.parameters()), lr=params['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                              'min', 
                                              min_lr=0.9e-7, 
                                              verbose=True, factor = 0.5, patience=50,
                                              threshold=1e-5)

    target_rdf11_torch = torch.Tensor(target_rdf11).to(device)
    target_rdf12_torch = torch.Tensor(target_rdf12).to(device)
    target_rdf22_torch = torch.Tensor(target_rdf22).to(device)
    
    print(f"start training for {params['nepochs']} epochs")
    for i in range(params['nepochs']): 
        v_t, q_t, pv_t = sim.simulate(steps=50, dt=0.005, frequency=50)

        # check for NaN
        if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
            print("encounter NaN")
            return 10.0, True 

        _, _, sim_rdf11 = rdf11(q_t)
        _, _, sim_rdf12 = rdf12(q_t)
        _, _, sim_rdf22 = rdf22(q_t)

        loss = (sim_rdf11 - target_rdf11_torch).pow(2).mean() + \
               (sim_rdf12 - target_rdf12_torch).pow(2).mean() + \
                (sim_rdf22 - target_rdf22_torch).pow(2).mean() 

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 5 == 0:
            plot_pairs(sim, pair11, pair12, pair22, fn=f'{model_path}/{str(i).zfill(3)}')
            plot_sim_rdfs(sim_rdf11.detach().cpu(), sim_rdf12.detach().cpu(), sim_rdf22.detach().cpu(), 
                          target_rdf11, target_rdf12, target_rdf22, 
                          rdf_range,
                          f'{model_path}/{str(i).zfill(3)}')

        print(loss.item())

        scheduler.step(loss)
        
        
    n_equi = 10
    all_rdf11 = []
    all_rdf12 = []
    all_rdf22 = []

    for i in range(n_equi): 
        v_t, q_t, pv_t = sim.simulate(steps=50, dt=0.005, frequency=50)

        _, _, sim_rdf11 = rdf11(q_t)
        _, _, sim_rdf12 = rdf12(q_t)
        _, _, sim_rdf22 = rdf22(q_t)

        all_rdf11.append(sim_rdf11.detach().cpu().numpy())
        all_rdf12.append(sim_rdf12.detach().cpu().numpy())
        all_rdf22.append(sim_rdf22.detach().cpu().numpy())

    equi_rdf11 = np.array(all_rdf11).mean(0)
    equi_rdf12 = np.array(all_rdf12).mean(0)
    equi_rdf22 = np.array(all_rdf22).mean(0)

    save_rdf(equi_rdf11, (0.5, 2.5), f'{model_path}/pair11')
    save_rdf(equi_rdf12, (0.5, 2.5), f'{model_path}/pair12')
    save_rdf(equi_rdf22, (0.5, 2.5), f'{model_path}/pair22')

    save_rdf(target_rdf11, (0.5, 2.5), f'{model_path}/pair11_target')
    save_rdf(target_rdf12, (0.5, 2.5), f'{model_path}/pair12_target')
    save_rdf(target_rdf22, (0.5, 2.5), f'{model_path}/pair22_target')

    plot_sim_rdfs(equi_rdf11, equi_rdf12, equi_rdf22, 
                  target_rdf11, target_rdf12, target_rdf22, 
                  rdf_range,
                  f'{model_path}/{str(i).zfill(3)}')

    # compute loss 
    rdf_dev = np.abs(equi_rdf11 - target_rdf11).mean() + np.abs(equi_rdf12 - target_rdf12).mean() + \
                np.abs(equi_rdf22 - target_rdf22).mean()
    
    return rdf_dev, failed



if __name__ == "__main__": 

    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))
    # run 
    params = {'logdir': './multi_test', 'subjob':"test" + dt_string, 'device': 2, 'nepochs': 500, 'n_sim': 800, 'size': 4, 'T': 1.2, 'rho':0.8, 'x':0.5, 'lr': 1e-3, 
              'gaussian_width': 0.25, 'n_width': 128, 'n_layers': 4, 'nonlinear': 'Tanh', 'sigma':0.9}

    rdf_dev = run(params)

    print(f"rdf dev is {rdf_dev}")