import argparse
from fit_2_comp import *
from gen_mix_data import * 
from datetime import datetime
from datetime import date
import random
import numpy as np
from ase import io 
from munch import Munch


def pretrain(x_list, params, pairmlp11, pairmlp12, pairmlp22, kT=1.0):
    'fit to the BI of the first system'


    device = params['device']

    all_pot11 = []
    all_pot12 = []
    all_pot22 = []

    for x in x_list:
        # load target rdfs 
        rdf_range11, target_rdf11 = np.loadtxt(mix_data[x]['rdf11'], delimiter=',')
        rdf_range12, target_rdf12 = np.loadtxt(mix_data[x]['rdf12'], delimiter=',')
        rdf_range22, target_rdf22 = np.loadtxt(mix_data[x]['rdf22'], delimiter=',')

        pot_11 = - kT * torch.log(torch.Tensor(target_rdf11))#.to(device)
        pot_12 = - kT * torch.log(torch.Tensor(target_rdf12))#.to(device)
        pot_22 = - kT * torch.log(torch.Tensor(target_rdf22))#.to(device)
        
        all_pot11.append(pot_11)
        all_pot12.append(pot_12)
        all_pot22.append(pot_22)

    bi_11 = torch.stack(all_pot11).mean(0)
    bi_12 = torch.stack(all_pot12).mean(0)
    bi_22 = torch.stack(all_pot22).mean(0)

    bi_11 = torch.nan_to_num(bi_11,  posinf=100.0)
    bi_12 = torch.nan_to_num(bi_12,  posinf=100.0)
    bi_22 = torch.nan_to_num(bi_22,  posinf=100.0)

    range11, range12, range22 = torch.Tensor(rdf_range11).to(device), torch.Tensor(rdf_range12).to(device), torch.Tensor(rdf_range22).to(device)

    pair = LJFamily(epsilon=2.0, sigma=params['sigma'], rep_pow=6, attr_pow=3).to(device)

    optimizer =  torch.optim.Adam(list(pairmlp11.parameters()) + list(pairmlp22.parameters()) + \
                                 list(pairmlp12.parameters()), lr=1e-3)


    print("pretraining pair potentials ")
    for i in range(4000): 
        pred11 = pairmlp11(range11.reshape(-1, 1)) + pair(range11.reshape(-1, 1))
        pred12 = pairmlp12(range12.reshape(-1, 1)) + pair(range12.reshape(-1, 1))
        pred22 = pairmlp22(range22.reshape(-1, 1)) + pair(range22.reshape(-1, 1))

        loss = (pred11.squeeze() - bi_11.to(device )).pow(2).mean() + \
                (pred12.squeeze() - bi_12.to(device )).pow(2).mean() + \
                (pred22.squeeze() - bi_22.to(device )).pow(2).mean() 

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if i % 50 == 0:
            print(i, loss.item())


def prepare_sim(sys, x, params, pairmlp11, pairmlp12, pairmlp22):

    device = params['device']
    sys[x] = Munch()

    # load target rdfs 
    sys[x].target_rdf11 = np.loadtxt(mix_data[x]['rdf11'], delimiter=',')[1]
    sys[x].target_rdf12 = np.loadtxt(mix_data[x]['rdf12'], delimiter=',')[1]
    sys[x].target_rdf22 = np.loadtxt(mix_data[x]['rdf22'], delimiter=',')[1]

    rdf_range = mix_data[x]['rdf_range']


    L = get_unit_len(rho=mix_data[x]['rho'], N_unitcell=4)
    size = mix_data[x]['size']

    # get system 
    atoms = io.read(mix_data[x]['xyz'])
    system = System(atoms, device=device)
    system.set_temperature(mix_data[x]['T'])

    atom1_index = torch.nonzero(torch.Tensor(system.get_atomic_numbers() == 1)).squeeze()
    atom2_index = torch.nonzero(torch.Tensor(system.get_atomic_numbers() == 2)).squeeze()

    sys[x].system = system 

    atom1_index = torch.LongTensor(atom1_index)
    atom2_index = torch.LongTensor(atom2_index)
        
    pair = LJFamily(epsilon=2.0, sigma=params['sigma'], rep_pow=6, attr_pow=3) 

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

    model = Stack({'mlppot11': mlp11, 'mlppot22': mlp22, 'mlppot12': mlp12,
                     'prior': prior
                    })

    # define 
    diffeq = NoseHooverChain(model, 
            system,
            Q=50.0, 
            T=1.2,
            num_chains=5, 
            adjoint=True,
            topology_update_freq=10).to(device)

    sys[x].diffeq = diffeq

    sim = Simulations(system, diffeq)

    sys[x].sim = sim 

    sys[x].rdf11 = rdf(system, nbins=100, r_range=rdf_range, index_tuple=(atom1_index, atom1_index))
    sys[x].rdf22 = rdf(system, nbins=100, r_range=rdf_range, index_tuple=(atom2_index, atom2_index))
    sys[x].rdf12 = rdf(system, nbins=100, r_range=rdf_range, index_tuple=(atom1_index, atom2_index))   

    return sys

def run_mix(params):


    now = datetime.now()
    dt_string = now.strftime("%m-%d-%H-%M-%S") + str(random.randint(0, 100))

    subjob = params['subjob'] + dt_string

    model_path = '{}/{}'.format(params['logdir'], subjob)
    os.makedirs(model_path)

    device = params['device']

    # initialize pair neural nets 
    mlp_parmas = {'n_gauss': int(params['cutoff']//params['gaussian_width']), 
              'r_start': 0.0,
              'r_end': params['cutoff'], 
              'n_width': params['n_width'],
              'n_layers': params['n_layers'],
              'nonlinear': params['nonlinear'],
              'res': params['res']}


    # # Define prior potential
    pairmlp11 = pairMLP(**mlp_parmas).to(device)
    pairmlp22 = pairMLP(**mlp_parmas).to(device)
    pairmlp12 = pairMLP(**mlp_parmas).to(device)

    # Define potentials for the ground truth 
    pair11 = LennardJones(epsilon=1.0, sigma=0.9).to(device)
    pair22 = LennardJones(epsilon=1.0, sigma=1.1).to(device)
    pair12 = LennardJones(epsilon=1.0, sigma=1.0).to(device)

    train_sys = {} 
    val_sys = {}

    pretrain(params['trainx'], params, pairmlp11, pairmlp12, pairmlp22, kT=1.0)

    for i, x in  enumerate(params['trainx']): 
        train_sys = prepare_sim(train_sys, x, params, pairmlp11, pairmlp12, pairmlp22)


    if params['valx'] != None:
        for x in  params['valx']: 
            val_sys = prepare_sim(val_sys, x, params, pairmlp11, pairmlp12, pairmlp22)


    # try simulating 
    optimizer = torch.optim.SGD(list(pairmlp11.parameters()) + list(pairmlp22.parameters()) + \
                                 list(pairmlp12.parameters()), lr=params['lr'])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                              'min', 
                                              min_lr=0.9e-7, 
                                              verbose=True, factor = 0.5, patience=50,
                                              threshold=1e-5)

    print(f"start training for {params['nepochs']} epochs")

    loss_log = []
    for i in range(params['nepochs']): 

        loss = torch.Tensor([0.0]).to(device)

        for epoch in range(params['update_epoch']):
            
            all_rdf11 = []
            all_rdf12 = []
            all_rdf22 = [] 

            for x in params['trainx']:
                tau = params['nsteps']
                v_t, q_t, pv_t = train_sys[x].sim.simulate(steps=tau, dt=0.005, frequency=tau)

                # check for NaN
                if torch.isnan(q_t.reshape(-1)).sum().item() > 0:
                    print("encounter NaN")
                    return 10.0, True 

                _, _, sim_rdf11 = train_sys[x].rdf11(q_t)
                _, _, sim_rdf12 = train_sys[x].rdf12(q_t)
                _, _, sim_rdf22 = train_sys[x].rdf22(q_t)

                loss_ = (sim_rdf11 - torch.Tensor(train_sys[x].target_rdf11).to(device) ).pow(2).mean() + \
                        (sim_rdf12 - torch.Tensor(train_sys[x].target_rdf12).to(device) ).pow(2).mean() + \
                        (sim_rdf22 - torch.Tensor(train_sys[x].target_rdf22).to(device) ).pow(2).mean() 

                all_rdf11.append(sim_rdf11)
                all_rdf12.append(sim_rdf12)
                all_rdf22.append(sim_rdf22)

                loss_.backward()

                loss += loss_.item()

                mean_all_rdf11 = torch.stack(all_rdf11).detach().cpu().mean(0)
                mean_all_rdf12 = torch.stack(all_rdf12).detach().cpu().mean(0)
                mean_all_rdf22 = torch.stack(all_rdf22).detach().cpu().mean(0)

                if i % 5 == 0:  
                    plot_pairs(train_sys[x].sim, pair11, pair12, pair22, fn=f'{model_path}/x_{x}_{str(i).zfill(3)}_pot.pdf')
                    plot_sim_rdfs(mean_all_rdf11, mean_all_rdf12, mean_all_rdf22, 
                                  train_sys[x].target_rdf11, train_sys[x].target_rdf12, train_sys[x].target_rdf22, 
                                  mix_data[x]['rdf_range'],
                                  f'{model_path}/x_{x}_{str(i).zfill(3)}_rdf.pdf')
 
        optimizer.step()
        optimizer.zero_grad()
 
        print(loss.item() / params['update_epoch'])
        loss_log.append(loss.item() / params['update_epoch'] )

        scheduler.step(loss)


    # save loss log 

    loss_log = np.array(loss_log)
    np.savetxt(f"{model_path}/loss_log.txt" ,loss_log)
    plt.plot(loss_log)
    plt.savefig(f"{model_path}/loss_log.pdf")
    plt.show()
    plt.close()

    # run equilibrabtion 

    all_sys = {**train_sys, **val_sys}

    rdf_devs = 0.0

    for x in all_sys.keys():
        print(f"simulating system {x}" )
        for i in range(params['nsim']):
            v_t, q_t, pv_t = all_sys[x].sim.simulate(steps=50, dt=0.005, frequency=50)


        # loop over to compute observables 
        trajs = torch.Tensor( np.stack( all_sys[x].sim.log['positions'])).to(device).detach()

        skip = trajs.shape[0] // 3

        xrange, sim_rdf11 = collect_equilibrium_rdf(trajs[skip:], all_sys[x].rdf11)
        xrange, sim_rdf12 = collect_equilibrium_rdf(trajs[skip:], all_sys[x].rdf12)
        xrange, sim_rdf22 = collect_equilibrium_rdf(trajs[skip:], all_sys[x].rdf22)

        # combine save rdf 
        save_rdf(sim_rdf11, mix_data[x]['rdf_range'], f"{model_path}/equi_x_{x}_rdf11.csv")
        save_rdf(sim_rdf12, mix_data[x]['rdf_range'], f"{model_path}/equi_x_{x}_rdf12.csv")
        save_rdf(sim_rdf22, mix_data[x]['rdf_range'], f"{model_path}/equi_x_{x}_rdf22.csv")

        plot_sim_rdfs(sim_rdf11, sim_rdf12, sim_rdf22, 
                    all_sys[x].target_rdf11, all_sys[x].target_rdf12, all_sys[x].target_rdf22, 
                    mix_data[x]['rdf_range'],
                    f"{model_path}/equi_x_{x}_rdf.pdf")

        plot_pairs(all_sys[x].sim, pair11, pair12, pair22, f"{model_path}/equi_x_{x}_pot", save=True, nbins=1000)

        # save potentials 

        if x in train_sys.keys():
            rdf_devs += np.abs(sim_rdf11 - all_sys[x].target_rdf11).mean() + np.abs(sim_rdf12 - all_sys[x].target_rdf12).mean() + \
                         np.abs(sim_rdf22 - all_sys[x].target_rdf22).mean()

        print(rdf_devs)
    return rdf_devs, False


if __name__ == '__main__': 

    parser = argparse.ArgumentParser()
    parser.add_argument("-logdir", type=str)
    parser.add_argument("-subjob", type=str)
    parser.add_argument("-nruns", type=int, default=1)
    parser.add_argument("-device", type=int, default=0)
    parser.add_argument("-nepochs", type=int, default=30)
    parser.add_argument("-nsteps", type=int, default=50)
    parser.add_argument("-nsim", type=int, default=40)
    parser.add_argument("-update_epoch", type=int, default=1)
    parser.add_argument("-trainx", type=float, nargs='+')
    parser.add_argument("-valx", type=float, nargs='+')
    parser.add_argument("-lr", type=float, default=3e-3)
    parser.add_argument("-gaussian_width", type=float, default=0.25)
    parser.add_argument("-cutoff", type=float, default=2.5)
    parser.add_argument("-sigma", type=float, default=1.0)
    parser.add_argument("-n_width", type=int, default=128)
    parser.add_argument("-n_layers", type=int, default=2)
    parser.add_argument("-nonlinear", type=str, default='SELU')
    parser.add_argument("--res", action='store_true', default=False)

    params = vars(parser.parse_args())
    for i in range(params['nruns']):
        print(f"run {i}")
        run_mix(params)
