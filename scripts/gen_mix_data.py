from fit_2_comp import *
from ase import io 


device = 3

mix_data = {0.5: 
                  {'rdf11': '../data/mix_data/x0.5_rdf11.csv', 
                  'rdf22': '../data/mix_data/x0.5_rdf22.csv',
                  'rdf12': '../data/mix_data/x0.5_rdf12.csv', 
                  'xyz': '../data/mix_data/x0.5.xyz',
                  'img': '../data/mix_data/x0.5_rdf.pdf',
                  'rho': 0.8, 
                  'size': 4,
                  'T': 1.2,
                  'n_sim': 1000,
                  'rdf_range': (0.6, 3.3)
             },
         0.3: 
              {'rdf11': '../data/mix_data/x0.3_rdf11.csv', 
              'rdf22': '../data/mix_data/x0.3_rdf22.csv',
              'rdf12': '../data/mix_data/x0.3_rdf12.csv', 
              'xyz': '../data/mix_data/x0.3.xyz',
              'img': '../data/mix_data/x0.3_rdf.pdf',
              'rho': 0.8, 
              'size': 4,
              'T': 1.2,
              'n_sim': 1000,
              'rdf_range': (0.6, 3.3)
         },

         0.7: 
              {'rdf11': '../data/mix_data/x0.7_rdf11.csv', 
              'rdf22': '../data/mix_data/x0.7_rdf22.csv',
              'rdf12': '../data/mix_data/x0.7_rdf12.csv', 
              'xyz': '../data/mix_data/x0.7.xyz',
              'img': '../data/mix_data/x0.7_rdf.pdf',
              'rho': 0.8, 
              'size': 4,
              'T': 1.2,
              'n_sim': 1000,
              'rdf_range': (0.6, 3.3)
         },


           }


if __name__ == "__main__": 

    for x in mix_data.keys(): 

        print(x)
        rho = mix_data[x]['rho']
        size = mix_data[x]['size']
        T = mix_data[x]['T']
        n_sim = mix_data[x]['n_sim']
        rdf_range = mix_data[x]['rdf_range']

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
        save_rdf(target_rdf11, rdf_range, mix_data[x]['rdf11'])
        save_rdf(target_rdf12, rdf_range, mix_data[x]['rdf12'])
        save_rdf(target_rdf22, rdf_range, mix_data[x]['rdf22'])


        plot_sim_rdfs(target_rdf11, target_rdf12, target_rdf22, 
                    target_rdf11, target_rdf12, target_rdf22, 
                    rdf_range,
                    mix_data[x]['img'])

        io.write(mix_data[x]['xyz'], system)