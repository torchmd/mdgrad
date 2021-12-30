
import torch 
import numpy as np 
from scipy import interpolate

from ase.lattice.cubic import FaceCenteredCubic, Diamond
from torchmd.observable import generate_vol_bins

from torchmd.potentials import ExcludedVolume, LennardJones, SplineOverlap, LJFamily, ModifiedMorse

def get_exp_rdf(data, nbins, r_range, device, dim=3):
    # load RDF data 
    if data.shape[0] == 2:
        f = interpolate.interp1d(data[0], data[1])
    elif data.shape[1] == 2:
        f = interpolate.interp1d(data[:,0], data[:,1])

    start = r_range[0]
    end = r_range[1]
    xnew = np.linspace(start, end, nbins)
        
    # generate volume bins 
    V, vol_bins, _ = generate_vol_bins(start, end, nbins, dim=dim)
    vol_bins = vol_bins.to(device)

    g_obs = torch.Tensor(f(xnew)).to(device)
    g_obs_norm = ((g_obs.detach() * vol_bins).sum()).item()
    g_obs = g_obs * (V/g_obs_norm)
    count_obs = g_obs * vol_bins / V

    return xnew, g_obs

def exp_angle_data(nbins, angle_range, fn='../data/water_angle_pccp.csv'):
    angle_data = np.loadtxt(fn, delimiter=',')
    # convert angle to cos(phi)
    cos = angle_data[:, 0] * np.pi / 180
    density = angle_data[:, 1]
    f = interpolate.interp1d(cos, density)
    start = angle_range[0]
    end = angle_range[1]
    xnew = np.linspace(start, end, nbins)
    density = f(xnew)
    density /= density.sum()
    
    return density

def get_unit_len(rho, mass, N_unitcell):
    
    Na = 6.02214086 * 10**23 # avogadro number 

    N = (rho * 10**6 / mass) * Na  # number of molecules in 1m^3 of water 

    rho = N / (10 ** 30) # number density in 1 A^3
 
    L = (N_unitcell / rho) ** (1/3)
    
    return L 

pair_data_dict = {
    'lj_0.845_1.5': { 
                      'rdf_fn': '../data/LJ_data/rdf_rho0.845_T1.5_dt0.01.csv' ,
                      'vacf_fn': '../data/LJ_data/vacf_rho0.845_T1.5_dt0.01.csv',
                       'rho': 0.845,
                        'T': 1.5, 
                        'start': 0.75, 
                        'end': 3.3,
                        'element': "H",
                        'mass': 1.0,
                        "N_unitcell": 4,
                        "cell": FaceCenteredCubic,
                        "target_pot": LennardJones()
                        },

    'lj_0.845_1.0': {
                    'rdf_fn': '../data/LJ_data/rdf_rho0.845_T1.0_dt0.01.csv' ,
                    'vacf_fn': '../data/LJ_data/vacf_rho0.845_T1.0_dt0.01.csv' ,
                   'rho': 0.845,
                    'T': 1.0, 
                    'start': 0.75, 
                    'end': 3.3,
                    'element': "H",
                    'mass': 1.0,
                    "N_unitcell": 4,
                    "cell": FaceCenteredCubic,
                    "target_pot": LennardJones()
                    },

    'lj_0.845_0.75': {
                    'rdf_fn': '../data/LJ_data/rdf_rho0.845_T0.75_dt0.01.csv' ,
                    'vacf_fn': '../data/LJ_data/vacf_rho0.845_T0.75_dt0.01.csv' ,
                   'rho': 0.845,
                    'T': 0.75, 
                    'start': 0.75, 
                    'end': 3.3,
                    'element': "H",
                    'mass': 1.0,
                    "N_unitcell": 4,
                    "cell": FaceCenteredCubic,
                    "target_pot": LennardJones()
                    },

    'lj_0.7_1.2': {
                'rdf_fn': '../data/LJ_data/rdf_rho0.7_T1.2_dt0.01.csv' ,
                'vacf_fn': '../data/LJ_data/vacf_rho0.7_T1.2_dt0.01.csv' ,
                'rho': 0.7,
                'T': 1.2, 
                'start': 0.75, 
                'end': 3.3,
                'element': "H",
                'mass': 1.0,
                "N_unitcell": 4,
                "cell": FaceCenteredCubic,
                "target_pot": LennardJones()
                },

    'lj_1.2_1.2': {
                'rdf_fn': '../data/LJ_data/rdf_rho1.2_T1.2_dt0.01.csv' ,
                'vacf_fn': '../data/LJ_data/vacf_rho1.2_T1.2_dt0.01.csv' ,
                'rho': 1.2,
                'T': 1.2, 
                'start': 0.75, 
                'end': 3.3,
                'element': "H",
                'mass': 1.0,
                "N_unitcell": 4,
                "cell": FaceCenteredCubic,
                "target_pot": LennardJones()
                },

    'lj_0.9_1.2': {
                'rdf_fn': '../data/LJ_data/rdf_rho0.9_T1.2_dt0.01.csv' ,
                'vacf_fn': '../data/LJ_data/vacf_rho0.9_T1.2_dt0.01.csv' ,
                'rho': 0.9,
                'T': 1.2, 
                'start': 0.75, 
                'end': 3.3,
                'element': "H",
                'mass': 1.0,
                "N_unitcell": 4,
                "cell": FaceCenteredCubic,
                "target_pot": LennardJones()
                },

    'lj_1.0_1.2': {
            'rdf_fn': '../data/LJ_data/rdf_rho1.0_T1.2_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho1.0_T1.2_dt0.01.csv' ,
            'rho': 1.0,
            'T': 1.2, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()

            },
    'lj_0.5_1.2': {
            'rdf_fn': '../data/LJ_data/rdf_rho0.5_T1.2_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho0.5_T1.2_dt0.01.csv' ,
            'rho': 0.5,
            'T': 1.2, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            }, 

    'lj_1.2_0.75': {
            'rdf_fn': '../data/LJ_data/rdf_rho1.2_T0.75_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho1.2_T0.75_dt0.01.csv' ,
            'rho': 1.2,
            'T': 0.75, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            },

    'lj_1.0_0.75': {
            'rdf_fn': '../data/LJ_data/rdf_rho1.0_T0.75_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho1.0_T0.75_dt0.01.csv' ,
            'rho': 1.0,
            'T': 0.75, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            },

    'lj_0.3_1.2': {
            'rdf_fn': '../data/LJ_data/rdf_rho0.3_T1.2_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho0.3_T1.2_dt0.01.csv' ,
            'rho': 0.3,
            'T': 1.2, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            },


    'lj_0.1_1.2': {
            'rdf_fn': '../data/LJ_data/rdf_rho0.1_T1.2_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho0.1_T1.2_dt0.01.csv' ,
            'rho': 0.1,
            'T': 1.2, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            },


    'lj_0.7_1.0': {
            'rdf_fn': '../data/LJ_data/rdf_rho0.7_T1.0_dt0.01.csv' ,
            'vacf_fn': '../data/LJ_data/vacf_rho0.7_T1.0_dt0.01.csv' ,
            'rho': 0.7,
            'T': 1.0, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            },

    'softsphere_0.7_1.0': {
            'rdf_fn': '../data/softsphere_data/rdf_rho0.7_T1.0_dt0.01.csv' ,
            'vacf_fn': '../data/softsphere_data/vacf_rho0.7_T1.0_dt0.01.csv' ,
            'rho': 0.7,
            'T': 1.0, 
            'start': 0.75, 
            'end': 3.3,
            'element': "H",
            'mass': 1.0,
            "N_unitcell": 4,
            "cell": FaceCenteredCubic,
            "target_pot": LennardJones()
            }, 

    'yukawa_0.7_1.0': {
        'rdf_fn': '../data/Yukawa_data/rdf_rho0.7_T1.0_dt0.01.csv' ,
        'vacf_fn': '../data/Yukawa_data/vacf_rho0.7_T1.0_dt0.01.csv' ,
        'rho': 0.7,
        'T': 1.0, 
        'start': 0.5, 
        'end': 3.0,
        'element': "H",
        'mass': 1.0,
        "N_unitcell": 4,
        "cell": FaceCenteredCubic
        }, 

    'yukawa_0.5_1.0': {
        'rdf_fn': '../data/Yukawa_data/rdf_rho0.5_T1.0_dt0.01.csv' ,
        'vacf_fn': '../data/Yukawa_data/vacf_rho0.5_T1.0_dt0.01.csv' ,
        'rho': 0.5,
        'T': 1.0, 
        'start': 0.5, 
        'end': 3.0,
        'element': "H",
        'mass': 1.0,
        "N_unitcell": 4,
        "cell": FaceCenteredCubic
        }, 

    'yukawa_0.3_1.0': {
        'rdf_fn': '../data/Yukawa_data/rdf_rho0.3_T1.0_dt0.01.csv' ,
        'vacf_fn': '../data/Yukawa_data/vacf_rho0.3_T1.0_dt0.01.csv' ,
        'rho': 0.3,
        'T': 1.0, 
        'start': 0.5, 
        'end': 3.0,
        'element': "H",
        'mass': 1.0,
        "N_unitcell": 4,
        "cell": FaceCenteredCubic
        }, 

    'overalp_0.9766_T0.07':
        {
        'rdf_fn': '../data/stripe_data/overalp_0.9766_k4.7896_V01000_0.07.csv' ,
        'rho': 0.9766,
        'T': 0.07,
        'dim': 2,  
        'start': 0.5, 
        'end': 7.5,
        'size': 25,
        'element': "H",
        'cufoff': 8.0,
        'ref': 'https://aip.scitation.org/doi/pdf/10.1063/5.0021475',
        'target_pot': SplineOverlap(K=4.7896, V0=1000, device="cpu")
        },

    'overalp_0.9766_T0.07_cut12':
        {
        'rdf_fn': '../data/stripe_data/overalp_0.9766_k4.7896_V01000_0.07_cutoff12.0.csv',
        'rho': 0.9766,
        'T': 0.07,
        'dim': 2,  
        'start': 0.6, 
        'end': 9.75,
        'size': 24,
        'element': "H",
        'cufoff': 12.0,
        'target_pot': SplineOverlap(K=4.7896, V0=1000, device="cpu")
        },

       "lj_rep_6_attr4_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_6_4_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_6_4_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=4, rep_pow=6) 
       },

       "lj_rep_8_attr4_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_8_4_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_8_4_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=4, rep_pow=8) 
       },
       "lj_rep_10_attr4_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_10_4_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_10_4_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=4, rep_pow=10) 
       },
       "lj_rep_12_attr4_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_12_4_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_12_4_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=4, rep_pow=12) 
       },
       "lj_rep_8_attr6_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_8_6_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_8_6_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=6, rep_pow=8) 
       },
       "lj_rep_10_attr6_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_10_6_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_10_6_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=6, rep_pow=10) 
       },
       "lj_rep_12_attr6_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_12_6_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_12_6_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=6, rep_pow=12) 
       },
       "lj_rep_10_attr8_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_10_8_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_10_8_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=8, rep_pow=10) 
       },
       "lj_rep_12_attr8_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_12_8_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_12_8_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=8, rep_pow=12) 
       },
       "lj_rep_12_attr10_rho0.5_T1.0_dt0.01": {
          "vacf_fn": "../data/LJfam_data/rdf_12_10_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/LJfam_data/vacf_12_10_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.75,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "cell": FaceCenteredCubic,
          "target_pot":  LJFamily(epsilon=1.0, sigma=1.0, attr_pow=10, rep_pow=12) 
       },

       "morse_a_12_phi1.0_rho0.3_T1.0_dt0.01": {
          "tag": "morse_a_12_phi1.0_rho0.3_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_12_1.0_rho0.3_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_12_1.0_rho0.3_T1.0_dt0.01.csv",
          "rho": 0.3,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=12, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_10_phi1.0_rho0.3_T1.0_dt0.01": {
          "tag": "morse_a_10_phi1.0_rho0.3_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_10_1.0_rho0.3_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_10_1.0_rho0.3_T1.0_dt0.01.csv",
          "rho": 0.3,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=10, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_8_phi1.0_rho0.3_T1.0_dt0.01": {
          "tag": "morse_a_8_phi1.0_rho0.3_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_8_1.0_rho0.3_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_8_1.0_rho0.3_T1.0_dt0.01.csv",
          "rho": 0.3,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=8, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_6_phi1.0_rho0.3_T1.0_dt0.01": {
          "tag": "morse_a_6_phi1.0_rho0.3_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_6_1.0_rho0.3_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_6_1.0_rho0.3_T1.0_dt0.01.csv",
          "rho": 0.3,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=6, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_12_phi1.0_rho0.5_T1.0_dt0.01": {
          "tag": "morse_a_12_phi1.0_rho0.5_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_12_1.0_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_12_1.0_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=12, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_10_phi1.0_rho0.5_T1.0_dt0.01": {
          "tag": "morse_a_10_phi1.0_rho0.5_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_10_1.0_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_10_1.0_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=10, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_8_phi1.0_rho0.5_T1.0_dt0.01": {
          "tag": "morse_a_8_phi1.0_rho0.5_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_8_1.0_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_8_1.0_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=8, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_6_phi1.0_rho0.5_T1.0_dt0.01": {
          "tag": "morse_a_6_phi1.0_rho0.5_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_6_1.0_rho0.5_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_6_1.0_rho0.5_T1.0_dt0.01.csv",
          "rho": 0.5,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=6, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_12_phi1.0_rho0.75_T1.0_dt0.01": {
          "tag": "morse_a_12_phi1.0_rho0.75_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_12_1.0_rho0.75_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_12_1.0_rho0.75_T1.0_dt0.01.csv",
          "rho": 0.75,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=12, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_10_phi1.0_rho0.75_T1.0_dt0.01": {
          "tag": "morse_a_10_phi1.0_rho0.75_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_10_1.0_rho0.75_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_10_1.0_rho0.75_T1.0_dt0.01.csv",
          "rho": 0.75,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=10, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_8_phi1.0_rho0.75_T1.0_dt0.01": {
          "tag": "morse_a_8_phi1.0_rho0.75_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_8_1.0_rho0.75_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_8_1.0_rho0.75_T1.0_dt0.01.csv",
          "rho": 0.75,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=8, phi=1),
          "cell": FaceCenteredCubic
       },
       "morse_a_6_phi1.0_rho0.75_T1.0_dt0.01": {
          "tag": "morse_a_6_phi1.0_rho0.75_T1.0_dt0.01",
          "vacf_fn": "../data/Morse_data/rdf_6_1.0_rho0.75_T1.0_dt0.01.csv",
          "rdf_fn": "../data/Morse_data/vacf_6_1.0_rho0.75_T1.0_dt0.01.csv",
          "rho": 0.75,
          "T": 1.0,
          "start": 0.6,
          "end": 3.3,
          "element": "H",
          "mass": 1.0,
          "N_unitcell": 4,
          "target_pot": ModifiedMorse(a=6, phi=1),
          "cell": FaceCenteredCubic
       },
          "morse_a_12_phi1.0_rho0.7_T1.0_dt0.01": {
      "tag": "morse_a_12_phi1.0_rho0.7_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_12_1.0_rho0.7_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_12_1.0_rho0.7_T1.0_dt0.01.csv",
      "rho": 0.7,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=12, phi=1)
   },
   "morse_a_10_phi1.0_rho0.7_T1.0_dt0.01": {
      "tag": "morse_a_10_phi1.0_rho0.7_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_10_1.0_rho0.7_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_10_1.0_rho0.7_T1.0_dt0.01.csv",
      "rho": 0.7,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=10, phi=1)
   },
   "morse_a_8_phi1.0_rho0.7_T1.0_dt0.01": {
      "tag": "morse_a_8_phi1.0_rho0.7_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_8_1.0_rho0.7_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_8_1.0_rho0.7_T1.0_dt0.01.csv",
      "rho": 0.7,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=8, phi=1)
   },
   "morse_a_6_phi1.0_rho0.7_T1.0_dt0.01": {
      "tag": "morse_a_6_phi1.0_rho0.7_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_6_1.0_rho0.7_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_6_1.0_rho0.7_T1.0_dt0.01.csv",
      "rho": 0.7,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6, phi=1)
   },
   "morse_a_12_phi1.0_rho0.9_T1.0_dt0.01": {
      "tag": "morse_a_12_phi1.0_rho0.9_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_12_1.0_rho0.9_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_12_1.0_rho0.9_T1.0_dt0.01.csv",
      "rho": 0.9,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=12, phi=1)
   },
   "morse_a_10_phi1.0_rho0.9_T1.0_dt0.01": {
      "tag": "morse_a_10_phi1.0_rho0.9_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_10_1.0_rho0.9_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_10_1.0_rho0.9_T1.0_dt0.01.csv",
      "rho": 0.9,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=10, phi=1)
   },
   "morse_a_8_phi1.0_rho0.9_T1.0_dt0.01": {
      "tag": "morse_a_8_phi1.0_rho0.9_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_8_1.0_rho0.9_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_8_1.0_rho0.9_T1.0_dt0.01.csv",
      "rho": 0.9,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=8, phi=1)
   },
   "morse_a_6_phi1.0_rho0.9_T1.0_dt0.01": {
      "tag": "morse_a_6_phi1.0_rho0.9_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_6_1.0_rho0.9_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_6_1.0_rho0.9_T1.0_dt0.01.csv",
      "rho": 0.9,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6, phi=1)
   },
   "morse_a_12_phi1.0_rho0.2_T1.0_dt0.01": {
      "tag": "morse_a_12_phi1.0_rho0.2_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_12_1.0_rho0.2_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_12_1.0_rho0.2_T1.0_dt0.01.csv",
      "rho": 0.2,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=12, phi=1)
   },
   "morse_a_10_phi1.0_rho0.2_T1.0_dt0.01": {
      "tag": "morse_a_10_phi1.0_rho0.2_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_10_1.0_rho0.2_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_10_1.0_rho0.2_T1.0_dt0.01.csv",
      "rho": 0.2,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=10, phi=1)
   },
   "morse_a_8_phi1.0_rho0.2_T1.0_dt0.01": {
      "tag": "morse_a_8_phi1.0_rho0.2_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_8_1.0_rho0.2_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_8_1.0_rho0.2_T1.0_dt0.01.csv",
      "rho": 0.2,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=8, phi=1)
   },
   "morse_a_6_phi1.0_rho0.2_T1.0_dt0.01": {
      "tag": "morse_a_6_phi1.0_rho0.2_T1.0_dt0.01",
      "vacf_fn": "../data/Morse_data/rdf_6_1.0_rho0.2_T1.0_dt0.01.csv",
      "rdf_fn": "../data/Morse_data/vacf_6_1.0_rho0.2_T1.0_dt0.01.csv",
      "rho": 0.2,
      "T": 1.0,
      "start": 0.6,
      "end": 3.3,
      "element": "H",
      "mass": 1.0,
      "N_unitcell": 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6, phi=1)
   },
   'morse_a_8.0_phi3.06_rho0.3_T1.0_dt0.01': 
      {'tag': 'morse_a_8.0_phi3.06_rho0.3_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_8.0_3.06_rho0.3_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_8.0_3.06_rho0.3_T1.0_dt0.01.csv',
      'rho': 0.3,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=8.0, phi=3.06)},
     'morse_a_7.5_phi2.33_rho0.3_T1.0_dt0.01':
      {'tag': 'morse_a_7.5_phi2.33_rho0.3_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_7.5_2.33_rho0.3_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_7.5_2.33_rho0.3_T1.0_dt0.01.csv',
      'rho': 0.3,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=7.5, phi=2.33)},
     'morse_a_7.0_phi1.58_rho0.3_T1.0_dt0.01': 
     {'tag': 'morse_a_7.0_phi1.58_rho0.3_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_7.0_1.58_rho0.3_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_7.0_1.58_rho0.3_T1.0_dt0.01.csv',
      'rho': 0.3,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=7.0, phi=1.58)},
     'morse_a_6.5_phi0.81_rho0.3_T1.0_dt0.01': 
     {'tag': 'morse_a_6.5_phi0.81_rho0.3_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_6.5_0.81_rho0.3_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_6.5_0.81_rho0.3_T1.0_dt0.01.csv',
      'rho': 0.3,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6.5, phi=0.81)},
     'morse_a_5.5_phi-0.84_rho0.3_T1.0_dt0.01': 
     {'tag': 'morse_a_5.5_phi-0.84_rho0.3_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_5.5_-0.84_rho0.3_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_5.5_-0.84_rho0.3_T1.0_dt0.01.csv',
      'rho': 0.3,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=5.5, phi=-0.84)},
     'morse_a_8.0_phi3.06_rho0.5_T1.0_dt0.01': 
     {'tag': 'morse_a_8.0_phi3.06_rho0.5_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_8.0_3.06_rho0.5_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_8.0_3.06_rho0.5_T1.0_dt0.01.csv',
      'rho': 0.5,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=8.0, phi=3.06)},
     'morse_a_7.5_phi2.33_rho0.5_T1.0_dt0.01': 
     {'tag': 'morse_a_7.5_phi2.33_rho0.5_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_7.5_2.33_rho0.5_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_7.5_2.33_rho0.5_T1.0_dt0.01.csv',
      'rho': 0.5,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=7.5, phi=2.33)},
     'morse_a_7.0_phi1.58_rho0.5_T1.0_dt0.01': 
     {'tag': 'morse_a_7.0_phi1.58_rho0.5_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_7.0_1.58_rho0.5_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_7.0_1.58_rho0.5_T1.0_dt0.01.csv',
      'rho': 0.5,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=7.0, phi=1.58)},
     'morse_a_6.5_phi0.81_rho0.5_T1.0_dt0.01': 
     {'tag': 'morse_a_6.5_phi0.81_rho0.5_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_6.5_0.81_rho0.5_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_6.5_0.81_rho0.5_T1.0_dt0.01.csv',
      'rho': 0.5,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6.5, phi=0.81)},
     'morse_a_5.5_phi-0.84_rho0.5_T1.0_dt0.01': 
     {'tag': 'morse_a_5.5_phi-0.84_rho0.5_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_5.5_-0.84_rho0.5_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_5.5_-0.84_rho0.5_T1.0_dt0.01.csv',
      'rho': 0.5,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=5.5, phi=-0.84)},
     'morse_a_8.0_phi3.06_rho0.7_T1.0_dt0.01': 
     {'tag': 'morse_a_8.0_phi3.06_rho0.7_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_8.0_3.06_rho0.7_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_8.0_3.06_rho0.7_T1.0_dt0.01.csv',
      'rho': 0.7,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=8.0, phi=3.06)},
     'morse_a_7.5_phi2.33_rho0.7_T1.0_dt0.01': 
     {'tag': 'morse_a_7.5_phi2.33_rho0.7_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_7.5_2.33_rho0.7_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_7.5_2.33_rho0.7_T1.0_dt0.01.csv',
      'rho': 0.7,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=7.5, phi=2.33)},
     'morse_a_7.0_phi1.58_rho0.7_T1.0_dt0.01': 
     {'tag': 'morse_a_7.0_phi1.58_rho0.7_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_7.0_1.58_rho0.7_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_7.0_1.58_rho0.7_T1.0_dt0.01.csv',
      'rho': 0.7,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=7.0, phi=1.58)},
     'morse_a_6.5_phi0.81_rho0.7_T1.0_dt0.01':
      {'tag': 'morse_a_6.5_phi0.81_rho0.7_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_6.5_0.81_rho0.7_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_6.5_0.81_rho0.7_T1.0_dt0.01.csv',
      'rho': 0.7,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6.5, phi=0.81)},
     'morse_a_5.5_phi-0.84_rho0.7_T1.0_dt0.01':
      {'tag': 'morse_a_5.5_phi-0.84_rho0.7_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_5.5_-0.84_rho0.7_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_5.5_-0.84_rho0.7_T1.0_dt0.01.csv',
      'rho': 0.7,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=5.5, phi=-0.84)},

    'morse_a_10.0_phi5.5_rho0.3_T1.0_dt0.01': 
    {'tag': 'morse_a_10.0_phi5.5_rho0.3_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_10.0_5.5_rho0.3_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_10.0_5.5_rho0.3_T1.0_dt0.01.csv',
      'rho': 0.3,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=10.0, phi=5.5)},

     'morse_a_10.0_phi5.5_rho0.5_T1.0_dt0.01': 
     {'tag': 'morse_a_10.0_phi5.5_rho0.5_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_10.0_5.5_rho0.5_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_10.0_5.5_rho0.5_T1.0_dt0.01.csv',
      'rho': 0.5,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4, 
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=10.0, phi=5.5)},

     'morse_a_10.0_phi5.5_rho0.7_T1.0_dt0.01': 
     {'tag': 'morse_a_10.0_phi5.5_rho0.7_T1.0_dt0.01',
      'vacf_fn': '../data/Morse_data/rdf_10.0_5.5_rho0.7_T1.0_dt0.01.csv',
      'rdf_fn': '../data/Morse_data/vacf_10.0_5.5_rho0.7_T1.0_dt0.01.csv',
      'rho': 0.7,
      'T': 1.0,
      'start': 0.6,
      'end': 3.3,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=10.0, phi=5.5)},

      # soft to hard potential 
      # Morse(4.0, 2.2) : 1.0, 0.7, 0.5, 0.3
     'morse_a_4.0_phi2.2_rho1.0_T1.0': 
     {'tag': 'morse_a_4.0_phi2.2_rho1.0_T1.0',
      'dt': 0.005,
      'rho': 1.0,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=4.0, phi=2.2)},

    'morse_a_4.0_phi2.2_rho0.7_T1.0': 
     {'tag': 'morse_a_4.0_phi2.2_rho0.7_T1.0',
      'dt': 0.005,
      'rho': 0.7,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=4.0, phi=2.2)},

    'morse_a_4.0_phi2.2_rho0.5_T1.0': 
     {'tag': 'morse_a_4.0_phi2.2_rho0.5_T1.0',
      'dt': 0.005,
      'rho': 0.5,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=4.0, phi=2.2)},

    'morse_a_4.0_phi2.2_rho0.3_T1.0': 
     {'tag': 'morse_a_4.0_phi2.2_rho0.3_T1.0',
      'dt': 0.005,
      'rho': 0.3,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=4.0, phi=2.2)},


     'morse_a_6.5_phi-0.45_rho1.0_T1.0': 
     {'tag': 'morse_a_6.5_phi-0.45_rho1.0_T1.0',
      'dt': 0.005,
      'rho': 1.0,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6.5, phi=-0.45)},


           'morse_a_6.5_phi-0.45_rho0.7_T1.0': 
     {'tag': 'morse_a_6.5_phi-0.45_rho0.7_T1.0',
      'dt': 0.005,
      'rho': 0.7,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6.5, phi=-0.45)},

           'morse_a_6.5_phi-0.45_rho0.5_T1.0': 
     {'tag': 'morse_a_6.5_phi-0.45_rho0.5_T1.0',
      'dt': 0.005,
      'rho': 0.5,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6.5, phi=-0.45)},


    'morse_a_6.5_phi-0.45_rho0.3_T1.0': 
     {'tag': 'morse_a_6.5_phi-0.45_rho0.3_T1.0',
      'dt': 0.005,
      'rho': 0.3,
      'T': 1.0,
      'start': 0.5,
      'end': 3.0,
      'element': 'H',
      'mass': 1.0,
      'N_unitcell': 4,
      "cell": FaceCenteredCubic,
      "target_pot": ModifiedMorse(a=6.5, phi=-0.45)}

    }

exp_rdf_data_dict = {
    'Si_2.293_100K': { 'fn': '../data/a-Si/100K_2.293.csv' ,
                       'rho': 2.293,
                        'T': 100.0, 
                        'start': 1.8, 
                        'end': 7.9,
                        'element': "H",
                        'mass': 28.0855,
                        "N_unitcell": 8,
                        "cell": Diamond
                        },
                        
    'Si_2.287_83K': { 'fn': '../data/a-Si/83K_2.287_exp.csv' ,
                       'rho': 2.287,
                        'T': 83.0, 
                        'start': 1.8, 
                        'end': 10.0,
                        'element': "H",
                        'mass': 28.0855,
                        "N_unitcell": 8,
                        "cell": Diamond
                        },

    'Si_2.327_102K_cry': { 'fn': '../data/a-Si/102K_2.327_exp.csv' ,
                       'rho': 2.3267,
                        'T': 102.0, 
                        'start': 1.8, 
                        'end': 8.0,
                        'element': "H",
                        'mass': 28.0855,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        'anneal_flag': True
                        },

    'H20_0.997_298K': { 'fn': "../data/water_exp/water_exp_pccp.csv",
                        'rho': 0.997,
                        'T': 298.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond, #FaceCenteredCubic
                        "pressure": 1.0 # MPa
                        },

    'H20_0.978_342K': { 'fn': "../data/water_exp/water_exp_skinner_342K_0.978.csv",
                       'rho': 0.978,
                        'T': 342.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1,  #MPa
                        "ref": "https://doi.org/10.1063/1.4902412"
                        },

    'H20_0.921_423K_soper': { 'fn': "../data/water_exp/water_exp_Soper_423K_0.9213.csv",
                       'rho': 0.9213,
                        'T': 423.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 10.0, # MPa
                        "ref": "https://doi.org/10.1016/S0301-0104(00)00179-8"
                        },

    'H20_0.999_423K_soper': { 'fn': "../data/water_exp/water_exp_Soper_423K_0.999.csv",
                       'rho': 0.999,
                        'T': 423.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 190, 
                        "ref": "https://doi.org/10.1016/S0301-0104(00)00179-8"
                        },

    'H20_298K_redd': { 'fn': "../data/water_exp/water_exp_298K_redd.csv",
                       'rho': 0.99749,
                        'T': 298.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H20_308K_redd': { 'fn': "../data/water_exp/water_exp_308K_redd.csv",
                       'rho': 0.99448,
                        'T': 308.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H20_338K_redd': { 'fn': "../data/water_exp/water_exp_338K_redd.csv",
                       'rho': 0.98103,
                        'T': 338.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H20_368K_redd': { 'fn': "../data/water_exp/water_exp_368K_redd.csv",
                       'rho': 0.96241,
                        'T': 368.0, 
                        'start': 1.8, 
                        'end': 7.5,
                        'element': "H" ,
                        'mass': 18.01528,
                        "N_unitcell": 8,
                        "cell": Diamond,
                        "pressure": 1, 
                        "ref": "https://aip.scitation.org/doi/pdf/10.1063/1.4967719"
                        },

    'H2O_long_correlation' : {
                        'ref': 'https://aip.scitation.org/doi/pdf/10.1063/1.4961404'
    },

    'H2O_soper': {
                        'ref': 'https://doi.org/10.1016/S0301-0104(00)00179-8'
    },

    'Argon_1.417_298k': { 'fn': "../data/argon_exp/argon_exp.csv",
                       'rho': 1.417,
                        'T': 298.0, 
                        'start': 2.0, 
                        'end': 9.0,
                        'element': "H",
                        'mass': 39.948,
                        "N_unitcell": 4,
                        "cell": FaceCenteredCubic
                        }
}


angle_data_dict = {
   "water":
        {
        2.7: '../data/water_angle_deepcg_2.7.csv',
        3.7: '../data/water_angle_deepcg_3.7.csv', 
        }
}
