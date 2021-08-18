# Import dependencies
import numpy as np

from . import psm


def sim_spectrum(theta, vturb=False):
    """
    Simulates a spectrum using PSM given parameters.
        
    Parameters
    -----------
    theta : array_like
            Stellar parameters and abundances for simulation.
        
    Returns
    --------
    synspec: ndarray
             Simulated (synthetic) spectrum.
    """
    # Spectrum: Vary Carbon, Magnesium, Aluminium, Silicon, Manganese, Iron abundances
    # LABEL ORDER Teff [1000K], logg, vturb [km/s] (micro), ch, nh, oh, nah, mgh, alh, sih, sh, kh, cah, 
    # tih, vh, mnh, nih, feh, log10(c12c13).
    if vturb:
        synspec = psm.generate_spectrum(Teff=theta[0]/1000, logg=theta[1], ch=theta[2], nh=theta[3],
                                        oh=theta[4], nah=theta[5], mgh=theta[6], alh=theta[7], sih=theta[8],
                                        sh=theta[9], kh=theta[10], cah=theta[11], tih=theta[12], vh=theta[13],
                                        mnh=theta[14], nih=theta[15], feh=theta[16], vturb=theta[17])
    else:
        synspec = psm.generate_spectrum(Teff=theta[0]/1000, logg=theta[1], ch=theta[2], nh=theta[3],
                                        oh=theta[4], nah=theta[5], mgh=theta[6], alh=theta[7], sih=theta[8],
                                        sh=theta[9], kh=theta[10], cah=theta[11], tih=theta[12], vh=theta[13],
                                        mnh=theta[14], nih=theta[15], feh=theta[16])
    return synspec


def sim_spectra(apogee_cat_cut, k=None, seed=0, vturb=False):
    """
    Simulate k spectra given a catalog of spectroscopic parameters.
    These can be used as basis functions for FPCA of spectral data
    or to simulate theoretical spectra for spectroscopic inference.
    """

    nstars = apogee_cat_cut.shape[0]
    pixels = 7214
    
    if k is None:
        k = nstars
    else:
        # Read K random APOGEE spectra from a selected catalogue.
        np.random.seed(seed)
        rand_ind = np.random.random_integers(0, nstars-1, k)
        apogee_cat_cut = apogee_cat_cut[rand_ind]

    # Simulate spectra using the catalogue parameters
    phi_t = np.zeros((k, pixels))
    
    if vturb:
        theta = np.zeros((k, 18))
        theta[:, 17] = apogee_cat_cut['VMICRO']
    else:
        theta = np.zeros((k, 17))

    theta[:, 0] = apogee_cat_cut['TEFF']
    theta[:, 1] = apogee_cat_cut['LOGG']

    # 15 elements: ch, nh, oh, nah, mgh, alh, sih, sh, kh, cah, tih, vh, mnh, nih, feh
    theta[:, 16] = apogee_cat_cut['FE_H']
    abundances = ['C_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE', 'SI_FE', 'S_FE',
                  'K_FE', 'CA_FE', 'TI_FE', 'V_FE', 'MN_FE', 'NI_FE']
    for i in range(2, 16):
        # [X/H] = [X/Fe] + [Fe/H]
        theta[:, i] = apogee_cat_cut[abundances[i-2]] + theta[:, 16]

    for ind in range(k):
        phi_t[ind] = sim_spectrum(theta[ind])

    return phi_t