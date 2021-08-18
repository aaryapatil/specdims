__author__      = "Aarya Patil"
__copyright__   = "Copyright 2021"

"""
Author: Aarya Patil, 2021

Perform Sequential Neural Likelihood with Functional Principal
Component Analysis on an M67 Giant Member spectrum.

Reference: Sequential Neural Likelihood: Fast Likelihood-free Inference with
           Autoregressive Flows by George Papamakarios, David Sterratt, and
           Iain Murray, PMLR, 2019
           Fast likelihood-free cosmology with neural density estimators and
           active learning by Justin Alsing, Tom Charnock, Stephen Feeney, and
           Benjamin Wandelt, MNRAS, 2019
"""

import os
import warnings

import numpy as np
import matplotlib as mpl
import empca
import tensorflow as tf
import pydelfi.priors as priors
import pydelfi.ndes as ndes
import pydelfi.delfi as delfi
from astropy.io import fits
from apogee.tools import bitmask as bm
from apogee.tools.path import change_dr

from delfiSpec import util, specproc, psm


# Matplotlib setting for no plotting on server
mpl.use('Agg')
# Ignore warnings
warnings.filterwarnings("ignore")


class SytheticSpectra():

    def __init__(self, star=None, badcombpixmask=bm.badpixmask(), turbospectrum=False,
                 fPC=None, spec_mean=0):
        '''
        This class defines the setup required for modeling stellar spectra.
        Given APOGEE data of a star, it preprocesses the observed spectrum ``d_obs``
        and generates its mask ``m_obs`` and uncertainties ``e_obs``. It also provides
        a ``Simulator`` that forward models spectra given stellar parameters and abundances.
        
        Parameters
        ----------
        star: array_like
              APOGEE data of a star
        
        turbospectrum: bool(scalar), optional
                       If True, Turbospectrum will be used in the ``Simulator``.
                       Defaults to False, in which case, PSM will be used
        
        fPC: array_like
             If not None, functional principal components are provided.
             Defaults to None, in which case, no dimensional reduction is performed
        '''
        
        # Read ASPCAP stellar parameters for given star
        self.teff = star['TEFF']
        self.logg = star['LOGG']
        
        # Read APOGEE spectrum
        spec_info = apCat.read_allStar_spectra(apogee_cat_cut=[star])
        
        # Preprocess spectrum and obtain uncertainties and weights(mask)
        self.spec, self.spec_err, self.spec_weight = specproc.process_spectra(
            spectra_info=spec_info, badcombpixmask=badcombpixmask)
        self.spec_mask = np.ma.masked_array(self.spec, mask=(self.spec_weight==0))

        # Errors for masked values are constrained to 3 sigma, because ASPCAP significantly
        # bumps up errors on masked values -- this can be ignored because FPCA decomposition
        # handles this using an EM step.
        self.spec_err[self.spec_weight==0] = np.median(self.spec_err)*3
        
        # Define covariance/error matrix
        self.C = np.diag(self.spec_err**2)
        self.L = np.linalg.cholesky(self.C)

        # Functional Principal Component Analysis (FPCA)
        if fPC is not None:
            # Subtract mean from the spectrum
            spec_res = self.spec - spec_mean
            
            # Model the spectrum using FPCs with an EM step
            empca_fit = empca.Model(fPC.T, spec_res[np.newaxis, :],
                                    self.spec_weight[np.newaxis, :])
        
            # Coefficients of model represent dimensionally reduced data
            self.reduced_spec = np.squeeze(empca_fit.coeff)
            
        # Set the Simulator
        if turbospectrum:
            self.turbospectrum = True
            
            # Local import turbospectrum package which requires DR12
            path.change_dr('12')
            import apogee.modelspec.turbospec
            from apogee.modelatm import atlas9
            
            # Generate model atmosphere using stellar parameters of given star
            self.atm = atlas9.Atlas9Atmosphere(teff=self.teff, logg=self.logg, metals=star['METALS'],
                                               am=star['ALPHA_M'], cm=star['X_M'][0])
            
            # Default back to DR14
            change_dr('14')
        else:
            self.turbospectrum = False


    def simulate(self, theta, seed):
        '''
        Simulate or forward model spectra. This ``Simulator`` generates
        a spectrum using stellar parameters and abundances as input, and
        adds noise to mimic APOGEE data.
        '''

        # Random seed
        if seed:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random.RandomState()

        # Simulate using PSM
        if not self.turbospectrum:
            self.npar = 17
            # Spectrum
            synspec = psm.generate_spectrum(Teff=theta[0]/1000, logg=theta[1], ch=theta[2], nh=theta[3],
                                            oh=theta[4], nah=theta[5], mgh=theta[6], alh=theta[7], sih=theta[8],
                                            sh=theta[9], kh=theta[10], cah=theta[11], tih=theta[12], vh=theta[13],
                                            mnh=theta[14], nih=theta[15], feh=theta[16])
            # Noise
            noise = np.dot(self.L, rng.normal(0, 1, len(self.L)))
            return synspec + noise
        else:
            change_dr('12')
            self.npar = 15
            # Atomic numbers 6: Carbon, 7: Nitrogen, 8: Oxygen, 11: Sodium, 12: Magnesium,
            # 13: Aluminium, 14: Silicon, 16: Sulphur, 19: Potassium, 20: Calcium,
            # 22: Titanium, 23: Vanadium, 25: Manganese, 28: Nickel and 26: Iron
            synspec = apogee.modelspec.turbospec.synth([6, theta[0]], [7, theta[1]], [8, theta[2]], [11, theta[3]],
                                                       [12, theta[4]], [13, theta[5]], [14, theta[6]], [16, theta[7]],
                                                       [19, theta[8]], [20, theta[9]], [22, theta[10]], [23, theta[11]],
                                                       [25, theta[12]], [28, theta[13]], [26, theta[14]],
                                                       modelatm=self.atm, linelist='201404080919',
                                                       lsf='all', cont='cannon', vmacro=6.,
                                                       isotopes='solar')
            sim_spec = util.get_DR_slice(synspec[0])
            # Noise
            noise = np.dot(self.L, np.random.normal(0, 1, len(self.L)))
            change_dr('14')
            return sim_spec + noise


# Read OCCAM M67 cluster data from APOGEE DR 14
change_dr('14')
apCat = util.ApogeeCat()
M67_apogee_DM, M67_apogee_GM = apCat.read_OCCAM_cluster()

# Read FPCs and spectrum mean
psi_cap_t = np.loadtxt('data/FPCA_apogee/fpca_dat_eigenvectors_psi_t.dat')
spec_mean = np.loadtxt('data/FPCA_apogee/fpca_dat_spec_mean.dat')

# Define the bitmask to be used
badcombpixmask = bm.badpixmask()
pix_err = np.array([bm.apogee_pixmask_int("SIG_SKYLINE"), bm.apogee_pixmask_int("SIG_TELLURIC"),
                    bm.apogee_pixmask_int("PERSIST_HIGH"), bm.apogee_pixmask_int("PERSIST_MED"),
                    bm.apogee_pixmask_int("PERSIST_LOW")])
badcombpixmask += np.sum(2**pix_err)

# Index of M67 star
star_ind = 16

# Setup for modeling
SpectraSimulator = SytheticSpectra(star=M67_apogee_GM[star_ind],
                                   badcombpixmask = badcombpixmask,
                                   fPC=psi_cap_t, spec_mean=spec_mean)


# ------------- Simulator -------------
def simulator(theta, seed, simulator_args, batch=1):
    '''
    Simulator function.
    
    Parameters
    ----------
    theta: array_like
           Parameters for the simulator
        
    seed: float
          Seed for random generator
              
    simulator_args: array_like
                    Additional arguments
    
    Returns
    -------
    Simulated data vector
    '''
    return SpectraSimulator.simulate(theta, seed)


# ------- Dimensional Reduction -------
def compressor(d, compressor_args):
    '''
    Simulator function.
    
    Parameters
    ----------
    d: array_like
       Data vector
              
    compressor_args: array_like
                     Additional arguments
    
    Returns
    -------
    Compressed/dimensionally reduced data vector
    '''    
    # Subtract mean from the spectrum
    d_res = d - spec_mean
        
    # Model the spectrum using FPCs with an EM step
    empca_fit = empca.Model(psi_cap_t.T, d_res[np.newaxis, :],
                            SpectraSimulator.spec_weight[np.newaxis, :])
        
    # Coefficients of model represent dimensionally reduced data
    return np.squeeze(empca_fit.coeff)


# Set additional arguments (optional)
simulator_args = None
compressor_args = None

# Test simulator
spec = simulator([SpectraSimulator.teff, SpectraSimulator.logg, 0, 0, 0,
                  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 0, simulator_args)

# Generate compressed/dimensionally reduced data
compressed_data = SpectraSimulator.reduced_spec


# ------------- Prior -------------------
# Hard prior boundaries on the parameters
lower = np.array([4000, 0, -0.5, -1, -0.5, -0.5, -0.5,
                  -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5,
                  -0.5, -0.5, -1])
upper = np.array([5500, 4, 0.5, 1, 0.5, 0.5, 0.5,
                  0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
                  0.5, 0.5, 1])
# Uniform prior distribution using the above boundaries
prior = priors.Uniform(lower, upper)

# Reset tensorflow graph
tf.reset_default_graph()


# ------ Neural Density Estimators ------
# High no. of MADEs in MAF accounts for more flexibility, but also adds
# multi-modalities in the fits. Therefore, we tune the NDEs for each star
# by adding/removing MDNs or MADEs
NDEs = [ndes.ConditionalMaskedAutoregressiveFlow(n_parameters=SpectraSimulator.npar,
                                                 n_data=compressed_data.shape[0],
                                                 n_hiddens=[10, 10], n_mades=1,
                                                 act_fun=tf.tanh, index=0),
        ndes.MixtureDensityNetwork(n_parameters=SpectraSimulator.npar,
                                   n_data=compressed_data.shape[0],
                                   n_components=1, n_hidden=[15, 15],
                                   activations=[tf.tanh, tf.tanh], index=1),
        ndes.MixtureDensityNetwork(n_parameters=SpectraSimulator.npar,
                                   n_data=compressed_data.shape[0],
                                   n_components=3, n_hidden=[15, 15],
                                   activations=[tf.tanh, tf.tanh], index=2)]
    
# Make directory to store posteriors
os.mkdir('GM_{}'.format(star_ind))
# Write the NDE graph for storing NDE state
writer = tf.summary.FileWriter('GM_{}/graphs'.format(star_ind), tf.get_default_graph())


# ---- Sequential Neural Likelihood ----
DelfiEnsemble = delfi.Delfi(compressed_data, prior, NDEs,
                            param_limits = [lower, upper],
                            param_names = [r'T_\mathrm{eff}', r'\log g', r'[\mathrm{C/H}]',
                                           r'[\mathrm{N/H}]', r'[\mathrm{O/H}]', r'[\mathrm{Na/H}]',
                                           r'[\mathrm{Mg/H}]', r'[\mathrm{Al/H}]', r'[\mathrm{Si/H}]',
                                           r'[\mathrm{S/H}]', r'[\mathrm{K/H}]', r'[\mathrm{Ca/H}]',
                                           r'[\mathrm{Ti/H}]', r'[\mathrm{V/H}]', r'[\mathrm{Mn/H}]]',
                                           r'[\mathrm{Ni/H}]', r'[\mathrm{Fe/H}]'],
                            results_dir = 'GM_{}/'.format(star_ind),
                            show_plot=False)

# Number of simulations to be run in each round of SNL
n_initial = 600
n_batch = 300
n_populations = 8

# Patience for early stopping
DelfiEnsemble.sequential_training(simulator, compressor, n_initial,
                                  n_batch, n_populations, patience=10,
                                  save_intermediate_posteriors=True)