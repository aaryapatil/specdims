__author__      = "Aarya Patil"
__copyright__   = "Copyright 2020"

"""
This module generates MCMC samples for hierarchical modeling of abundances
in M67. The functional form used for modeling is the generalized t-distribution,
but any other form can be used -- modify ``fun_alpha`` and ``log_prior``
to get the desired distribution.

Reference: Inferring the eccentricity distribution by David Hogg,
Adam Myers and Jo Bovy, ApJ, 2010
"""

import numpy as np
import emcee
from tqdm import tqdm
from scipy.stats import uniform, t
from scipy.optimize import minimize


def fun_alpha(X_H, alpha):
    '''
    PDF of [X/H] samples given that the functional form of
    M67 abundance distributions is the generalized t-distribution.

    Parameters
    ----------
    X_H : array_like
          Samples of abundance [X/H]
    alpha : array_like
            degrees of freedom, location, scale parameters
            of t-distribution
    '''
    return t.pdf(X_H, alpha[0], loc=alpha[1], scale=alpha[2])


def log_prior(alpha):
    '''
    Log prior on parameters of hierarchical generalized t-distribution.

    Parameters
    ----------
    alpha : array_like
            degrees of freedom, location, scale parameters
            of t-distribution
    '''
    if 1 < alpha[0] < 10 and -0.5 < alpha[1] < 0.5 and 0 < alpha[2] < 2:
        return 0
    return -np.inf


def uninformative_prior(X_H):
    '''
    Uninformative prior used to get posterior samples
    for each star - uniform distribution from [-1, 1]
    or [-0.5, 0.5] in the form of [loc, loc+scale].

    Parameters
    ----------
    X_H : array_like
          Samples of abundance [X/H]
    '''
    if ele == 3 or ele == 16:
        loc = -1
        scale = 2
    else:
        loc = -0.5
        scale = 1
    return uniform.pdf(X_H, loc=-1, scale=2)


def log_likelihood(alpha):
    '''
    Log likelihood of hierarchical generalized t-distribution.
    Here samples represent the [X/H]_nk posterior samples where
    n = 1,...,28 and k=1,...,100,000.

    Parameters
    ----------
    alpha : array_like
            degrees of freedom, location, scale parameters
            of t-distribution
    '''
    # Total posterior samples
    K = samples.shape[1]
    # Can be thought of as an importance-sampling approximation
    ratio = fun_alpha(samples, alpha)/uninformative_prior(samples)
    sum_of_ratios = 1/K*np.sum(ratio, axis=1)
    log_sum_of_ratios = np.log(sum_of_ratios)
    return np.sum(log_sum_of_ratios)


def log_posterior(alpha):
    '''
    Log posterior of hierarchical generalized t-distribution.

    Parameters
    ----------
    alpha : array_like
            degrees of freedom, location, scale parameters
            of t-distribution
    '''
    # Prior
    lp = log_prior(alpha)
    if not np.isfinite(lp):
        return -np.inf
    # Posterior =  Prior * Likelihood
    return lp + log_likelihood(alpha)


M67_constrain = np.zeros(shape=(28, 100000, 17))

for ind in np.arange(28):
    M67_constrain[ind] = np.loadtxt(f'data/SNL/posteriors_M67_{ind}.dat')

# Stellar parameters and abundances
labels = ['Teff', 'logg', 'C', 'N', 'O', 'Na', 'Mg', 'Al',
          'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Mn', 'Ni', 'Fe']

for ele in np.arange(2, 17):
    # Posterior samples of element
    samples = M67_constrain[:, :, ele]

    # Set seed for reproducibility
    np.random.seed(42)
    nll = lambda *args: -log_posterior(*args)

    # Initial guess for MCMC is optimized
    initial = [1.5, 0.0, 0.03]
    soln = minimize(nll, initial)

    # Generate values around the optimized start for different walkers
    pos = soln.x + 0.05*np.random.randn(6, 3)
    # df and std must be positive
    pos[:, 0] = np.abs(pos[:, 0])
    pos[:, 2] = np.abs(pos[:, 2])
    nwalkers, ndim = pos.shape

    # Sample using MCMC
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, threads=6)

    # Show progress
    for pos, lnp, rstate in tqdm(sampler.sample(pos, iterations=750)):
        pass

    samples = sampler.chain
    flat_samples = samples[:, :, :].reshape(-1, samples.shape[-1])

    np.savetxt('data/hierarch/{}_H_hierarch_t.dat'.format(labels[ele]), flat_samples)
