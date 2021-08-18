__author__      = "Aarya Patil"
__copyright__   = "Copyright 2020"

"""
This module provides a Functional Principal Component Analysis (FPCA) class
to obtain functional PCs of any given data. It also includes Legendre Polynomials
as default orthonormal basis functions; the user can choose to use any other basis
function set depending on their problem.

Reference: Functional Data Analysis by Bernard Silverman and James O. Ramsay, 1997
Inspiration: FPCA code by Sebastian Jaimungal at 
https://gist.github.com/sebjai/5f8376dee07af3e5efe1285c98bc50c6#file-sta4505-fpca-ipynb
"""

import multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import linalg
from tqdm import tqdm


def GenerateLegendrePolynomials(t, Kmax, makeplot=0):
    '''
    Evaluate Legendre Polynomials up to order Kmax at the
    sampling points t. These are used as default orthonormal
    basis functions.
    
    Parameters
    ----------
    t        : Points at which the polynomials are evaluated
    Kmax     : Maximum order of the polynomials to be generated
    makeplot : If True, plot the polynomials
    
    Returns
    -------
    lpoly    : Legendre Polynomials
    '''
    
    lpoly = np.zeros((Kmax, t.size))
    
    for k in range(0, Kmax):
        c = np.zeros(k+1)
        # Generate Legendre Polynomial of order k
        c[k] = 1
        lpoly[k,:] = np.polynomial.legendre.legval(t, c)*np.sqrt(k+1/2)

        # Plot the ploynomials
        if makeplot:
            plt.subplot(1, Kmax, k+1)
            plt.plot(t, LP[k,:])
            plt.title(r'$L_' + str(k) +'(t)$')

    if makeplot:
        plt.tight_layout()
        plt.show()
    
    return lpoly


class FPCA:
    """
    Functional Principal Component Analysis of noisy observations.
    """

    def __init__(self, x, k, xerr=None, phi=None, seed=0):
        """
        Load data and basis functions. The raw data is comprised of
        discrete, noisy observations of a function in "time" dimension.
        Basis functions are defined using domain knowledge.
        
        Parameters:
        ----------
        x : ndarray (N, M)
            Raw data.
        k : int
            Number of basis functions.
        xerr : ndarray (N, M), optional
               Uncertainties on the raw data.
        basis : ndarrat (K, M)
                Basis functions. If None, Legendre Polynomials
                are used by default.
        seed  : int
                Random number generator seed. Defaults to 0.
        """
        
        self.x = x
        self.n = x.shape[0]
        self.m = x.shape[1]
        self.k = k
        self.xerr = xerr
        
        if phi is not None:
            if not k > phi.shape[0]:
                self.phi_t = phi[:k, :]
            else:
                raise ValueError('Number of basis functions to be used should \
                                  be less than or equal to the provided basis.')
        else:
            # Normalise "time" for generating Legendre polynomials
            t = np.arange(self.m)
            mid = (self.m - 1)/2
            t = (t - mid)/mid
            
            self.phi_t = GenerateLengdrePolynomials(t, k)
            
        # Center the observations
        self.sample_mu = np.array(np.mean(self.x, axis=0))
        self.x_centered = self.x - self.sample_mu
    
    @staticmethod
    def _run_fit(x, phi_t, xerr=None, rlm=None):
        '''
        Helper function for regression of data onto basis functions.
        Performs regression in one of three different ways:
        1) Weighted Least Squares
        2) Robust Linear Modeling
        3) Ordinary Least Squares
        4) FUTURE: Smoothing parameter - PENSSE
        '''
        x = np.array(x)
        phi_t = np.array(phi_t)
        
        if xerr:
            regr = sm.WLS(x, phi.T, weights=1./(xerr**2))
        elif rlm:
            regr = sm.RLM(x, phi_t.T)
        else:
            regr = sm.OLS(x, phi_t.T)

        regr_results = regr.fit()
        return regr_results.params
    
    def alpha_regression(self, parallel=0, rlm=False, M=None):
        """
        Regress discrete observations onto the basis to generate
        smooth continuous functions. Coefficients of regression
        are alpha.
        
        Parameters:
        -----------
        parallel : int
                   Number of processes to perform regression parallely
                   using multitprocessing. Sequential execution if zero.
        rlm : bool
              If True, Robust Linear Model will be used for regression
              else, Ordinary Least Squares regression will be done.
        M : statsmodels.robust.norms.RobustNorm, optional
            The robust criterion function for downweighting outliers.
            The current options are LeastSquares, HuberT, RamsayE, AndrewWave,
            TrimmedMean, Hampel, and TukeyBiweight.  The default is HuberT().
            See statsmodels.robust.norms for more information. 
        """
        
        self.regressed_alpha = np.zeros((self.n, self.k))
        
        # Observation mask
        try:
            mask = self.x_centered.mask
        except AttributeError:
            mask = np.full(self.x_centered.shape, False)
        
        # Parallel processing
        if parallel:
            # Create process pool
            pool = mp.Pool(parallel)
            # Assign tasks to processes
            regressed_alpha_list = pool.starmap(FPCA._run_fit,
                                                [(self.x_centered[i, ~mask[i, :]],
                                                  self.phi_t[:, ~mask[i, :]], rlm) for i in range(0, self.n)])
            # Close process pool
            pool.close()
            self.regressed_alpha = np.array(regressed_alpha_list)
       
        # Linear processing
        for i in tqdm(range(self.n)):
            dat = np.array(self.x_centered[i, ~mask[i, :]])
            phi = np.array(self.phi_t[:, ~mask[i, :]])
            if self.xerr is not None:
                # Weighted Least Squares
                sigma_err = np.array(self.xerr[i, ~mask[i, :]])
                regr = sm.WLS(dat, phi.T, weights=1./(sigma_err**2))
            elif rlm:
                 # Robust Linear Model
                regr = sm.RLM(dat, phi.T, M=M)
            else:
                # Ordinary Least Squares
                regr = sm.OLS(dat, phi.T)
            regr_results = regr.fit()
            self.regressed_alpha[i, :] = regr_results.params


    def solve_eigenproblem(self):
        """
        Solve the following finite dimensional eigenproblem to
        estimate eigenfunctions and eigenvalues.

        [1/N * W^(1/2) * alpha.T * alpha * W^(1/2)] * u_j = k_j * u_j
        where b = W^(-1/2) * u.
        
        Solve the above to get the eigenvalues k_j and b_j, and then
        solve for the eigenfunctions psi_j_t = b_j * phi_t.
        """
        
        phi_t = self.phi_t[:self.k]
        regressed_alpha = self.regressed_alpha[:, :self.k]
        
        # Basis weight matrix
        W = np.matmul(phi_t, phi_t.T)
        W_1_2 = linalg.fractional_matrix_power(W, 0.5)
        
        # subtract mean curve
        #self.regressed_alpha0 = self.regressed_alpha - np.mean(self.regressed_alpha, axis=0)
        
        # Solve eigenproblem and estimate eigenvalues
        A = (1/self.n)*np.linalg.multi_dot([W_1_2, regressed_alpha.T, regressed_alpha, W_1_2])        
        self.k_cap, u = np.linalg.eigh(A)
        
        # Estimate eigenfunctions
        beta = np.matmul(linalg.fractional_matrix_power(W, -0.5), u)
        self.psi_cap_t = np.matmul(phi_t.T, beta)
        
        # Get percentage of variation in the direction of psi_t
        self.perc_var = np.floor(1000 * self.k_cap/np.sum(self.k_cap))/10