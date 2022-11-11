specdims
========

Code for A. A. Patil, J. Bovy, G. Eadie, and S. Jaimungal, "Functional Data Analysis for Extracting the Intrinsic Dimensionality of Spectra: Application to Chemical Homogeneity in the Open Cluster M67", The Astrophysical Journal. 926, 51 (2022) [ADS](https://ui.adsabs.harvard.edu/abs/2022ApJ...926...51P/abstract)

AUTHOR
------

Main author: Aarya A. Patil

CITING THIS CODE
----------------

Aarya A. Patil et al. 2022 ApJ 926 51

INTRODUCTION
------------

We apply Functional Principal Component Analysis (FPCA) to a large sample of APOGEE giants to extract the spectral structure instrinsic to the stars embedded in systematics, and determine its dimensionality. We then apply our Functional Principal Components (FPCs) to constrain chemical homogeneity in open cluster M67. This proves that the FPCs incorporate abundance information, and in turn helps validate chemical tagging. Thus, we have the opportunity to perform chemical tagging in the instrinsic spectral structure defined by our FPCs, which likely has information beyond a limited number of error-prone model-derived abundances.

fpca
----

One of the main contributions of this code is ``fpca.py``, a general-purpose python module for performing FPCA. ``fpca`` is currently not yet available on PyPI, but it can be used by downloading the source code or cloning the GitHub repository and importing the module as follows:

```
import fpca

fpca_train_dat = FPCA(data, 50, phi=basis, xerr=data_err)
fpca_train_dat.alpha_regression()
fpca_train_dat.solve_eigenproblem()

print(fpca_train_dat.psi_cap_t, fpca_train_dat.perc_var)
```

Here ``data`` and ``data_err`` represent the data you want to perform FPCA on and its error. ``50`` represents the number of basis functions. ``basis`` represents the basis functions you want to use; by deafult these are Legendre Polynomials. ``alpha_regression()`` provides the coefficients of regression for generating the functional approximation of data, whereas solve_eigenproblem provides FPCs.

Refer to the notebook ``fpca_test.ipynb`` to understand the basics of FPCA and the usage of ``fpca``.

delfiSpec
---------

``delfiSpec`` is a module that provides useful functions to read, process and simulate APOGEE spectra.

This builds on top of [`apogee`](https://github.com/jobovy/apogee).

apogee_fpca
-----------

Code to compute FPCs of the APOGEE giants.

apogee_SNL
----------

Code to perform Sequential Neural Likelihood on M67 APOGEE giants for inference of stellar parameters and abundances.

This requires [`pydelfi`](https://github.com/justinalsing/pydelfi).

hierarch_t_distribution
-----------------------

Computing hierarchical distributions using the method Hogg, 2010 to constrain chemical homogeneity in M67.

gen_plots_tables
----------------

Code to generate plots and tables in the paper.
