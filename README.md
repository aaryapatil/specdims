# specdims
Code for Patil et al. 2021, "Functional Data Analysis for extracting the Intrinsic Dimensionality of Spectra
Application:  Chemical Homogeneity in Open Cluster M67"

We apply Functional Principal Component Analysis (FPCA) to a large sample of APOGEE giants to extract spectral structure instrinsic to the stars from systematics, and determine its dimensionality. We then apply our Functional Principal Components (FPCs) to constrain chemical homogeneity in open cluster M67. This proves that the FPCs incorporate abundance information and in turn helps validate chemical tagging. Thus, we have the opportunity to chemical tagging in the instrinsic spectral structure defined by our FPCs, which likely has information beyond a limited number of inferred abundances.

One of the main constributions of this code is ``fpca.py`` a general-purpose python module for performing FPCA. You can use this code by cloning this repository and importing the module as follows:

```
import fpca

fpca_train_dat = FPCA(data, 50, phi=basis, xerr=data_err)
fpca_train_dat.alpha_regression()
fpca_train_dat.solve_eigenproblem()

print(fpca_train_dat.psi_cap_t, fpca_train_dat.perc_var)
```

Here ``data`` and ``data_err`` represent the data you want to perform FPCA on and its error. ``50`` represents the number of basis functions. ``basis`` represents the basis functions you want to use; by deafult these are Legendre Polynomials. ``alpha_regression()`` provides the coefficients of regression for generating the functional approximation of data, whereas solve_eigenproblem provides FPCs.