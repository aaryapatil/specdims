import warnings

import numpy as np
import empca
from apogee.tools.path import change_dr
from apogee.tools import bitmask as bm
from sklearn.decomposition import PCA

from delfiSpec import util, specproc, specsim
from fpca import FPCA


# Ignore warnings
warnings.filterwarnings("ignore")

# Read APOGEE DR14 catalogue
change_dr('14')
apCat = util.ApogeeCat()

# Read M67 cluster APOGEE spectra    
M67_DM_apogee, M67_GM_apogee = apCat.read_OCCAM_cluster()

# Perform APOGEE cuts: Giant members with Iron abundance within M67 limits
print(r'M67 GM FE/H limits: [{:.3f}, {:.3f}]'.format(np.min(M67_GM_apogee['FE_H']),
                                                     np.max(M67_GM_apogee['FE_H'])))
print(r'Our GM FE/H limits: [{:.3f}, {:.3f}]'.format(-0.15, 0.15))
apogee_cat_cut = apCat.apogee_cat[(apCat.apogee_cat['FE_H'] > -0.15) &
                                  (apCat.apogee_cat['FE_H'] < 0.15) &
                                  (apCat.apogee_cat['LOGG'] < 4) & (apCat.apogee_cat['LOGG'] > -1)]

# High Signal-to-Noise Ratio cut
indx = apogee_cat_cut['SNR'] > 200
apogee_cat_cut = apogee_cat_cut[indx]
print('APOGEE giants spectra after FE_H cut: {}'.format(len(apogee_cat_cut)))

# Make sure all abundances have "physical values"
abundances = ['FE_H', 'C_FE', 'N_FE', 'O_FE', 'NA_FE', 'MG_FE', 'AL_FE', 'SI_FE', 'S_FE',
              'K_FE', 'CA_FE', 'TI_FE', 'V_FE', 'MN_FE', 'NI_FE']

for i in range(1, len(abundances)):
    apogee_cat_cut = apogee_cat_cut[(apogee_cat_cut[abundances[i]] > -1) &
                                    (apogee_cat_cut[abundances[i]] < 1)]

# Read APOGEE spectra for the above APOGEE cut
data_set = apCat.read_allStar_spectra(apogee_cat_cut)

# Mask bad pixels using APOGEE bitmask
badcombpixmask = bm.badpixmask()
pix_err = np.array([bm.apogee_pixmask_int("SIG_SKYLINE"), bm.apogee_pixmask_int("SIG_TELLURIC"),
                    bm.apogee_pixmask_int("PERSIST_HIGH"), bm.apogee_pixmask_int("PERSIST_MED"),
                    bm.apogee_pixmask_int("PERSIST_LOW")])
badcombpixmask += np.sum(2**pix_err)

data_set_specproc, data_set_specerr, data_set_specweight = specproc.process_spectra(spectra_info=data_set,
                                                                             badcombpixmask=badcombpixmask)

# Mask the spectra based on APOGEE bitmask
data_set_specmasked = np.ma.masked_array(data_set_specproc, mask=(data_set_specweight==0))

# Remove spectra with more than 50% masked pixels
data_set_maskpixels = np.sum(data_set_specmasked.mask, axis=1)
data_set_specmasked = data_set_specmasked[data_set_maskpixels < 50/100*7214]
data_set_specerr = data_set_specerr[data_set_maskpixels < 50/100*7214]

# Generate an APOGEE data cut that corresponds exactly to the data
apogee_cat_cut = apogee_cat_cut[data_set_maskpixels < 50/100*7214]

# Simulate theoretical spectra analogous to the training set using its APOGEE parameters
data_set_sim = specsim.sim_spectra(apogee_cat_cut)

# --------------- FPCA on APOGEE spectra data ---------------
# Choose 50 random basis functions
rand_ind = np.random.random_integers(0, len(data_set_sim)-1, 50)
basis = data_set_sim[rand_ind]

# FPCA of masked APOGEE spectra
fpca_train_dat = FPCA(data_set_specmasked, 50, phi=basis, xerr=data_set_specerr)
fpca_train_dat.alpha_regression()
fpca_train_dat.solve_eigenproblem()

np.savetxt('data/FPCA_apogee/fpca_dat_eigenvectors_psi_t.dat', fpca_train_dat.psi_cap_t.real[:, ::-1])
np.savetxt('data/FPCA_apogee/fpca_dat_eigenvalues_k_cap.dat', fpca_train_dat.perc_var[::-1])
np.savetxt('data/FPCA_apogee/fpca_dat_spec_mean.dat', fpca_train_dat.sample_mu)

# --------------- EMPCA on APOGEE spectral data ---------------
# EMPCA of masked APOGEE spectra after mean subtraction
data_set_data_mean = data_set_specmasked.mean(axis=0, keepdims=True)
data_set_spec_meansub = data_set_specproc - data_set_data_mean
model_empca = empca.empca(data_set_spec_meansub, data_set_weight, deltR2=2e-07, niter=10, nvec=50)

var_r2 = np.zeros(50)
for nvec in np.arange(1, 51):
    var_r2[nvec-1] = model_empca.R2(nvec=nvec)

np.savetxt('data/EMPCA_apogee/empca_dat_eigenvectors_PCs.dat', model_empca.eigvec)
np.savetxt('data/EMPCA_apogee/empca_dat_eigenvalues_R2.dat', var_r2)

# ---------------- PCA on PSM simulated spectra ----------------
# PCA of simulated spectra
data_set_sim_mean = np.mean(data_set_sim, axis=0)
data_set_sim = data_set_sim - data_set_sim_mean
pca_sim = PCA(n_components=50)
pca_sim.fit(data_set_sim)

np.savetxt('data/PCA_sim/pca_sim_eigenvectors_PCs.dat', pca_sim.components_)
np.savetxt('data/PCA_sim/pca_sim_eigenvalues_percvar.dat', pca_sim.explained_variance_ratio_*100)

# Correlations
eigenfun_dat = fpca_train_dat.psi_cap_t.real[:, ::-1]
eigenvec_dat = model_empca.eigvec
eigenvec_sim = pca_sim.components_

# First 10 PCs
fpca_corr = np.zeros(10)
empca_corr = np.zeros(10)
for i in range(10):
    fpca_corr[i] = 100*np.abs(np.corrcoef(eigenfun_dat[:, i], eigenvec_sim[i])[0][1])
    empca_corr[i] = 100*np.abs(np.corrcoef(eigenvec_dat[i], eigenvec_sim[i])[0][1])

np.savetxt('data/fpca_correlations.dat', fpca_corr)
np.savetxt('data/empca_correlations.dat', empca_corr)