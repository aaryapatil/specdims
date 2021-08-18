import numpy as np
import empca
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from apogee.tools.path import change_dr
from apogee.tools import pix2wv
from apogee.tools import bitmask as bm
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from delfiSpec import util, specproc, specsim


# Read APOGEE DR14 catalogue
change_dr('14')
apCat = util.ApogeeCat()

# Read M67 cluster APOGEE spectra    
_, M67_GM_apogee = apCat.read_OCCAM_cluster()

# Perform APOGEE cuts: Giant members with Iron abundance within M67 limits
apogee_cat_cut = apCat.apogee_cat[(apCat.apogee_cat['FE_H'] > -0.15) &
                                  (apCat.apogee_cat['FE_H'] < 0.15) &
                                  (apCat.apogee_cat['LOGG'] < 4) & (apCat.apogee_cat['LOGG'] > -1)]

# High Signal-to-Noise Ratio
indx = apogee_cat_cut['SNR'] > 200
apogee_cat_cut = apogee_cat_cut[indx]

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

data_set_specproc, _, data_set_weight = specproc.process_spectra(spectra_info=data_set,
                                                                 badcombpixmask=badcombpixmask)

# Mask the spectra based on APOGEE bitmask
data_set_specmasked = np.ma.masked_array(data_set_specproc, mask=(data_set_weight==0))

# Remove spectra with more than 50% masked pixels
data_set_maskpixels = np.sum(data_set_specmasked.mask, axis=1)

# Final APOGEE cut
apogee_cat_cut = apogee_cat_cut[data_set_maskpixels < 50/100*7214]

# ------------------- Case Study Sample -------------------
'''
Teff and log g cut leads to a total of 20 spectra in ``case_cut``, the first then of these
are the case study "original" sample, S_org, whereas the rest are used either as basis
functions or for validation purposes and denoted as B. In most of the case study, we use 5
basis functions which are generated using ASPCAP estimates of spectra in B. At the end of
the case study, we illustrate how to choose K and J by using the third last spectrum in B
as a "test" or "validation" set, given that S_org is our "training" set, i.e., functional
PCs are computed for S_org; we need more basis functions in this case, which come from
ASPCAP estimates of ``case_cut_sep``.
'''
case_cut = apogee_cat_cut[np.where((4270 < apogee_cat_cut['TEFF']) & (apogee_cat_cut['TEFF'] < 4300))]
case_cut = case_cut[np.where((1.4 < case_cut['LOGG']) & (case_cut['LOGG'] < 1.6))]

# A separate cut is used for generating extra basis functions
case_cut_sep = apogee_cat_cut[np.where((4240 < apogee_cat_cut['TEFF']) & (apogee_cat_cut['TEFF'] < 4280))]
case_cut_sep = case_cut_sep[np.where((1.4 < case_cut_sep['LOGG']) & (case_cut_sep['LOGG'] < 1.6))]

# Read the two sample
case_cut_spec = apCat.read_allStar_spectra(case_cut)
case_cut_sep_spec = apCat.read_allStar_spectra(case_cut_sep)

case_cut_specproc, case_cut_specerr, case_cut_specweight = specproc.process_spectra(
    spectra_info=case_cut_spec, badcombpixmask=badcombpixmask)

case_cut_sep_specproc, case_cut_sep_specerr, case_cut_sep_specweight = specproc.process_spectra(
    spectra_info=case_cut_sep_spec, badcombpixmask=badcombpixmask)

# Mask the spectra based on APOGEE bitmask
case_cut_spec_masked = np.ma.masked_array(case_cut_specproc, mask=(test_cut_weight==0))
case_cut_sep_spec_masked = np.ma.masked_array(case_cut_sep_specproc, mask=(test_cut_sep_weight==0))

# APOGEE wavelengths
pix = np.arange(7214)
wave = pix2wv(pix, dr='12')
wave_mask = (((wave > 15799) & (wave < 15865)) | ((wave > 16425) & (wave < 16485)))
masked_wave = np.ma.masked_array(wave, wave_mask)

# Create the "original+systematic" sample
case_cut_sys_spec_masked = np.ma.zeros((10, 200))
for ind in range(5):
    case_cut_sys_spec_masked[i] = case_cut_spec_masked[ind, :200]  - (ind+1)*0.00025*(
        masked_wave[:200] - masked_wave[0])

for ind in range(5, 10):
    case_cut_sys_spec_masked[i] = case_cut_spec_masked[ind, :200]  + (ind-4)*0.00025*(
        masked_wave[:200] - masked_wave[0])

# ------------------- Plot Case Study Sample -------------------
mpl.rcParams["axes.labelsize"] = 28
mpl.rcParams['xtick.labelsize']= 20
mpl.rcParams['ytick.labelsize']= 20

fig, ax = plt.subplots(1, 1, figsize=(14, 12))

# Plot spectra for the two samples
for i in range(9):
    ax.scatter(masked_wave[:200], case_cut_spec_masked[i, 0:200] - i*0.3,
               color='orange', marker='o', s=12)
    ax.scatter(masked_wave[:200], case_cut_sys_spec_masked[i] - i*0.3,
               color='black', marker='o', s=12)

# Label the samples
ax.scatter(masked_wave[:200], case_cut_spec_masked[9, 0:200] - 9*0.3, color='orange',
           label='original', marker='o', s=12)
ax.scatter(masked_wave[:200], case_cut_sys_spec_masked[9] - 9*0.3, color='black',
           label='original + systematic', marker='o', s=12)

ax.legend(loc='upper right', fontsize=21)
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))

ax.set_yticks(np.linspace(1.0, -1.7, 10))
ax.set_yticklabels(np.arange(1, 11))
ax.set_xlabel(r'Wavelength $\lambda(\AA)$')
ax.set_ylabel(r'Spectrum number $n$ $[\mathbf{y}_n]$')
ax.set_ylim(-2.1, 1.5)

plt.savefig('data_case_study.png')


# ------------------- FPCA of Samples -------------------
# Choose 5 random basis functions from ``case_cut[10:]``
basis = specsim.sim_spectra(case_cut[10:])

# FPCA of masked case study spectra S_org
fpca_org = fpca.FPCA(case_cut_spec_masked[:10, :200], 5, phi=basis[:, :200],
                     xerr=case_cut_specerr[:10, :200])
fpca_org.alpha_regression()
fpca_org.solve_eigenproblem()

# Here we use same basis functions and same error, but different data.
# Essentially, the data has an added systematic.
fpca_org_sys = fpca.FPCA(case_cut_sys_spec_masked[:10], 5, phi=basis[:, :200],
                         xerr=case_cut_specerr[:10, :200])
fpca_org_sys.alpha_regression()
fpca_org_sys.solve_eigenproblem()


# ------------------- Plot Basis Functions -------------------
mpl.rcParams["axes.labelsize"] = 23
mpl.rcParams['xtick.labelsize']= 18
mpl.rcParams['ytick.labelsize']= 18

fig, ax = plt.subplots(1, 1, figsize=(10, 8))

color = ['indigo', 'green', 'orange', 'maroon', 'red']
axins = inset_axes(ax, width="60%", height="25%", loc=1)

# Plot basis functions
for i in range(5):
    ax.plot(masked_wave[:200], basis[i, :200] - i*0.4, color=color[i])

# Connect the inset to the relevant wavelength
ax.vlines(masked_wave[22], -1, 1.5, linestyle='--', linewidth=2, color='silver')
ax.plot([masked_wave[22], masked_wave[78]], [1.5, 1.97], linestyle='--', linewidth=2, color='silver')
ax.plot([masked_wave[22], masked_wave[78]], [1.5, 1.20], linestyle='--', linewidth=2, color='silver')

ax.set_yticks(np.linspace(1, -0.6, 5))
ax.set_yticklabels(np.arange(1, 6))
ax.xaxis.set_major_locator(MultipleLocator(10))
ax.xaxis.set_minor_locator(MultipleLocator(2))
ax.set_xlabel(r'Wavelength $\lambda(\AA)$')
ax.set_ylabel(r'Basis number $k$ [$\phi_k(\lambda)$]')
ax.set_ylim(-1, 2)

# Plot the inset
axins.scatter(np.arange(5), basis[:5, 22], color=color[:5], marker='o', s=85)
axins.set_ylim(0.785, 0.835)
axins.set_ylabel(r'$f/f_c(\lambda_x)$', fontsize=16)
axins.set_yticks([0.80, 0.82])
axins.set_yticklabels([0.80, 0.82], fontsize=12)
axins.set_xticks(np.arange(0, 5))
axins.set_xticklabels(np.arange(1, 6), fontsize=12)

# Indicate the wavelength for which we create an inset
plt.text(-2.3, 0.64, r'$\lambda_x$', fontsize=18, color='black')

plt.savefig('basis_case_study.png')


# ------------------- Plot Functional Approximation -------------------
mpl.rcParams["axes.labelsize"] = 23
mpl.rcParams['xtick.labelsize']= 18
mpl.rcParams['ytick.labelsize']= 18

masked_data_ind = masked_wave[0:200][(case_cut_spec_masked[0, 0:200].mask == True)]
mask_lower_ind = masked_data_ind[0]
mask_higher_ind = masked_data_ind[-1]

fig, ax = plt.subplots(2, 1, sharex=True, figsize=(14, 6),
                       gridspec_kw={'height_ratios': [3, 1.5]})

# Plot spectrum and the functional approx
ax[0].errorbar(masked_wave[0:200], case_cut_spec_masked[0, 0:200],
               yerr=case_cut_specerr[0, :200], color='black',
               fmt='.', label=r'spectral data $\mathbf{y}_{o,1}$')
ax[0].plot(masked_wave[0:200], fpca_org.sample_mu + np.dot(fpca_org.regressed_alpha,
                                                           fpca_org.phi_t)[0],
           color='steelblue', linestyle='--', linewidth=2,
           label=r'smooth function $\hat f_{o,1}(\lambda)$')

# Highlight the masked region
ax[0].axvspan(mask_lower_ind, mask_higher_ind, alpha=0.8,
              color='lavender')

ax[0].legend(loc='upper right', fontsize=15)
ax[0].set_ylim([0.775, 1.175])
ax[0].yaxis.set_major_locator(MultipleLocator(0.1))
ax[0].yaxis.set_minor_locator(MultipleLocator(0.025))
ax[0].get_yaxis().labelpad = 26
ax[0].set_ylabel(r'$f/f_c (\lambda)$')

# Plot residuals
ax[1].axhspan(-0.005, 0.005, alpha=0.35,
              color='silver', label='APOGEE base uncertainty')
ax[1].scatter(masked_wave[0:200], (case_cut_spec_masked[0, 0:200] - \
                                   (fpca_org.sample_mu + np.dot(fpca_org.regressed_alpha,
                                                                fpca_org.phi_t)[0])),
              color='black', s=8)
# Highlight the masked region
ax[1].axvspan(mask_lower_ind, mask_higher_ind, alpha=0.8,
              color='lavender', label='mask')

ax[1].legend(ncol=2, loc='lower left', fontsize=15)
ax[1].set_ylim([-0.075, 0.075])
ax[1].xaxis.set_major_locator(MultipleLocator(10))
ax[1].xaxis.set_minor_locator(MultipleLocator(2))
ax[1].yaxis.set_major_locator(MultipleLocator(0.05))
ax[1].yaxis.set_minor_locator(MultipleLocator(0.025))
ax[1].set_xlabel(r'Wavelength $\lambda(\AA)$')
ax[1].set_ylabel('Residuals')

plt.savefig('func_approx_case_study.png')


# ------------------- PCA of Samples -------------------
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

X_org = (case_cut_spec_masked[:10, :200]).filled(np.nan)
X_org = imp_mean.fit_transform(X_org)
X_org = X_org - np.mean(X_org, axis=0)

X_org_sys = (case_cut_sys_spec_masked[:10]).filled(np.nan)
X_org_sys = imp_mean.fit_transform(X_org_sys)
X_org_sys = X_org_sys - np.mean(X_org_sys, axis=0)

pca_org = PCA(n_components=5)
pca_org.fit(X_org)

pca_org_sys = PCA(n_components=5)
pca_org_sys.fit(X_org_sys)

# ------------------- Plot eigenanalysis results -------------------
mpl.rcParams["axes.labelsize"] = 34
mpl.rcParams['xtick.labelsize']= 28
mpl.rcParams['ytick.labelsize']= 28

fig, ax = plt.subplots(2, 2, figsize=(32, 20), gridspec_kw={'width_ratios': [3, 2.5]})

fig.tight_layout(pad=14, h_pad=4)

# Plot the FPCs
for eigen in np.arange(2):
    ax[0][0].plot(masked_wave[:200], fpca_org.psi_cap_t.real[:, -(eigen+1)] - eigen*0.5,
                  color='orange', linewidth=2.5)
    ax[0][0].plot(masked_wave[:200], fpca_org_sys.psi_cap_t.real[:, -(eigen+1)] - eigen*0.5,
                  color='black', linestyle='-.', linewidth=2.5)

ax[0][0].plot(masked_wave[:200], fpca_org.psi_cap_t.real[:, -3] - 2*0.5,
              color='orange', linewidth=2.5, label='original')
ax[0][0].plot(masked_wave[:200], fpca_org_sys.psi_cap_t.real[:, -3] - 2*0.5,
              color='black', linestyle='-.', linewidth=2.5, label='original + systematic')

ax[0][0].set_xlim(masked_wave[0], masked_wave[200])
ax[0][0].set_ylim(-1.5, 0.3)
ax[0][0].xaxis.set_major_locator(MultipleLocator(10))
ax[0][0].xaxis.set_minor_locator(MultipleLocator(2))
ax[0][0].set_yticks([0, -0.5, -1])
ax[0][0].set_yticklabels(['$\Psi_{}(\lambda)$'.format(i) for i in range(1, 4)])
ax[0][0].set_ylabel(r'Principal Component')
ax[0][0].legend(loc='lower right', fontsize=25)

# Plot the PCs
for eigen in np.arange(2):
    ax[1][0].plot(masked_wave[:200], pca_org.components_[eigen] - eigen*0.5,
                  color='orange', linewidth=2.5)
    ax[1][0].plot(masked_wave[:200], pca_org_sys.components_[eigen] - eigen*0.5,
                  color='black', linestyle='-.', linewidth=2.5)

ax[1][0].plot(masked_wave[:200], pca_org.components_[2] - 2*0.5,
              color='orange', linewidth=2.5, label='original')
ax[1][0].plot(masked_wave[:200], pca_org_sys.components_[2] - 2*0.5,
              color='black', linestyle='-.', linewidth=2.5, label='original + systematic')

ax[1][0].set_xlim(masked_wave[0], masked_wave[200])
ax[1][0].set_ylim(-1.5, 0.3)
ax[1][0].xaxis.set_major_locator(MultipleLocator(10))
ax[1][0].xaxis.set_minor_locator(MultipleLocator(2))
ax[1][0].set_yticks([0, -0.5, -1])
ax[1][0].set_yticklabels([r'$\overrightarrow{PC}_1$',
                          r'$\overrightarrow{PC}_2$',
                          r'$\overrightarrow{PC}_3$'])
ax[1][0].set_xlabel(r'Wavelength $\lambda(\AA)$')
ax[1][0].set_ylabel(r'Principal Component')
ax[1][0].get_yaxis().labelpad = 23
ax[1][0].legend(loc='lower right', fontsize=25)

# Scree plots for FPCA
ax[0][1].plot(fpca_org.perc_var[::-1], marker='o', color='orange',
              linewidth=2.5, label='original')
ax[0][1].plot(fpca_org_sys.perc_var[::-1], marker='o', linestyle='-.',
              linewidth=2.5, color='black', label='original + systematic')

ax[0][1].set_ylim([-5, 101])
ax[0][1].set_xticks(np.arange(5))
ax[0][1].set_xticklabels(['$\Psi_{}(\lambda)$'.format(i) for i in range(1, 6)])
ax[0][1].yaxis.set_major_locator(MultipleLocator(20))
ax[0][1].yaxis.set_minor_locator(MultipleLocator(5))
ax[0][1].set_ylabel('% Variance Explained \n' + r'(100*$\zeta_j)$')
ax[0][1].get_yaxis().labelpad = 0
ax[0][1].legend(loc='upper right', fontsize=25)

# Scree plots for PCA
ax[1][1].plot(pca_org.explained_variance_ratio_*100,
              marker='o', color='orange', linewidth=2.5, label='original')
ax[1][1].plot(pca_org_sys.explained_variance_ratio_*100,
              marker='o', linestyle='-.', linewidth=2.5,
              color='black', label='original + systematic')

ax[1][1].set_ylim([-5, 101])
ax[1][1].set_xticks(np.arange(5))
ax[1][1].set_xticklabels([r'$\overrightarrow{PC}_1$', r'$\overrightarrow{PC}_2$', r'$\overrightarrow{PC}_3$',
                          r'$\overrightarrow{PC}_4$', r'$\overrightarrow{PC}_5$'])
ax[1][1].yaxis.set_major_locator(MultipleLocator(20))
ax[1][1].yaxis.set_minor_locator(MultipleLocator(5))
ax[1][1].set_xlabel(r'Principal Component')
ax[1][1].set_ylabel('% Variance Explained \n' + r'(100*$\zeta_j)$')
ax[1][1].get_yaxis().labelpad = 0
ax[1][1].legend(loc='upper right', fontsize=25)

plt.text(-8.3, 170, 'FPCA', fontsize=50, weight='bold')
plt.text(-8.3, 45, 'PCA', fontsize=50, weight='bold')

plt.savefig('pc_case_study.png')

print(f'Mean Slope of PC1 for the original + systematic sample'
       'is {(pca_org_sys.components_[0][-1] - pca_org_sys.components_[0][0])/(wave[200] - wave[0])}')


# ------------------- Covariance Structures -------------------
# Covariance functions
v_org = 1/fpca_org.n * np.dot(np.dot(fpca_org.phi_t.T, fpca_org.regressed_alpha.T),
                              np.dot(fpca_org.regressed_alpha, fpca_org.phi_t))
v_org_sys = 1/fpca_org_sys.n * np.dot(np.dot(fpca_org_sys.phi_t.T, fpca_org_sys.regressed_alpha.T),
                                      np.dot(fpca_org_sys.regressed_alpha, fpca_org_sys.phi_t))

# Covariance matrix
C_org = 1/fpca_org.n * np.dot(X_org.T, X_org)
C_org_sys = 1/fpca_org_sys.n * np.dot(X_org_sys.T, X_org_sys)


# ------------------- Plot Covariance Structures -------------------
mpl.rcParams["axes.labelsize"] = 32
mpl.rcParams['xtick.labelsize']= 26
mpl.rcParams['ytick.labelsize']= 26

fig, ax = plt.subplots(2, 2, figsize=(22, 15))

minmin = np.min([np.min(v_org), np.min(C_org)])
maxmax = np.max([np.max(v_org), np.max(C_org)])

# Covariance function for original
im1 = ax[0][0].imshow(v_org, vmin=minmin, vmax=maxmax,
                      cmap='afmhot', aspect='auto')

ax[0][0].set_xticks(np.arange(39, 200, 47))
ax[0][0].set_yticks(np.arange(39, 200, 47))
ax[0][0].set_xticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[0][0].set_yticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[0][0].set_ylabel(r'Wavelength $\lambda(\AA)$')
ax[0][0].set_title('ORIGINAL', fontsize=30, pad=20)

# Covariance matrix for original
im2 = ax[1][0].imshow(C_org, cmap='afmhot', vmin=minmin,
                      vmax=maxmax, aspect='auto')

ax[1][0].set_xticks(np.arange(39, 200, 47))
ax[1][0].set_yticks(np.arange(39, 200, 47))
ax[1][0].set_xticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[1][0].set_yticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[1][0].set_xlabel(r'Wavelength $\lambda(\AA)$')
ax[1][0].set_ylabel(r'Wavelength $\lambda(\AA)$')

minmin = np.min([np.min(v_sys), np.min(C_pca_sys)])
maxmax = np.max([np.max(v_sys), np.max(C_pca_sys)])

# Covariance function for original + systematic
im3 = ax[0][1].imshow(v_org_sys, vmin=minmin, vmax=maxmax,
                      cmap='afmhot', aspect='auto')

ax[0][1].set_xticks(np.arange(39, 200, 47))
ax[0][1].set_yticks(np.arange(39, 200, 47))
ax[0][1].set_xticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[0][1].set_yticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[0][1].set_ylabel(r'Wavelength $\lambda(\AA)$')
ax[0][1].set_title('ORIGINAL + SYSTEMATIC', fontsize=30, pad=20)

# Covariance matrix for original + systematic
im4 = ax[1][1].imshow(C_org_sys, cmap='afmhot', vmin=minmin,
                      vmax=maxmax, aspect='auto')

ax[1][1].set_xticks(np.arange(39, 200, 47))
ax[1][1].set_yticks(np.arange(39, 200, 47))
ax[1][1].set_xticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[1][1].set_yticklabels(['{:.0f}'.format(masked_wave[i]) for i in np.arange(39, 200, 47)])
ax[1][1].set_xlabel(r'Wavelength $\lambda(\AA)$')
ax[1][1].set_ylabel(r'Wavelength $\lambda(\AA)$')

fig.tight_layout(h_pad=2, pad=16)

# Plot the colorbars
fig.subplots_adjust(right=0.9)

cbar_ax_1 = fig.add_axes([0.475, 0.26, 0.015, 0.5])
cbar_1 = fig.colorbar(im1, cax=cbar_ax_1, ticks=[-0.0002, 0, 0.0002, 0.0004], orientation='vertical')
cbar_1.ax.ticklabel_format(axis='y', scilimits=[-4, 4])
cbar_1.ax.set_ylabel('Covariance')

cbar_ax_2 = fig.add_axes([0.93, 0.26, 0.015, 0.5])
cbar_2 = fig.colorbar(im3, cax=cbar_ax_2, ticks=[0, 0.0004, 0.0008, 0.0012], orientation='vertical')
cbar_2.ax.ticklabel_format(axis='y', scilimits=[-3, 3])
cbar_2.ax.set_ylabel('Covariance')

plt.text(-0.098, 0.0011, 'FPCA', fontsize=36, weight='bold')
plt.text(-0.098, 0.00008, 'PCA', fontsize=36, weight='bold')

plt.savefig('cov_case_study.png')


# ------------------- How to choose K and J? -------------------
# Choose 50 random basis functions from ``case_cut_sep``
basis = specsim.sim_spectra(case_cut_sep)

# FPCA of masked case study spectra S_org
fpca_org_train = fpca.FPCA(case_cut_spec_masked[:10, :200], 50, phi=basis[:, :200],
                           xerr=case_cut_specerr[:10, :200])
fpca_org_train.alpha_regression()
fpca_org_train.solve_eigenproblem()

# Residual error statistics
mae = np.zeros(50)
mad = np.zeros(50)

# FPCs
psi_t = fpca_org_train.psi_cap_t[::-1]

# Mean subtracted validation spectrum
test_mean_sub = case_cut_spec_masked[-3, :200] - fpca_org_train.sample_mu
test_weight = case_cut_specweight[-3, :200]

for i in range(0, 51):
    # Fit an FPC model using the EM algorithm
    empca_fit = empca.Model(psi_t[:, :i].T, test_mean_sub[np.newaxis, :],
                            test_weight[np.newaxis, :])
    mae[i-1] = np.mean(np.abs((test_mean_sub - empca_fit.model[0])/case_cut_spec_masked[-3, :200]))
    mad[i-1] = np.median(np.abs((test_mean_sub - empca_fit.model[0])/case_cut_spec_masked[-3, :200]))


# ------------------- Plot MAE and MAD -------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 7))

ax.plot(mae, color='red', marker='o', label='mean absolute error')
ax.plot(mad, color='black', marker='o', label='median absolute deviation')

# Plot SNR of validation star as well as APOGEE base systematic uncertainty
ax.axhline((1/case_cut['SNR'][-3]), color='orange', linestyle='-.',
           linewidth=4, label='test star noise level', alpha=0.65)
ax.axhline((1/200), color='silver', linestyle='-.', linewidth=4,
           label='APOGEE systematic noise level', alpha=0.65)

ax.set_xticks(np.arange(0, 60, 10))
ax.set_xticklabels(np.arange(1, 61, 10))
ax.set_yticks([3e-3, 4e-3, 5e-3, 6e-3, 7e-3])
ax.set_xlabel(r'Number of eigenfunctions $J$ (dimensionality)')
ax.set_ylabel('Residual Error $e$')
ax.legend(fontsize=15)

# Plot a secondary y-axis
ax1 = ax.twinx()
ax1.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax1.set_yticklabels([3/5, 4/5, 1, 6/5, 7/5])
ax1.set_ylabel('$e/0.005$')

plt.savefig('mse_case_study.png')
# --------------------------------------------------------
