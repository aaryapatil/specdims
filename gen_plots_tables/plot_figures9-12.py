import numpy as np
import empca
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from apogee.tools.path import change_dr

from delfiSpec import util, specproc


mpl.rcParams["axes.labelsize"] = 36
mpl.rcParams['xtick.labelsize']= 30
mpl.rcParams['ytick.labelsize']= 30

def plot_eigenvectors(eigvec, pca_type='fpca'):
    """
    Plot FPCs and EM PCs of the APOGEE spectral sample
    """
    fig, ax0 = plt.subplots(1, 1, figsize=(18, 15))
    
    if pca_type == 'fpca':
        color = ['black', 'orange']*10
    else:
        color = ['midnightblue', 'coral']*10

    for eigen in range(10):
        ax0.plot(masked_wave, eigvec[(eigen)] - eigen*0.15,
                 color=color[eigen], linewidth=1.25)
        ax0.hlines(-eigen*0.15, masked_wave[0], masked_wave[-1],
                   linestyle='--', color='grey', linewidth=0.5)

    ax0.set_xlabel(r'Wavelength $\lambda$ ($\AA$)')

    if pca_type == 'fpca':
        ax0.set_ylabel('Functional Principal Components')
    else:
        ax0.set_ylabel('Expetation Maximized Principal Components')

    ax0.xaxis.set_major_locator(MultipleLocator(250))
    ax0.xaxis.set_minor_locator(MultipleLocator(50))

    ax0.set_ylim([-1.5, 0.15])
    ax0.set_yticks(np.linspace(0, -1.35, 10))

    if pca_type == 'fpca':
        ax0.set_yticklabels([r'$\Psi_1(\lambda)$', r'$\Psi_2(\lambda)$',
                             r'$\Psi_3(\lambda)$', r'$\Psi_4(\lambda)$',
                             r'$\Psi_5(\lambda)$', r'$\Psi_6(\lambda)$',
                             r'$\Psi_7(\lambda)$', r'$\Psi_8(\lambda)$',
                             r'$\Psi_9(\lambda)$', r'$\Psi_{10}(\lambda)$'])
        plt.savefig('specdims/data/fpca_comp.png')
    else:
        ax0.set_yticklabels([r'$PC_1$', r'$PC_2$', r'$PC_3$', r'$PC_4$',
                             r'$PC_5$', r'$PC_6$', r'$PC_7$', r'$PC_8$',
                             r'$PC_9$', r'$PC_{10}$'])
        plt.savefig('specdims/data/empca_comp.png')


eigenfun_dat = np.loadtxt('specdims/data/FPCA_apogee/fpca_dat_eigenvectors_psi_t.dat')
eigenval_dat = np.loadtxt('specdims/data/FPCA_apogee/fpca_dat_eigenvalues_k_cap.dat')
spec_mean_dat = np.loadtxt('specdims/data/FPCA_apogee/fpca_dat_spec_mean.dat')

eigenvec_dat = np.loadtxt('specdims/data/EMPCA_apogee/empca_dat_eigenvectors_PCs.dat')
eigenR2_dat = np.loadtxt('specdims/data/EMPCA_apogee/empca_dat_eigenvalues_R2.dat')

eigenvec_sim = np.loadtxt('specdims/data/PCA_sim/pca_sim_eigenvectors_PCs.dat')
eigenval_sim = np.loadtxt('specdims/data/PCA_sim/pca_sim_eigenvalues_percvar.dat')


# ---------------- Plot PCs ----------------
plot_eigenvectors(np.swapaxes(eigenfun_dat, 1, 0))
plot_eigenvectors(eigenvec_dat, pca_type='pca')


# ---------------- Plot eigenvalues ----------------
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(20, 15),
                             gridspec_kw={'height_ratios': [8, 1]})

ax1.bar(np.arange(0, 10), np.cumsum(eigenval_dat[:10]),
        color='maroon', width=0.7, label='FPCA $\zeta_{\mathrm{j}}$')
ax1.bar(np.arange(1, 10), np.cumsum(eigenval_dat[:10])[:-1],
        color='wheat', width=0.7, label='FPCA $\sum_{i=1}^{\mathrm{j}-1} \zeta_{i}$')
ax1.bar(np.arange(0, 10), 100*eigenR2_dat[:10], color='tab:blue',
        width=0.5, label='EMPCA $\zeta_{\mathrm{j}}$')
ax1.bar(np.arange(1, 10), 100*eigenR2_dat[0:9], color='lavender',
        width=0.5, label='EMPCA $\sum_{i=1}^{\mathrm{j}-1} \zeta_{i}$')
ax2.bar(np.arange(0, 10), np.cumsum(eigenval_dat[:10]), color='maroon',
        width=0.7, label='FPCA $\zeta_{\mathrm{j}}$')
ax2.bar(np.arange(1, 10), np.cumsum(eigenval_dat[:10])[:-1],
        color='wheat', width=0.7, label='FPCA $\sum_{i=1}^{\mathrm{j}-1} \zeta_{i}$')
ax2.bar(np.arange(0, 10), 100*eigenR2_dat[:10], color='tab:blue',
        width=0.5, label='EMPCA $\zeta_{\mathrm{j}}$')
ax2.bar(np.arange(1, 10), 100*eigenR2_dat[0:9], color='lavender', width=0.5,
        label='EMPCA $\sum_{i=1}^{\mathrm{j}-1} \zeta_{i}$')

# Show two different data portions
ax1.set_ylim(78, 105)  # upper portion
ax2.set_ylim(0, 2.5)   # bottom portion

# Hide the spines between ax1 and ax2
ax1.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)

d = .015               # diagonal lines size
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
ax1.plot((-d, +d), (-d, +d), **kwargs, linewidth=2.5)        # top-left diagonal
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs, linewidth=2.5)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (0.92 - d, 1.15 + d), **kwargs,
         linewidth=2.5)        # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (0.92 - d, 1.15 + d), **kwargs,
         linewidth=2.5)        # bottom-right diagonal

ax2.set_xlim(-1, 10)
ax2.set_xticks(np.arange(0, 10, 1))
ax2.set_xticklabels(np.concatenate(([f'$PC_{i}$' for i in range(1, 10, 1)], [r'$PC_{10}$'])))
ax2.set_xlabel('Principal Component')
ax2.set_yticks([0, 1])
ax2.set_yticklabels([0, 1])
ax2.yaxis.set_minor_locator(MultipleLocator(1))

ax1.set_yticks(np.arange(80, 102, 5))
ax1.set_yticklabels(np.arange(80, 102, 5))
ax1.yaxis.set_minor_locator(MultipleLocator(1))

ax1.legend(ncol=4, fontsize=29)
f.text(0.06, 0.51, 'Cumulative % Variance', va='center', rotation='vertical', fontsize=36)

plt.subplots_adjust(hspace=0.15)

plt.savefig('specdims/data/eigvals.png')


# ---------------- Model M67 GM member ----------------
# Read apogee DR14 catalogue
change_dr('14')
apCat = util.ApogeeCat()

# Read M67 data
_, M67_GM_apogee = apCat.read_OCCAM_cluster()
M67_GM_spectra = apCat.read_allStar_spectra(M67_GM_apogee)

# Mask bad pixels using APOGEE bitmask
badcombpixmask = bm.badpixmask()
pix_err = np.array([bm.apogee_pixmask_int("SIG_SKYLINE"),
                    bm.apogee_pixmask_int("SIG_TELLURIC"),
                    bm.apogee_pixmask_int("PERSIST_HIGH"),
                    bm.apogee_pixmask_int("PERSIST_MED"),
                    bm.apogee_pixmask_int("PERSIST_LOW")])
badcombpixmask += np.sum(2**pix_err)

# Read processed spectra
M67_GM_specproc, M67_GM_specerr, M67_GM_specweight = specproc.process_spectra(
    spectra_info=M67_GM_spectra, badcombpixmask=badcombpixmask)

# Project mean subtracted spectral data onto FPCs
M67_GM_specres = M67_GM_specproc - spec_mean_dat
empca_fit_data = empca.Model(eigenfun_dat[:, :50].T, M67_GM_specres,
                             M67_GM_weight)

fit_data = empca_fit_data.model


# ---------------- Plot FPCA model ----------------
mpl.rcParams["axes.labelsize"] = 22
mpl.rcParams['xtick.labelsize']= 18
mpl.rcParams['ytick.labelsize']= 18

f, (a0, a1) = plt.subplots(2, 1, sharex=True, figsize=(20, 6),
                           gridspec_kw={'height_ratios': [4, 1.5]})

# Plot spectral data and FPCA model
a0.plot(masked_wave, M67_GM_spec_masked[0], color='black',
        label='original spectrum', linewidth=1.5, marker='.', markersize=5)
a0.plot(masked_wave, spec_mean_dat + fit_data[0], color='orange',
        label='model spectrum (10 FPCs)', linewidth=1.5)

# Highlight regions
a0.axvspan(15269, 15272, alpha=0.3, color='skyblue', label='noise')
a0.axvspan(15314, 15317, alpha=0.8, color='lavender', label='mask')

a0.xaxis.set_major_locator(MultipleLocator(20))
a0.xaxis.set_minor_locator(MultipleLocator(5))
a0.yaxis.set_major_locator(MultipleLocator(0.1))
a0.yaxis.set_minor_locator(MultipleLocator(0.05))
a0.set_ylabel(r'f/$f_c(\lambda)$')
a0.legend(fontsize=15)

# Plot residuals
a1.plot(masked_wave, M67_GM_spec_masked[0] - (spec_mean_dat + fit_data[0]),
        color='black', linewidth=1, marker='.', markersize=4)

a1.axvspan(15269.5, 15271.5, alpha=0.3, color='skyblue')
a1.axvspan(15314.5, 15316.5, alpha=0.8, color='lavender')
a1.axhspan(-0.005, 0.005, alpha=0.25, color='dimgray', label='APOGEE base uncertainty')

a1.set_xlim(masked_wave[500], masked_wave[1000])
a1.set_ylim(-0.06, 0.06)
a1.set_xticks(np.arange(15260, 15370, 20))
a1.set_xticklabels(np.arange(15260, 15370, 20))
a1.yaxis.set_major_locator(MultipleLocator(0.05))
a1.yaxis.set_minor_locator(MultipleLocator(0.025))
a1.set_xlabel(r'Wavelength $\lambda$($\AA$)')
a1.set_ylabel(r'Residuals')

a1.legend(loc='lower right', fontsize=15)
f.align_ylabels()

plt.savefig('specdims/data/fpca_model.png')
# -------------------------------------------------
