import numpy as np
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from astropy.io import fits
from apogee.tools import elemIndx, paramIndx, path
from apogee.tools import bitmask as bm

from delfiSpec import util, specproc


# ------------- Read M67 information -------------
# Read APOGEE DR14 catalogue
path.change_dr('14')
apCat = util.ApogeeCat()

# Read M67 cluster APOGEE spectra    
_, M67_GM_apogee = apCat.read_OCCAM_cluster()
M67_GM_spectra = apCat.read_allStar_spectra(apogee_cat_cut=M67_GM_apogee)

# Preprocess M67 GM spectra
badcombpixmask = bm.badpixmask()
pix_err = np.array([bm.apogee_pixmask_int("SIG_SKYLINE"), bm.apogee_pixmask_int("SIG_TELLURIC"),
                    bm.apogee_pixmask_int("PERSIST_HIGH"), bm.apogee_pixmask_int("PERSIST_MED"),
                    bm.apogee_pixmask_int("PERSIST_LOW")])
badcombpixmask += np.sum(2**pix_err)
M67_GM_specproc, M67_GM_specerr, M67_GM_specweight = specproc.process_spectra(
    spectra_info=M67_GM_spectra, badcombpixmask=badcombpixmask)

# Get stellar parameters and abundances for M67 from APOGEE spectroscopic fits
indexArrays = fits.getdata(path.allStarPath(), 3)

# Elements used
elems = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K',
         'Ca', 'Ti', 'V', 'Mn', 'Ni', 'Fe']

truths = np.zeros((28, 17))
truths_error = np.zeros((28, 17))
X_H = np.zeros(len(elems))
X_H_err = np.zeros(len(elems))

for star_ind, star in enumerate(M67_GM_apogee):    
    for j, ele in enumerate(elems):
        ele_ind = elemIndx(ele)
        X_H[j] = star['FELEM'][ele_ind]
        X_H_err[j] = star['FELEM_ERR'][ele_ind]**2
        if not indexArrays['ELEMTOH'][0][ele_ind]:
            X_H[j] += star['FELEM'][elemIndx('Fe')]
            X_H_err[j] += star['FELEM_ERR'][elemIndx('Fe')]**2
        X_H_err[j] = np.sqrt(X_H_err[j])
            
    truths[star_ind] = np.concatenate(([star['FPARAM'][paramIndx('Teff')],
                                        star['FPARAM'][paramIndx('logg')]], X_H))
    truths_error[star_ind] = np.concatenate(([np.diag(star['FPARAM_COV'])[paramIndx('Teff')],
                                              np.diag(star['FPARAM_COV'])[paramIndx('logg')]], X_H_err))


# ------------- Read M67 posteriors -------------
M67_constrain = np.zeros(shape=(28, 100000, 17))

for ind in np.arange(28):
    M67_constrain[ind] = np.loadtxt(f'data/posteriors_M67_{ind}.dat')


def lkpost(ele_no):
    '''
    Posterior samples in a format understandable by violinplots
    '''
    return [M67_constrain[i][:, ele_no] for i in range(len(M67_constrain))]


# Signal-to-Noise Ratio
M67_GM_APOGEE_SNR = M67_GM_apogee['SNR']

# Percentage masking of data
M67_perc_masked = np.sum(M67_GM_specweight==0, axis=1)/7214*100

# Elements of interest
plot_ele = np.array([-1, -3, 8, 7, 6, 4, 3, 2])

# ------------- Plot M67 posteriors -------------
fig, ax = plt.subplots(nrows=10, ncols=1, figsize=(22, 27), sharex = True)
fig.tight_layout(h_pad=0, pad=0)

ax[0].bar(np.arange(1, 29), M67_GM_APOGEE_SNR, width=0.25, color='black')

ax[0].set_ylim([0, 2000])
ax[0].set_ylabel('SNR', fontsize=26)
ax[0].yaxis.set_major_locator(MultipleLocator(1000))
ax[0].yaxis.set_minor_locator(MultipleLocator(250))

ax[1].bar(np.arange(1, 29), M67_perc_masked, width=0.25, color='darkgrey')

ax[1].set_ylim([0, 65])
ax[1].set_ylabel('% masked pixels')
ax[1].yaxis.set_major_locator(MultipleLocator(30))
ax[1].yaxis.set_minor_locator(MultipleLocator(10))

labels = [r'$T_\mathrm{eff}$', r'$\log g$', 'C', 'N', 'O', 'Na', 'Mg',
          'Al', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Mn', 'Ni', 'Fe']
color = ['chocolate', 'darkgrey', 'orange', 'blue', 'red',
         'lightcoral', 'teal', 'green']

for ind, ele in enumerate(plot_ele):
    # Plot violins for posteriors
    parts = ax[ind+2].violinplot(lkpost(ele), showmeans=False, showmedians=False, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor(color[ind])
        pc.set_alpha(0.2)

    # Plot quantiles and medians
    quartile1, medians, quartile3 = np.percentile(M67_constrain[:, :, ele],
                                                  [16, 50, 84], axis=1)
    num_est = np.arange(1, len(medians) + 1)
    ax[ind+2].scatter(num_est, medians, marker='_', color=color[ind], s=30,
                      zorder=3, label='Patil et al. (2021; this paper)')
    ax[ind+2].scatter(num_est, quartile1, marker='_', color=color[ind], s=30, zorder=3)
    ax[ind+2].scatter(num_est, quartile3, marker='_', color=color[ind], s=30, zorder=3)
    ax[ind+2].vlines(num_est, quartile1, quartile3, color=color[ind], linewidth=1.5)    
    
    # Plot APOGEE FELEM
    ax[ind+2].axhline(0, -1, 1, color='silver', linestyle='--', alpha=0.5, linewidth=1.5)  
    ax[ind+2].errorbar(np.arange(1, 29) + 0.25, truths[:, ele], yerr=truths_error[:, ele],
                       color='k', label=r'APOGEE $FELEM$', fmt='o')

    ax[ind+2].set_ylabel(f'{labels[ele]}')
    ax[ind+2].set_ylim([-1.2, 1.2])
    ax[ind+2].set_yticks(np.linspace(-1, 1, 9))
    ax[ind+2].set_yticklabels([-1, '', '', '', 0, '', '', '', 1])
    ax[ind+2].legend(ncol=2, loc='lower right', fontsize=16)

ax[-1].set_xlim([0.1, 28.9])
ax[-1].xaxis.set_major_locator(MultipleLocator(2))
ax[-1].xaxis.set_minor_locator(MultipleLocator(1))
ax[-1].set_xticklabels(np.arange(0, 29, 2))
ax[-1].set_xlabel('M67 Giant Member')

fig.align_ylabels()