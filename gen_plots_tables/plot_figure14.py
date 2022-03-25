__author__      = "Aarya Patil"
__copyright__   = "Copyright 2021"

"""
Plot corner plot or posterior distributions of an M67 Giant Member.
These distributions are obtained using Sequential Neural Likelihood
with Functional Principal Component Analysis.
"""

import numpy as np
import seaborn as sns
import matplotlib as mpl
import pandas as pd
from apogee.tools.path import change_dr
from astropy.io import fits
from apogee.tools import path, elemIndx, paramIndx

from delfiSpec import util


# Set matplotlib plot parameters -- labels and ticks
mpl.rcParams["axes.labelsize"] = 38
mpl.rcParams['xtick.labelsize']= 24
mpl.rcParams['ytick.labelsize']= 24

# Read apogee DR14 catalogue
change_dr('14')
apCat = util.ApogeeCat()

# Read M67 GM spectra
M67_apogee_DM, M67_apogee_GM = apCat.read_OCCAM_cluster()

# Elements to be used
elems = ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K',
         'Ca', 'Ti', 'V', 'Mn', 'Ni', 'Fe']

# Get stellar parameters and abundances for M67 from APOGEE spectroscopic fits
indexArrays = fits.getdata(path.allStarPath(), 3)

# APOGEE fits
apogee_fits = np.zeros((28, 17))

# Abundances need to be converted to X/H first
X_H = np.zeros(len(elems))

for star_ind, star in enumerate(M67_apogee_GM):    
    for j, ele in enumerate(elems):
        # Element index in FELEM, FELEM_ERR
        ele_ind = elemIndx(ele)
        X_H[j] = star['FELEM'][ele_ind]
        # ELEMTOH is True when abundance is X/H, False if X/Fe
        if not indexArrays['ELEMTOH'][0][ele_ind]:
            X_H[j] += star['FELEM'][elemIndx('Fe')]
            
    apogee_fits[star_ind] = np.concatenate(([star['FPARAM'][paramIndx('Teff')],
                                             star['FPARAM'][paramIndx('logg')]], X_H))

giant_id = 11

# Posterior Distribution for M67 Giant Member 11
# Change the path to data according to directory structure
posterior = np.loadtxt(f'specdims/data/SNL/posteriors_M67_{giant_id}.dat')

# Labels for plots
labels = [r'$T_\mathrm{eff}$', r'$\log g$', 'C', 'N', 'O', 'Na', 'Mg',
          'Al', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Mn', 'Fe', 'Ni']

# Switch the order of Fe and Ni to maintain the ascending atomic number order
fe = np.array(posterior[:, -1], copy=True)
posterior[:, -1] = np.array(posterior[:, -2], copy=True)
posterior[:, -2] = fe
posterior_df = pd.DataFrame(posterior, columns=labels)

# Marginal distributions of elements of interest are highlighted using given colors
plot_ele = np.array([-2, 2, 6, 8, -3, 7, 3, 4])
color = ['chocolate', 'green', 'red', 'orange', 'grey', 'blue', 'teal', 'lightcoral']

# Create Grid
g = sns.PairGrid(posterior_df, diag_sharey=False, corner=True)

# Set axes limits and ticks
g.axes[0, 0].set_xlim([4400, 5300])
g.axes[1, 1].set_xlim([1.0, 3.8])
g.axes[0, 0].set_xticks([4600, 4850, 5100])
g.axes[0, 0].set_xticklabels([4600, '', 5100])
g.axes[1, 1].set_xticks([1.2, 1.8, 2.4, 3.0, 3.6])
g.axes[1, 1].set_xticklabels(['', 1.8, '', 3.0, ''])

g.axes[1, 0].set_ylim([1.0, 3.8])
g.axes[1, 0].set_yticks([1.2, 1.8, 2.4, 3.0, 3.6])
g.axes[1, 0].set_yticklabels(['', 1.8, '', 3.0, ''])

for ind in range(2, 17):
    if ind==3:
        g.axes[ind, ind].set_xlim([-0.4, 1.0])
        g.axes[ind, ind].set_xticks([-0.3, 0.0, 0.3, 0.6, 0.9])
        g.axes[ind, ind].set_xticklabels(['', 0.0, '', 0.6, ''])
        g.axes[ind, 0].set_ylim([-0.4, 1.0])
        g.axes[ind, 0].set_yticks([-0.3, 0.0, 0.3, 0.6, 0.9])
        g.axes[ind, 0].set_yticklabels(['', 0.0, '', 0.6, ''])
    else:
        g.axes[ind, ind].set_xlim([-0.7, 0.7])
        g.axes[ind, ind].set_xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        g.axes[ind, ind].set_xticklabels([-0.5, '', 0.0, '', 0.5])
        g.axes[ind, 0].set_ylim([-0.7, 0.7])
        g.axes[ind, 0].set_yticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        g.axes[ind, 0].set_yticklabels([-0.5, '', 0.0, '', 0.5])

# Padding between subplots
g.tight_layout(h_pad=0, w_pad=0, pad=0)
mpl.pyplot.subplots_adjust(hspace=0.01, wspace=0.01)

# Plot marginal distributions in the diagonal using KDE
g.map_diag(sns.kdeplot, lw=2, color='black')

# Highlight elements of interest in the marginals
for i in range(17):
    for ax in g.axes[i:, i]:
        ax.axvline(apogee_fits[giant_id][i], color='black', linestyle='--', lw=1)
    for ax in g.axes[i, :i]:
        ax.axhline(apogee_fits[giant_id][i], color='black', linestyle='--', lw=1)

for ind, ele in enumerate(plot_ele):
    sns.kdeplot(data=posterior[:, ele], ax=g.diag_axes[ele], color=color[ind], lw=6, alpha=0.2)

# Plot joint posteriors with contours -- mark the 68 and 95 % credible intervals
g.map_lower(sns.kdeplot, fill=True, levels=15, cmap='mako')
g.map_lower(sns.kdeplot, fill=False, levels=[1-0.95, 1-0.68], color='w', linewidths=0.75)

# Align y axis labels
g.fig.align_ylabels(g.axes[:, 0])

# Save figure
g.savefig('plots/M67_posterior_star.png')