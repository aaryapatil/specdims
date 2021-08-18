import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from apogee.tools.path import change_dr
from apogee.tools import bitmask as bm

from delfiSpec import util, specproc


# Set matplotlib parameters
mpl.rcParams["axes.labelsize"] = 26
mpl.rcParams['xtick.labelsize']= 20
mpl.rcParams['ytick.labelsize']= 20


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

# ------------------- APOGEE Spectral Sample -------------------
fig, ax = plt.subplots(figsize=(8, 6))

s = 16  # size of points
im = ax.scatter(apogee_cat_cut['TEFF'], apogee_cat_cut['LOGG'],
                c=apogee_cat_cut['FE_H'], s=s, cmap=plt.cm.viridis)

# Set limits and labels
ax.set_xlim([5650, 3350])
ax.set_ylim([4, 0])
ax.set_xlabel(r'$T_{eff}$ (K)')
ax.set_ylabel(r'$\log\,\;g$')

# Major and minor labels
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# Add a colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.ax.get_yaxis().set_ticks([-0.10, 0.00, 0.10])
cbar.ax.get_yaxis().set_ticklabels([-0.10, 0.00, 0.10])
cbar.ax.get_yaxis().set_minor_locator(MultipleLocator(0.025))
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('[Fe/H]', rotation=270)

# Save APOGEE sample figure
plt.savefig('data/APOGEE_sample.png')

# ------------------- M67 Spectral Sample -------------------

fig, ax = plt.subplots(figsize=(8, 6))

s = 70 # size of points
im = ax.scatter(M67_GM_apogee['TEFF'], M67_GM_apogee['LOGG'],
                c=M67_GM_apogee['FE_H'], s=s, cmap=plt.cm.viridis)

# Set limits and labels
ax.set_xlim([5650, 3750])
ax.set_ylim([4, 1.5])
ax.set_xlabel(r'$T_{eff}$ (K)')
ax.set_ylabel(r'$\log\,\;g$')

# Set major and minor ticks
ax.xaxis.set_major_locator(MultipleLocator(500))
ax.xaxis.set_minor_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_minor_locator(MultipleLocator(0.1))

# Add a colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.ax.get_yaxis().set_ticks([-0.05, 0, 0.05, 0.10])
cbar.ax.get_yaxis().set_ticklabels([-0.05, 0, 0.05, 0.10])
cbar.ax.get_yaxis().set_minor_locator(MultipleLocator(0.025))
cbar.ax.get_yaxis().labelpad = 25
cbar.ax.set_ylabel('[Fe/H]', rotation=270)

# Save M67 sample figure
plt.savefig('data/M67_sample.png')