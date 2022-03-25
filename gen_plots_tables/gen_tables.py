import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column
from apogee.tools.path import change_dr

from delfiSpec import util


# ------- FPCs and EM PCs Correlations with theoretical PCs -------
eigenfun_dat = np.loadtxt('specdims/data/FPCA_apogee/fpca_dat_eigenvectors_psi_t.dat')
eigenvec_dat = np.loadtxt('specdims/data/EMPCA_apogee/empca_dat_eigenvectors_PCs.dat')
eigenvec_sim = np.loadtxt('specdims/data/PCA_sim/pca_sim_eigenvectors_PCs.dat')

for i in range(10):
    print(r'$PC_{}$ & {:.1f} & {:.1f} \\'.format(
        i+1, 100*np.abs(np.corrcoef(eigenfun_dat[:, i], eigenvec_sim[i])[0][1]),
        100*np.abs(np.corrcoef(eigenvec_dat[i], eigenvec_sim[i])[0][1])))


# --------------------- Hierarchical Modeling ---------------------
for ele in ['C', 'N', 'O', 'Na', 'Mg', 'Al', 'Si', 'S', 'K', 'Ca', 'Ti', 'V', 'Mn', 'Ni', 'Fe']:
    posterior_t = np.loadtxt(f'specdims/data/hierarch/{ele}_H_hierarch_t.dat')

    intervals_t = t.interval(0.68, posterior_t[50:, 0],
                             loc=posterior_t[50:, 1], scale=posterior_t[50:, 2])
    perc_t = np.percentile(np.abs(intervals_t[0] - posterior_t[50:, 1]),
                           [16, 50, 84])

    q_t = np.diff(perc_t)
    txt_t = "{0:.3f}_{{-{1:.3f}}}^{{+{2:.3f}}}"
    txt_t = txt_t.format(perc_t[1], q_t[0], q_t[1])
    print(ele + ' & $' + txt_t + '$ \\\\')


# ------------------ APPENDIX table of abundances ------------------
# Read APOGEE DR14 catalogue
change_dr('14')
apCat = util.ApogeeCat()

# Read M67 GM spectra
_, M67_GM_apogee = apCat.read_OCCAM_cluster()

# Read posteriors
M67_constrain = np.zeros(shape=(28, 100000, 17))

for ind in np.arange(28):
    M67_constrain[ind] = np.loadtxt(f'specdims/data/posteriors_M67_{ind}.dat')

# Labels for columns
labels = [r'$T_{eff}$', r'log $g$', r'$[$C/H$]$', r'$[$N/H$]$', r'$[$O/H$]$', r'$[$Na/H$]$', r'$[$Mg/H$]$', r'$[$Al/H$]$',
          r'$[$Si/H$]$', r'$[$S/H$]$', r'$[$K/H$]$', r'$[$Ca/H$]$', r'$[$Ti/H$]$', r'$[$V/H$]$', r'$[$Mn/H$]$', r'$[$Ni/H$]$', r'$[$Fe/H$]$']

# Table with median values of abundances except FE/H
appendix_table = Table(np.around(np.median(M67_constrain[:, :, 2:-1], axis=1), decimals=2),
                       names=labels[2:-1])

# Add ID Column at the start
col_id = Column(name='2Mass ID', data=M67_GM_apogee['APOGEE_ID'])
appendix_table.add_column(col_id, 0)

# Add [FE/H] Column at second position
col_fe = Column(name=labels[-1], data=np.around(np.median(M67_constrain[:, :, -1], axis=1),
                                                decimals=2))
appendix_table.add_column(col_fe, 1)

# Table with median along with uncertainties based on 16th, 50th, 84th percentiles
for ind, lb in enumerate(labels[2:]):
    perc = np.percentile(M67_constrain[:, :, ind+2], [25, 50, 75], axis=1)
    q = np.diff(perc, axis=0)
    txt = "{0:.2f}_{{-{1:.2f}}}^{{+{2:.2f}}}"
    appendix_table[lb] = [txt.format(perc[1][star], q[0][star], q[1][star]) for star in range(28)]

print(ascii.write(appendix_table, format='latex'))