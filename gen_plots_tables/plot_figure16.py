import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import t


mpl.rcParams["axes.labelsize"] = 28
mpl.rcParams['xtick.labelsize']= 20
mpl.rcParams['ytick.labelsize']= 20


def plot_hierarch(posterior, hist=False):
    # Labels for plots
    labels = [r'$\nu$', r'$\hat\mu$', r'$\hat\sigma$']
    # Remove burn-in
    posterior_df = pd.DataFrame(posterior[50:], columns=labels)
    
    g = sns.PairGrid(posterior_df, diag_sharey=False, corner=True)

    # Set limits, ticks and labels
    g.axes[0, 0].set_xlim([-1, 12])
    g.axes[1, 1].set_xlim([-0.15, 0.1])    
    g.axes[0, 0].set_xticks([0, 2.5, 5, 7.5, 10])
    g.axes[0, 0].set_xticklabels([0, '', 5, '', 10])
    g.axes[1, 1].set_xticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
    g.axes[1, 1].set_xticklabels(['', -0.1, '', 0, '', 0.1])
    g.axes[1, 0].set_ylim([-0.15, 0.1])
    g.axes[1, 0].set_yticks([-0.15, -0.1, -0.05, 0, 0.05, 0.1])
    g.axes[1, 0].set_yticklabels(['', -0.1, '', 0, '', 0.1])
    g.axes[2, 2].set_xlim([0, 0.1])
    g.axes[2, 2].set_xticks([0, 0.025, 0.05, 0.075, 0.1])
    g.axes[2, 2].set_xticklabels([0, '', 0.05, '', 0.1])
    g.axes[2, 0].set_ylim([0, 0.1])
    g.axes[2, 0].set_yticks([0, 0.025, 0.05, 0.075, 0.1])
    g.axes[2, 0].set_yticklabels([0, '', 0.05, '', 0.1])
    
    if hist:
        g.map_diag(sns.histplot)
    else:
        g.map_diag(sns.kdeplot, lw=3, color='black', fill=False)

    g.map_lower(sns.kdeplot, fill=True, levels=50, cmap='mako')
    g.map_lower(sns.kdeplot, fill=False, levels=[1-0.95, 1-0.68], color='w')
    
    g.fig.align_ylabels()
    
    for ind, ax in enumerate(g.diag_axes):
        quartile1, median, quartile3 = np.percentile(posterior[50:, ind], [16, 50, 84])
        ax.vlines(median, 0, ax.get_ylim()[1], color='teal', lw=2)
        ax.fill_between(np.array([quartile1, quartile3]), y1=0, y2=ax.get_ylim()[1], color='lightblue', alpha=0.3)
    
    plt.savefig('plots/Fe_H.png')


posterior = np.loadtxt('hierarch/Fe_H_hierarch_t_v1to10.dat')
plot_hierarch(posterior, hist=False)