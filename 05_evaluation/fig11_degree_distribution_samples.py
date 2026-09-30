"""Figure 11 (Section 5.2.2): degree distribution of the synthetic grids (boxplots over all
retained samples, whiskers without outliers) against the real grid (red line), 15 countries
sorted by n; the last bin collects all higher degrees.

Input:  results/evaluation/degree_distribution_<CC>.npz (degree_distribution_samples.py)
Output: figures/generalization_degree_gof.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import MODELED_COUNTRIES
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

if __name__ == '__main__':
    n_of = pd.read_csv(RESULTS_DIR / 'tables' / 'table5_beta_bar.csv').set_index('country')['n']
    fig, axes = plt.subplots(3, 5, figsize=(20, 11))
    for ax, c in zip(axes.flat, MODELED_COUNTRIES):
        z = np.load(RESULTS_DIR / 'evaluation' / f'degree_distribution_{c}.npz')
        obs, sim = z['observed'], z['samples']; dmax = len(obs) - 1; bins = np.arange(dmax + 1)
        ax.boxplot([sim[:, k] for k in bins], positions=bins, widths=0.6, showfliers=False,
                   patch_artist=True, boxprops=dict(facecolor='lightgrey', linewidth=0.5),
                   medianprops=dict(color='grey', linewidth=0.8),
                   whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5))
        ax.plot(bins, obs, 'r-', lw=1.3, label='observed')
        ax.set_xticks(bins); ax.set_xticklabels([str(k) for k in bins])
        ax.set_xlim(-0.6, dmax + 0.6)
        ax.tick_params(labelsize=7); ax.set_ylim(bottom=0)
        ax.set_title(f'{c} ($n={n_of[c]}$)', fontsize=10)
    axes.flat[0].legend(fontsize=8, loc='upper right')
    fig.supxlabel('degree $k$'); fig.supylabel('fraction of nodes $P(k)$')
    plt.tight_layout(rect=[0.015, 0.02, 1, 1])
    plt.savefig(ensure(FIGURES_DIR) / 'generalization_degree_gof.png', dpi=160, bbox_inches='tight')
    print('written')
