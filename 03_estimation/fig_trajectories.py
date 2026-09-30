"""Trajectory figures of the EE fits: Figure 10 (Section 5.2.1) and Figures 15-16 (Appendix B).

Figure 10: raw trajectories of all six parameters for Germany, Portugal and Serbia; the dashed
line is the mean over the last quarter of the chain.
Figure 15: all 15 countries x 6 parameters, normalized as (beta(t) - m) / d, where m is the mean
over the last quarter and d = |m|, or the standard deviation of that window if it is larger
than |m| (avoids dividing by a near-zero reference); y range [-3, 3].
Figure 16: the same trajectories on each parameter's own scale.

Input: results/trajectories/trajectory_<CC>.npz (export_trajectories.py)
Output: figures/fitting_traceplot_curated.png, figures/fitting_traceplot_normalized.png,
        figures/fitting_traceplot_raw.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

LABELS = ['m_LL', 'm_LH', 'm_HH', 'akstar', 't1', 't2']
TEX_LABELS = ['$m_{LL}$', '$m_{LH}$', '$m_{HH}$', '$\\mathrm{AKS}(G)$', '$t_1(G)$', '$t_2(G)$']


def load(c):
    z = np.load(RESULTS_DIR / 'trajectories' / f'trajectory_{c}.npz')
    h = z['beta_hist']
    return h, z['beta_last_quarter'], np.linspace(0, 100, len(h))


def curated(n_of):
    fig, axes = plt.subplots(6, 3, figsize=(9, 11), sharex=True)
    for j, c in enumerate(['DE', 'PT', 'RS']):
        h, m, x = load(c)
        axes[0, j].set_title(f'{c} (n={n_of[c]})')
        for i in range(6):
            ax = axes[i, j]
            ax.plot(x, h[:, i], color='#2c6a8f', lw=0.6)
            ax.axhline(m[i], color='red', ls='--', lw=0.8)
            if j == 0:
                ax.set_ylabel(TEX_LABELS[i])
        axes[-1, j].set_xlabel('% fit phase')
    fig.suptitle('Fit-phase trajectories: Germany, Portugal, Serbia (raw scale)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'fitting_traceplot_curated.png', dpi=200)
    plt.close(fig)


def grid(n_of, normalized):
    fig, axes = plt.subplots(6, 15, figsize=(30, 9.6), sharex=True)
    for j, c in enumerate(MODELED_COUNTRIES):
        h, m, x = load(c)
        tail = h[int(0.75 * len(h)):]
        axes[0, j].set_title(f'{c}\n(n={n_of[c]})', fontsize=8)
        for i in range(6):
            ax = axes[i, j]
            if normalized:
                sd = tail[:, i].std()
                d = abs(m[i]) if abs(m[i]) >= sd else sd
                y = (h[:, i] - m[i]) / d if d > 0 else h[:, i] * 0
                ax.plot(x, y, color='#2563eb', lw=0.5)
                ax.axhline(0, color='red', ls=':', lw=0.8)
                ax.set_ylim(-3, 3)
            else:
                ax.plot(x, h[:, i], color='#16a34a', lw=0.5)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel(LABELS[i], fontsize=8)
        axes[-1, j].set_xlabel('% Fit-Phase', fontsize=6)
    plt.tight_layout()
    name = 'fitting_traceplot_normalized.png' if normalized else 'fitting_traceplot_raw.png'
    plt.savefig(FIGURES_DIR / name, dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    ensure(FIGURES_DIR)
    n_of = {c: build_country_graph(c).number_of_nodes() for c in MODELED_COUNTRIES}
    curated(n_of)
    grid(n_of, normalized=True)
    grid(n_of, normalized=False)
    print('written to', FIGURES_DIR)
