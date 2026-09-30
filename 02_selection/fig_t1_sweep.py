"""Figure 8 (Section 5.1): mean simulated triangle count t1(G) against beta_t1, swept around the
fitted value, with t2 active (blue) and with beta_t2 = 0 (red), for the eight countries of the
stability test. Symmetric-log y axis (linear below 1).

Input:  results/selection/t1_sweep_thesis.csv (thesis run); use --input for the output of
        stability_tests.py (results/selection/t1_sweep.csv)
Output: figures/model_t1_cliff_sweep_symlog.png
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

COUNTRIES = ['HR', 'BA', 'SK', 'CZ', 'AT', 'BG', 'PT', 'CH']

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--input', type=Path, default=RESULTS_DIR / 'selection' / 't1_sweep_thesis.csv')
    df = pd.read_csv(ap.parse_args().input)
    n_of = pd.read_csv(RESULTS_DIR / 'tables' / 'table5_beta_bar.csv').set_index('country')['n']
    fig, axes = plt.subplots(2, 4, figsize=(15.9, 7.4))
    for k, (ax, c) in enumerate(zip(axes.flat, COUNTRIES)):
        sub = df[df.country == c]
        for variant, style in [('without_t2', dict(color='#c8553d', ls='--', lw=3, label='without $t_2$')),
                               ('with_t2', dict(color='#2c6a8f', ls='-', lw=1.5,
                                                label='with $t_2$ (production)'))]:
            g = sub[sub.variant == variant].groupby('beta_t1')['t1'].mean()
            ax.plot(g.index, g.values, marker='o', ms=3.5, **style)
        ax.axvline(sub.beta_t1_fit.iloc[0], color='grey', ls=':', lw=1, label='fitted $\\beta_{t_1}$')
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.3)
        ax.set_title(f'({"abcdefgh"[k]}) {c} ($n={n_of[c]}$)', fontsize=10)
        ax.set_xlabel('$\\beta_{t_1}$'); ax.set_ylabel('$t_1(G)$, simulated (log scale)')
    axes.flat[0].legend(fontsize=7, loc='upper left')
    fig.suptitle('Simulated triangle count $t_1(G)$ vs. $\\beta_{t_1}$, with and without $t_2$',
                 fontweight='bold')
    plt.tight_layout()
    plt.savefig(ensure(FIGURES_DIR) / 'model_t1_cliff_sweep_symlog.png', dpi=200)
    print('written')
