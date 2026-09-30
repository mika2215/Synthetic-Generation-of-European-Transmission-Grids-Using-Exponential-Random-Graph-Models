"""Figure 5 (Section 3.3): degree distribution P(k) of the 29 countries with n >= 20, three panels
by network size, semi-logarithmic axes. Dashed: exponential fit P(k) = alpha exp(-beta k) to the
pooled degrees of all 29 countries (log-linear least squares; alpha = 1.21, beta = 0.71).
Degrees with no node are omitted, and the line is broken there.

Output: figures/descriptive_degree_distribution.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import N_MIN, build_country_graph, layer, load_tables
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

ensure(FIGURES_DIR)
import numpy as np
import matplotlib.pyplot as plt

graphs = {}
for c in sorted(load_tables()[0]['country'].unique()):
    G = build_country_graph(c)
    if G.number_of_nodes() >= N_MIN:
        graphs[c] = G
order = sorted(graphs, key=lambda c: graphs[c].number_of_nodes())
KMAX = 10

pooled = np.concatenate([[d for _, d in graphs[c].degree()] for c in order])
cnt = np.bincount(pooled)[1:]; kk = np.arange(1, len(cnt) + 1); p = cnt / cnt.sum()
b, loga = np.polyfit(kk[p > 0], np.log(p[p > 0]), 1)
alpha, beta = np.exp(loga), -b
print(f'pooled exponential fit: alpha={alpha:.2f}, beta={beta:.2f}; k_max={pooled.max()}')

groups = [order[:10], order[10:20], order[20:]]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
ks = np.arange(1, KMAX + 1)
for ax, grp in zip(axes, groups):
    for c in grp:
        deg = np.array([d for _, d in graphs[c].degree()])
        pk = np.array([(deg == k).mean() for k in ks])
        pk[pk == 0] = np.nan
        ax.plot(ks, pk, 'o-', ms=3.5, lw=1, label=f'{c} ({len(deg)})')
    ax.plot(ks, alpha * np.exp(-beta * ks), '--', color='black', lw=1.2,
            label=f'fit $\\alpha e^{{-\\beta k}}$' if grp is groups[0] else None)
    ax.set_yscale('log'); ax.set_xticks(ks); ax.set_xlim(0.6, KMAX + 0.4)
    ax.set_xlabel('degree $k$'); ax.grid(alpha=0.3, which='both')
    n_lo, n_hi = graphs[grp[0]].number_of_nodes(), graphs[grp[-1]].number_of_nodes()
    ax.set_title(f'$n={n_lo}$ to ${n_hi}$', fontsize=11)
    ax.legend(fontsize=7, ncol=2, frameon=False, loc='lower left')
axes[0].set_ylabel('fraction of nodes $P(k)$')
plt.tight_layout()
out = FIGURES_DIR / 'descriptive_degree_distribution.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('written', out)
