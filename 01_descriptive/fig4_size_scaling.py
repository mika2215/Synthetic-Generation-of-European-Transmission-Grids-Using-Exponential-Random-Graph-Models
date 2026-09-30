"""Figure 4 (Section 3.3): number of edges m against number of nodes n, 29 countries, with the
least-squares line through the origin (m = 1.28 n, R^2 = 0.997), compared in the text with
Espejo et al. (2018).

Input:  results/descriptive/descriptive_metrics.csv (descriptive_metrics.py)
Output: figures/descriptive_size_scaling.png
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
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv(RESULTS_DIR / 'descriptive' / 'descriptive_metrics.csv')
n, m = df['n'].values.astype(float), df['m'].values.astype(float)
a = (n @ m) / (n @ n)
r2 = 1 - ((m - a * n) ** 2).sum() / ((m - m.mean()) ** 2).sum()
print(f'm = {a:.3f} n, R2 = {r2:.4f}')
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.labelweight': 'bold'})
fig, ax = plt.subplots(figsize=(7.5, 5.2))
xx = np.array([0, n.max() * 1.05])
ax.plot(xx, a * xx, '-', color='#6B7280', lw=1.3, label=f'$m={a:.2f}\\,n$ ($R^2={r2:.3f}$)')
ax.scatter(n, m, s=85, color='#0033CC', edgecolor='black', linewidth=0.5, zorder=3)
for c, side in {'FR': 'l', 'ES': 'l', 'DE': 'l', 'IT': 'l', 'GB': 'r', 'PT': 'l'}.items():
    i = int(np.where(df['country'] == c)[0][0])
    off, ha = ((-8, 0), 'right') if side == 'l' else ((8, 0), 'left')
    ax.annotate(c, (n[i], m[i]), xytext=off, textcoords='offset points', ha=ha, va='center',
                fontsize=11, fontweight='bold', color='#0033CC')
ax.set_xlabel('number of nodes $n$'); ax.set_ylabel('number of edges $m$')
ax.set_xlim(0, 1300); ax.set_ylim(0, 1750); ax.grid(alpha=0.35)
ax.legend(frameon=False, fontsize=10, loc='upper left')
plt.tight_layout(); plt.savefig(FIGURES_DIR / 'descriptive_size_scaling.png', dpi=200, bbox_inches='tight')
print('written')
