"""Figure 6 (Section 3.3): clustering coefficient C against network size, 29 countries; only
extreme or outlying countries are labeled (style after Espejo et al. 2018, Fig. 6).

Input:  results/descriptive/descriptive_metrics.csv (descriptive_metrics.py)
Output: figures/descriptive_clustering_vs_n.png
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
from scipy import stats

df = pd.read_csv(RESULTS_DIR / 'descriptive' / 'descriptive_metrics.csv')
n = df['n'].values.astype(float)
BLUE, ORANGE, GREY = '#0033CC', '#D9480F', '#6B7280'
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.labelweight': 'bold'})

def n_axis(ax):
    ax.set_xscale('log'); ax.set_xlabel('number of nodes $n$')
    ax.grid(alpha=0.35, which='major'); ax.grid(alpha=0.15, which='minor', ls=':')

POS = {'r': ((7, 0), 'left', 'center'), 'l': ((-7, 0), 'right', 'center'),
       'u': ((0, 8), 'center', 'bottom'), 'd': ((0, -9), 'center', 'top')}

def label(ax, x, y, spec, color):
    """spec: {country code: side}; label placed directly next to the marker, as in Espejo et al."""
    code2i = {c: i for i, c in enumerate(df['country'])}
    for c, side in spec.items():
        i = code2i[c]; off, ha, va = POS[side]
        ax.annotate(c, (x[i], y[i]), xytext=off, textcoords='offset points', ha=ha, va=va,
                    fontsize=11, fontweight='bold', color=color)

# ---------- (3) clustering ----------
fig, ax = plt.subplots(figsize=(7.5, 5.2))
y = df['clustering'].values.astype(float)
ax.scatter(n, y, s=85, marker='^', color=BLUE, edgecolor='black', linewidth=0.5, zorder=3)
label(ax, n, y, {'AL': 'l', 'HR': 'l', 'LV': 'l', 'LT': 'r', 'PT': 'r', 'AT': 'd', 'RO': 'd', 'NO': 'd', 'IT': 'u', 'DE': 'r', 'ES': 'r', 'FR': 'r'}, BLUE)
ax.set_xlim(15, 2200)
r_c, p_c = stats.pearsonr(n, y)
n_axis(ax); ax.set_ylabel('clustering coefficient $C$'); ax.set_ylim(bottom=-0.013)
print(f'clustering: r={r_c:.2f} p={p_c:.2f}')
plt.tight_layout(); plt.savefig(FIGURES_DIR / 'descriptive_clustering_vs_n.png', dpi=200, bbox_inches='tight'); plt.close()
print('written')
