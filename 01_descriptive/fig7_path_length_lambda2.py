"""Figure 7 (Section 3.3): (a) average path length and diameter, (b) algebraic connectivity,
both against network size (log-log), 29 countries; only extreme or outlying countries labeled.

Input:  results/descriptive/descriptive_metrics.csv (descriptive_metrics.py)
Output: figures/descriptive_pathlength_lambda2.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import N_MIN, build_country_graph, layer, load_tables
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

ensure(FIGURES_DIR)
import numpy as np, pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv(RESULTS_DIR / 'descriptive' / 'descriptive_metrics.csv')
n = df['n'].values.astype(float)
BLUE, ORANGE = '#0033CC', '#D9480F'
plt.rcParams.update({'font.size': 15, 'axes.labelsize': 16, 'axes.labelweight': 'bold'})
POS = {'r': ((7, 0), 'left', 'center'), 'l': ((-7, 0), 'right', 'center'),
       'u': ((0, 8), 'center', 'bottom'), 'd': ((0, -9), 'center', 'top')}
def label(ax, x, y, spec, color):
    idx = {c: i for i, c in enumerate(df['country'])}
    for c, side in spec.items():
        off, ha, va = POS[side]
        ax.annotate(c, (x[idx[c]], y[idx[c]]), xytext=off, textcoords='offset points', ha=ha, va=va,
                    fontsize=13, fontweight='bold', color=color)
def n_axis(ax):
    ax.set_xscale('log'); ax.set_xlabel('number of nodes $n$'); ax.set_xlim(15, 2200)
    ax.grid(alpha=0.35, which='major'); ax.grid(alpha=0.15, which='minor', ls=':')
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.6))
l, d = df['avg_path_len'].values.astype(float), df['diameter'].values.astype(float)
a1.scatter(n, d, s=110, marker='^', color=ORANGE, edgecolor='black', linewidth=0.5, zorder=3, label='diameter')
a1.scatter(n, l, s=110, marker='o', color=BLUE, edgecolor='black', linewidth=0.5, zorder=3,
           label='average path length $\\langle l\\rangle$')
label(a1, n, d, {'AL': 'l', 'BG': 'u', 'PL': 'u', 'NO': 'r', 'GB': 'r', 'IT': 'r', 'DE': 'r', 'ES': 'u', 'FR': 'r'}, ORANGE)
label(a1, n, l, {'AL': 'l', 'NO': 'r', 'GB': 'r', 'FR': 'r'}, BLUE)
n_axis(a1); a1.set_yscale('log'); a1.set_yticks([3, 5, 10, 20, 50]); a1.set_yticklabels(['3', '5', '10', '20', '50'])
a1.set_ylabel('graph distance (hops)'); a1.legend(frameon=False, fontsize=13, loc='upper left')
a1.set_title('(a) average path length and diameter', fontsize=16)
y = df['lambda2'].values.astype(float)
a2.scatter(n, y, s=110, marker='o', color=BLUE, edgecolor='black', linewidth=0.5, zorder=3)
label(a2, n, y, {'AL': 'l', 'HR': 'r', 'CZ': 'r', 'NL': 'l', 'BE': 'r', 'AT': 'd', 'PT': 'u', 'PL': 'l',
                 'NO': 'r', 'GB': 'l', 'DE': 'u', 'FR': 'r'}, BLUE)
n_axis(a2); a2.set_yscale('log'); a2.set_ylabel('algebraic connectivity $\\lambda_2$')
a2.set_title('(b) algebraic connectivity', fontsize=16)
plt.tight_layout(w_pad=3)
plt.savefig(FIGURES_DIR / 'descriptive_pathlength_lambda2.png', dpi=200, bbox_inches='tight')
print('written')
