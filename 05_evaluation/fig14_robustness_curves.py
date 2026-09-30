"""Figure 14 (Section 5.3): edge-removal curves S(f) of Bosnia and Herzegovina (n = 39), Germany
(n = 780) and Spain (n = 1066): real grid (black) and 30 synthetic samples (mean +- SD).
Left: random edge failure, real curve averaged over 20 random orders. Right: targeted edge
attack by static edge betweenness. Also prints the largest single-removal drops of S under
attack quoted in Section 5.3.

Input:  results/evaluation/robustness_curves.pkl (robustness_curves.py)
Output: figures/robustness_edge_BA_DE_ES.png
"""
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

ROWS = [('BA', 'Bosnia and Herzegovina'), ('DE', 'Germany'), ('ES', 'Spain')]

if __name__ == '__main__':
    with open(RESULTS_DIR / 'evaluation' / 'robustness_curves.pkl', 'rb') as fh:
        C = pickle.load(fh)
    f = np.linspace(0, 1, 200); m = len(f)
    interp = lambda y: np.interp(f, np.arange(len(y)) / len(y), y)
    fig, axes = plt.subplots(3, 2, figsize=(13, 14), sharey=True, sharex=True)
    for row, (code, name) in enumerate(ROWS):
        c = C[code]
        for col, (key_g, key_s, color, lab) in enumerate([('g_r', 's_r', '#D97706', 'random edge failure'),
                                                          ('g_a', 's_a', '#2563EB', 'targeted edge attack')]):
            ax = axes[row, col]
            S = np.array([interp(y) for y in c[key_s]]); mu, sd = S.mean(0), S.std(0)
            pts = np.linspace(0, m - 1, 24).astype(int)
            ax.plot(f, interp(c[key_g]), color='black', lw=2, label='real network')
            ax.plot(f, mu, color=color, lw=1, ls='-.', alpha=0.8)
            ax.errorbar(f[pts], mu[pts], yerr=sd[pts], fmt='o', color=color, ms=6, lw=1.2,
                        label='synthetic (mean $\\pm$ SD, 30 samples)')
            ax.set_title(f'({"abcdef"[2 * row + col]}) {name} ($n={c["n"]}$), {lab}', fontsize=12)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.03); ax.grid(alpha=0.3)
            ax.legend(frameon=False, fontsize=10, loc='upper right')
            if row == 2:
                ax.set_xlabel('fraction of removed edges $f$')
        axes[row, 0].set_ylabel('$S(f)$')
    plt.tight_layout()
    plt.savefig(ensure(FIGURES_DIR) / 'robustness_edge_BA_DE_ES.png', dpi=200, bbox_inches='tight')
    for code in ['DE', 'ES']:
        g = np.asarray(C[code]['g_a']); d = g[:-1] - g[1:]; k = int(np.argmax(d))
        print(f'{code}, real grid under attack: largest drop at removal {k + 1} of {len(g)} '
              f'(f = {(k + 1) / len(g):.3f}): S {g[k]:.3f} -> {g[k + 1]:.3f}')
