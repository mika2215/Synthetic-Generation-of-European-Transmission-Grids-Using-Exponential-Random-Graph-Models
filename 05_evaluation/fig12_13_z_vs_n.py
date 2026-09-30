"""Figures 12 and 13 (Sections 5.2.2, 5.3): standardized deviation z of the held-out statistics of
the final model against network size, 30 samples per country.
Figure 12 (fig_heldout_z_vs_n): structure (<l>, diameter, max. betweenness, lambda_2: global;
assortativity: local). Figure 13 (fig_robustness_z_vs_n): robustness index R.
Shaded regions: small (n <= 67), medium (113-246) and large (>= 572) grids; green band |z| < 1.
Also prints the Spearman rank correlations quoted in the captions.

Input:  results/evaluation/heldout_per_statistic.csv (evaluate_heldout.py)
Output: figures/fig_heldout_z_vs_n.pdf/.png, figures/fig_robustness_z_vs_n.pdf/.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

ensure(FIGURES_DIR)
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

DATA = RESULTS_DIR / 'evaluation' / 'heldout_per_statistic.csv'
OUT = FIGURES_DIR

PANEL_A = [("avg_path_len", r"$\langle l\rangle$", "o"),
           ("diameter", "diameter", "s"),
           ("betw_max", "max. betweenness", "^"),
           ("lambda2", r"$\lambda_2$", "D"),
           ("assortativity", "assortativity", "v")]
PANEL_B = [("R_edge_random", "$R$, random failure", "o"),
           ("R_edge_attack", "$R$, targeted attack", "s")]
# Size groups used in Chapter 5 (shaded regions): small n<=67, medium 113-246, large >=572
GROUPS = [((20, 80), "0.95", 38, "small"),
          ((100, 280), "0.90", 165, "medium"),
          ((480, 1300), "0.85", 800, "large")]


def style(ax, ytext):
    for (x0, x1), col, xt, label in GROUPS:
        ax.axvspan(x0, x1, color=col)
        ax.text(xt, ytext, label, ha="center", fontsize=8, color="0.3")
    ax.axhline(0, color="k", lw=0.6)
    ax.axhspan(-1, 1, color="tab:green", alpha=0.08)
    ax.set_xscale("log")
    ax.set_xlim(20, 1300)
    ax.set_xlabel("number of nodes $n$")
    ax.spines[["top", "right"]].set_visible(False)


def plot_panel(p, panel, ylim, ytext, legend_anchor, name):
    """One single-panel figure; same style as the former two-panel version."""
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    style(ax, ytext=ytext)
    for stat, label, marker in panel:
        ax.plot(p["n"], p[stat], marker=marker, ms=4, lw=0.9, label=label)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=legend_anchor)
    ax.set_ylabel("standardized deviation $z$")
    ax.set_ylim(*ylim)
    plt.tight_layout()
    OUT.mkdir(exist_ok=True)
    plt.savefig(OUT / f"{name}.pdf")
    plt.savefig(OUT / f"{name}.png", dpi=180)
    plt.close(fig)


def main():
    a = pd.read_csv(DATA)
    p = (a[a["model"] == "final"]
         .pivot_table(index=["country", "n"], columns="stat", values="z")
         .reset_index().sort_values("n"))

    # Split into two figures (2026-09-29): structure in Sec. 5.2.2, robustness in Sec. 5.3.
    # Structure keeps the former common scale; robustness gets its own, narrower scale.
    plot_panel(p, PANEL_A, (-17, 7), 5.8, (0.0, 0.35), "fig_heldout_z_vs_n")
    plot_panel(p, PANEL_B, (-2, 7), 6.3, (0.0, 0.75), "fig_robustness_z_vs_n")

    # Spearman rank correlations quoted in the captions
    for stat, label, _ in PANEL_A + PANEL_B:
        r, pval = spearmanr(p["n"], p[stat], nan_policy="omit")
        print(f"{stat:15s} rho = {r:+.2f}, p = {pval:.3g}")


if __name__ == "__main__":
    main()
