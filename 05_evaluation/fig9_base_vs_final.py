"""Figure 9 (Section 5.1): mean |z| over the seven held-out statistics, base model vs. final
model, 15 countries (mean betweenness is omitted: for fixed n it is an affine function of the
average path length and has the same z). Also prints the numbers quoted in the text.

Input:  results/evaluation/heldout_per_statistic.csv (evaluate_heldout.py)
Output: figures/fig_ablation_heldout_z.pdf/.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

ensure(FIGURES_DIR)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATA = RESULTS_DIR / 'evaluation' / 'heldout_per_statistic.csv'
OUT = FIGURES_DIR

# Seven held-out statistics (mean betweenness is omitted: for fixed n it is an
# affine function of the average path length and has the same z).
STATS = ["assortativity", "avg_path_len", "diameter", "lambda2",
         "betw_max", "R_edge_random", "R_edge_attack"]


def main():
    a = pd.read_csv(DATA)
    q = a[a["stat"].isin(STATS)].copy()
    q["absz"] = q["z"].abs()
    m = (q.groupby(["country", "n", "model"])["absz"].mean()
           .unstack().reset_index().sort_values("n", ascending=False))

    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    y = np.arange(len(m))
    h = 0.38
    ax.barh(y + h / 2, m["base"], h, color="#c0513a",
            label="Base model ($m_{LL}$, $m_{LH}$, $m_{HH}$)")
    ax.barh(y - h / 2, m["final"], h, color="#2f6b8f",
            label="Final model (+ AKS, $t_1$, $t_2$)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c} (n={n})" for c, n in zip(m["country"], m["n"])])
    ax.set_xlabel("Mean $|z|$ over seven held-out statistics")
    ax.axvline(1, color="grey", lw=0.8, ls=":")
    ax.legend(loc="lower right", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    OUT.mkdir(exist_ok=True)
    plt.savefig(OUT / "fig_ablation_heldout_z.pdf")
    plt.savefig(OUT / "fig_ablation_heldout_z.png", dpi=200)

    # Numbers quoted in the text
    m["final_better"] = m["final"] < m["base"]
    print(m.round(2).to_string(index=False))
    print("final model better in", int(m["final_better"].sum()), "of", len(m), "countries")
    per_stat = (q.pivot_table(index=["country", "stat"], columns="model", values="absz")
                  .reset_index())
    per_stat["better"] = per_stat["final"] < per_stat["base"]
    print(per_stat.groupby("stat")["better"].sum())
    print("|z|>2: final", int((per_stat["final"] > 2).sum()),
          "base", int((per_stat["base"] > 2).sum()), "of", len(per_stat))


if __name__ == "__main__":
    main()
