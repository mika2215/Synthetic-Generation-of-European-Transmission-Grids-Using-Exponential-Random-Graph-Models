"""Edge-removal curves S(f) and robustness index R, real grid and final model (Section 5.3).

Real grid: random failure averaged over 20 orders, targeted attack by static edge betweenness.
Synthetic grids: the 30-sample evaluation subset, 5 random orders each.

Outputs: results/evaluation/robustness_R.csv (R values; Table 7 is built from heldout_per_statistic.csv) and robustness_curves.pkl (Figure 14)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.heldout import REP_REAL, REP_SAMPLE, evaluation_subset
from common.paths import RESULTS_DIR, ensure
from common.robustness import attack_curve, random_failure_curve, robustness_index
from samples_io import load_samples

if __name__ == '__main__':
    rows, curves = [], {}
    for c in MODELED_COUNTRIES:
        G0 = build_country_graph(c)
        sub, _ = evaluation_subset(load_samples('final', c))
        g_r, g_a = random_failure_curve(G0, REP_REAL), attack_curve(G0)
        s_r = [random_failure_curve(G, REP_SAMPLE) for G in sub]; s_a = [attack_curve(G) for G in sub]
        curves[c] = dict(n=G0.number_of_nodes(), g_r=g_r, g_a=g_a, s_r=s_r, s_a=s_a)
        Rr = np.array([robustness_index(x) for x in s_r]); Ra = np.array([robustness_index(x) for x in s_a])
        rows.append(dict(country=c, n=G0.number_of_nodes(), m=G0.number_of_edges(),
                         R_real_random=robustness_index(g_r), R_sim_random=Rr.mean(), sd_random=Rr.std(),
                         R_real_attack=robustness_index(g_a), R_sim_attack=Ra.mean(), sd_attack=Ra.std()))
        print(c, 'done', flush=True)
    out = ensure(RESULTS_DIR / 'evaluation')
    pd.DataFrame(rows).to_csv(out / 'robustness_R.csv', index=False)
    with open(out / 'robustness_curves.pkl', 'wb') as f:
        pickle.dump(curves, f)
