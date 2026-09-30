"""Held-out statistics of the real grids and of the synthetic grids of both models
(Sections 4.5, 5.1-5.3; Figures 9, 12, 13; Tables 6-7).

For each country and model, the 30-sample evaluation subset (np.random.default_rng(0)) is
evaluated on the held-out statistics of common/heldout.py: assortativity, average path length,
diameter, algebraic connectivity, maximum and mean betweenness, and the robustness index R under
random edge failure (20 orders for the real grid, 5 per sample) and targeted attack. The
standardized deviation is z = (mean - observed) / SD (Equation 26).

Output: results/evaluation/heldout_per_statistic.csv
Runtime: about 1 h with 6 processes (dominated by betweenness and lambda_2 of the large grids).
"""
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.heldout import HELDOUT_STATS, REP_REAL, REP_SAMPLE, evaluation_subset, heldout_statistics
from common.paths import RESULTS_DIR, ensure
from samples_io import load_samples


def run_country(c):
    G0 = build_country_graph(c)
    obs = heldout_statistics(G0, REP_REAL)
    rows = []
    for model in ['base', 'final']:
        sub, _ = evaluation_subset(load_samples(model, c))
        vals = pd.DataFrame([heldout_statistics(G, REP_SAMPLE) for G in sub])
        for s in HELDOUT_STATS:
            v = vals[s].to_numpy(float); m, sd = v.mean(), v.std(ddof=1)
            rows.append(dict(country=c, n=G0.number_of_nodes(), model=model, stat=s, observed=obs[s],
                             mean=m, sd=sd, z=(m - obs[s]) / sd if sd > 0 else np.nan))
    print(c, 'done', flush=True)
    return rows


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=6) as ex:
        rows = [r for rs in ex.map(run_country, MODELED_COUNTRIES[::-1]) for r in rs]
    pd.DataFrame(rows).to_csv(ensure(RESULTS_DIR / 'evaluation') / 'heldout_per_statistic.csv', index=False)
