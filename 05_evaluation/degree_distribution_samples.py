"""Degree distributions of the real grids and of all retained samples (Figure 11).

For each country, the fraction of nodes with degree k = 0, ..., d_max is computed for the real
grid and for every sample of the final model, where d_max is the larger of the maximum degree of
the real grid and the 95th percentile of the maximum degree of the samples; the last bin collects
all higher degrees.

Output: results/evaluation/degree_distribution_<CC>.npz (arrays `observed`, `samples`)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.paths import RESULTS_DIR, ensure
from samples_io import load_samples


def proportions(G, dmax):
    v = np.zeros(dmax + 1)
    for _, d in G.degree():
        v[min(d, dmax)] += 1
    return v / G.number_of_nodes()


if __name__ == '__main__':
    out = ensure(RESULTS_DIR / 'evaluation')
    for c in MODELED_COUNTRIES:
        G0 = build_country_graph(c); S = load_samples('final', c)
        dmax = int(max(max(d for _, d in G0.degree()),
                       np.percentile([max(d for _, d in G.degree()) for G in S], 95)))
        np.savez_compressed(out / f'degree_distribution_{c}.npz', observed=proportions(G0, dmax),
                            samples=np.array([proportions(G, dmax) for G in S], dtype=np.float32))
        print(c, 'done', flush=True)
