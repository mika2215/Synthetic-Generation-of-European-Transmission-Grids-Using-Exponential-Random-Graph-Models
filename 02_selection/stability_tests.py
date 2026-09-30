"""Stability tests of the selected raw terms (Sections 4.3, 4.4, 5.1).

Part A, t1 (Figure 8): for the eight countries whose fitted model contains both t1 and t2 with
a nonzero target, beta_t1 is varied around its fitted value (+-1.5 in steps of 0.5), with all
other parameters fixed at beta_bar, once with beta_t2 at its fitted value and once with
beta_t2 = 0. For each grid point, 5 graphs are sampled from the observed grid (burn-in 30,000,
spacing 8,000, seed 5000 + grid index) and their triangle counts recorded.

Part B, sigma^2 and AKS (Serbia and Czech Republic): reduced models H = beta_e |E| + beta x,
with beta_e fixed at the logit of the observed edge density, for x = t1 (beta in [-1, 1.2]),
sigma^2 (beta in [-4, 16]) and AKS (beta in [-30, 15]). For each value, 5 graphs are sampled
(burn-in 100,000, spacing 20,000, seed = grid index) and m, t1, sigma^2 and AKS recorded.
The mean of sigma^2 jumps by orders of magnitude beyond beta_sigma2 = 3.5-5.0, whereas AKS
changes gradually over the whole range.

The results of the thesis runs are stored as *_thesis.csv.

Outputs: results/selection/t1_sweep.csv, results/selection/reduced_model_sweeps.csv
Runtime: part A about 1 h, part B a few minutes.
"""
import argparse
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import build_country_graph
from common.paths import RESULTS_DIR, ensure
from common.sampler import sample_erg
from common.statistics import (FINAL_KEYS, CountryModel, akstar, delta_akstar, delta_sig2,
                               delta_t1)

T1_COUNTRIES = ['HR', 'BA', 'SK', 'CZ', 'AT', 'BG', 'PT', 'CH']


def t1_grid(center, half_width=1.5, step=0.5):
    grid = np.arange(center - half_width, center + half_width + 1e-9, step)
    return np.unique(np.round(np.concatenate([grid, [center]]), 4))


def part_a():
    rows = []
    for c in T1_COUNTRIES:
        model = CountryModel(build_country_graph(c), FINAL_KEYS)
        beta_bar = np.load(RESULTS_DIR / 'beta_bar' / 'final' / f'beta_{c}.npy')
        for variant in ['with_t2', 'without_t2']:
            for i, val in enumerate(t1_grid(beta_bar[4])):
                beta = beta_bar.copy(); beta[4] = val
                if variant == 'without_t2':
                    beta[5] = 0.0
                for G in sample_erg(model.G, model.vg, beta, model.x_target, model.delta_fn, model.n,
                                    n_samples=5, steps_between=8_000, burn_in=30_000,
                                    rng_seed=5000 + i):
                    rows.append(dict(country=c, variant=variant, beta_t1=val, beta_t1_fit=beta_bar[4],
                                     m=G.number_of_edges(), t1=sum(nx.triangles(G).values()) // 3))
        print(f'{c} done', flush=True)
    pd.DataFrame(rows).to_csv(ensure(RESULTS_DIR / 'selection') / 't1_sweep.csv', index=False)


def reduced_delta_fn(term, n):
    """Change statistics (edge count, term) for a reduced model; x_curr[0] is the edge count."""
    fns = {'t1': lambda G, u, v, a, m: delta_t1(G, u, v, a),
           'sig2': lambda G, u, v, a, m: delta_sig2(G, u, v, a, n, m),
           'akstar': lambda G, u, v, a, m: delta_akstar(G, u, v, a)}
    f = fns[term]
    return lambda G, u, v, adding, vg, x, _n: np.array([1.0 if adding else -1.0, f(G, u, v, adding, x[0])])


GRIDS = {'t1': np.round(np.arange(-1.0, 1.21, 0.15), 3),
         'sig2': np.round(np.arange(-4.0, 16.1, 1.5), 3),
         'akstar': np.round(np.arange(-30.0, 15.1, 3.0), 3)}


def part_b():
    rows = []
    for c in ['RS', 'CZ']:
        G0 = build_country_graph(c); n = G0.number_of_nodes(); m0 = G0.number_of_edges()
        p = np.clip(m0 / (n * (n - 1) // 2), 1e-6, 1 - 1e-6); beta_e = float(np.log(p / (1 - p)))
        for term, grid in GRIDS.items():
            dfn = reduced_delta_fn(term, n)
            for i, b in enumerate(grid):
                for G in sample_erg(G0, None, [beta_e, b], [m0, 0.0], dfn, n, n_samples=5,
                                    steps_between=20_000, burn_in=100_000, rng_seed=i):
                    degs = np.array([d for _, d in G.degree()])
                    rows.append(dict(country=c, term=term, beta=b, beta_e=beta_e, m=G.number_of_edges(),
                                     t1=sum(nx.triangles(G).values()) // 3, sig2=float(degs.var()),
                                     akstar=akstar(G)))
            print(f'{c} {term} done', flush=True)
    pd.DataFrame(rows).to_csv(ensure(RESULTS_DIR / 'selection') / 'reduced_model_sweeps.csv', index=False)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--part', choices=['a', 'b', 'both'], default='both')
    args = ap.parse_args()
    if args.part in ('a', 'both'):
        part_a()
    if args.part in ('b', 'both'):
        part_b()
