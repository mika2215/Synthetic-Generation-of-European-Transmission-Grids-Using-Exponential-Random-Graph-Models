"""Stage 2 of the term selection: forward selection on the German grid (Sections 4.4, 5.1).

Starting from the base model (m_LL, m_LH, m_HH), each remaining candidate of the pre-filter is
added in turn, the model is fitted with the EE algorithm, 50 graphs are sampled, and the model
is scored on the eight selection statistics (Table 4, Equation 24):
    Score = mean_i |mean_sample(x_i) - x_i^*| / |x_i^*|  +  0.005 * (number of added terms).
The best candidate is added if it lowers the score; otherwise the selection stops (at most 5
added terms).

Settings as in the thesis: EE with T = 30M, theta = 500, alpha = c = 0.001, seed 42; sampling
from the end of the EE chain with 50 graphs, 50,000 steps apart, burn-in 200,000, seed 43.

Output: results/selection/forward_selection_trace.csv (one row per step and candidate).
Runtime: several hours (about 16 EE fits of the German grid).

The trace of the thesis run is stored in results/selection/forward_selection_trace_thesis.csv.
"""
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ee import fit_country
from common.grid import build_country_graph
from common.paths import RESULTS_DIR, ensure
from common.sampler import sample_erg
from common.statistics import CountryModel

LAMBDA_PEN = 0.005
MAX_ADDED = 5
N_SAMPLES, STEPS_BETWEEN, BURN_IN = 50, 50_000, 200_000
SELECTION_STATS = ['m', 'mean_k', 'var_k', 'clustering', 'art_points', 'bridges', 'lambda_max',
                   'lambda_ratio']


def selection_statistics(G):
    degs = [d for _, d in G.degree()]
    mean_k = np.mean(degs)
    try:
        A = nx.to_scipy_sparse_array(G, format='csr', dtype=float)
        lam_max = float(eigsh(A, k=1, which='LM', return_eigenvectors=False)[0])
    except Exception:
        lam_max = np.nan
    return dict(m=G.number_of_edges(), mean_k=mean_k, var_k=float(np.var(degs)),
                clustering=nx.average_clustering(G), art_points=sum(1 for _ in nx.articulation_points(G)),
                bridges=sum(1 for _ in nx.bridges(G)), lambda_max=lam_max,
                lambda_ratio=lam_max / mean_k if mean_k > 0 else np.nan)


def score(samples, target):
    rows = [selection_statistics(G) for G in samples]
    errs = [abs(np.nanmean([r[s] for r in rows]) - target[s]) / abs(target[s])
            for s in SELECTION_STATS if abs(target[s]) > 1e-9]
    return float(np.mean(errs))


def evaluate(G, keys):
    model = CountryModel(G, keys)
    beta_bar, G_end, _ = fit_country(model, rng_seed=42)
    samples = sample_erg(G_end, model.vg, beta_bar, model.x_target, model.delta_fn, model.n,
                         n_samples=N_SAMPLES, steps_between=STEPS_BETWEEN, burn_in=BURN_IN,
                         rng_seed=43)
    return beta_bar, samples


def main():
    G = build_country_graph('DE')
    target = selection_statistics(G)
    lasso = pd.read_csv(RESULTS_DIR / 'selection' / 'lasso_coefficients.csv')
    lasso = lasso[(lasso.variant == 'original') & lasso.retained]
    remaining = list(lasso.reindex(lasso.coefficient.abs().sort_values(ascending=False).index).candidate)
    print('candidates after the pre-filter:', remaining)

    selected, rows = [], []
    _, samples = evaluate(G, [])
    current = score(samples, target)
    rows.append(dict(step=0, candidate='(base)', raw_score=current, pen_score=current, added=True))
    print(f'step 0 (base model): score = {current:.4f}')
    for step in range(1, MAX_ADDED + 1):
        best, best_pen = None, current
        for ck in remaining:
            t0 = time.perf_counter()
            _, samples = evaluate(G, selected + [ck])
            raw = score(samples, target); pen = raw + LAMBDA_PEN * (len(selected) + 1)
            rows.append(dict(step=step, candidate=ck, raw_score=raw, pen_score=pen, added=False))
            print(f'step {step}: + {ck:6s} raw = {raw:.4f}  pen = {pen:.4f}  ({(time.perf_counter()-t0)/60:.0f} min)')
            if pen < best_pen:
                best, best_pen = ck, pen
        if best is None:
            print('no candidate improves the score -> stop'); break
        selected.append(best); remaining.remove(best); current = best_pen
        for r in rows:
            if r['step'] == step and r['candidate'] == best:
                r['added'] = True
    print('selected terms:', selected)
    pd.DataFrame(rows).to_csv(ensure(RESULTS_DIR / 'selection') / 'forward_selection_trace.csv', index=False)


if __name__ == '__main__':
    main()
