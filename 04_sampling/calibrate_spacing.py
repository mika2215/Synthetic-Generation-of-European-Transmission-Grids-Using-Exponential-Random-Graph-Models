"""Calibration of the spacing between retained samples on the German grid (Section 4.1).

With beta fixed, the chain is run until the retained graph differs from the previous one in at
least 2n node pairs (Hamming distance of the edge sets, counted as the set of node pairs whose
net change since the last retained graph is nonzero). The number of steps needed
for each of 1,100 samples is recorded; its mean, 683,254 steps, is the anchor s_DE of the
spacing rule s(n) in common/thinning.py (Bhamidi et al. 2011: s(n) ~ n^2 ln n).

The calibration uses a preliminary estimate of beta for the German grid (EE fit with
T = 90M, seed 101; results/sampling/beta_DE_calibration_run.npy) and starts from the end of that
EE chain (results/sampling/G_start_DE_calibration.pkl), with a burn-in of 200,000 steps and seed
5101. The steps per sample recorded for the thesis are in
results/sampling/steps_per_sample_DE_thesis.npy (mean 683,254, median 682,216, SD 33,975).

Output: results/sampling/steps_per_sample_DE.npy
Runtime: about 8 h (7.5e8 steps on the German grid).
"""
import pickle
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import build_country_graph
from common.paths import RESULTS_DIR, ensure
from common.sampler import is_bridge
from common.statistics import FINAL_KEYS, CountryModel


def steps_until_hamming(G_start, vg, beta, delta_fn, n, n_samples, threshold, burn_in, rng_seed):
    """Number of steps between consecutive retained graphs that differ in >= threshold pairs."""
    rng = random.Random(rng_seed)
    G = G_start.copy(); nodes_ = list(G.nodes()); beta = np.array(beta, dtype=float)
    x_curr = np.zeros(len(beta))     # the change statistics of AKS, t1, t2 do not use x_curr
    t = 0
    steps, diff, t_last = [], set(), 0
    while len(steps) < n_samples:
        u, v = rng.sample(nodes_, 2)
        blocked = False
        if G.has_edge(u, v):
            if is_bridge(G, u, v):
                blocked = True
            else:
                adding = False
        else:
            adding = True
        if not blocked:
            dx = delta_fn(G, u, v, adding, vg, x_curr, n)
            dH = float(np.dot(beta, dx))
            if dH >= 0 or rng.random() < np.exp(dH):
                (G.add_edge if adding else G.remove_edge)(u, v)
                x_curr += dx
                if t >= burn_in:
                    diff ^= {frozenset((u, v))}
        t += 1
        if t == burn_in:
            t_last = t
        if t > burn_in and len(diff) >= threshold:
            steps.append(t - t_last); t_last = t; diff = set()
    return np.array(steps)


if __name__ == '__main__':
    model = CountryModel(build_country_graph('DE'), FINAL_KEYS)
    beta = np.load(RESULTS_DIR / 'sampling' / 'beta_DE_calibration_run.npy')
    with open(RESULTS_DIR / 'sampling' / 'G_start_DE_calibration.pkl', 'rb') as f:
        G_start = pickle.load(f)
    steps = steps_until_hamming(G_start, model.vg, beta, model.delta_fn, model.n, n_samples=1100,
                                threshold=2 * model.n, burn_in=200_000, rng_seed=5101)
    np.save(ensure(RESULTS_DIR / 'sampling') / 'steps_per_sample_DE.npy', steps)
    print(f'steps per sample: mean {steps.mean():.0f}, median {np.median(steps):.0f}, SD {steps.std():.0f}')
