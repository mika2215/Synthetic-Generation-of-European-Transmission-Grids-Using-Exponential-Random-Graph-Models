"""Check of the sample spacing (Section 4.1): mean Hamming distance between consecutive retained
samples, relative to the calibration target 2n of the German grid.

In the thesis samples, the mean distance lies between 0.55 * 2n (Bosnia and Herzegovina) and
1.06 * 2n (Poland), so the samples of most countries are somewhat more strongly correlated than
those of the German grid (results/sampling/hamming_per_country_thesis.csv).

Input:  output/samples/final/samples_<CC>.pkl
Output: results/sampling/hamming_per_country.csv
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.paths import OUTPUT_DIR, RESULTS_DIR, ensure
from common.thinning import scaled_spacing


def hamming(samples):
    E = [set(map(frozenset, G.edges())) for G in samples]
    return np.array([len(a ^ b) for a, b in zip(E[:-1], E[1:])])


if __name__ == '__main__':
    rows = []
    for c in MODELED_COUNTRIES:
        G0 = build_country_graph(c); n = G0.number_of_nodes()
        with open(OUTPUT_DIR / 'samples' / 'final' / f'samples_{c}.pkl', 'rb') as f:
            h = hamming(pickle.load(f))
        rows.append(dict(country=c, n=n, spacing_s=scaled_spacing(n), N=len(h) + 1,
                         hamming_mean=h.mean(), hamming_median=np.median(h), two_n=2 * n,
                         hamming_over_2n=h.mean() / (2 * n)))
    df = pd.DataFrame(rows)
    df.to_csv(ensure(RESULTS_DIR / 'sampling') / 'hamming_per_country.csv', index=False)
    print(df.round(3).to_string(index=False))
