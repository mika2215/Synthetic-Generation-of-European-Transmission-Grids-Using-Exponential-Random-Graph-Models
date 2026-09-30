"""Reproduction of the fitted terms (Section 5.2.1, Tables 8-9 in Appendix A).

For each country, the six terms of the final model are computed for all retained samples, and
tau_i = (mean - x_i^*) / SD (Equation 25) is reported. Values |tau| < 1 indicate adequate
reproduction.

Output: results/evaluation/tau_fitted_terms.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.grid import MODELED_COUNTRIES, build_country_graph, voltage_layers
from common.paths import RESULTS_DIR, ensure
from common.statistics import observed_statistics
from samples_io import load_samples

TERMS = ['m_LL', 'm_LH', 'm_HH', 'akstar', 't1', 't2']

if __name__ == '__main__':
    rows = []
    for c in MODELED_COUNTRIES:
        G0 = build_country_graph(c); vg = voltage_layers(G0); target = observed_statistics(G0, vg)
        X = pd.DataFrame([observed_statistics(G, vg) for G in load_samples('final', c)])
        for t in TERMS:
            x = X[t].to_numpy(float); sd = x.std(ddof=1)
            rows.append(dict(country=c, n=G0.number_of_nodes(), term=t, target=target[t], mean=x.mean(),
                             sd=sd, N=len(x), tau_SD=(x.mean() - target[t]) / sd if sd > 0 else np.nan))
        print(c, 'done', flush=True)
    pd.DataFrame(rows).to_csv(ensure(RESULTS_DIR / 'evaluation') / 'tau_fitted_terms.csv', index=False)
