"""EE fits of the final model and the base model for the 15 countries (Sections 4.2, 5.2.1).

Final model: m_LL, m_LH, m_HH, AKS, t1, t2 with the sign constraint beta_t1 * beta_t2 <= 0.
Base model:  m_LL, m_LH, m_HH only (used for the comparison in Section 5.1, Figure 9).
Settings: T = 30M steps, theta = 500, alpha = c = 0.001, seed 42, beta_bar = mean over the last
quarter of the chain. Serbia (final model) is refitted with alpha = 2e-4 (--alpha 0.0002).

Outputs
  results/beta_bar/<model>/beta_<CC>.npy     beta_bar (small, part of the repository)
  output/ee/<model>/beta_hist_<CC>.npy       beta after every update (trajectories, Figs. 10, 15, 16)
  output/ee/<model>/G_end_<CC>.pkl           last graph of the chain (start of the sampling chain)

Runtime: from about 2 min (Albania) to several hours (Spain) per country.

Usage
  python 03_estimation/fit_models.py --model final
  python 03_estimation/fit_models.py --model final --countries RS --alpha 0.0002
  python 03_estimation/fit_models.py --model base
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.ee import fit_country
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.paths import OUTPUT_DIR, RESULTS_DIR, ensure
from common.statistics import FINAL_KEYS, CountryModel

MODEL_KEYS = {'final': FINAL_KEYS, 'base': []}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', choices=MODEL_KEYS, default='final')
    ap.add_argument('--countries', nargs='*', default=MODELED_COUNTRIES)
    ap.add_argument('--alpha', type=float, default=0.001)
    ap.add_argument('--T', type=int, default=30_000_000)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    out_beta = ensure(RESULTS_DIR / 'beta_bar' / args.model)
    out_ee = ensure(OUTPUT_DIR / 'ee' / args.model)
    for c in args.countries:
        model = CountryModel(build_country_graph(c), MODEL_KEYS[args.model])
        t0 = time.perf_counter()
        print(f'{c}: n = {model.n}, terms = {model.labels}, alpha = {args.alpha} ...', flush=True)
        beta_bar, G_end, beta_hist = fit_country(model, T=args.T, alpha=args.alpha,
                                                 rng_seed=args.seed)
        np.save(out_beta / f'beta_{c}.npy', beta_bar)
        np.save(out_ee / f'beta_hist_{c}.npy', beta_hist)
        with open(out_ee / f'G_end_{c}.pkl', 'wb') as f:
            pickle.dump(G_end, f)
        print(f'   beta_bar = {np.round(beta_bar, 3)}  ({(time.perf_counter() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
