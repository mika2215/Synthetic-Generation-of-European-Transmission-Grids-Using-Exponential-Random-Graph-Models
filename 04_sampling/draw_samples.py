"""Draw the synthetic grids from the fitted models (Section 4.1).

For each country, N = 1,200 graphs (n <= 246) or 1,100 graphs (three largest grids) are drawn
with beta fixed at beta_bar, keeping every s(n)-th state of the chain (common/thinning.py)
after a burn-in of s(n) steps. The chain starts from the last graph of the EE chain.

Inputs   results/beta_bar/<model>/beta_<CC>.npy, output/ee/<model>/G_end_<CC>.pkl
Output   output/samples/<model>/samples_<CC>.pkl   (list of networkx graphs)

Usage
  python 04_sampling/draw_samples.py --model final
  python 04_sampling/draw_samples.py --model base

Seeds: country CC uses 51000 + (index of CC in MODELED_COUNTRIES) for the final model and 51000
for the base model.
"""
import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.paths import OUTPUT_DIR, RESULTS_DIR, ensure
from common.sampler import sample_erg
from common.statistics import FINAL_KEYS, CountryModel
from common.thinning import n_samples, scaled_spacing

MODEL_KEYS = {'final': FINAL_KEYS, 'base': []}


def seed_for(model, country):
    return 51000 + MODELED_COUNTRIES.index(country) if model == 'final' else 51000


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--model', choices=MODEL_KEYS, default='final')
    ap.add_argument('--countries', nargs='*', default=MODELED_COUNTRIES)
    args = ap.parse_args()

    out = ensure(OUTPUT_DIR / 'samples' / args.model)
    for c in args.countries:
        model = CountryModel(build_country_graph(c), MODEL_KEYS[args.model])
        beta_bar = np.load(RESULTS_DIR / 'beta_bar' / args.model / f'beta_{c}.npy')
        with open(OUTPUT_DIR / 'ee' / args.model / f'G_end_{c}.pkl', 'rb') as f:
            G_start = pickle.load(f)
        s, N = scaled_spacing(model.n), n_samples(model.n)
        t0 = time.perf_counter()
        print(f'{c}: n = {model.n}, spacing = {s:,}, N = {N} ...', flush=True)
        samples = sample_erg(G_start, model.vg, beta_bar, model.x_target, model.delta_fn, model.n,
                             n_samples=N, steps_between=s, burn_in=s,
                             rng_seed=seed_for(args.model, c))
        with open(out / f'samples_{c}.pkl', 'wb') as f:
            pickle.dump(samples, f)
        print(f'   {len(samples)} samples ({(time.perf_counter() - t0) / 60:.1f} min)')


if __name__ == '__main__':
    main()
