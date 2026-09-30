"""Store the EE trajectories in a compact form for the trajectory figures (Figs. 10, 15, 16).

The full trajectories (beta after each of the 60,000 updates, output/ee/final/beta_hist_<CC>.npy)
are about 3 MB per country. This script keeps every 10th update and the mean over the last
quarter of the full chain, and writes results/trajectories/trajectory_<CC>.npz (arrays
`beta_hist` of shape (6000, 6), `beta_last_quarter`, `theta`, `every`).

For the thesis, the trajectories of the fits with alpha = 0.001 are used for all 15 countries,
including the frozen Serbia chain shown in Figure 10; pass --source to read them from another
folder.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import MODELED_COUNTRIES
from common.paths import OUTPUT_DIR, RESULTS_DIR, ensure

EVERY = 10


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--source', type=Path, default=OUTPUT_DIR / 'ee' / 'final')
    args = ap.parse_args()
    out = ensure(RESULTS_DIR / 'trajectories')
    for c in MODELED_COUNTRIES:
        h = np.load(args.source / f'beta_hist_{c}.npy')
        np.savez_compressed(out / f'trajectory_{c}.npz', beta_hist=h[::EVERY].astype(np.float32),
                            beta_last_quarter=h[int(0.75 * len(h)):].mean(axis=0), theta=500, every=EVERY)
        print(c, h.shape)


if __name__ == '__main__':
    main()
