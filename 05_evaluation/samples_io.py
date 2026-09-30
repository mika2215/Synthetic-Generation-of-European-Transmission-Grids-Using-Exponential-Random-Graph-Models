"""Location of the sample files of both models (output/samples/<model>/samples_<CC>.pkl).

The folder can be changed with the environment variable ERGM_GRIDS_OUTPUT.
"""
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import OUTPUT_DIR


def load_samples(model, country):
    with open(OUTPUT_DIR / 'samples' / model / f'samples_{country}.pkl', 'rb') as f:
        return pickle.load(f)
