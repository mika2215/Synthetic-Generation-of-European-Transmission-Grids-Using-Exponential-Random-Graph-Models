"""Locations of input data, stored results and figures.

All paths are relative to the repository root. The raw PyPSA-Eur tables are not part of the
repository; see README.md for the download. Large intermediate files (EE trajectories,
samples) go to `output/`, which is not version-controlled.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Raw data: buses.csv, lines.csv, transformers.csv of the prebuilt PyPSA-Eur network (v0.7).
DATA_DIR = Path(os.environ.get('ERGM_GRIDS_DATA', ROOT / 'data' / 'pypsa_eur'))
# Small result files that are part of the repository (beta_bar, tables).
RESULTS_DIR = ROOT / 'results'
# Large intermediate files (samples, trajectories); not version-controlled.
OUTPUT_DIR = Path(os.environ.get('ERGM_GRIDS_OUTPUT', ROOT / 'output'))
# Figures as used in the thesis.
FIGURES_DIR = ROOT / 'figures'


def ensure(path):
    """Create a directory if necessary and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
