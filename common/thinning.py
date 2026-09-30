"""Spacing between retained samples (Section 4.1).

The spacing is calibrated on the German grid (n = 780): consecutive retained graphs differ in
at least 2n node pairs after on average 683,000 steps. For the other countries it is scaled
with n^2 ln n (Bhamidi et al. 2011), with a lower bound of 1,000 steps.
"""
import numpy as np

N_REF, SPACING_REF = 780, 683_000
K_SCALE = SPACING_REF / (N_REF ** 2 * np.log(N_REF))


def scaled_spacing(n):
    return max(int(K_SCALE * n ** 2 * np.log(n)), 1000)


def n_samples(n):
    """1,200 samples for n <= 246, 1,100 for the three largest grids."""
    return 1200 if n <= 246 else 1100
