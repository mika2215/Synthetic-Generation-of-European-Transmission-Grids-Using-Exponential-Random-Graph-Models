"""Equilibrium-expectation estimation (Byshkin et al. 2018; Giacomarra et al. 2024), Section 4.2.

The chain starts at the observed graph. Every theta steps, each parameter is updated as
    beta_i <- beta_i + alpha * max(|beta_i|, c) * sign(x_i^* - x_i(G_t))      (Equation 16).
If beta_t1 and beta_t2 have the same sign after an update, the one with the smaller absolute
value is multiplied by -0.1 (sign constraint, Equation 20). The estimate beta_bar is the mean
of beta over the last quarter of the chain.
"""
import random

import numpy as np

from .sampler import is_bridge


def ee_algorithm(G_ref, vg, beta_0, x_target, delta_fn, n, T=30_000_000, theta=500, alpha=0.001,
                 c=0.001, burn_in_frac=0.75, sign_constraint_idx=None, rng_seed=42):
    """Return (beta_bar, G_end, beta_hist); beta_hist has shape (T // theta, number of terms)
    and stores beta after every update. A rejected bridge removal counts as a step, so the
    update schedule is purely time-based."""
    rng = random.Random(rng_seed)
    G = G_ref.copy()
    nodes_ = list(G.nodes())
    beta = np.array(beta_0, dtype=float)
    x_curr = x_target.copy()
    t_B = int(burn_in_frac * T)
    beta_acc = np.zeros_like(beta)
    n_post = 0
    n_updates = T // theta
    beta_hist = np.zeros((n_updates, len(beta)), dtype=float)
    update_idx = 0

    for t in range(T):
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
            delta_H = float(np.dot(beta, dx))
            if delta_H >= 0 or rng.random() < np.exp(delta_H):
                if adding:
                    G.add_edge(u, v)
                else:
                    G.remove_edge(u, v)
                x_curr += dx
        if (t + 1) % theta == 0:
            step = alpha * np.maximum(np.abs(beta), c)
            beta += step * np.sign(x_target - x_curr)
            if sign_constraint_idx is not None:
                i, j = sign_constraint_idx
                if beta[i] * beta[j] > 0:
                    if abs(beta[i]) < abs(beta[j]):
                        beta[i] *= -0.1
                    else:
                        beta[j] *= -0.1
            if update_idx < n_updates:
                beta_hist[update_idx] = beta
                update_idx += 1
        if t >= t_B:
            beta_acc += beta
            n_post += 1

    return beta_acc / max(n_post, 1), G, beta_hist


def fit_country(model, T=30_000_000, theta=500, alpha=0.001, c=0.001, rng_seed=42):
    """EE fit of a `statistics.CountryModel` with the settings of the thesis."""
    return ee_algorithm(model.G, model.vg, model.initial_beta(), model.x_target, model.delta_fn,
                        model.n, T=T, theta=theta, alpha=alpha, c=c,
                        sign_constraint_idx=model.sign_constraint_idx, rng_seed=rng_seed)
