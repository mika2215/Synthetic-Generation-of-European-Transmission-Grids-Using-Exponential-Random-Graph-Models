"""Metropolis-Hastings sampler on connected graphs (Sections 4.1, 4.2).

A proposal toggles a uniformly chosen node pair. A removal is rejected if the edge is a bridge,
so the chain stays on connected graphs (Giacomarra et al. 2024; Gray et al. 2019). Otherwise
the proposal is accepted with probability min{1, exp(H(G') - H(G_t))}.
"""
import random

import numpy as np


def is_bridge(G, u, v):
    """True if removing the edge {u, v} disconnects G (depth-first search from u to v)."""
    if G.degree(u) == 1 or G.degree(v) == 1:
        return True
    G.remove_edge(u, v)
    visited, stack, found = set(), [u], False
    while stack and not found:
        node = stack.pop()
        if node == v:
            found = True
        elif node not in visited:
            visited.add(node)
            stack.extend(G.neighbors(node))
    G.add_edge(u, v)
    return not found


def sample_erg(G_start, vg, beta, x_init, delta_fn, n, n_samples=10, steps_between=50_000,
               burn_in=200_000, rng_seed=0):
    """Draw `n_samples` graphs with fixed beta, keeping every `steps_between`-th state after
    `burn_in` steps. A rejected bridge removal still counts as a step, so the spacing between
    retained graphs is exact."""
    rng = random.Random(rng_seed)
    G = G_start.copy()
    nodes_ = list(G.nodes())
    beta = np.array(beta, dtype=float)
    x_curr = np.array(x_init, dtype=float)
    Gs = []
    for t in range(burn_in + n_samples * steps_between):
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
        if t >= burn_in and (t - burn_in + 1) % steps_between == 0:
            Gs.append(G.copy())
    return Gs
