"""Held-out statistics of a graph (Section 4.5, Tables 6-7).

The number of random removal orders is 20 for the real grids and 5 for each synthetic grid.
"""
import networkx as nx
import numpy as np

from .robustness import attack_curve, random_failure_curve, robustness_index

REP_REAL, REP_SAMPLE = 20, 5
L2_SEED = 0
HELDOUT_STATS = ['assortativity', 'avg_path_len', 'diameter', 'lambda2', 'betw_max',
                 'betw_mean', 'R_edge_random', 'R_edge_attack']


def heldout_statistics(G, reps):
    bc = np.array(list(nx.betweenness_centrality(G, normalized=True).values()))
    return dict(assortativity=nx.degree_assortativity_coefficient(G),
                avg_path_len=nx.average_shortest_path_length(G),
                diameter=nx.diameter(G),
                lambda2=nx.algebraic_connectivity(G, seed=L2_SEED),
                betw_max=bc.max(), betw_mean=bc.mean(),
                R_edge_random=robustness_index(random_failure_curve(G, reps)),
                R_edge_attack=robustness_index(attack_curve(G)))


def evaluation_subset(samples, n_sub=30, seed=0):
    """The 30 samples per country and model used for all held-out statistics:
    np.random.default_rng(0).choice(len(samples), 30, replace=False)."""
    idx = np.random.default_rng(seed).choice(len(samples), size=n_sub, replace=False)
    return [samples[i] for i in idx], idx
