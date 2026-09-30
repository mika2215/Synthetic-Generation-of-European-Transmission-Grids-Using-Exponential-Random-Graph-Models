"""Edge-removal robustness (Sections 2.2, 5.3).

S(f) is the relative size of the largest connected component after a fraction f of the edges
has been removed (Equation 6); all nodes stay in the graph. Random failure removes edges in a
uniformly random order, targeted attack in decreasing order of the edge betweenness of the
intact graph (static). The robustness index R is the area under S(f) (Equation 7), evaluated
with the trapezoidal rule over the m removal steps and divided by m.
Curves are computed in reverse with a union-find structure: adding the edges back in reverse
removal order gives the largest component after every removal in O(m alpha(n)).
"""
import networkx as nx
import numpy as np


class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n)); self.sz = [1] * n; self.mx = 1

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a; self.sz[a] += self.sz[b]; self.mx = max(self.mx, self.sz[a])


def removal_curve(G, order):
    """S after k = 0, ..., m-1 removals of the edges in `order`."""
    idx = {v: i for i, v in enumerate(G.nodes())}
    uf = _UnionFind(len(idx)); rev = [1]
    for a, b in reversed(order):
        uf.union(idx[a], idx[b]); rev.append(uf.mx)
    return np.array(list(reversed(rev))[:-1]) / len(idx)


def random_failure_curve(G, reps):
    """Mean curve over `reps` random orders; order r uses np.random.default_rng(r)."""
    E = list(G.edges()); curves = []
    for r in range(reps):
        o = list(E); np.random.default_rng(r).shuffle(o); curves.append(removal_curve(G, o))
    return np.mean(curves, axis=0)


def attack_curve(G):
    """Removal in decreasing order of the static edge betweenness."""
    eb = nx.edge_betweenness_centrality(G)
    return removal_curve(G, [e for e, _ in sorted(eb.items(), key=lambda x: -x[1])])


_trapezoid = getattr(np, 'trapezoid', None) or np.trapz     # NumPy >= 2.0 / < 2.0


def robustness_index(curve):
    return _trapezoid(curve) / len(curve)
