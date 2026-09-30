"""Observed statistics and change statistics of the ERGM terms (Sections 2.4, 4.3, 4.4).

For every term x_k the Markov chain needs the change statistic Delta x_k(i, j), the change of
x_k when the node pair {i, j} is toggled. All change statistics depend only on the
neighborhoods of i and j and are computed locally.

Term keys
---------
m_LL, m_LH, m_HH : edges within L, between L and H, within H (mixing terms)
akstar           : AKS = sum_i 0.5^{k_i}, the alternating k-star statistic with lambda = 2
t1               : number of triangles
t2               : number of 2-triangles (pairs of triangles sharing an edge)
sig2             : degree variance
sig2L, sig2H     : degree variance within the L / H layer
sG               : s(G) = sum over edges of k_u k_v
S3, S4           : number of 3-stars and 4-stars
t1_LL, t1_HH     : triangles with all three nodes in L / in H
"""
import networkx as nx
import numpy as np


def comb2(k): return k * (k - 1) // 2
def comb3(k): return k * (k - 1) * (k - 2) // 6
def comb4(k): return k * (k - 1) * (k - 2) * (k - 3) // 24


def _triangles(G):
    for u in G.nodes():
        Nu = set(G.neighbors(u))
        for v in Nu:
            if v <= u:
                continue
            for w in (Nu & set(G.neighbors(v))):
                if w > v:
                    yield (u, v, w)


def akstar(G, lam=2.0):
    """AKS = sum_i (1 - 1/lam)^{k_i}; with lam = 2 this is sum_i 0.5^{k_i} (Equation 21)."""
    base = 1 - 1 / lam
    return sum(base ** k for _, k in G.degree())


# --------------------------------------------------------------------------------------------
# Observed statistics
# --------------------------------------------------------------------------------------------
def observed_statistics(G, vg):
    """All candidate statistics of a graph G with voltage layers vg (node -> 'L'/'H')."""
    degs = dict(G.degree())
    deg_arr = np.array(list(degs.values()))
    t1 = sum(nx.triangles(G).values()) // 3
    t2 = sum(comb2(len(set(G.neighbors(u)) & set(G.neighbors(v)))) for u, v in G.edges())
    sig2 = float(np.var(deg_arr))
    sG = float(sum(degs[u] * degs[v] for u, v in G.edges()))
    dL = [degs[nd] for nd in G if vg[nd] == 'L']
    dH = [degs[nd] for nd in G if vg[nd] == 'H']
    sig2L = float(np.var(dL)) if len(dL) > 1 else 0.0
    sig2H = float(np.var(dH)) if len(dH) > 1 else 0.0
    S3 = int(sum(comb3(k) for k in deg_arr))
    S4 = int(sum(comb4(k) for k in deg_arr))
    t1_LL = sum(1 for u, v, w in _triangles(G) if vg[u] == 'L' and vg[v] == 'L' and vg[w] == 'L')
    t1_HH = sum(1 for u, v, w in _triangles(G) if vg[u] == 'H' and vg[v] == 'H' and vg[w] == 'H')
    m_LL = sum(1 for u, v in G.edges() if vg[u] == 'L' and vg[v] == 'L')
    m_LH = sum(1 for u, v in G.edges() if vg[u] != vg[v])
    m_HH = sum(1 for u, v in G.edges() if vg[u] == 'H' and vg[v] == 'H')
    return dict(t1=t1, t2=t2, sig2=sig2, sG=sG, sig2L=sig2L, sig2H=sig2H, S3=S3, S4=S4,
                t1_LL=t1_LL, t1_HH=t1_HH, akstar=akstar(G), m_LL=m_LL, m_LH=m_LH, m_HH=m_HH)


# --------------------------------------------------------------------------------------------
# Change statistics. `adding` is True if the edge {u, v} is added, False if it is removed.
# For a removal, the edge is still present in G when the function is called.
# --------------------------------------------------------------------------------------------
def delta_mixing(u, v, adding, vg):
    """Changes of (m_LL, m_LH, m_HH)."""
    s = 1 if adding else -1
    u_vg, v_vg = vg[u], vg[v]
    return (s if u_vg == 'L' and v_vg == 'L' else 0,
            s if u_vg != v_vg else 0,
            s if u_vg == 'H' and v_vg == 'H' else 0)


def delta_t1(G, u, v, adding):
    Nu = set(G.neighbors(u)) - ({v} if not adding else set())
    Nv = set(G.neighbors(v)) - ({u} if not adding else set())
    c = len(Nu & Nv)
    return c if adding else -c


def delta_t2(G, u, v, adding):
    if adding:
        Nu = set(G.neighbors(u)); Nv = set(G.neighbors(v))
    else:
        Nu = set(G.neighbors(u)) - {v}; Nv = set(G.neighbors(v)) - {u}
    common = Nu & Nv
    c = len(common)
    return (c * (c - 1) // 2 + sum(len(Nu & set(G.neighbors(w))) + len(Nv & set(G.neighbors(w)))
                                   for w in common)) * (1 if adding else -1)


def delta_sig2(G, u, v, adding, n, m_curr):
    k_u, k_v = G.degree(u), G.degree(v)
    mean_k = 2 * m_curr / n
    if adding:
        return 2 * (k_u + k_v + 1 - 2 * mean_k) / n - 4 / n ** 2
    return -2 * (k_u + k_v - 1 - 2 * mean_k) / n - 4 / n ** 2


def delta_sig2_layer(G, u, v, adding, vg_target, n_vg, mean_k_vg, u_vg, v_vg):
    """Change of the degree variance within one layer (vg_target = 'L' or 'H')."""
    if n_vg == 0:
        return 0.0
    k_u, k_v = G.degree(u), G.degree(v)
    u_in = (u_vg == vg_target); v_in = (v_vg == vg_target)
    if u_in and v_in:
        if adding:
            return 2 * (k_u + k_v + 1 - 2 * mean_k_vg) / n_vg - 4 / n_vg ** 2
        return -2 * (k_u + k_v - 1 - 2 * mean_k_vg) / n_vg - 4 / n_vg ** 2
    if u_in:
        if adding:
            return (2 * (k_u - mean_k_vg) + 1) / n_vg - 1 / n_vg ** 2
        return -(2 * (k_u - mean_k_vg) - 1) / n_vg - 1 / n_vg ** 2
    if v_in:
        if adding:
            return (2 * (k_v - mean_k_vg) + 1) / n_vg - 1 / n_vg ** 2
        return -(2 * (k_v - mean_k_vg) - 1) / n_vg - 1 / n_vg ** 2
    return 0.0


def delta_sG(G, u, v, adding):
    if adding:
        Nu = set(G.neighbors(u)); Nv = set(G.neighbors(v))
        k_u, k_v = G.degree(u), G.degree(v)
        return float((k_u + 1) * (k_v + 1) + sum(G.degree(w) for w in Nu)
                     + sum(G.degree(w) for w in Nv))
    Nu = set(G.neighbors(u)) - {v}; Nv = set(G.neighbors(v)) - {u}
    k_u, k_v = G.degree(u), G.degree(v)
    return float(-k_u * k_v - sum(G.degree(w) for w in Nu) - sum(G.degree(w) for w in Nv))


def delta_S3(G, u, v, adding):
    k_u, k_v = G.degree(u), G.degree(v)
    return (comb2(k_u) + comb2(k_v)) if adding else -(comb2(k_u - 1) + comb2(k_v - 1))


def delta_S4(G, u, v, adding):
    k_u, k_v = G.degree(u), G.degree(v)
    return (comb3(k_u) + comb3(k_v)) if adding else -(comb3(k_u - 1) + comb3(k_v - 1))


def delta_t1_layer(G, u, v, adding, vg, layer):
    """Change of the number of triangles with all three nodes in `layer`."""
    if not (vg[u] == layer and vg[v] == layer):
        return 0
    if adding:
        common = set(G.neighbors(u)) & set(G.neighbors(v))
    else:
        common = (set(G.neighbors(u)) - {v}) & (set(G.neighbors(v)) - {u})
    c = sum(1 for w in common if vg[w] == layer)
    return c if adding else -c


def delta_akstar(G, u, v, adding, lam=2.0):
    """Change of AKS = sum_i (1-1/lam)^{k_i}; depends only on the degrees of u and v."""
    base = 1 - 1 / lam
    k_u, k_v = G.degree(u), G.degree(v)
    if adding:
        return -(1 / lam) * (base ** k_u + base ** k_v)
    return (1 / lam) * (base ** (k_u - 1) + base ** (k_v - 1))


# --------------------------------------------------------------------------------------------
# Terms of one country and the combined change-statistic function of a Hamiltonian
# --------------------------------------------------------------------------------------------
def candidate_terms(n, n_L, n_H, vg, ref):
    """Dict key -> (name, observed value, change-statistic function).

    The change-statistic functions take (G, u, v, adding, x_curr), where x_curr is the current
    vector of statistics (its first three entries are m_LL, m_LH, m_HH). Default-argument
    binding avoids late binding inside loops over countries.
    """
    return {
        't1': ('t1', ref['t1'], lambda G, u, v, add, xc: float(delta_t1(G, u, v, add))),
        't2': ('t2', ref['t2'], lambda G, u, v, add, xc: float(delta_t2(G, u, v, add))),
        'sig2': ('sig2', ref['sig2'],
                 lambda G, u, v, add, xc, _n=n: delta_sig2(G, u, v, add, _n, xc[0] + xc[1] + xc[2])),
        'akstar': ('akstar', ref['akstar'], lambda G, u, v, add, xc: delta_akstar(G, u, v, add)),
        'sG': ('sG', ref['sG'], lambda G, u, v, add, xc: delta_sG(G, u, v, add)),
        'sig2L': ('sig2L', ref['sig2L'],
                  lambda G, u, v, add, xc, _nL=n_L, _vg=vg:
                      delta_sig2_layer(G, u, v, add, 'L', _nL,
                                       (2 * xc[0] + xc[1]) / _nL if _nL > 0 else 0, _vg[u], _vg[v])),
        'sig2H': ('sig2H', ref['sig2H'],
                  lambda G, u, v, add, xc, _nH=n_H, _vg=vg:
                      delta_sig2_layer(G, u, v, add, 'H', _nH,
                                       (2 * xc[2] + xc[1]) / _nH if _nH > 0 else 0, _vg[u], _vg[v])),
        'S3': ('S3', ref['S3'], lambda G, u, v, add, xc: float(delta_S3(G, u, v, add))),
        'S4': ('S4', ref['S4'], lambda G, u, v, add, xc: float(delta_S4(G, u, v, add))),
        't1_LL': ('t1_LL', ref['t1_LL'],
                  lambda G, u, v, add, xc, _vg=vg: float(delta_t1_layer(G, u, v, add, _vg, 'L'))),
        't1_HH': ('t1_HH', ref['t1_HH'],
                  lambda G, u, v, add, xc, _vg=vg: float(delta_t1_layer(G, u, v, add, _vg, 'H'))),
    }


def make_delta_fn(selected_keys, candidates):
    """Change statistics of the Hamiltonian m_LL, m_LH, m_HH + selected_keys, as one vector."""
    def delta_fn(G, u, v, adding, vg, x_curr, n):
        dm = list(delta_mixing(u, v, adding, vg))
        extras = [candidates[ck][2](G, u, v, adding, x_curr) for ck in selected_keys]
        return np.array(dm + extras, dtype=float)
    return delta_fn


class CountryModel:
    """Everything the sampler and the EE algorithm need for one country and one Hamiltonian.

    Parameters
    ----------
    G : networkx.Graph, the observed country graph
    selected_keys : terms beyond the mixing terms, e.g. ['akstar', 't1', 't2'] (final model)
                    or [] (base model)
    """
    def __init__(self, G, selected_keys):
        from .grid import voltage_layers
        self.G = G
        self.vg = voltage_layers(G)
        self.n = G.number_of_nodes()
        self.n_L = sum(1 for v in self.vg.values() if v == 'L')
        self.n_H = self.n - self.n_L
        self.keys = list(selected_keys)
        self.ref = observed_statistics(G, self.vg)
        self.candidates = candidate_terms(self.n, self.n_L, self.n_H, self.vg, self.ref)
        self.delta_fn = make_delta_fn(self.keys, self.candidates)
        self.labels = ['m_LL', 'm_LH', 'm_HH'] + self.keys
        self.x_target = np.array([self.ref['m_LL'], self.ref['m_LH'], self.ref['m_HH']]
                                 + [self.candidates[k][1] for k in self.keys], dtype=float)
        # Sign constraint beta_t1 * beta_t2 <= 0 (Equation 20), if both terms are present.
        self.sign_constraint_idx = ((3 + self.keys.index('t1'), 3 + self.keys.index('t2'))
                                    if 't1' in self.keys and 't2' in self.keys else None)

    def initial_beta(self):
        """Start values: logit of the edge densities for the mixing terms, 0 otherwise."""
        def safe_logit(p, eps=1e-6):
            p = np.clip(p, eps, 1 - eps)
            return float(np.log(p / (1 - p)))
        b_LL = safe_logit(self.ref['m_LL'] / max(1, self.n_L * (self.n_L - 1) // 2))
        b_LH = safe_logit(self.ref['m_LH'] / max(1, self.n_L * self.n_H))
        b_HH = safe_logit(self.ref['m_HH'] / max(1, self.n_H * (self.n_H - 1) // 2))
        return [b_LL, b_LH, b_HH] + [0.0] * len(self.keys)


FINAL_KEYS = ['akstar', 't1', 't2']     # Equation 27
