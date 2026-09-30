"""Stage 1 of the term selection: LASSO-penalized pseudo-likelihood on the German grid (Sec. 4.4).

Every node pair of the observed grid is one observation: the outcome is whether the edge is
present, the predictors are the change statistics of the ten candidate terms (Table 3) for
adding the edge (Equation 23). The change statistics are standardized, and an L1-penalized
logistic regression is fitted with the penalty chosen by five-fold cross-validation and the two
classes reweighted to equal total weight (only 0.33 % of the pairs are edges). Candidates with
a zero coefficient are discarded.

The design matrix uses two approximations: for node pairs that are edges, the degrees
are not reduced by one before "adding" the edge, and Delta t2 uses only the C(c, 2) term. Both
only affect the ranking of this pre-filter.

The variants --grid-extended and --no-reweight reproduce the robustness check of the footnote
in Section 4.4 (the same five candidates are retained).

Output: results/selection/lasso_coefficients.csv (one row per variant and candidate).
Runtime: about 2 min per variant (design matrix with 303,810 node pairs).
"""
import argparse
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import build_country_graph
from common.paths import RESULTS_DIR, ensure
from common.statistics import CountryModel, comb2, comb3, delta_sig2_layer

CAND_KEYS = ['t1', 't2', 'sig2', 'sG', 'sig2L', 'sig2H', 'S3', 'S4', 't1_LL', 't1_HH']


def design_matrix(G, vg):
    nodes = list(G.nodes()); n = len(nodes)
    degs = dict(G.degree())
    n_L = sum(1 for v in vg.values() if v == 'L'); n_H = n - n_L
    m_LL = sum(1 for u, v in G.edges() if vg[u] == 'L' and vg[v] == 'L')
    m_HH = sum(1 for u, v in G.edges() if vg[u] == 'H' and vg[v] == 'H')
    m_LH = G.number_of_edges() - m_LL - m_HH
    A = nx.to_scipy_sparse_array(G, nodelist=nodes, format='csr', dtype=np.float32)
    A2 = (A @ A).toarray()                                   # A2[i, j] = |N(i) & N(j)|
    L_nodes = [nd for nd in nodes if vg[nd] == 'L']; H_nodes = [nd for nd in nodes if vg[nd] == 'H']
    L_idx = {nd: i for i, nd in enumerate(L_nodes)}; H_idx = {nd: i for i, nd in enumerate(H_nodes)}
    AL = nx.to_scipy_sparse_array(G.subgraph(L_nodes), nodelist=L_nodes, format='csr', dtype=np.float32)
    AH = nx.to_scipy_sparse_array(G.subgraph(H_nodes), nodelist=H_nodes, format='csr', dtype=np.float32)
    AL2, AH2 = (AL @ AL).toarray(), (AH @ AH).toarray()
    mean_k = 2 * G.number_of_edges() / n
    mean_kL = (2 * m_LL + m_LH) / n_L; mean_kH = (2 * m_HH + m_LH) / n_H

    X = np.zeros((n * (n - 1) // 2, len(CAND_KEYS)), dtype=np.float32)
    y = np.zeros(n * (n - 1) // 2, dtype=np.int8)
    row = 0
    for i, u in enumerate(nodes):
        ku, uvg = degs[u], vg[u]
        for j in range(i + 1, n):
            v = nodes[j]; kv, vvg = degs[v], vg[v]
            y[row] = 1 if G.has_edge(u, v) else 0
            ku_e = ku - 1 if y[row] else ku
            kv_e = kv - 1 if y[row] else kv
            c = int(A2[i, j])
            X[row] = [
                A2[i, j],                                              # t1
                c * (c - 1) / 2,                                       # t2 (approximation)
                2 * (ku_e + kv_e + 1 - 2 * mean_k) / n - 4 / n ** 2,   # sig2
                (ku_e + 1) * (kv_e + 1),                               # sG
                delta_sig2_layer(G, u, v, True, 'L', n_L, mean_kL, uvg, vvg),
                delta_sig2_layer(G, u, v, True, 'H', n_H, mean_kH, uvg, vvg),
                comb2(ku_e) + comb2(kv_e),                             # S3
                comb3(ku_e) + comb3(kv_e),                             # S4
                AL2[L_idx[u], L_idx[v]] if uvg == 'L' and vvg == 'L' else 0,
                AH2[H_idx[u], H_idx[v]] if uvg == 'H' and vvg == 'H' else 0,
            ]
            row += 1
    return X, y


VARIANTS = {'original': (np.logspace(-3, 2, 20), 'balanced'),
            'grid-extended': (np.logspace(-5, 2, 30), 'balanced'),
            'no-reweight': (np.logspace(-5, 2, 30), None)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--variants', nargs='*', default=['original'], choices=list(VARIANTS))
    args = ap.parse_args()
    model = CountryModel(build_country_graph('DE'), [])
    t0 = time.perf_counter()
    X, y = design_matrix(model.G, model.vg)
    print(f'design matrix {X.shape}, {int(y.sum())} edges ({time.perf_counter() - t0:.0f} s)')
    Xs = StandardScaler().fit_transform(X)
    rows = []
    for name in args.variants:
        Cs, cw = VARIANTS[name]
        lr = LogisticRegressionCV(Cs=Cs, penalty='l1', solver='liblinear', cv=5, class_weight=cw,
                                  max_iter=1000, random_state=42).fit(Xs, y)
        for k, coef in zip(CAND_KEYS, lr.coef_[0]):
            rows.append(dict(variant=name, candidate=k, coefficient=coef, retained=abs(coef) > 1e-6,
                             C_best=lr.C_[0]))
        kept = [k for k, cf in zip(CAND_KEYS, lr.coef_[0]) if abs(cf) > 1e-6]
        print(f'{name}: C* = {lr.C_[0]:.3g}, retained = {kept}')
    pd.DataFrame(rows).to_csv(ensure(RESULTS_DIR / 'selection') / 'lasso_coefficients.csv', index=False)


if __name__ == '__main__':
    main()
