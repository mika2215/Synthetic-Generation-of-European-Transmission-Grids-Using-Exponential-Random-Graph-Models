"""Figure 2 (Section 2.2): edge-removal robustness of the real Portuguese grid (after
Rosas-Casals et al. 2007, Fig. 6). Middle: S(f) under random edge failure (mean +- SD over 20
random orders) and targeted edge attack (static edge betweenness). Top/bottom: the grid after
removing a fraction f of the edges (random row: one fixed order, seed 0) in the force-directed
layout of the intact grid; the next 5 % of edges to be removed are red, nodes and lines outside
the largest component are faded. Only the real grid is shown, no synthetic samples.

Usage:  python 01_descriptive/fig2_edge_removal_example.py [country code, default PT]
Output: figures/robustness_edge_snapshots_<CC>.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import N_MIN, build_country_graph, layer, load_tables
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

ensure(FIGURES_DIR)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

COUNTRY = sys.argv[1] if len(sys.argv) > 1 else 'PT'
G0 = build_country_graph(COUNTRY)
n0, E = G0.number_of_nodes(), list(G0.edges()); m0 = len(E)
pos = nx.spring_layout(G0, seed=42)

class UF:
    def __init__(s, n): s.p = list(range(n)); s.sz = [1]*n; s.mx = 1
    def f(s, x):
        while s.p[x] != x: s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b):
        a, b = s.f(a), s.f(b)
        if a == b: return
        if s.sz[a] < s.sz[b]: a, b = b, a
        s.p[b] = a; s.sz[a] += s.sz[b]; s.mx = max(s.mx, s.sz[a])

def curve(order):
    idx = {v: i for i, v in enumerate(G0.nodes())}; uf = UF(n0); rev = [1]
    for a, b in reversed(order):
        uf.u(idx[a], idx[b]); rev.append(uf.mx)
    return np.array(list(reversed(rev))[:-1]) / n0

orders = []
for r in range(20):
    o = list(E); np.random.default_rng(r).shuffle(o); orders.append(o)
rc = np.array([curve(o) for o in orders]); r_mean, r_sd = rc.mean(0), rc.std(0)
eb = nx.edge_betweenness_centrality(G0)
att = [e for e, _ in sorted(eb.items(), key=lambda x: -x[1])]
a_curve = curve(att)
f_axis = np.arange(m0) / m0

def draw_snapshot(ax, order, f, next_frac=0.05):
    k = int(f * m0); kn = max(1, int(next_frac * m0))
    nxt = order[k:k + kn]
    H = G0.copy(); H.remove_edges_from(order[:k])
    nxt_set = {frozenset(e) for e in nxt}
    lcc = max(nx.connected_components(H), key=len)
    rest_in = [e for e in H.edges() if frozenset(e) not in nxt_set and e[0] in lcc]
    rest_out = [e for e in H.edges() if frozenset(e) not in nxt_set and e[0] not in lcc]
    nx.draw_networkx_edges(H, pos, edgelist=rest_in, ax=ax, edge_color='#9CA3AF', width=0.6)
    nx.draw_networkx_edges(H, pos, edgelist=rest_out, ax=ax, edge_color='#C4C9D1', width=0.6, style='dashed')
    nx.draw_networkx_edges(H, pos, edgelist=nxt, ax=ax, edge_color='#DC2626', width=1.8)
    # variant (b): nodes outside the largest component are faded (white fill, light outline)
    nx.draw_networkx_nodes(H, pos, nodelist=[v for v in H if v in lcc], ax=ax, node_size=25,
                           node_color='#E5E7EB', edgecolors='#4B5563', linewidths=0.4)
    nx.draw_networkx_nodes(H, pos, nodelist=[v for v in H if v not in lcc], ax=ax, node_size=25,
                           node_color='white', edgecolors='#8B94A3', linewidths=0.7)
    ax.set_title(f'f={f:.2f}', fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)

fig = plt.figure(figsize=(13, 11))
gs = gridspec.GridSpec(3, 3, height_ratios=[1, 1.6, 1], hspace=0.45, wspace=0.15)
f_random_snaps = [0.05, 0.15, 0.30]
f_attack_snaps = {'AT': [0.02, 0.10, 0.25], 'PT': [0.05, 0.10, 0.15]}.get(COUNTRY, [0.02, 0.10, 0.25])
for i, f in enumerate(f_random_snaps):
    draw_snapshot(fig.add_subplot(gs[0, i]), orders[0], f)
fig.text(0.5, 0.955, 'random edge failure', ha='center', fontsize=11, color='#D97706')

ax = fig.add_subplot(gs[1, :])
ax.errorbar(f_axis, r_mean, yerr=r_sd, color='#D97706', ecolor='#D97706', elinewidth=0.6,
            capsize=0, marker='o', markersize=2, linewidth=1,
            label='random edge failure (mean $\\pm$ SD, 20 repeats)')
ax.plot(f_axis, a_curve, color='#2563EB', marker='o', markersize=2, linewidth=1,
        label='targeted edge attack')
ax.set_xlabel('f'); ax.set_ylabel('S(f)'); ax.legend(fontsize=9); ax.grid(alpha=0.3)
for sp in ['top', 'right']: ax.spines[sp].set_visible(False)

for i, f in enumerate(f_attack_snaps):
    draw_snapshot(fig.add_subplot(gs[2, i]), att, f)
fig.text(0.5, 0.295, 'targeted edge attack', ha='center', fontsize=11, color='#2563EB')

plt.savefig(FIGURES_DIR / f'robustness_edge_snapshots_{COUNTRY}.png', dpi=150, bbox_inches='tight')
plt.close()
print('written | random S at snaps', [round(float(r_mean[int(f*m0)]), 2) for f in f_random_snaps],
      '| attack S at snaps', [round(float(a_curve[int(f*m0)]), 2) for f in f_attack_snaps])
