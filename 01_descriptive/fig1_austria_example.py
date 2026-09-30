"""Figure 1 (Section 2.1): (a) the real Austrian grid at geographic bus coordinates, (b) one
synthetic grid of the fitted ERGM in a force-directed layout (after Sole et al. 2008, Fig. 1).
The sample shown is retained sample no. 599 of the Austrian chain, stored in
results/descriptive/austria_sample_599.pkl.

Output: figures/theory_austria_example.png
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import N_MIN, build_country_graph, layer, load_tables
from common.paths import FIGURES_DIR, RESULTS_DIR, ensure

ensure(FIGURES_DIR)
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

G0 = build_country_graph('AT')
with open(RESULTS_DIR / 'descriptive' / 'austria_sample_599.pkl', 'rb') as f:
    S = pickle.load(f)


fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={'width_ratios': [1.25, 1]})

ax = axes[0]
pos_geo = {v: (G0.nodes[v]['x'], G0.nodes[v]['y']) for v in G0}
ax.add_collection(LineCollection([[pos_geo[u], pos_geo[v]] for u, v in G0.edges()], colors='black', linewidths=0.8))
ax.scatter(*zip(*pos_geo.values()), s=14, c='#D1D5DB', edgecolors='black', linewidths=0.5, zorder=3)
ax.autoscale(); ax.set_aspect(1 / np.cos(np.deg2rad(47.5)))
ax.set_title('(a) real network, geographic', fontsize=11)

ax = axes[1]
pos = nx.spring_layout(S, seed=42)
ax.add_collection(LineCollection([[pos[u], pos[v]] for u, v in S.edges()], colors='black', linewidths=0.8))
ax.scatter(*zip(*pos.values()), s=14, c='#D1D5DB', edgecolors='black', linewidths=0.5, zorder=3)
ax.autoscale(); ax.set_title('(b) ERGM sample, topological', fontsize=11)
for a in axes[:2]:
    a.set_anchor('N')
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values(): s.set_visible(False)

plt.tight_layout()
out = FIGURES_DIR / 'theory_austria_example.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('written', out, '| real n,m =', G0.number_of_nodes(), G0.number_of_edges(),
      '| sample n,m =', S.number_of_nodes(), S.number_of_edges())
