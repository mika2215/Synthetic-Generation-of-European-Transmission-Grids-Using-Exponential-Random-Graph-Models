"""Figure 3 (Section 3.1): the European transmission grid of the dataset (AC lines and
transformers, HVDC excluded), drawn as straight bus-to-bus segments at geographic coordinates,
with zoom panels for Germany and Austria that colour each node by its degree (after
Rosas-Casals et al. 2007, Fig. 2).

Output: figures/descriptive_europe_map.png
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
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

G_eur = build_country_graph(None)
print('pan-European graph:', G_eur.number_of_nodes(), 'nodes,', G_eur.number_of_edges(), 'edges')

def segments(G):
    return [[(G.nodes[u]['x'], G.nodes[u]['y']), (G.nodes[v]['x'], G.nodes[v]['y'])] for u, v in G.edges()]

DEG_BINS = [(1, 2, '#FACC15', '1–2'), (3, 4, '#A855F7', '3–4'), (5, 6, '#22C55E', '5–6'), (7, 99, '#DC2626', '≥7')]
def deg_color(k):
    for lo, hi, c, _ in DEG_BINS:
        if lo <= k <= hi: return c

fig = plt.figure(figsize=(11, 12.5))
ax = fig.add_axes([0.03, 0.40, 0.94, 0.58])
ax.add_collection(LineCollection(segments(G_eur), colors='black', linewidths=0.35))
ax.set_xlim(-11, 42); ax.set_ylim(34.5, 71.5)
ax.set_aspect(1 / np.cos(np.deg2rad(52)))
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values(): s.set_visible(False)

zooms = {'DE': ('Germany', [0.05, 0.02, 0.43, 0.36]), 'AT': ('Austria', [0.53, 0.02, 0.43, 0.36])}
for code, (name, rect) in zooms.items():
    G = build_country_graph(code)
    xs = np.array([G.nodes[v]['x'] for v in G]); ys = np.array([G.nodes[v]['y'] for v in G])
    pad = 0.3
    x0, x1, y0, y1 = xs.min() - pad, xs.max() + pad, ys.min() - pad, ys.max() + pad
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#DC2626', lw=1.3))
    ty, va = (y1 + 0.3, 'bottom') if code == 'DE' else (y0 - 0.3, 'top')
    ax.text(x0, ty, name, color='#DC2626', fontsize=9, va=va)
    zx = fig.add_axes(rect)
    zx.add_collection(LineCollection(segments(G), colors='#374151', linewidths=0.6))
    deg = dict(G.degree())
    zx.scatter(xs, ys, s=[10 + 4 * deg[v] for v in G], c=[deg_color(deg[v]) for v in G],
               edgecolors='black', linewidths=0.3, zorder=3)
    zx.set_xlim(x0, x1); zx.set_ylim(y0, y1)
    zx.set_aspect(1 / np.cos(np.deg2rad(ys.mean())))
    zx.set_xticks([]); zx.set_yticks([])
    for s in zx.spines.values(): s.set_edgecolor('#DC2626')
    zx.set_title(f'{name} ($n={G.number_of_nodes()}$, $m={G.number_of_edges()}$)', fontsize=10)

handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=c, markeredgecolor='black',
                  markersize=7, label=f'degree {lab}') for _, _, c, lab in DEG_BINS]
fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.01))
out = FIGURES_DIR / 'descriptive_europe_map.png'
plt.savefig(out, dpi=200, bbox_inches='tight')
print('written', out)
