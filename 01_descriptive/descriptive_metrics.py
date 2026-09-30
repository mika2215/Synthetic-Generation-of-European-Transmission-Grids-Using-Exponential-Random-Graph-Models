"""Topological descriptors of the 29 country grids with n >= 20 (Chapter 3).

For every country graph (common/grid.py): size, degree statistics, clustering coefficient,
assortativity, average path length, diameter, algebraic connectivity, triangle counts, voltage
layers and mixing counts, bridges and articulation points. Also the pan-European cross-check of
Section 3.3: the continent-wide graph (largest component) and the share of national bridges that
remain bridges when cross-border lines are taken into account.

Outputs: results/descriptive/descriptive_metrics.csv (one row per country, sorted by n)
         results/descriptive/pan_european_bridges.csv
"""
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import N_MIN, build_country_graph, load_tables, voltage_layers
from common.paths import RESULTS_DIR, ensure
from common.statistics import comb2


def descriptors(G):
    vg = voltage_layers(G)
    n, m = G.number_of_nodes(), G.number_of_edges()
    deg = np.array([d for _, d in G.degree()])
    br = list(nx.bridges(G)); ap = list(nx.articulation_points(G))
    return dict(
        n=n, m=m, k_mean=2 * m / n, m_n=m / n, var_k=deg.var(), k_max=int(deg.max()),
        p_deg1=(deg == 1).mean(), p_deg2=(deg == 2).mean(), p_deg3=(deg >= 3).mean(),
        clustering=nx.average_clustering(G), avg_path_len=nx.average_shortest_path_length(G),
        diameter=nx.diameter(G),
        assortativity=nx.degree_assortativity_coefficient(G) if m > 1 else 0.0,
        lambda2=nx.algebraic_connectivity(G),
        t1=sum(nx.triangles(G).values()) // 3,
        t2=sum(comb2(len(set(G.neighbors(u)) & set(G.neighbors(v)))) for u, v in G.edges()),
        n_L=sum(1 for x in vg.values() if x == 'L'), n_H=sum(1 for x in vg.values() if x == 'H'),
        m_LL=sum(1 for u, v in G.edges() if vg[u] == 'L' and vg[v] == 'L'),
        m_LH=sum(1 for u, v in G.edges() if vg[u] != vg[v]),
        m_HH=sum(1 for u, v in G.edges() if vg[u] == 'H' and vg[v] == 'H'),
        bridges=len(br), frac_bridge=len(br) / m, art_points=len(ap))


def pan_european_bridges(countries):
    """Bridges of the continent-wide graph vs. bridges of the national graphs cut out of it."""
    G_eur = build_country_graph(None)
    G_eur = G_eur.subgraph(max(nx.connected_components(G_eur), key=len)).copy()
    eur_bridges = set(frozenset(e) for e in nx.bridges(G_eur))
    total_m, total_br, survive = 0, 0, 0
    for c in countries:
        S = G_eur.subgraph([x for x, d in G_eur.nodes(data=True) if d['country'] == c]).copy()
        S.remove_nodes_from(list(nx.isolates(S)))
        if S.number_of_nodes() < N_MIN:
            continue
        S = S.subgraph(max(nx.connected_components(S), key=len)).copy()
        br = list(nx.bridges(S))
        total_m += S.number_of_edges(); total_br += len(br)
        survive += sum(1 for e in br if frozenset(e) in eur_bridges)
    return dict(n_europe=G_eur.number_of_nodes(), m_europe=G_eur.number_of_edges(),
                bridges_europe=len(eur_bridges), frac_bridges_europe=len(eur_bridges) / G_eur.number_of_edges(),
                national_bridges=total_br, national_edges=total_m, still_bridges_in_europe=survive,
                share_still_bridges=survive / total_br)


if __name__ == '__main__':
    buses, _, _ = load_tables()
    rows = []
    for c in sorted(buses['country'].unique()):
        G = build_country_graph(c)
        if G.number_of_nodes() >= N_MIN:
            rows.append(dict(country=c, **descriptors(G)))
    # sorted by n; ties (HR/LV, BG/PT) by decreasing m, as in Tables 1 and 2
    df = pd.DataFrame(rows).sort_values(['n', 'm'], ascending=[True, False]).reset_index(drop=True)
    out = ensure(RESULTS_DIR / 'descriptive')
    df.to_csv(out / 'descriptive_metrics.csv', index=False)
    pe = pan_european_bridges(df.country)
    pd.DataFrame([pe]).to_csv(out / 'pan_european_bridges.csv', index=False)
    print(f'{len(df)} countries; assortativity negative in {(df.assortativity < 0).sum()}')
    print(pe)
