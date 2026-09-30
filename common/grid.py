"""Construction of the country graphs from the PyPSA-Eur tables (Section 3.2).

Nodes are buses, edges are AC lines and transformers; parallel lines collapse into one edge.
Voltages 400 kV and 225 kV are remapped to 380 kV and 220 kV, HVDC buses are removed, and
each country graph is restricted to its largest connected component. Buses below 300 kV form
the L layer, all others the H layer.
"""
from functools import lru_cache

import networkx as nx
import pandas as pd

from .paths import DATA_DIR

VOLT_REMAP = {400: 380, 225: 220}
HVDC_VOLTS = {320, 450, 500, 515, 525, 600, 750}
VOLT_THRESH = 300          # kV; below: L layer, at or above: H layer
N_MIN = 20                 # minimum number of nodes for the descriptive sample (29 countries)

# The 15 countries fitted with the ERGM, sorted by n (Section 5.1).
MODELED_COUNTRIES = ['AL', 'HR', 'BA', 'SK', 'RS', 'CZ', 'AT', 'BG', 'PT', 'RO', 'CH', 'PL',
                     'IT', 'DE', 'ES']


@lru_cache(maxsize=1)
def load_tables():
    """Read buses, lines and transformers of the PyPSA-Eur network."""
    buses = pd.read_csv(DATA_DIR / 'buses.csv')
    lines = pd.read_csv(DATA_DIR / 'lines.csv', engine='python', quotechar="'",
                        on_bad_lines='skip')
    transformers = pd.read_csv(DATA_DIR / 'transformers.csv', engine='python', quotechar="'",
                               on_bad_lines='skip')
    return buses, lines, transformers


def build_country_graph(country_code=None):
    """Graph of one country, or of the whole continent if `country_code` is None.

    The pan-European graph is not restricted to its largest component (Section 3.3 does this
    explicitly where needed). Node attributes: voltage, x, y, country.
    """
    buses, lines, transformers = load_tables()
    cb = buses.copy() if country_code is None else buses[buses['country'] == country_code].copy()
    cb['voltage'] = cb['voltage'].replace(VOLT_REMAP)
    cb = cb[~cb['voltage'].isin(HVDC_VOLTS)]
    cb_ids = set(cb['bus_id'])
    cl = lines[lines['bus0'].isin(cb_ids) & lines['bus1'].isin(cb_ids)]
    ct = transformers[transformers['bus0'].isin(cb_ids) & transformers['bus1'].isin(cb_ids)]
    G = nx.Graph()
    for _, row in cb.iterrows():
        G.add_node(row['bus_id'], voltage=row['voltage'], x=row['x'], y=row['y'],
                   country=row['country'])
    for _, row in cl.iterrows():
        G.add_edge(row['bus0'], row['bus1'])
    for _, row in ct.iterrows():
        G.add_edge(row['bus0'], row['bus1'])
    G.remove_nodes_from(list(nx.isolates(G)))
    if country_code is not None and G.number_of_nodes() > 0 and not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    return G


def voltage_layers(G):
    """Map node -> 'L' or 'H'."""
    return {nd: ('L' if G.nodes[nd]['voltage'] < VOLT_THRESH else 'H') for nd in G.nodes()}


def layer(G, node):
    return 'H' if G.nodes[node]['voltage'] >= VOLT_THRESH else 'L'
