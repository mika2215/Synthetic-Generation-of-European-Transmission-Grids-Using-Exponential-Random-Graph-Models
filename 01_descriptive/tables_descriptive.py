"""Tables 1 and 2 (Chapter 3) from results/descriptive/descriptive_metrics.csv.

Table 1: voltage-layer composition (n_L, n_H, m_LL, m_LH, m_HH), 29 countries, sorted by n.
Table 2: core descriptors (n, m, <k>, C, assortativity, <l>, diameter, lambda_2, bridges %).
Output: results/tables/table1_voltage_layers.tex, results/tables/table2_descriptors.tex (rows)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.paths import RESULTS_DIR, ensure

NAMES = {'AL': 'Albania', 'AT': 'Austria', 'BA': 'Bosnia and Herzegovina', 'BE': 'Belgium',
         'BG': 'Bulgaria', 'CH': 'Switzerland', 'CZ': 'Czech Republic', 'DE': 'Germany',
         'DK': 'Denmark', 'ES': 'Spain', 'FI': 'Finland', 'FR': 'France', 'GB': 'United Kingdom',
         'GR': 'Greece', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland', 'IT': 'Italy',
         'LT': 'Lithuania', 'LV': 'Latvia', 'NL': 'Netherlands', 'NO': 'Norway', 'PL': 'Poland',
         'PT': 'Portugal', 'RO': 'Romania', 'RS': 'Serbia', 'SE': 'Sweden', 'SK': 'Slovakia',
         'UA': 'Ukraine'}

if __name__ == '__main__':
    df = pd.read_csv(RESULTS_DIR / 'descriptive' / 'descriptive_metrics.csv')
    t1 = [f"{r.country} & {r.n_L} & {r.n_H} & {r.m_LL} & {r.m_LH} & {r.m_HH} \\\\" for r in df.itertuples()]
    t2 = [f"{NAMES[r.country]} & {r.n} & {r.m} & {r.k_mean:.2f} & {r.clustering:.3f} & "
          f"{r.assortativity:.3f} & {r.avg_path_len:.2f} & {r.diameter} & {r.lambda2:.3f} & "
          f"{100 * r.frac_bridge:.1f} \\\\" for r in df.itertuples()]
    out = ensure(RESULTS_DIR / 'tables')
    (out / 'table1_voltage_layers.tex').write_text('\n'.join(t1) + '\n')
    (out / 'table2_descriptors.tex').write_text('\n'.join(t2) + '\n')
    print('\n'.join(t1[:2] + t2[:2]))
