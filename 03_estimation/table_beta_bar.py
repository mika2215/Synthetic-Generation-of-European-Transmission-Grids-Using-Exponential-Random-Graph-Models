"""Table 5 (Section 5.2.1): estimated parameters beta_bar of the final model, 15 countries.

Input:  results/beta_bar/final/beta_<CC>.npy
Output: results/tables/table5_beta_bar.csv and table5_beta_bar.tex (LaTeX rows)
An asterisk marks terms whose observed value is zero, so that the maximum-likelihood estimate
does not exist and the reported value is the one the EE chain reached within its run.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import MODELED_COUNTRIES, build_country_graph
from common.paths import RESULTS_DIR, ensure
from common.statistics import FINAL_KEYS, CountryModel

NAMES = {'AL': 'Albania', 'HR': 'Croatia', 'BA': 'Bosnia and Herzegovina', 'SK': 'Slovakia',
         'RS': 'Serbia$^\\dagger$', 'CZ': 'Czech Republic', 'AT': 'Austria', 'BG': 'Bulgaria',
         'PT': 'Portugal', 'RO': 'Romania', 'CH': 'Switzerland', 'PL': 'Poland', 'IT': 'Italy',
         'DE': 'Germany', 'ES': 'Spain'}
TERMS = ['LL', 'LH', 'HH', 'AKS', 't1', 't2']

if __name__ == '__main__':
    rows, tex = [], []
    for c in MODELED_COUNTRIES:
        model = CountryModel(build_country_graph(c), FINAL_KEYS)
        b = np.load(RESULTS_DIR / 'beta_bar' / 'final' / f'beta_{c}.npy')
        zero = dict(t1=model.ref['t1'] == 0, t2=model.ref['t2'] == 0)
        rows.append(dict(country=c, n=model.n, **{f'beta_{t}': v for t, v in zip(TERMS, b)},
                         t1_target_zero=zero['t1'], t2_target_zero=zero['t2']))
        cells = [f'{(0.0 if round(v, 2) == 0 else v):.2f}'.replace('-', '$-$') + ('$^{*}$' if t in zero and zero[t] else '')
                 for t, v in zip(TERMS, b)]
        tex.append(f'{NAMES[c]} & {model.n} & ' + ' & '.join(cells) + ' \\\\')
    out = ensure(RESULTS_DIR / 'tables')
    pd.DataFrame(rows).to_csv(out / 'table5_beta_bar.csv', index=False)
    (out / 'table5_beta_bar.tex').write_text('\n'.join(tex) + '\n')
    print('\n'.join(tex))
