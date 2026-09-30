"""Tables 6-9 as LaTeX rows.

Table 6 (Section 5.2.2): held-out statistics <l>, diameter, lambda_2, maximum betweenness and
         assortativity: real grid (Act.) and mean (SD) over 30 synthetic samples (Sim.)
Table 7 (Section 5.3): robustness index R, random failure and targeted attack, Act. and Sim.
Table 8 (Appendix A): fitted mixing terms (observed, mean over all retained samples, tau)
Table 9 (Appendix A): fitted terms AKS, t1, t2 (tau undefined if all samples are equal)
The standardized deviations z of Tables 6-7 are shown in Figures 12-13.

Inputs:  results/evaluation/tau_fitted_terms.csv, heldout_per_statistic.csv
Outputs: results/tables/table6_heldout.tex ... table9_fitted_other.tex
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.grid import MODELED_COUNTRIES
from common.paths import RESULTS_DIR, ensure

NAMES = {'AL': 'Albania', 'HR': 'Croatia', 'BA': 'Bosnia and Herzegovina', 'SK': 'Slovakia',
         'RS': 'Serbia$^\\dagger$', 'CZ': 'Czech Republic', 'AT': 'Austria', 'BG': 'Bulgaria',
         'PT': 'Portugal', 'RO': 'Romania', 'CH': 'Switzerland', 'PL': 'Poland', 'IT': 'Italy',
         'DE': 'Germany', 'ES': 'Spain'}


def f(v, d):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '--'
    s = f'{v:.{d}f}'
    if float(s) == 0:
        s = s.lstrip('-')
    return s.replace('-', '$-$')


def rows(fmt):
    return [fmt(c) for c in MODELED_COUNTRIES]


if __name__ == '__main__':
    tau = pd.read_csv(RESULTS_DIR / 'evaluation' / 'tau_fitted_terms.csv').set_index(['country', 'term'])
    ho = pd.read_csv(RESULTS_DIR / 'evaluation' / 'heldout_per_statistic.csv')
    ho = ho[ho.model == 'final'].set_index(['country', 'stat'])
    n = tau.reset_index().groupby('country').n.first()

    def fitted(c, terms, act_d):
        cells = []
        for t in terms:
            r = tau.loc[(c, t)]
            cells += [f(r.target, act_d[t]), f(r['mean'], 1), f(r.tau_SD, 2)]
        return f'{NAMES[c]} & {n[c]} & ' + ' & '.join(cells) + ' \\\\'

    def act_sim(c, spec):
        cells = []
        for s, d in spec:
            r = ho.loc[(c, s)]
            cells += [f(r.observed, d), f'{f(r["mean"], d)} ({f(r.sd, d)})']
        return f'{NAMES[c]} & {n[c]} & ' + ' & '.join(cells) + ' \\\\'

    tables = {
        'table6_heldout': rows(lambda c: act_sim(c, [('avg_path_len', 2), ('diameter', 1), ('lambda2', 3),
                                                   ('betw_max', 3), ('assortativity', 2)])),
        'table7_robustness': rows(lambda c: act_sim(c, [('R_edge_random', 3), ('R_edge_attack', 3)])),
        'table8_fitted_mixing': rows(lambda c: fitted(c, ['m_LL', 'm_LH', 'm_HH'], dict(m_LL=0, m_LH=0, m_HH=0))),
        'table9_fitted_other': rows(lambda c: fitted(c, ['akstar', 't1', 't2'], dict(akstar=1, t1=0, t2=0))),
    }
    out = ensure(RESULTS_DIR / 'tables')
    for name, lines in tables.items():
        (out / f'{name}.tex').write_text('\n'.join(lines) + '\n')
    print('written', ', '.join(tables))
