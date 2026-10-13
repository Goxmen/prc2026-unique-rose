"""Exact stratified permutation null: independent finite enumeration fixture.

Synthetic rows only. No producer imports, actual data, model or fit. The closed
form is the expectation of ordered within-airline/day residual products under
independent uniform permutations inside fixed nuisance strata.
"""
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/reports/cohort_learning_20261013/correlated_modes_analytic_null_fixture.json'


def expectation(frame):
    stats = {}
    for stratum, rows in frame.groupby('stratum', sort=True):
        values = rows.u.to_numpy(float)
        n, total, squares = len(values), values.sum(), values@values
        stats[stratum] = (total/n, (total*total-squares)/(n*(n-1)) if n > 1 else 0.)
    result = {}
    for group, rows in frame.groupby(['airline','day'], sort=True):
        counts = rows.stratum.value_counts()
        loads = [count*stats[stratum][0] for stratum, count in counts.items()]
        same_stratum = sum(count*(count-1)*stats[stratum][1] for stratum,count in counts.items())
        cross_strata = sum(left*right for i,left in enumerate(loads) for j,right in enumerate(loads) if i != j)
        result[group] = float(same_stratum+cross_strata)
    return result


def main():
    assert not OUT.exists()
    # Two nonconstant four-slot strata, with unequal per-day counts, plus one
    # singleton stratum and another airline. Constants/zeros exercise signs.
    frame = pd.DataFrame(dict(airline=['A']*9+['B'],day=['01','01','02','03','01','02','03','04','01','01'],
        stratum=['s1']*4+['s2']*4+['s3','s4'],u=[-.5,.2,.7,-.1,.8,-.3,.4,-.9,.6,0.]))
    expected = expectation(frame)
    blocks = [list(rows.index) for _,rows in frame.groupby('stratum',sort=True)]
    permutations = [list(itertools.permutations(block)) for block in blocks]
    totals = {key:0. for key in expected}
    count = 0
    for choices in itertools.product(*permutations):
        value = frame.u.to_numpy().copy()
        for slots, order in zip(blocks,choices):
            value[slots] = frame.u.to_numpy()[list(order)]
        for key, rows in frame.groupby(['airline','day'],sort=True):
            ix = list(rows.index)
            # Explicit ordered pairs: no sum-of-squares identity reused.
            totals[key] += sum(value[i]*value[j] for i in ix for j in ix if i != j)
        count += 1
    errors = {str(key):abs(totals[key]/count-expected[key]) for key in expected}
    assert count == 576 and max(errors.values()) < 1e-13
    # Common constant residual bias produces its exact nonzero null; do not
    # subtract independent mean-zero innovations unless supported by evidence.
    constant = frame.copy();constant.u = .4
    exact = expectation(constant)
    for key,rows in constant.groupby(['airline','day'],sort=True):
        np.testing.assert_allclose(exact[key],len(rows)*(len(rows)-1)*.16,rtol=0,atol=1e-14)
    report = dict(status='pass',source=dict(path=str(Path(__file__).relative_to(ROOT)),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),synthetic_rows=len(frame),
        exhaustive_independent_permutations=count,maximum_error=max(errors.values()),
        formula='sum_s n_gs(n_gs-1)*(S_s^2-Q_s)/(N_s(N_s-1)) + sum_{s!=t} n_gs*n_gt*(S_s/N_s)*(S_t/N_t)',
        singleton_strata_same_stratum_term_zero=True,unequal_day_counts_and_cross_stratum_terms_checked=True,
        constant_propensity_mean_preserved=True,no_actual_labels_predictions_or_models_read=True,
        bootstrap_scope='Future day bootstrap conditions on these empirical stratum marginals; it does not propagate nuisance-estimation uncertainty.')
    with OUT.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='pass',receipt=dict(path=str(OUT.relative_to(ROOT)),
        sha256=hashlib.sha256(OUT.read_bytes()).hexdigest()))))


if __name__=='__main__':main()
