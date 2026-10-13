"""Independent synthetic ordered-pair statistic oracle; no experimental data.

This checks the algebra beneath a proposed same-day versus cross-day residual
dependence diagnostic. It does not define or authorize the experiment's gate,
nuisance strata, bootstrap, population or production dependence model.
"""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'VALIDATION/reports/cohort_learning_20261013/rome_mode_dependence_pair_math_fixture.json'
BASE = ['airline', 'month', 'stratum']
GROUP = [*BASE, 'day']
SQL = '''WITH g AS (
    SELECT airline,month,stratum,day,count(*) AS n,sum(residual) AS s,
        sum(residual*residual) AS q FROM f GROUP BY 1,2,3,4),
    a AS (SELECT airline,month,stratum,sum(n) AS n,sum(s) AS s,
        sum(s*s) AS sum_s2,sum(n*n) AS sum_n2,
        sum(s*s-q) AS within_num,sum(n*(n-1)) AS within_den
        FROM g GROUP BY 1,2,3)
    SELECT airline,month,stratum,within_num,within_den,
        s*s-sum_s2 AS cross_num,n*n-sum_n2 AS cross_den
        FROM a ORDER BY 1,2,3'''


def explicit_pairs(frame):
    """Deliberately slow reference: all ordered i!=j pairs inside each stratum."""
    rows = []
    for keys, group in frame.groupby(BASE, sort=True):
        within_num = cross_num = 0.
        within_den = cross_den = 0
        tuples = list(group.itertuples(index=False))
        for i, left in enumerate(tuples):
            for j, right in enumerate(tuples):
                if i == j:
                    continue
                product = left.residual*right.residual
                if left.day == right.day:
                    within_num += product;within_den += 1
                else:
                    cross_num += product;cross_den += 1
        rows.append(dict(zip(BASE, keys), within_num=within_num, within_den=within_den,
                         cross_num=cross_num, cross_den=cross_den))
    return pd.DataFrame(rows)


def numpy_groups(frame):
    rows = []
    for keys, group in frame.groupby(BASE, sort=True):
        counts, sums, squares = [], [], []
        for _, day in group.groupby('day', sort=False):
            x = day.residual.to_numpy(float)
            counts.append(len(x));sums.append(x.sum());squares.append(x@x)
        n, s, q = map(np.asarray, [counts, sums, squares])
        rows.append(dict(zip(BASE,keys), within_num=float(np.sum(s*s-q)),
            within_den=int(np.sum(n*(n-1))),cross_num=float(s.sum()**2-s@s),
            cross_den=int(n.sum()**2-n@n)))
    return pd.DataFrame(rows)


def compare(a, b):
    pd.testing.assert_frame_equal(a[BASE], b[BASE], check_dtype=False)
    np.testing.assert_allclose(a.drop(columns=BASE).to_numpy(float),
                               b.drop(columns=BASE).to_numpy(float),rtol=0,atol=1e-12)


def main():
    assert not OUT.exists()
    # Unequal groups, unsorted dates, two airlines, two months, and sparse strata.
    f = pd.DataFrame(dict(airline=['A']*9+['B']*4+['A']*4,
        month=['2025-03']*13+['2025-04']*4,
        stratum=[0]*6+[1]*3+[0]*4+[0]*4,
        day=['03','01','03','01','02','02','01','02','03','01','01','02','03','02','01','02','01'],
        residual=[.7,-.1,-.3,.2,.8,-.4,.5,-.6,.2,-.1,.3,-.2,.4,.1,.1,-.2,.3]))
    oracle = explicit_pairs(f)
    compare(oracle,numpy_groups(f))
    with duckdb.connect(config={'threads':1,'memory_limit':'128MB'}) as con:
        con.register('f',f);compare(oracle,con.execute(SQL).fetchdf())
        shuffled = f.sample(frac=1,random_state=5).reset_index(drop=True)
        con.register('f',shuffled);compare(oracle,con.execute(SQL).fetchdf())
    sparse = oracle[(oracle.airline=='A') & (oracle.month=='2025-03') & (oracle.stratum==1)].iloc[0]
    assert sparse.within_den == 0 and sparse.cross_den == 6
    # A common airline/month residual bias alone is removed by cross-day control.
    bias = pd.DataFrame(dict(airline=['A']*6,month=['2025-03']*6,stratum=[0]*6,
                            day=['01','01','02','02','03','03'],residual=[.4]*6))
    row = explicit_pairs(bias).iloc[0]
    assert abs(row.within_num/row.within_den-row.cross_num/row.cross_den) < 1e-14
    # Same-day positive dependence must survive the cross-day contrast.
    burst = bias.copy();burst.residual = [.4,.4,-.4,-.4,0.,0.]
    row = explicit_pairs(burst).iloc[0]
    assert row.within_num/row.within_den-row.cross_num/row.cross_den > 0
    report = dict(status='pass',source=dict(path=str(Path(__file__).relative_to(ROOT)),
        sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),synthetic_rows=len(f),
        independent_ordered_pair_SQL_NumPy_agree=True,unequal_group_and_sparse_strata_checked=True,
        order_invariance=True,constant_propensity_bias_cancels=True,positive_burst_detected=True,
        zero_within_pair_denominator_is_insufficient_coverage=True,
        no_actual_labels_predictions_or_checkpoint_access=True,
        scope='Sufficient-statistic algebra only; proposed experiment normalization/bootstrap/gate remain separately reviewed.')
    with OUT.open('x') as stream:json.dump(report,stream,indent=2);stream.write('\n')
    print(json.dumps(dict(status='pass',receipt=dict(path=str(OUT.relative_to(ROOT)),
        sha256=hashlib.sha256(OUT.read_bytes()).hexdigest()))))


if __name__=='__main__':
    main()
