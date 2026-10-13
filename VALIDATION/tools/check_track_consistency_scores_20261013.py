"""Tiny independent SQL/bootstrap/gate fixtures; no real labels or predictions."""
import ast
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import duckdb
import audit_track_consistency_scores_20261013 as audit

ROOT=audit.ROOT
OUT=ROOT/'VALIDATION/reports/cohort_learning_20261013/track_consistency_score_fixture.json'


def main():
    assert not OUT.exists()
    ast.parse(Path(audit.__file__).read_text());ast.parse(audit.SCORER.read_text())
    # Deliberately unsorted days and fractional truth exercise first-occurrence
    # cluster order and floating target arithmetic, independent of real data.
    f=pd.DataFrame(dict(mvt_id=np.arange(8,dtype=float),month=['2025-01']*4+['2025-07']*4,
        day=['2025-01-03','2025-01-01','2025-01-03','2025-01-02',
             '2025-07-02','2025-07-01','2025-07-02','2025-07-03'],
        y=[1.5,4.25,7.75,8.25,3.5,1.5,5.25,8.75],v80=[0.,4.,9.,12.,2.,4.,8.,13.],
        supervised_control=[0.,3.,9.,11.,1.,3.,8.,12.],training_consistency=[0.,4.,8.,12.,2.,4.,7.,11.],
        target_consistency=[2.,4.,8.,8.,4.,2.,5.,9.]))
    for arm in audit.ARMS:f[arm+'_component_delta']=f[arm]-f.v80
    o=f[['mvt_id','month','day','y']].copy();o['selected']=10.;o['one_tabm']=11.;o['three_tabm']=20.
    c=f[['mvt_id','month','day','y']].copy();c['three_tabm_runway']=f.v80
    r=f[['mvt_id','month','day','y']].copy()
    parsed=0
    with duckdb.connect(config={'threads':1,'memory_limit':'128MB'}) as con:
        for name,frame in [('f',f),('o',o),('c',c),('r',r)]:con.register(name,frame)
        for arm in audit.ARMS:
            d=o.copy();d['three_tabm']+=f[arm+'_component_delta'];con.register('d',d)
            query=audit.transport_sql(arm);parsed+=len(con.extract_statements(query))
            assert con.execute(query).fetchone()==(8,8,0)
            corrupted=d.copy();corrupted.loc[0,'selected']+=1;con.register('d',corrupted)
            assert con.execute(query).fetchone()[2]==1
            con.unregister('d')
        metrics={row[0]:dict(zip(['v80',*audit.ARMS],row[1:])) for row in con.execute(audit.metrics_sql(['v80',*audit.ARMS])).fetchall()}
        parsed+=len(con.extract_statements(audit.metrics_sql(['v80',*audit.ARMS])))
        day=con.execute(audit.days_sql()).fetchdf();parsed+=len(con.extract_statements(audit.days_sql()))
        assert day.day.tolist()==['2025-01-03','2025-01-01','2025-01-02','2025-07-02','2025-07-01','2025-07-03']
        rounded=con.execute('SELECT round_even(greatest(x,0.),0) FROM (VALUES (-1.5),(.5),(1.5),(2.5),(3.5)) t(x)').fetchnumpy()
        np.testing.assert_array_equal(next(iter(rounded.values())),[0,0,2,2,4])
    actual=audit.bootstrap(day,metrics)
    # Brute independent oracle resamples original row groups, not SQL sums.
    for ref in audit.REFERENCES:
        rng=np.random.default_rng(0);draws=np.zeros(2000)
        for month in audit.MONTHS:
            g=f[f.month.eq(month)];labels=list(dict.fromkeys(g.day));groups=[g[g.day.eq(d)] for d in labels]
            chosen=rng.integers(0,len(groups),size=(2000,len(groups)))
            for repetition,which in enumerate(chosen):
                sample=pd.concat([groups[int(k)] for k in which],ignore_index=True)
                draws[repetition]+=(np.sqrt(np.mean((sample.target_consistency-sample.y)**2))-
                    np.sqrt(np.mean((sample[ref]-sample.y)**2)))/2
        expected=np.percentile(draws,[2.5,97.5]);audit.close(actual[ref]['ci95'],expected,1e-12)
    passed=audit.gate(metrics,actual);assert passed['passed']
    weak={m:dict(values,target_consistency=values['v80']-.05) for m,values in metrics.items()}
    assert not audit.gate(weak,actual)['passed'] and not audit.gate(weak,actual)['minimum_gain']
    worse={m:dict(values) for m,values in metrics.items()};worse['2025-01']['target_consistency']=worse['2025-01']['training_consistency']+.1
    assert not audit.gate(worse,actual)['passed']
    uncertain={k:dict(v) for k,v in actual.items()};uncertain['supervised_control']=dict(uncertain['supervised_control'],ci95=[-1.,.00001])
    assert not audit.gate(metrics,uncertain)['passed']
    value=dict(status='pass',source=audit.art(Path(__file__)),auditor=audit.art(Path(audit.__file__)),scorer=audit.art(audit.SCORER),
        SQL_statements_parsed=parsed,synthetic_rows=8,all_three_transports_and_original_controls_exact=True,
        corrupted_original_controls_rejected=True,nonnegative_nearest_even_fixture=True,
        unsorted_first_occurrence_day_order_exact=True,independent_raw_group_bootstrap_2000_draws_exact=True,
        gate_minimum_both_months_all_three_references_checked=True,no_real_data_no_fit_no_inference=True)
    with OUT.open('x') as stream:stream.write(json.dumps(value,indent=2)+'\n')
    print(json.dumps(dict(status='pass',receipt=audit.art(OUT))))


if __name__=='__main__':main()
