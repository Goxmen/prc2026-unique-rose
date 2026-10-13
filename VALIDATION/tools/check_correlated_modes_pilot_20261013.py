"""Tiny synthetic production/auditor statistic parity before real-data freeze."""
import ast
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

import audit_correlated_modes_pilot_20261013 as audit

OUT=audit.ROOT/'VALIDATION/reports/cohort_learning_20261013/correlated_modes_pilot_numeric_fixture.json'


def producer_functions():
    tree=ast.parse(audit.PRODUCER.read_text())
    names=['group_cross','analytic_null','bootstrap_days','evaluate']
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert len(nodes)==4
    space=dict(np=np,pd=pd,PERMUTATIONS=199,BOOTSTRAPS=2000)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(audit.PRODUCER),'exec'),space)
    return space


def data():
    random=np.random.default_rng(20261013)
    dates=pd.date_range('2025-03-01',periods=60).strftime('%Y-%m-%d')
    rows=[]
    for date in dates:
        for airline in ['A','B']:
            for stratum in range(3):
                for repeat in range(3):
                    p=[.2,.5,.8][stratum];gap=9000.+33*repeat
                    z=int(random.random()<p)
                    rows.append(dict(airline=airline,day=date,month=date[:7],p_bin=stratum,
                        gap_band=0,clock_band=2,p=p,gap=gap,z=z,u=gap*(z-p),v=gap*gap*p*(1-p)))
    frame=pd.DataFrame(rows)
    frame['mvt_id']=np.arange(len(frame))
    frame['stratum']=pd.factorize(pd.MultiIndex.from_frame(frame[audit.STRATA]),sort=True)[0]
    frame['group']=pd.factorize(pd.MultiIndex.from_frame(frame[['airline','day']]),sort=True)[0]
    count=dict(retained_rows=len(frame),groups_with_pairs=frame.group.nunique(),days=frame.day.nunique())
    return frame,count


def compare_one(frame,count,namespace,con):
    report,days,permutations,draws=namespace['evaluate'](frame,count)
    actual_days,actual_permutations,actual_draws,metrics=audit.statistics(frame,con)
    for column in ['observed_cross','null_mean','mc_null_mean','denominator','numerator']:
        audit.close(days[column],actual_days[column])
    np.testing.assert_array_equal(permutations.iteration,actual_permutations.iteration)
    np.testing.assert_array_equal(permutations.null_cross,actual_permutations.null_cross)
    np.testing.assert_array_equal(permutations.excess_variance_ratio,actual_permutations.excess_variance_ratio)
    audit.close(draws,actual_draws,atol=1e-12,rtol=0)
    for key in ['denominator','observed_cross','null_mean_cross','mc_null_mean_cross','excess_variance_ratio','ci95','permutation_p']:
        audit.close(report[key],metrics[key],atol=1e-6,rtol=1e-12)
    assert report['gate']==metrics['gate']
    # Separate direct row-cluster bootstrap oracle; no daily SQL sums reused.
    random=np.random.default_rng(20261014)
    numerator=np.zeros(2000);denominator=np.zeros(2000)
    expected_null=dict(zip(actual_days.day,actual_days.null_mean))
    for month in sorted(frame.month.unique()):
        part=frame[frame.month.eq(month)];dates=sorted(part.day.unique())
        n=[];d=[]
        for date in dates:
            group=part[part.day.eq(date)]
            cross=0.
            for _,airline in group.groupby('airline',sort=True):
                values=airline.u.to_numpy(float)
                cross+=sum(left*right for i,left in enumerate(values) for j,right in enumerate(values) if i!=j)
            n.append(cross-expected_null[date]);d.append(float(group.v.sum()))
        take=random.integers(0,len(dates),size=(2000,len(dates)))
        numerator+=np.asarray(n)[take].sum(1);denominator+=np.asarray(d)[take].sum(1)
    audit.close(draws,numerator/denominator,atol=1e-12,rtol=0)
    return report


def main():
    assert not OUT.exists()
    namespace=producer_functions();frame,counts=data()
    with duckdb.connect(config={'threads':1,'memory_limit':'128MB'}) as con:
        con.extract_statements(audit.PAIR_SQL)
        result=compare_one(frame,counts,namespace,con)
        # All null statistics tie exactly: the >= plus-one convention returns1.
        tied=frame.copy();tied.u=4500.;tied.v=20250000.
        tied_result=compare_one(tied,counts,namespace,con)
        assert tied_result['permutation_p']==1. and not tied_result['gate']
    insufficient=namespace['evaluate'](frame.iloc[:3],dict(retained_rows=3,groups_with_pairs=1,days=1))
    assert insufficient[0]['status']=='inconclusive_insufficient_support' and not insufficient[0]['gate']
    assert insufficient[1:]==(None,None,None)
    value=dict(status='pass',source=audit.art(__file__),producer=audit.art(audit.PRODUCER),
        auditor=audit.art(audit.__file__),design=audit.art(audit.PLAN),synthetic_rows=len(frame),
        independent_SQL_analytic_null_and_observed_pairs=True,
        all_199_permutation_statistics_bit_exact=True,all_2000_bootstrap_draws_reconstructed=True,
        separate_explicit_raw_pair_day_bootstrap=True,all_tied_randomization_p_equals_one=True,
        sparse_support_is_inconclusive_not_null=True,no_actual_labels_predictions_models_read=True)
    audit.write(OUT,value)
    print(json.dumps(dict(status='pass',receipt=audit.art(OUT))))


if __name__=='__main__':main()
