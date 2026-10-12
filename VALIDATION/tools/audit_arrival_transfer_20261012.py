"""Independent small saved-projection arithmetic audit; no original raw reread."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'tmp/research_20261012/signal_scout/arrival_departure_transfer'


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def airline(value):
    if not isinstance(value,str) or len(value)<3: return None
    return value[:3] if value[:3].isalpha() else value[:2]


def means(frame,value):
    airports=defaultdict(lambda:[0.,0]);groups=defaultdict(lambda:[0.,0])
    for airport,air,number in frame[['airport','airline',value]].itertuples(index=False,name=None):
        airports[airport][0]+=number;airports[airport][1]+=1
        if isinstance(air,str):groups[(airport,air)][0]+=number;groups[(airport,air)][1]+=1
    def prediction(airport,air):
        total,n=groups[(airport,air)];asum,an=airports[airport]
        return (total+20*asum/an)/(n+20)
    return prediction


def run():
    started=time.monotonic();record=json.loads((OUT/'result.json').read_text())
    for name,expected in record['outputs'].items():assert sha(OUT/name)==expected
    arr=pd.read_parquet(OUT/'shared_training_arrival_clocks.parquet')
    arr=arr[arr.delta.notna()].copy();arr['airline']=[airline(x)for x in arr.flight];arr['z']=arr.delta.clip(-600,600)
    dep=pd.read_parquet(OUT/'shared_training_departure_clocks.parquet').set_index('mvt_id')
    q=pd.read_parquet(OUT/'diagnostic_rows.parquet');source=dep.loc[q.mvt_id]
    assert set(q.date)==set(record['design']['dates'])
    np.testing.assert_array_equal(q.raw_target,source.raw_target)
    np.testing.assert_array_equal(q.target,np.clip(source.raw_target,-3600,3600))
    sample=q.sort_values('mvt_id').groupby('date',sort=True).head(3)
    worst=0.;cases=0
    for date,queries in sample.groupby('date'):
        static=means(arr[arr.date.ne(date)],'z')
        for row in queries.itertuples():
            donors=arr[arr.date.eq(date)&arr.airport.eq(row.airport)&arr.t.ge(row.t-7200)&arr.t.lt(row.t)]
            prior=static(row.airport,row.airline);n=len(donors);total=float(donors.z.sum())
            x=(total+20*prior)/(n+20)-prior
            np.testing.assert_allclose([prior,n,total,x],[row.arrival_static,row.support,row.arrival_sum,row.x],atol=1e-10,rtol=0)
            worst=max(worst,abs(x-row.x));cases+=1
    dates=[min(q.date),max(q.date)];slopes=[]
    for date in dates:
        tr=q[q.date.ne(date)];held=q[q.date.eq(date)];static=means(tr,'target')
        trbase=np.array([static(a,b)for a,b in tr[['airport','airline']].itertuples(index=False,name=None)])
        beta=float(np.sum(tr.x.to_numpy()*(tr.target.to_numpy()-trbase))/np.sum(tr.x.to_numpy()**2))
        heldbase=np.array([static(a,b)for a,b in held[['airport','airline']].itertuples(index=False,name=None)])
        np.testing.assert_allclose(heldbase,held.baseline,atol=1e-10,rtol=0)
        np.testing.assert_allclose(heldbase+beta*held.x,held.candidate,atol=1e-10,rtol=0)
        expected=next(r['beta']for r in record['coefficients']if r['held_date']==date)
        np.testing.assert_allclose(beta,expected,atol=1e-12,rtol=0);slopes.append(dict(date=date,beta=beta,held_rows=len(held)))
    result=dict(status='pass',source_sha256=sha(__file__),result_sha256=sha(OUT/'result.json'),
        rows=len(q),all_saved_projection_target_relationships_exact=True,causal_context_queries=cases,
        manual_dictionary_static_and_slope_checks=slopes,maximum_context_difference=worst,
        scope='saved-projection independent arithmetic; raw-source binding inherited from prior audited pass',
        seconds=time.monotonic()-started)
    (OUT/'independent_arithmetic_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':run()
