"""Fixed input-only follow-up: does ID order add more than supplied IOBT?"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

import research_flight_id_order_20261012 as parent

ROOT = parent.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_flight_id_clock_residual'
DESIGN = dict(parent='passed monotonic flight-ID chronology pilot',
    sources='unchanged original January/July2025 and January/July2026 inputs',
    query_dates=parent.DATES, membership='unchanged per-month unique, consistent flight-ID eligibility',
    estimate='linear interpolation between the nearest strictly smaller and larger eligible flight IDs with known original IOBT; remove the query ID entirely; no extrapolation',
    clocks='Compare interpolated IOBT to supplied IOBT and EOBT1; compare EOBT1-IOBT to EOBT1-interpolated IOBT.',
    strata='each fixed month and origin, missingness; fixed absolute residual cutoffs60/300/900seconds',
    purpose='Detect redundant minute-resolution ordering versus a distinct clock anomaly; no creation-time assertion.',
    no_target_columns=True, no_fit=True, no_cv=True, no_feature_promotion=True, no_threshold_search=True,
    resources=dict(threads=1,duckdb_memory='256MB',external_timeout_seconds=60))


def art(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def freeze():
    assert not (OUT/'identity.json').exists()
    assert read(parent.OUT/'result.json')['decision']['retain_monotonic_chronology_hypothesis']
    assert read(ROOT/'VALIDATION/reports/source_clocks_20261012/flight_id_order_audit.json')['status']=='pass'
    sources=[Path(__file__),Path(parent.__file__),parent.OUT/'design.json',parent.OUT/'result.json',
             ROOT/'VALIDATION/reports/source_clocks_20261012/flight_id_order_audit.json']
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'identity.json').write_text(json.dumps(dict(design=DESIGN,sources=[art(p) for p in sources]),indent=2)+'\n')


def interpolate(ids, clocks):
    ids,clocks=np.asarray(ids),np.asarray(clocks,float)
    assert np.all(np.diff(ids)>0)
    reference=np.flatnonzero(np.isfinite(clocks));refids=ids[reference]
    left=np.searchsorted(refids,ids,side='left')-1
    right=np.searchsorted(refids,ids,side='right')
    valid=(left>=0)&(right<len(reference));positions=np.flatnonzero(valid)
    lower=reference[left[valid]];upper=reference[right[valid]]
    assert np.all(ids[lower]<ids[positions]) and np.all(ids[upper]>ids[positions])
    estimate=np.full(len(ids),np.nan)
    weight=(ids[positions]-ids[lower])/(ids[upper]-ids[lower])
    estimate[valid]=clocks[lower]+weight*(clocks[upper]-clocks[lower])
    # Independent scalar interpolation on deterministic locations, including missing queries.
    for at in positions[np.linspace(0,len(positions)-1,min(100,len(positions)),dtype=int)]:
        lo=max(i for i in reference if ids[i]<ids[at]);hi=min(i for i in reference if ids[i]>ids[at])
        scalar=(float(ids[hi]-ids[at])*clocks[lo]+float(ids[at]-ids[lo])*clocks[hi])/float(ids[hi]-ids[lo])
        assert abs(scalar-estimate[at])<=1e-6
    return estimate


def describe(frame):
    records=[]
    for (month,airport),group in frame.groupby(['month_key','origin']):
        known=group[['iobt','peer_iobt']].notna().all(axis=1)
        error=group.loc[known,'peer_iobt']-group.loc[known,'iobt']
        valid=group[['iobt','eobt','peer_iobt']].notna().all(axis=1)
        old=group.loc[valid,'eobt']-group.loc[valid,'iobt'];new=group.loc[valid,'eobt']-group.loc[valid,'peer_iobt']
        records.append(dict(month=month,airport=airport,queries=len(group),known_iobt=int(group.iobt.notna().sum()),
            interpolated=int(group.peer_iobt.notna().sum()),missing_iobt_recovered=int((group.iobt.isna()&group.peer_iobt.notna()).sum()),
            residual_count=len(error),residual_quantiles=error.quantile([0,.01,.1,.5,.9,.99,1]).tolist(),
            abs_residual_quantiles=error.abs().quantile([.5,.9,.99,1]).tolist(),
            residual_counts={str(t):int(error.abs().gt(t).sum()) for t in [60,300,900]},
            eobt_vs_original_iobt_rmse=float(np.sqrt(np.mean(old**2))),
            eobt_vs_peer_iobt_rmse=float(np.sqrt(np.mean(new**2)))))
    return records


def pilot():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    for rec in identity['sources']:assert art(ROOT/rec['path'])==rec
    assert not (OUT/'result.json').exists()
    started=time.monotonic();frames=[]
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for record in read(parent.OUT/'design.json')['source_records']:
            path=ROOT/record['path'];assert path.stat().st_size==record['bytes'] and path.stat().st_mtime_ns==record['mtime_ns']
            months=parent.MONTHS[2:] if path.name=='ranking.parquet' else [m for m in parent.MONTHS[:2] if m in path.name]
            parent.make_source(con,path,months)
            frame=con.execute('SELECT month_key,flight_id,origin,dep_day,dep_id,iobt,eobt FROM eligible ORDER BY month_key,flight_id').df()
            for month,group in frame.groupby('month_key'):
                group=group.copy();group['peer_iobt']=interpolate(group.flight_id.to_numpy(),group.iobt.to_numpy())
                frames.append(group.loc[group.dep_day.isin(parent.DATES)])
    frame=pd.concat(frames,ignore_index=True);assert frame.dep_id.is_unique
    frame.to_parquet(OUT/'diagnostic_rows.parquet',index=False)
    result=dict(identity=art(OUT/'identity.json'),rows=len(frame),reports=describe(frame),
        diagnostic_rows=art(OUT/'diagnostic_rows.parquet'),seconds=time.monotonic()-started,
        no_targets_read=True,no_predictive_feature_selected=True)
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(dict(rows=len(frame),seconds=result['seconds']))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','pilot'])
    globals()[parser.parse_args().stage]()
