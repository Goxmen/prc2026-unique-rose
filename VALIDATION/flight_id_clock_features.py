"""Two label-free, month-local peer-clock features; no absolute identifier input."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'VALIDATION/tools'))
import research_flight_id_order_20261012 as source

OUT = ROOT/'data/derived/flight_id_clock_20261012'
FEATURES = ['idclock_iobt_residual', 'idclock_neighbor_span']
DESIGN = dict(features=FEATURES,
    population='all original2025 departures and accepted old Jan/Jul2026 departures; phases retained only for source consistency checks',
    eligible=source.DESIGN['reference'],
    reference='same movement-UTC-month eligible departures across all origins; known original IOBT only',
    peers='nearest strictly smaller/larger numeric flight ID, exclude own entire flight ID, require both sides, no extrapolation',
    idclock_iobt_residual='interpolated peer IOBT minus own supplied IOBT, seconds',
    idclock_neighbor_span='larger-ID peer IOBT minus smaller-ID peer IOBT, seconds; signed, no clipping',
    missing='NaN for ineligible/missing identifiers or absent bracketing peers; residual alsoNaN when own IOBT missing',
    availability='uses only supplied within-month inputs; transductive calendar context, no departure labels',
    no_raw_id_feature=True,no_labels=True,no_date_thresholds=True,no_source_redownload=True,
    interpretation='relative schedule-order anomalies; numeric ID is not asserted to encode a creation time')


def art(path):
    path=Path(path)
    return dict(path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def read(path):
    return json.loads(Path(path).read_text())


def sources():
    paths=sorted((ROOT/'data').glob('training_2025-*.parquet'))
    assert len(paths)==12
    return paths+[ROOT/'data/ranking.parquet']


def freeze():
    assert not (OUT/'identity.json').exists()
    OUT.mkdir(parents=True,exist_ok=True)
    paths=[Path(__file__),Path(source.__file__),ROOT/'VALIDATION/tools/probe_flight_id_clock_residual_20261012.py',
        ROOT/'VALIDATION/runs/v3_20261012_flight_id_clock_residual/result.json',
        ROOT/'VALIDATION/reports/source_clocks_20261012/flight_id_order_audit.json']
    value=dict(design=DESIGN,source=art(__file__),helpers=[art(p) for p in paths],raw_sources=[art(p) for p in sources()])
    (OUT/'identity.json').write_text(json.dumps(value,indent=2)+'\n')


def guard():
    frozen=read(OUT/'identity.json');assert frozen['design']==DESIGN and frozen['source']==art(__file__)
    assert [ROOT/r['path'] for r in frozen['raw_sources']]==sources()
    for rec in [*frozen['helpers'],*frozen['raw_sources']]:assert art(ROOT/rec['path'])==rec
    return frozen


def table(con):
    return con.execute('''WITH known AS (
        SELECT month_key,flight_id,iobt FROM eligible WHERE iobt IS NOT NULL
    ), joined AS (
        SELECT e.dep_id AS mvt_id,e.month_key,e.flight_id,e.iobt,
          l.flight_id lo_id,l.iobt lo_clock,r.flight_id hi_id,r.iobt hi_clock
        FROM eligible e
        ASOF LEFT JOIN known l ON e.month_key=l.month_key AND e.flight_id>l.flight_id
        ASOF LEFT JOIN known r ON e.month_key=r.month_key AND e.flight_id<r.flight_id
    ) SELECT mvt_id,month_key,flight_id,iobt,lo_id,lo_clock,hi_id,hi_clock,
        CASE WHEN lo_id IS NOT NULL AND hi_id IS NOT NULL THEN
          lo_clock+(flight_id-lo_id)::DOUBLE/(hi_id-lo_id)*(hi_clock-lo_clock)-iobt END idclock_iobt_residual,
        CASE WHEN lo_id IS NOT NULL AND hi_id IS NOT NULL THEN hi_clock-lo_clock END idclock_neighbor_span
        FROM joined''').df()


def build():
    identity=guard();assert not (OUT/'manifest.json').exists()
    outputs={'training':[],'old':[]};scalar_checks=0;coverage=[]
    with duckdb.connect(config={'threads':1,'memory_limit':'512MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for view,path,months in [
            ('training',ROOT/'data/training_2025-*.parquet',[f'2025-{m:02d}' for m in range(1,13)]),
            ('old',ROOT/'data/ranking.parquet',['2026-01','2026-07'])]:
            assert all(len(m)==7 and m[:4] in ['2025','2026'] for m in months)
            source.make_source(con,path,months)
            value=table(con)
            for month,part in value.groupby('month_key'):
                known=part.loc[part.iobt.notna()].sort_values('flight_id')
                ids=known.flight_id.to_numpy();clocks=known.iobt.to_numpy()
                sample=part.iloc[np.linspace(0,len(part)-1,min(100,len(part)),dtype=int)]
                for row in sample.itertuples():
                    left=np.flatnonzero(ids<row.flight_id);right=np.flatnonzero(ids>row.flight_id)
                    if not len(left) or not len(right):
                        assert pd.isna(row.idclock_iobt_residual) and pd.isna(row.idclock_neighbor_span)
                    else:
                        lo,hi=left[-1],right[0]
                        assert row.lo_id==ids[lo] and row.hi_id==ids[hi]
                        span=clocks[hi]-clocks[lo]
                        expected=clocks[lo]+(row.flight_id-ids[lo])/(ids[hi]-ids[lo])*span-row.iobt
                        np.testing.assert_allclose(row.idclock_iobt_residual,expected,rtol=0,atol=1e-6,equal_nan=True)
                        assert row.idclock_neighbor_span==span
                    scalar_checks+=1
            dep=con.execute("SELECT mvt_id,month_key FROM raw WHERE phase='DEP'").df()
            assert dep.mvt_id.is_unique and value.mvt_id.is_unique
            result=dep.merge(value[['mvt_id',*FEATURES]],on='mvt_id',how='left',validate='one_to_one')
            assert len(result)==len(dep) and not np.isinf(result[FEATURES].to_numpy()).any()
            result[FEATURES]=result[FEATURES].astype('float32')
            for month,part in result.groupby('month_key'):
                coverage.append(dict(month=month,rows=len(part),available={k:int(part[k].notna().sum()) for k in FEATURES}))
            outputs[view].append(result[['mvt_id',*FEATURES]])
    records=[]
    for view,frames in outputs.items():
        frame=pd.concat(frames,ignore_index=True);assert frame.mvt_id.is_unique
        path=OUT/f'features_{view}.parquet';frame.to_parquet(path,index=False);records.append(art(path))
    value=dict(design=DESIGN,source=art(__file__),identity=art(OUT/'identity.json'),outputs=records,coverage=coverage,
        scalar_checks=scalar_checks,no_target_column_projected=True,monthly_peers_only=True,own_id_excluded=True)
    (OUT/'manifest.json').write_text(json.dumps(value,indent=2)+'\n')
    (OUT/'audit.json').write_text(json.dumps(dict(status='pass',manifest=art(OUT/'manifest.json'),source=art(__file__),
        scalar_checks=scalar_checks,scope='Strict SQL ASOF peers compared to independent sorted-array scalar calculations; exact full departure membership and finite-or-missing values. Separate input-only chronology/source audit retained.'),indent=2)+'\n')
    print(dict(complete=True,coverage=coverage,scalar_checks=scalar_checks),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','build'])
    globals()[parser.parse_args().stage]()
