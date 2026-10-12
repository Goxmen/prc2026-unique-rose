"""Original source-clock phase inputs; no departure targets or block timestamps.

These are observations of a timestamp's phase, not inferred precision. A single
time falling on a minute boundary does not establish that its source was rounded.
Movement-time phase is diagnostic only: final-ranking movement jitter changes it.
"""
import argparse
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

from records import sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/derived/source_clock_phase_20261012'
CLOCKS = dict(aobt3='AOBT_3_flt', lobt='LOBT_flt', iobt='IOBT_flt',
              eobt1='EOBT_1_flt', arvt3='ARVT_3_flt')
FEATURES = [f'scp_{c}_second' for c in CLOCKS] + ['scp_aobt3_minute_mod5', 'scp_aobt3_minute_mod15']
DIAGNOSTIC = ['diagnostic_mvt_second', 'diagnostic_schedule_second']
DESIGN = dict(change='seven original flight-plan timestamp phase inputs', features=FEATURES,
    raw_clocks=CLOCKS, missing='NaN; never schedule-fill or recover original timestamps',
    seconds='UTC integer second of minute, range0..59',
    minute_phase='original AOBT3 UTC minute of hour modulo5 and15',
    source_precision='phase observations only; no claim that a single aligned time was quantized',
    movement_phase='diagnostic only, excluded because final-ranking movement jitter changes it',
    phase='DEP only', target_use='none', departure_block_use='none',
    reference_fit='none; every query is transformed independently',
    comparison='all2025 and same-seasonJan/Jul2025 against accepted oldJan/Jul2026',
    final_check='separate IDs/inputs; compare common original plan phases and movement diagnostic')


def artifact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), bytes=path.stat().st_size, sha256=sha256(path))


def transform(raw):
    """Independent scalar-friendly implementation for input-only frame checks."""
    result = {}
    for name, column in CLOCKS.items():
        clock = pd.to_datetime(raw[column], utc=True)
        result[f'scp_{name}_second'] = clock.dt.second.astype('float32')
    actual = pd.to_datetime(raw.AOBT_3_flt, utc=True)
    result['scp_aobt3_minute_mod5'] = (actual.dt.minute % 5).astype('float32')
    result['scp_aobt3_minute_mod15'] = (actual.dt.minute % 15).astype('float32')
    return pd.DataFrame(result, index=raw.index)[FEATURES]


def sources():
    train = sorted((ROOT / 'data').glob('training_2025-*.parquet'))
    assert len(train) == 12
    return dict(training=train, old=[ROOT/'data/ranking.parquet'], final=[ROOT/'data/final_ranking.parquet'])


def read_view(con, paths):
    expressions = [f'extract(second from {column})::FLOAT scp_{name}_second' for name,column in CLOCKS.items()]
    expressions += ['(extract(minute from AOBT_3_flt)%5)::FLOAT scp_aobt3_minute_mod5',
        '(extract(minute from AOBT_3_flt)%15)::FLOAT scp_aobt3_minute_mod15',
        'extract(second from MVT_TIME_UTC_mvt)::FLOAT diagnostic_mvt_second',
        'extract(second from SCHED_TIME_UTC_mvt)::FLOAT diagnostic_schedule_second']
    return con.execute('SELECT MVT_ID_mvt::BIGINT mvt_id,ADEP_mvt airport,'
        "strftime(MVT_TIME_UTC_mvt,'%Y-%m') AS \"month\"," + ','.join(expressions) +
        " FROM read_parquet(?) WHERE PHASE_mvt='DEP' ORDER BY mvt_id",
        [[str(p) for p in paths]]).df()


def histograms(con, frame, view):
    con.register('phase_frame', frame)
    pieces=[]
    for col in FEATURES+DIAGNOSTIC:
        piece=con.execute(f'''SELECT airport,"month",'{col}' AS feature,{col} AS "value",count(*) AS n
            FROM phase_frame GROUP BY airport,"month",{col} ORDER BY airport,"month",{col}''').df()
        piece['view']=view;pieces.append(piece)
    return pd.concat(pieces,ignore_index=True)


def comparisons(hist):
    """Input-distribution distances; nothing here is a label association."""
    output=[]
    for airport in sorted(hist.airport.unique()):
        for feature in FEATURES+DIAGNOSTIC:
            for season in ['01','07']:
                groups=[]
                for view,month in [('training','2025-'+season),('old','2026-'+season)]:
                    part=hist[(hist.view==view)&(hist.month==month)&(hist.airport==airport)&(hist.feature==feature)]
                    values=part.assign(value=part.value.fillna(-1)).set_index('value').n
                    groups.append(values/values.sum())
                union=groups[0].index.union(groups[1].index)
                p,q=[g.reindex(union,fill_value=0) for g in groups]
                output.append(dict(airport=airport,feature=feature,season=season,
                    total_variation=float(.5*np.abs(p-q).sum()),
                    train_missing=float(p.get(-1,0)),old_missing=float(q.get(-1,0)),
                    train_zero=float(p.get(0,0)),old_zero=float(q.get(0,0))))
    return output


def describe():
    """Single bounded source-only pass writes reusable label-free feature tables."""
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=OUT/'manifest.json'
    assert not manifest.exists(),'Preserve completed source description and inputs'
    design_path=OUT/'design.json'
    if design_path.exists():assert json.loads(design_path.read_text())==DESIGN
    else:write_json(design_path,DESIGN)
    started=time.monotonic(); inventory=sources(); tables={}; summaries={}; hist=[]; hashes={}
    with duckdb.connect(config={'threads':2}) as con:
        con.execute("SET TimeZone='UTC'");con.execute("SET memory_limit='2GB'")
        for view,paths in inventory.items():
            hashes[view]=[artifact(p) for p in paths]
            table=read_view(con,paths)
            assert table.mvt_id.is_unique and table.mvt_id.notna().all()
            for col in FEATURES+DIAGNOSTIC:
                hi=4 if col.endswith('mod5') else 14 if col.endswith('mod15') else 59
                assert table[col].dropna().between(0,hi).all()
            path=OUT/f'features_{view}.parquet'
            if path.exists():
                pd.testing.assert_frame_equal(pd.read_parquet(path),table[['mvt_id']+FEATURES],check_exact=True)
            else:table[['mvt_id']+FEATURES].to_parquet(path,index=False)
            # Compact metadata is diagnostic only, never appended to a model.
            meta=OUT/f'diagnostic_{view}.parquet'
            if meta.exists():
                pd.testing.assert_frame_equal(pd.read_parquet(meta),table[['mvt_id','airport','month']+DIAGNOSTIC],check_exact=True)
            else:table[['mvt_id','airport','month']+DIAGNOSTIC].to_parquet(meta,index=False)
            hist.append(histograms(con,table,view))
            summaries[view]=dict(rows=len(table),nonmissing={c:int(table[c].notna().sum()) for c in FEATURES},
                outputs=[artifact(path),artifact(meta)])
            if view!='training':tables[view]=table.set_index('mvt_id')
            print(view,len(table),'rows',round(time.monotonic()-started,2),'sec',flush=True)
        old,final=tables['old'],tables['final'];common=old.index.intersection(final.index)
        delta={}
        for col in FEATURES+DIAGNOSTIC:
            a,b=old.loc[common,col],final.loc[common,col]
            both=a.notna()&b.notna()
            delta[col]=dict(availability_changed=int((a.isna()!=b.isna()).sum()),
                both_known=int(both.sum()),known_value_changed=int((a[both]!=b[both]).sum()))
    histogram=pd.concat(hist,ignore_index=True)
    histogram.to_parquet(OUT/'histograms.parquet',index=False)
    write_json(OUT/'description.json',dict(design=DESIGN,views=summaries,
        common_old_final=len(common),old_final_phase_delta=delta,
        same_season_input_shift=comparisons(histogram),elapsed_seconds=time.monotonic()-started))
    write_json(manifest,dict(design=DESIGN,source=artifact(__file__),raw_sources=hashes,views=summaries,
        outputs=[artifact(OUT/p) for p in ['design.json','description.json','histograms.parquet']],
        no_labels_or_departure_blocks_read=True,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['describe'])
    globals()[parser.parse_args().stage]()
