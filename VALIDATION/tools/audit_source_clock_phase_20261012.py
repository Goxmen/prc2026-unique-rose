"""Independent narrow-column clock-phase audit; never reads departure labels."""
import json
from pathlib import Path
import sys
import time

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'VALIDATION'))
import source_clock_phase as builder
from records import sha256,write_json


def verify(record):
    path=ROOT/record['path']
    assert path.stat().st_size==record['bytes'] and sha256(path)==record['sha256']
    return path


def main():
    started=time.monotonic();out=builder.OUT
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['design']==builder.DESIGN and verify(manifest['source'])==Path(builder.__file__)
    assert not (out/'audit.json').exists(),'Preserve completed audit'
    rows=[];samples=[]
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for view,records in manifest['raw_sources'].items():
            paths=[verify(record) for record in records]
            source=con.read_parquet([str(p) for p in paths])
            source.create_view('raw',replace=True)
            clocks=list(builder.CLOCKS.values())
            con.execute('CREATE OR REPLACE VIEW source AS SELECT MVT_ID_mvt::BIGINT mvt_id,ADEP_mvt airport,'
                'MVT_TIME_UTC_mvt movement,'+','.join(clocks)+" FROM raw WHERE PHASE_mvt='DEP'")
            feature_record=manifest['views'][view]['outputs'][0]
            feature_path=verify(feature_record)
            con.read_parquet(str(feature_path)).create_view('feature',replace=True)
            assert [r[0] for r in con.execute('describe feature').fetchall()]==['mvt_id']+builder.FEATURES
            n,unique=con.execute('select count(*),count(distinct mvt_id) from source').fetchone()
            assert n==unique==manifest['views'][view]['rows']
            assert con.execute('select count(*),count(distinct mvt_id) from feature').fetchone()==(n,n)
            # Integer epoch arithmetic is independent of the builder's calendar extraction.
            expected={f'scp_{name}_second':f'((epoch_ns(s.{column})//1000000000)%60)::FLOAT'
                for name,column in builder.CLOCKS.items()}
            expected['scp_aobt3_minute_mod5']='((epoch_ns(s.AOBT_3_flt)//60000000000)%5)::FLOAT'
            expected['scp_aobt3_minute_mod15']='((epoch_ns(s.AOBT_3_flt)//60000000000)%15)::FLOAT'
            conditions=['s.mvt_id IS NULL','f.mvt_id IS NULL']+[f'f.{name} IS DISTINCT FROM {value}' for name,value in expected.items()]
            total,bad=con.execute('SELECT count(*),sum(CASE WHEN '+ ' OR '.join(conditions)+
                ' THEN 1 ELSE 0 END) FROM source s FULL JOIN feature f USING(mvt_id)').fetchone()
            assert total==n and bad==0,(view,total,n,bad)
            missing={name:int(con.execute(f'select count(*) from feature where {name} is null').fetchone()[0]) for name in builder.FEATURES}
            # Earliest/latest movement and one missing/nonzero observation per raw clock
            # in every airport/month, with no label-driven selection.
            choices=['arg_min(mvt_id,movement) AS first_id','arg_max(mvt_id,movement) AS last_id']
            for i,column in enumerate(clocks):
                choices += [f'min(mvt_id) FILTER(WHERE {column} IS NULL) AS missing_{i}',
                    f'min(mvt_id) FILTER(WHERE ((epoch_ns({column})//1000000000)%60)!=0) AS nonzero_{i}']
            selected=con.execute("SELECT airport,strftime(movement,'%Y-%m') AS \"month\","+','.join(choices)+
                " FROM source GROUP BY airport,strftime(movement,'%Y-%m')").df()
            ids=np.array(sorted({int(v) for v in selected.iloc[:,2:].to_numpy().ravel() if pd.notna(v)}),dtype=np.int64)
            con.register('selected_ids',pd.DataFrame(dict(mvt_id=ids)))
            sample=con.execute('SELECT s.*,f.* EXCLUDE(mvt_id) FROM source s JOIN selected_ids i USING(mvt_id) JOIN feature f USING(mvt_id) ORDER BY s.mvt_id').df()
            assert len(sample)==len(ids) and sample.mvt_id.is_unique
            for row in sample.itertuples(index=False):
                for name,column in builder.CLOCKS.items():
                    value=getattr(row,column);actual=getattr(row,f'scp_{name}_second')
                    if pd.isna(value):assert pd.isna(actual)
                    else:
                        clock=pd.Timestamp(value)
                        clock=clock.tz_localize('UTC') if clock.tzinfo is None else clock.tz_convert('UTC')
                        assert actual==clock.to_pydatetime(warn=False).second
                value=row.AOBT_3_flt
                for modulo in [5,15]:
                    actual=getattr(row,f'scp_aobt3_minute_mod{modulo}')
                    if pd.isna(value):assert pd.isna(actual)
                    else:assert actual==pd.Timestamp(value).to_pydatetime(warn=False).minute%modulo
            sample.insert(0,'view',view);samples.append(sample)
            rows.append(dict(view=view,rows=n,all_ids_unique_and_exact=True,
                all_feature_values_and_missing_masks_exact=True,scalar_sample_rows=len(sample),
                airport_month_strata=len(selected),missing_counts=missing,feature=feature_record))
            con.unregister('selected_ids')
        # Include nullable equality explicitly, preserving the two ranking namespaces.
        con.read_parquet(str(out/'features_old.parquet')).create_view('old')
        con.read_parquet(str(out/'features_final.parquet')).create_view('final')
        comparisons=','.join(f'sum(CASE WHEN o.{c} IS DISTINCT FROM f.{c} THEN 1 ELSE 0 END)' for c in builder.FEATURES)
        shared=con.execute(f'select count(*),{comparisons} from old o join final f using(mvt_id)').fetchone()
        assert shared[0]==344841 and not any(shared[1:]),shared
    sample_path=out/'independent_samples.parquet';pd.concat(samples,ignore_index=True).to_parquet(sample_path,index=False)
    write_json(out/'audit.json',dict(status='pass',source=builder.artifact(Path(__file__)),
        builder_source=manifest['source'],manifest=builder.artifact(out/'manifest.json'),
        checks=rows,scalar_values_checked=sum(len(s)*len(builder.FEATURES) for s in samples),
        old_final_common_rows=shared[0],all_seven_common_plan_phases_and_missing_masks_unchanged=True,
        samples=builder.artifact(sample_path),seconds=time.monotonic()-started,
        source_scope='Only movement IDs, phase/airport/calendar and five original plan clocks; no departure label or block field',
        raw_sources=manifest['raw_sources']))
    print(json.dumps(dict(status='pass',views=rows,seconds=time.monotonic()-started),indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
