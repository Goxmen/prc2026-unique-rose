"""Independent production source linkage, control, bound and serialization audit.

No model/core imports, fits or bootstrap. Scalar posterior mathematics is covered
by the separately retained CV block audit and immutable core; this auditor checks
its production inputs and every output's mapping and correction arithmetic.
"""
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import audit_public_cohort_scores_20261012 as common

ROOT,RUNS=common.ROOT,common.RUNS
read,art,verify=common.read,common.artifact,common.verify
OUT=RUNS/'v3_20261012_joint_public_other_runway_submit'
OLD=RUNS/'v3_20261011_joint_public_moments_submit'
RAW=RUNS/'v3_20261011_daily_delay/ranking_inputs.parquet'
MATCHES=RUNS/'v3_20261011_recovered_exclusion_fixed/ranking_matches.parquet'


def main():
    started=time.monotonic();assert not (OUT/'audit.json').exists()
    identity=read(OUT/'identity.json');result=read(OUT/'result.json');inf=read(OUT/'inference.json')
    assert result['identity']==inf['identity']==art(OUT/'identity.json')
    assert result['inference']==art(OUT/'inference.json')
    assert identity['source']==art(ROOT/'VALIDATION/joint_public_other_runway_submit.py')
    assert identity['design']==result['design'] and identity['design']['rows']==344841
    assert identity['actual_v78_file_exact'] and identity['training_calibration_reused']
    for record in [*identity['sources'],*result['artifacts'],*inf['artifacts']]:verify(record)
    old_audit=read(OLD/'audit.json')
    assert old_audit['status']=='pass' and old_audit['identity']==art(OLD/'identity.json')
    assert old_audit['result']==art(OLD/'result.json')
    verify(old_audit['source'])
    original_identity=read(OLD/'identity.json')
    for record in original_identity['sources']:verify(record)
    assert read(OLD/'parameters.json')['allowed']==[f'2025-{m:02d}' for m in range(1,13)]
    paths={'p':OUT/'inference.parquet','o':OLD/'inference.parquet','i':OLD/'inputs.parquet',
        'v':OUT/'predictions.parquet','v78':ROOT/identity['v78_file']['path'],
        'v79':ROOT/identity['v79_file']['path'],
        'rank':ROOT/'data/ranking.parquet',
        'quality':RUNS/'v3_20261011_daily_main_refinements_submit/predictions.parquet'}
    # Resolve the retained quality file by exact original receipt; avoid guessing
    # a renamed helper output or importing its model dependency tree.
    quality=[ROOT/a['path'] for a in original_identity['sources']
        if a['path'].endswith('/predictions.parquet') and 'daily_main_refinements' in a['path']]
    assert len(quality)==1;paths['quality']=quality[0]
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,path in paths.items():con.read_parquet(str(path)).create_view(name)
        n,u,bad,maxerr=con.execute('''SELECT count(*),count(distinct p.mvt_id),sum(CASE WHEN
            p.month IS DISTINCT FROM i.month OR p.touched IS DISTINCT FROM o.touched OR
            p.old_joint IS DISTINCT FROM o.joint OR p.delta IS DISTINCT FROM p.new_joint-p.old_joint OR
            (p.extra_cells=0 AND (p.delta<>0 OR p.new_joint IS DISTINCT FROM p.old_joint)) OR
            (NOT p.touched AND p.delta<>0) OR p.weight<=0 OR p.weight>1 OR p.extra_q<0 OR
            NOT isfinite(p.new_joint) OR abs(p.recomputed_old_joint-o.joint)>1e-8
            THEN 1 ELSE 0 END),max(abs(p.recomputed_old_joint-o.joint))
            FROM p JOIN o USING(mvt_id) JOIN i USING(mvt_id)''').fetchone()
        assert n==u==26899 and bad==0
        nr,ur,bad=con.execute('''SELECT count(*),count(distinct i.mvt_id),sum(CASE WHEN
            rank.PHASE_mvt!='DEP' OR rank.ADEP_mvt!='LIRF' OR
            i.gap IS DISTINCT FROM epoch(rank.MVT_TIME_UTC_mvt-rank.SCHED_TIME_UTC_mvt) OR
            i.fltrule IS DISTINCT FROM rank.FLIGHT_RULE_mvt OR i.runway IS DISTINCT FROM rank.RUNWAY_mvt OR
            i.local_day IS DISTINCT FROM strftime(rank.MVT_TIME_UTC_mvt AT TIME ZONE 'Europe/Rome','%Y-%m-%d') OR
            i.assigned_day IS DISTINCT FROM strftime(to_timestamp(epoch(rank.MVT_TIME_UTC_mvt)-i.prior),'%Y-%m-%d') OR
            i.eligible IS DISTINCT FROM (rank.STAND_mvt IS NOT NULL AND rank.RUNWAY_mvt IS NOT NULL) OR
            i.quality IS DISTINCT FROM quality.quality_eligible
            THEN 1 ELSE 0 END) FROM i JOIN rank ON i.mvt_id=rank.MVT_ID_mvt JOIN quality USING(mvt_id)''').fetchone()
        assert nr==ur==26899 and bad==0
        count,unique,bad=con.execute('''SELECT count(*),count(distinct v.mvt_id),sum(CASE WHEN
            v.v78 IS DISTINCT FROM v78.TAXITIME_SEC_mvt OR v.v79 IS DISTINCT FROM v79.TAXITIME_SEC_mvt OR
            v.posterior_delta IS DISTINCT FROM coalesce(p.delta,0) OR
            v.active IS DISTINCT FROM coalesce(p.extra_cells>0,false) OR
            (NOT v.active AND v.robust_v78 IS DISTINCT FROM v.v78) OR
            v.serialized_delta IS DISTINCT FROM v.robust_v78-v.v78 OR
            v.robust_v79 IS DISTINCT FROM round_even(greatest(0,v.v79+v.robust_v78-v.v78),0)
            THEN 1 ELSE 0 END) FROM v JOIN v78 ON v.mvt_id=v78.MVT_ID_mvt
            JOIN v79 ON v.mvt_id=v79.MVT_ID_mvt LEFT JOIN p ON v.mvt_id=p.mvt_id''').fetchone()
        assert count==unique==344841 and bad==0
    f=pd.read_parquet(OUT/'predictions.parquet');ids=f.mvt_id.to_numpy()
    raw=pd.read_parquet(RAW,columns=['mvt_id','has_id','nm_gap']).set_index('mvt_id').loc[ids]
    matches=pd.read_parquet(MATCHES);accepted=matches[matches.accepted]
    assert accepted.mvt_id.is_unique
    values=f.v78.to_numpy()+.5*f.posterior_delta.to_numpy()
    known=raw.has_id.to_numpy(bool)&np.isfinite(raw.nm_gap.to_numpy())
    gap=raw.nm_gap.to_numpy()
    values[known]=np.maximum(0,np.minimum(np.maximum(values[known],gap[known]-3606),gap[known]+3606))
    values=np.maximum(values,0);positions={v:i for i,v in enumerate(ids)}
    for row in accepted.itertuples():
        ix=positions[row.mvt_id];lo,hi=row.recovered_gap-3606,row.recovered_gap+3606
        if lo<values[ix]<hi:values[ix]=min([v for v in [lo,hi] if v>=0],key=lambda x:abs(x-values[ix]))
    np.testing.assert_array_equal(np.rint(np.maximum(values,0)),f.robust_v78)
    template=pq.read_table(ROOT/'data/submitting.parquet')
    for name in ['v78','v79']:
        table=pq.read_table(OUT/f'candidate_{name}.parquet')
        assert template.schema==table.schema and table.num_rows==344841
        for col in template.column_names:
            if col!='TAXITIME_SEC_mvt':assert table[col].equals(template[col])
        np.testing.assert_array_equal(table['MVT_ID_mvt'].to_numpy(),ids)
        np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),f['robust_'+name])
        assert np.isfinite(f['robust_'+name]).all() and f['robust_'+name].ge(0).all()
    receipt=dict(status='pass',source=art(__file__),helper=art(common.__file__),seconds=time.monotonic()-started,
        identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),inference=art(OUT/'inference.json'),
        rows=count,rome_rows=n,old_float_max_error_sec=maxerr,
        exact_inputs_memberships_saved_control_deltas=True,independent_final_bounds_and_transport=True,
        every_serialized_value_and_protected_column_exact=True,
        scope='Independent source linkage/raw input clocks, every delta/ID, old scalar control and all output constraints. '
            'Full2025 calibration and old inputs inherit the hash-bound original independent audit; '
            'new posterior arithmetic inherits the separately retained CV block audit and identical frozen core. '
            'No production scalar posterior or model refit is claimed.')
    (OUT/'audit.json').write_text(__import__('json').dumps(receipt,indent=2)+'\n');print(receipt,flush=True)


if __name__=='__main__':main()
