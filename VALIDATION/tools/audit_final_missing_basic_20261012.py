"""Independent SQL identity, rule, serialization and original-vector audit."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_final_missing_basic_reconstruct'


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def main(family):
    folder=OUT/family;result=read(folder/'result.json');identity=read(OUT/'identity.json')
    assert result['state']=='exact_old_replay_final_inference_complete'
    assert result['identity']['sha256']==sha(OUT/'identity.json')
    for x in identity['sources']:assert sha(ROOT/x['path'])==x['sha256'],x['path']
    trained=read(folder/'trained.json');preflight=read(folder/'preflight.json')
    assert trained['identity']==result['identity']==preflight['identity']
    for key in ['model','categories','preflight','source']:
        assert sha(ROOT/trained[key]['path'])==trained[key]['sha256']
    assert preflight['rows']==2084659 and preflight['original_order_and_labels_exact']
    assert preflight['independent_raw_SQL_membership_and_basic18_features_exact']
    cache=read(ROOT/'data/derived/scored_submission_cache_20261011/manifest.json')
    thresholds={'12':None,'13':21600,'14':21600,'15':14400,'16':14400}
    checks={}
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for view,versions in result['views'].items():
            source=ROOT/('data/ranking.parquet' if view=='old' else 'data/final_ranking.parquet')
            template=ROOT/('data/submitting.parquet' if view=='old' else 'data/final_submitting.parquet')
            con.read_parquet(str(source)).create_view('raw',replace=True)
            con.read_parquet(str(template)).create_view('template',replace=True)
            for version,rec in versions.items():
                assert sha(ROOT/rec['prediction']['path'])==rec['prediction']['sha256']
                con.read_parquet(str(ROOT/rec['prediction']['path'])).create_view('p',replace=True)
                bound=thresholds[version]
                gap='epoch(r.MVT_TIME_UTC_mvt-r.SCHED_TIME_UTC_mvt)'
                expression='p.pred_raw' if bound is None else f"CASE WHEN r.ADEP_mvt='LIRF' AND r.FLIGHT_ID_mvt IS NULL AND {gap}>={bound} THEN {gap} ELSE p.pred_raw END"
                n,u,bad=con.execute(f'''SELECT count(*),count(distinct p.mvt_id),sum(CASE WHEN
                    p.pred IS DISTINCT FROM ({expression}) OR p.submitted IS DISTINCT FROM round_even(greatest(0,p.pred),0)
                    OR NOT isfinite(p.pred_raw) OR NOT isfinite(p.pred) OR NOT isfinite(p.submitted)
                    OR p.pred_raw<0 OR p.submitted<0 OR p.month IS DISTINCT FROM strftime(r.MVT_TIME_UTC_mvt,'%Y-%m')
                    THEN 1 ELSE 0 END) FROM p JOIN raw r ON p.mvt_id=r.MVT_ID_mvt
                    JOIN template t ON p.mvt_id=t.MVT_ID_mvt WHERE r.PHASE_mvt='DEP' ''').fetchone()
                assert n==u==rec['rows']==(344841 if view=='old' else 670790) and bad==0,(view,version,n,u,bad)
                months=dict(con.execute('SELECT month,count(*) FROM p GROUP BY month ORDER BY month').fetchall())
                expected={'2026-01':152719,'2026-07':192122} if view=='old' else {'2026-01':152719,'2026-02':144158,'2026-06':181791,'2026-07':192122}
                assert months==expected
                if view=='old':
                    original=ROOT/cache[version]['file'];assert sha(original)==cache[version]['sha256']
                    con.read_parquet(str(original)).create_view('original',replace=True)
                    count,bad=con.execute('SELECT count(*),sum(CASE WHEN p.submitted IS DISTINCT FROM o.TAXITIME_SEC_mvt THEN 1 ELSE 0 END) FROM p JOIN original o ON p.mvt_id=o.MVT_ID_mvt').fetchone()
                    assert count==344841 and bad==0
                checks[view+'_v'+version]=dict(rows=n,all_SQL_rule_rounding_ids_dates_values_pass=True,original_vector_exact=(view=='old'))
    target=folder/'independent_audit.json';assert not target.exists()
    target.write_text(json.dumps(dict(status='pass',family=family,source_sha256=sha(__file__),
        result_sha256=sha(folder/'result.json'),checks=checks,
        source_and_model_hashes_pass=True,original_feature_preflight_checked=True,
        caveats=['Original B0 source equivalence is documented, not byte identity.',
                 'Original integer replay is exact; reconstructed model text hash need not be identical.',
                 'Held model outputs were independently reloaded by the producer; this auditor independently verifies rule arithmetic and source identity.']),indent=2)+'\n')
    print(json.dumps(dict(status='pass',family=family,rows=sum(x['rows'] for x in checks.values()))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('family',choices=['baseline','recovery'])
    main(parser.parse_args().family)
