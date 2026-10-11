"""Independent donor recovery and production replay checks before submission."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import public_constraints_candidate as candidate
from records import sha256,write_json

def donor_checks(paths,accepted):
    cols=['MVT_ID_mvt','FLIGHT_ID_mvt','FLIGHT_mvt','ADEP_mvt','ADES_mvt','PHASE_mvt',
          'AIRCRAFT_TYPE_mvt','MVT_TIME_UTC_mvt','LOBT_flt','ADEP_flt','ADES_flt']
    raw=pd.concat([pd.read_parquet(p,columns=cols,filters=[('ADEP_mvt','in',['LIRF','LTFM'])]) for p in paths],ignore_index=True)
    dep=raw[raw.PHASE_mvt.eq('DEP')].set_index('MVT_ID_mvt')
    arr=raw[raw.PHASE_mvt.eq('ARR')&raw.FLIGHT_ID_mvt.notna()&raw.LOBT_flt.notna()].copy()
    groups={key:f for key,f in arr.groupby(['FLIGHT_mvt','ADEP_mvt','ADES_mvt'])}
    errors=[]
    for row in accepted.itertuples():
        d=dep.loc[row.mvt_id];assert pd.isna(d.FLIGHT_ID_mvt)
        g=groups[(d.FLIGHT_mvt,d.ADEP_mvt,d.ADES_mvt)]
        dt=(g.MVT_TIME_UTC_mvt-d.MVT_TIME_UTC_mvt).dt.total_seconds()
        g=g[(dt>600)&(dt<72000)];assert len(g)==1
        a=g.iloc[0];assert a.MVT_ID_mvt==row.donor_mvt_id
        assert a.ADEP_flt==d.ADEP_mvt and a.ADES_flt==d.ADES_mvt
        assert pd.isna(a.AIRCRAFT_TYPE_mvt) or pd.isna(d.AIRCRAFT_TYPE_mvt) or a.AIRCRAFT_TYPE_mvt==d.AIRCRAFT_TYPE_mvt
        delta=(d.MVT_TIME_UTC_mvt-a.LOBT_flt).total_seconds()-row.recovered_gap
        assert abs(delta)<1e-6;errors.append(delta)
    return len(errors),float(np.max(np.abs(errors))) if errors else 0.

def run(path):
    path=Path(path).resolve();m=json.loads(path.read_text());assert m['state']=='prepared'
    base=candidate.base;root=base.ROOT;dest=path.parent
    for name,digest in m['sources'].items():assert sha256(root/name)==digest,name
    counts=[]
    for files,name in [(sorted((root/'data').glob('training_2025-*.parquet')),'matches2025.parquet'),
                       ([root/'data/ranking.parquet'],'ranking_matches.parquet')]:
        pairs=pd.read_parquet(candidate.exclusion.OUT/name);counts.append(donor_checks(files,pairs[pairs.accepted]))
    # Negative lower edges are unavailable, not an invitation to use zero.
    fake=pd.DataFrame(dict(mvt_id=[1.,2.,3.,4.],accepted=True,recovered_gap=[1000.,5000.,5000.,1000.],donor_mvt_id=[10.,20.,30.,40.]))
    q,_=candidate.exclusion.apply(np.array([1.,2.,3.,4.]),np.array([0.,2000.,8000.,10000.]),fake)
    np.testing.assert_array_equal(q,[4606.,1394.,8606.,10000.])
    saved=pd.read_parquet(dest/'predictions.parquet');ids=saved.mvt_id.to_numpy()
    rank=candidate.monthly.load([root/'data/ranking.parquet'],False).set_index('mvt_id').loc[ids].reset_index()
    brank=base.load_rows([root/'data/ranking.parquet'],False).set_index('mvt_id').loc[ids].reset_index()
    params=json.loads((dest/'parameters.json').read_text())['full_year']
    a,cells=candidate.monthly.apply(rank,saved.parent.to_numpy(),candidate.monthly.public(2026),params)
    b=candidate.matched_bound(brank,a)
    pairs=pd.read_parquet(candidate.exclusion.OUT/'ranking_matches.parquet')
    c,audit=candidate.exclusion.apply(ids,b,pairs)
    np.testing.assert_array_equal(a,saved.monthly);np.testing.assert_array_equal(b,saved.matched_bound);np.testing.assert_array_equal(c,saved.pred)
    table=pq.read_table(root/m['file']);template=pq.read_table(root/'data/submitting.parquet')
    assert table.schema==template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),np.rint(np.maximum(c,0)).astype(np.int32))
    report=dict(passed=True,independent_donor_checks=counts,negative_lower_edge_cases=4,
        replay_rows=len(saved),all_stages_exact=True,source=base.artifact(__file__),file_sha256=sha256(root/m['file']))
    ap=dest/'audit.json';write_json(ap,report)
    for p in [Path(__file__),ap]:m['sources'][str(p.relative_to(root))]=sha256(p)
    m['audit']['independent_review']=report;write_json(path,m);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest');a=p.parse_args();run(a.manifest)
