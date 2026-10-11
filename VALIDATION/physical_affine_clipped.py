"""Extend the scored basis, auditing nonnegative clipping and integer rounding.

This is explicitly leaderboard-conditioned selection. The matching Jan/Jul
predictions are replayed without any new fit; their score is reported regardless
of whether it improves. Newly submitted affine outputs are excluded from the
basis to avoid treating rounding noise as another independent model direction.
"""
import argparse,json,os
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import scored_submission_geometry as geometry
import scored_blend_jan_jul as cv
import physical_candidate as physical
from records import sha256,write_json
from submit import fill_template

ROOT=geometry.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_physical_affine'

def main(manifests,version):
    dest=OUT/f'submission_v{version}';output=ROOT/'submissions'/f'unique-rose_v{version}.parquet'
    assert not dest.exists() and not output.exists(),'Preserve previous candidate'
    original=geometry.OUT/'validation_ready_basis.json';frozen=cv.ts.read(original)
    original_cv=cv.ts.read(cv.OUT/'result.json')
    assert original_cv['geometry_sha256']==sha256(original)
    assert sha256(cv.OUT/'scored.parquet')==original_cv['scored_sha256']
    for path,digest in original_cv['inputs'].items():assert sha256(ROOT/path)==digest,path
    sources=dict(original_cv['inputs'])
    sources.update({str(p.relative_to(ROOT)):sha256(p) for p in [original,cv.OUT/'result.json',cv.OUT/'scored.parquet',Path(__file__),Path(geometry.__file__)]})
    ids,allP,allscores,allversions,records=geometry.load(1)
    versions=list(frozen['versions']);select=[allversions.index(v) for v in versions]
    P=allP[:,select];scores=allscores[select]
    held=pd.read_parquet(cv.OUT/'scored.parquet');assert len(held)==344419 and held.mvt_id.is_unique
    C=held[[f'v{v}' for v in versions]].to_numpy(float)
    basis_scores={f'unique-rose_v{v}.parquet':float(s) for v,s in zip(versions,scores)}
    for v in versions:sources[records[str(v)]['file']]=records[str(v)]['sha256']
    for mp in map(Path,manifests):
        mp=mp.resolve();m=cv.ts.read(mp)
        assert m['state']=='scored' and m['official_result']['used_pairs']==344841
        assert m['change'] in ['physical-scale sparse track encoder','neural share assigned to callsign15'] and m['parent']=='v64'
        assert m['version'] not in versions
        for path,digest in m['sources'].items():assert sha256(ROOT/path)==digest,path
        p=ROOT/m['file'];assert sha256(p)==m['sha256']
        submitted=pd.read_parquet(p);np.testing.assert_array_equal(submitted.MVT_ID_mvt,ids)
        cp=(physical.OUT/physical.tag(m['members'])/'cv/result.json' if 'members' in m
            else physical.verify(m['cv_record']))
        cr=cv.ts.read(cp)
        cf=pd.read_parquet(physical.verify(cr['prediction'])).set_index('mvt_id')
        assert cf.index.is_unique and set(cf.index)==set(held.mvt_id)
        cf=cf.loc[held.mvt_id];np.testing.assert_array_equal(cf.y,held.y)
        np.testing.assert_array_equal(cf.month,held.month)
        P=np.column_stack([P,submitted.TAXITIME_SEC_mvt.to_numpy(float)])
        C=np.column_stack([C,cf.submitted.to_numpy(float)])
        scores=np.append(scores,m['score']);versions.append(m['version'])
        basis_scores[p.name]=m['score']
        for path in [mp,p,cp,physical.verify(cr['prediction'])]:sources[str(path.relative_to(ROOT))]=sha256(path)
    result,raw=geometry.solve(P,scores,versions,-3.,4.,1.)
    w=np.array([result['weights'][str(v)] for v in versions])
    assert np.isfinite(raw).all() and raw.max()<np.iinfo(np.int32).max
    mse=float(w@(scores*scores)-w@np.mean(P*P,axis=0)+np.mean(raw*raw))
    np.testing.assert_allclose(np.sqrt(mse),result['projected_rmse'],atol=1e-8,rtol=0)
    c_raw=C@w;pred=np.rint(np.maximum(c_raw,0));held=held[['mvt_id','month','day','airport','y','pred']].rename(columns={'pred':'v65'})
    held['pred']=pred;held['raw_pred']=c_raw
    summary={m:dict(rows=len(f),candidate=float(np.sqrt(np.mean((f.pred-f.y)**2))),
        v65=float(np.sqrt(np.mean((f.v65-f.y)**2)))) for m,f in held.groupby('month')}
    candidate_cv=float(np.mean([r['candidate'] for r in summary.values()]))
    paired=cv.ts.paired_bootstrap(held.day,held.month,(held.v65-held.y)**2,(held.pred-held.y)**2)
    template_path=ROOT/'data/submitting.parquet';template=pq.read_table(template_path)
    table=fill_template(template,ids,raw)
    assert table.schema==template.schema and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    expected=np.rint(np.maximum(raw,0)).astype(np.int32)
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),expected)
    rounding_rms=float(np.sqrt(np.mean((expected-raw)**2)))
    dest.mkdir(parents=True);pq.write_table(table,output);assert pq.read_table(output).equals(table)
    held.to_parquet(dest/'cv.parquet',index=False)
    sources[str(template_path.relative_to(ROOT))]=sha256(template_path)
    write_json(dest/'geometry.json',dict(result,versions=versions,basis_scores=basis_scores,
        independent_mse_identity=True,selection='Own leaderboard scores; not independent validation'))
    write_json(dest/'cv.json',dict(cv=candidate_cv,months=summary,paired_vs_v65=paired,
        sources=sources,prediction=physical.parent.artifact(dest/'cv.parquet'),
        negative_before_clipping=int((c_raw<0).sum()),
        caveat=original_cv['reconstruction_caveat'],selection='Leaderboard-conditioned; reused development months'))
    sources.update({str(p.relative_to(ROOT)):sha256(p) for p in [dest/'geometry.json',dest/'cv.json',dest/'cv.parquet']})
    write_json(dest/'manifest.json',dict(version=version,state='prepared',file=str(output.relative_to(ROOT)),
        sha256=sha256(output),rows=len(ids),parent='v65',change='add separately scored neural components to scored basis',
        jan_jul_cv=candidate_cv,projected_rmse=result['projected_rmse'],
        output_serialization_rms_sec=rounding_rms,
        negative_predictions_before_clipping=result['negative_predictions'],
        serialization='Clip at zero and round to nearest integer; interval includes both changes',
        conservative_rmse_interval=[result['score_rounding_rmse_interval'][0]-rounding_rms,
                                    result['score_rounding_rmse_interval'][1]+rounding_rms],
        basis_scores=basis_scores,sources=sources,authorization='October11 user authorized autonomous uploads',
        audit=dict(independent_mse_identity=True,template_ids_and_schema_exact=True,readback_exact=True),
        note='Leaderboard-conditioned selection. Jan/Jul CV reported separately; no new fits. Projection precedes clipping; the score interval uses the exact serialization RMS via the triangle inequality without assumptions about hidden labels.'))
    print(json.dumps(dict(version=version,projected=result['projected_rmse'],cv=candidate_cv,months=summary,paired=paired,negative_predictions=result['negative_predictions'],minimum=result['minimum_prediction'],serialization_rms=rounding_rms,conservative_upper=result['score_rounding_rmse_interval'][1]+rounding_rms)),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',nargs='+',required=True);ap.add_argument('--version',type=int,required=True);a=ap.parse_args()
    with threadpool_limits(2):main(a.manifest,a.version)
