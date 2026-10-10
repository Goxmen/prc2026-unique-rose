"""Submit the retained, audited callsign-only neural ensemble without refitting.

This disentangles the callsign/Rome improvement from the raw-track increment.
It passed Jan/Jul versus v58 but is worse than v64 there. Its official result
can establish whether those two model directions deserve different weights.
"""
import argparse,os,shutil
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np,pandas as pd,pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import callsign_lgb_submit as candidate
import callsign_lgb_combo_score as score
import modern_prediction_replay as replay
from records import sha256,write_json
parent=candidate.parent;ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_calls_only_candidate'

def prepare(version):
    dest=OUT/f'submission_v{version}';output=ROOT/'submissions'/f'unique-rose_v{version}.parquet'
    assert not dest.exists() and not output.exists(),'Preserve retained candidate'
    candidate.guard()
    ap=candidate.OUT/'candidate_audit.json';audit=parent.read(ap)
    assert audit['passed'];parent.features.verify_artifact(audit['source'])
    v=audit['views']['old'];mp=parent.features.verify_artifact(v['manifest']);m=parent.read(mp)
    retained=parent.features.verify_artifact(v['output']);parts=pd.read_parquet(parent.features.verify_artifact(v['parts']))
    assert m['output']==v['output'] and m['view']=='old'
    data=parent.ranking_data('old');template=pq.read_table(parent.features.verify_artifact(m['template']))
    zero,masks=parent.final_frame(data,parts.nn_base.to_numpy(),parts.gbm_base.to_numpy(),parts.p.to_numpy())
    pd.testing.assert_frame_equal(zero,parts[zero.columns],check_exact=True)
    parent.assert_parent_output(zero,template,pq.read_table(retained),parts)
    assert masks==m['correction_masks']
    sr=parent.read(score.OUT/'result.json');assert sr['source_sha256']==sha256(score.__file__)
    for path,digest in sr['input_sha256'].items():assert sha256(ROOT/path)==digest,path
    assert sr['comparisons']['combined_calls_lgb_vs_matched_v58']['decision']=='improvement'
    expected=pd.read_parquet(score.OUT/'scored_errors.parquet').set_index('mvt_id')
    sc,rome,_,trees=replay.baseline.baseline();rows=[]
    for month in replay.baseline.ts.SPLITS:
        d=sc.d[month];ids=d['ids']
        calls=replay.values(replay.combo.sequence.CALLS_PRED,ids,'candidate_nn')
        main=replay.values(replay.combo.parent.control.candidate.OUT/'predictions'/f'{month}_s42.parquet',ids,'pred')
        tree=np.mean([main,trees['airport'][42][month],trees['catboost'][42][month]],axis=0)
        p=.5*replay.combo.parent.cat.probabilities(month,ids)+.5*rome[month]
        pred=sc.final(month,.71*calls+(1-.71)*tree,p)
        np.testing.assert_array_equal((pred-d['y'])**2,expected.loc[ids,'combined_calls_lgb'])
        rows.append(pd.DataFrame(dict(mvt_id=ids,month=month,day=d['day'],y=d['y'],pred=pred,
            submitted=np.rint(np.maximum(pred,0)))))
    held=pd.concat(rows,ignore_index=True);assert held.mvt_id.is_unique and len(held)==344419
    cv=float(np.mean([np.sqrt(np.mean((f.pred-f.y)**2)) for _,f in held.groupby('month')]))
    np.testing.assert_allclose(cv,sr['scores']['combined_calls_lgb']['cv'],atol=1e-10,rtol=0)
    sources=dict(sr['input_sha256']);sources.update(replay.SOURCES)
    for path in [Path(__file__),ap,mp,retained,parent.features.verify_artifact(v['parts']),
                 parent.features.verify_artifact(audit['source']),score.OUT/'result.json',score.OUT/'scored_errors.parquet',
                 parent.fs.WORK/'inputs_old.pkl',parent.fs.OUT/'inputs_old.json']:
        sources[str(path.relative_to(ROOT))]=sha256(path)
    dest.mkdir(parents=True);held.to_parquet(dest/'cv.parquet',index=False)
    cp=dest/'cv.json';write_json(cp,dict(cv=cv,prediction=parent.artifact(dest/'cv.parquet'),sources=sources,
        exact_reproduction=True,selection='Retained model family; reused development months'))
    sources[str(cp.relative_to(ROOT))]=sha256(cp);sources[str((dest/'cv.parquet').relative_to(ROOT))]=sha256(dest/'cv.parquet')
    shutil.copyfile(retained,output);assert sha256(output)==v['output']['sha256']
    write_json(dest/'manifest.json',dict(version=version,state='prepared',file=str(output.relative_to(ROOT)),sha256=sha256(output),
        rows=344841,parent='v64',change='neural share assigned to callsign15',jan_jul_cv=cv,cv_record=parent.artifact(cp),
        sources=sources,authorization='October11 user authorized autonomous uploads',
        audit=dict(retained_independent_audit=True,parent_all_rows_replayed_exact=True,cv_squared_errors_exact=True),
        limitation='Worse than v64 in Jan/Jul; submitted to measure complementarity of retained model families. No refitting.'))
    print(f'Prepared v{version}: retained callsign ensemble; CV {cv:.6f}',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--version',type=int,required=True);a=ap.parse_args()
    with threadpool_limits(2):prepare(a.version)
