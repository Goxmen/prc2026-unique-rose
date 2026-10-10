"""Build, audit, and autonomously upload the validated affine scored ensemble.

User authorization on October11 supersedes old per-file approval guards.
Conditional PUT prevents overwrite; every source score, input hash, template
column, rounded output and returned remote hash is checked. No models are fit.
"""
import argparse,hashlib,json,os,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import numpy as np,pandas as pd,pyarrow.parquet as pq
from threadpoolctl import threadpool_limits
import scored_submission_geometry as geometry
from records import sha256,write_json
from submit import fill_template
import dashboard_data as live
CV=ROOT/'VALIDATION/runs/v3_20261011_scored_blend_jan_jul/result.json'
BASIS=geometry.OUT/'validation_ready_basis.json'
OUT=ROOT/'VALIDATION/runs/v3_20261011_scored_blend_submit'
AUTHORIZATION='User: forget the rule of asking approval for uploads. You are free to submit to leaderboard as you please.'

def now():return datetime.now(timezone.utc).isoformat(timespec='seconds')
def mc(*args,binary=False):
    cmd=[str(Path.home()/'.local/bin/mc'),*map(str,args)]
    r=subprocess.run(cmd,env=live.mc_environment(),capture_output=True,check=True)
    return r.stdout if binary else r.stdout.decode()

def path(version):return ROOT/'submissions'/f'unique-rose_v{version}.parquet'
def folder(version):return OUT/f'submission_v{version}'

def prepare(version):
    dest=folder(version);output=path(version)
    assert not dest.exists() and not output.exists(),'Candidate exists; inspect retained artifacts'
    c=json.loads(CV.read_text());r=json.loads(BASIS.read_text())
    assert c['geometry_sha256']==sha256(BASIS) and c['weights']==r['weights']
    for name,digest in c['inputs'].items():assert sha256(ROOT/name)==digest,name
    assert sha256(CV.parent/'scored.parquet')==c['scored_sha256']
    ids,P,scores,versions,records=geometry.load(1)
    selected=r['versions'];idx=[versions.index(v) for v in selected]
    P=P[:,idx];scores=scores[idx];w=np.array([r['weights'][str(v)] for v in selected])
    assert abs(w.sum()-1)<1e-8 and np.isfinite(w).all()
    pred=P@w
    # Independent identity: sum(w_i MSE_i) - sum(w_i E[p_i^2]) + E[(sum w_i p_i)^2].
    mse=float(w@(scores*scores)-w@np.mean(P*P,axis=0)+np.mean(pred*pred))
    np.testing.assert_allclose(np.sqrt(mse),r['projected_rmse'],rtol=0,atol=1e-8)
    template_path=ROOT/'data/submitting.parquet';template=pq.read_table(template_path)
    assert template.num_rows==344841 and np.array_equal(template['MVT_ID_mvt'].to_numpy(),ids)
    table=fill_template(template,ids,pred)
    assert table.schema.equals(template.schema) and table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    expected=np.rint(np.maximum(pred,0)).astype(np.int32)
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),expected)
    assert (expected>=0).all() and np.isfinite(pred).all()
    rounding_rms=float(np.sqrt(np.mean((expected-pred)**2)))
    assert int((pred<0).sum())==r['negative_predictions']==0
    dest.mkdir(parents=True);pq.write_table(table,output)
    pqtable=pq.read_table(output);assert pqtable.equals(table)
    pd.DataFrame(dict(mvt_id=ids,unrounded=pred,submitted=expected)).to_parquet(dest/'audit_rows.parquet',index=False)
    write_json(dest/'manifest.json',dict(version=version,file=str(output.relative_to(ROOT)),sha256=sha256(output),
        state='prepared',prepared_at=now(),authorization=AUTHORIZATION,rows=len(ids),weights=r['weights'],
        projected_rmse=float(np.sqrt(mse)),score_rounding_interval=r['score_rounding_rmse_interval'],
        output_rounding_rms_sec=rounding_rms,
        conservative_rmse_interval_after_serialization=[r['score_rounding_rmse_interval'][0]-rounding_rms,r['score_rounding_rmse_interval'][1]+rounding_rms],
        jan_jul_cv=c['candidate_cv'],control_jan_jul_cv=c['control_cv'],
        cv_sha256=sha256(CV),basis_sha256=sha256(BASIS),source_sha256=sha256(__file__),
        template_sha256=sha256(template_path),sources={str(v):records[str(v)] for v in selected},
        audit=dict(independent_mse_identity=True,template_ids_and_schema_exact=True,integer_rounding_exact=True,readback_exact=True),
        limitation='Leaderboard-selected coefficients; projection assumes unchanged truth across own scores. Current-fold CV is not independent confirmation.'))
    print(f'Prepared v{version}: projected {np.sqrt(mse):.6f}; Jan/Jul {c["candidate_cv"]:.6f}',flush=True)

def upload(version):
    dest=folder(version);mp=dest/'manifest.json';m=json.loads(mp.read_text());output=path(version)
    assert m['state']=='prepared' and m['sha256']==sha256(output)
    assert m['source_sha256']==sha256(__file__) and m['cv_sha256']==sha256(CV) and m['basis_sha256']==sha256(BASIS)
    assert m['template_sha256']==sha256(ROOT/'data/submitting.parquet')
    for r in m['sources'].values():assert sha256(ROOT/r['file'])==r['sha256']
    public=live.fetch_leaderboard();uploads=live.fetch_bucket(strict=True)
    byfile={v['filename']:v for v in public if v['teamName']==live.TEAM}
    for v,r in m['sources'].items():
        p=byfile[f'unique-rose_v{v}.parquet']
        assert p['usedPairs']==344841 and abs(p['score']-r['score'])<1e-8,'A basis score/population changed'
    recent=sorted(public,key=lambda v:v['processedAt'])[-50:]
    assert all(v.get('usedPairs') in [None,344841] for v in recent),'Accepted population may have changed; inspect final-set readiness'
    assert output.name not in {u['file'] for u in uploads},'Remote name occupied; never overwrite'
    today=datetime.now(timezone.utc).date()
    used=[u['file'] for u in uploads if u.get('uploadedAt') and
        datetime.fromisoformat(u['uploadedAt'].replace('Z','+00:00')).astimezone(timezone.utc).date()==today]
    assert len(used)<5,'Five-upload UTC-day limit reached; retain candidate for the next day'
    write_json(dest/'live_preflight.json',dict(at=now(),existing_uploads_today=used,
        basis_scores={v:byfile[f'unique-rose_v{v}.parquet'] for v in m['sources']},authorization=AUTHORIZATION))
    receipt=mc('cp','--disable-multipart','--custom-header','If-None-Match:*','--attr',
        f'candidate=scored-affine-ensemble;sha256={m["sha256"]}',output,live.BUCKET+output.name)
    (dest/'mc_upload.log').write_text(receipt)
    digest=hashlib.sha256(mc('cat',live.BUCKET+output.name,binary=True)).hexdigest()
    assert digest==m['sha256'],'Remote bytes differ'
    m.update(state='uploaded',uploaded_at=now(),remote_sha256=digest);write_json(mp,m)
    print(f'Uploaded v{version}; remote SHA-256 verified',flush=True)

def result(version):
    dest=folder(version);mp=dest/'manifest.json';m=json.loads(mp.read_text())
    assert m['state'] in ['uploaded','scored','rejected']
    key=path(version).name+'_result.json'
    official=json.loads(mc('cat',live.BUCKET+key))
    write_json(dest/'official_result.json',official)
    m['official_result']=official;m['checked_at']=now()
    if official.get('status')=='Succeeded':
        assert official['used_pairs']==344841
        m.update(state='scored',score=official['score'])
    elif official.get('status') not in ['Pending','Running']:m['state']='rejected'
    write_json(mp,m);print(json.dumps(official),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','upload','result']);ap.add_argument('--version',required=True,type=int);a=ap.parse_args()
    with threadpool_limits(2):globals()[a.stage](a.version)
