"""Execute one explicitly authorized, audited upload after its UTC quota window.

No fits or candidate changes. Fresh source/schema/quota/population checks remain
in audited_candidate_upload. A fixed queue hash binds this job to the reviewed
manifest; a cancellation file or inactive objective prevents submission.
"""
import argparse,json,os,time,traceback
from datetime import datetime,timezone
from pathlib import Path
import audited_candidate_upload as uploader
from records import sha256,write_json
ROOT=uploader.ROOT

def now():return datetime.now(timezone.utc)
def load(path):return json.loads(Path(path).read_text())

def run(path):
    path=Path(path).resolve();q=load(path)
    assert q['state']=='queued' and q['authorization']=='User authorized autonomous leaderboard uploads'
    assert sha256(__file__)==q['runner_sha256'],'Queued uploader source changed'
    manifest=ROOT/q['manifest'];cancel=ROOT/q['cancel_file'];objective=ROOT/q['objective']
    lock=path.with_suffix('.lock');fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    def status(state,**extra):
        q.update(state=state,updated_at=now().isoformat(),**extra);write_json(path,q)
    try:
        start=datetime.fromisoformat(q['not_before_utc']);deadline=datetime.fromisoformat(q['expires_utc'])
        status('waiting',pid=os.getpid())
        while now()<start:
            if cancel.exists() or load(objective)['state']!='active':status('cancelled');return
            if now()>=deadline:status('expired');return
            time.sleep(min(30,max((start-now()).total_seconds(),.1)))
        if cancel.exists() or load(objective)['state']!='active':status('cancelled');return
        assert now()<deadline
        assert sha256(manifest)==q['manifest_sha256'],'Reviewed manifest changed'
        m=load(manifest);assert m['state']=='prepared' and m['audit']['independent_review']['status']=='pass'
        public=uploader.transport.live.fetch_leaderboard();ours=[r for r in public if r['teamName']==uploader.transport.live.TEAM and r.get('score') is not None and (r.get('usedPairs') or 0)>0]
        if ours and min(r['score'] for r in ours)<=load(objective)['target_rmse']:
            status('skipped_goal_met');return
        status('uploading')
        uploader.run(manifest,'upload');status('uploaded',uploaded_manifest_sha256=sha256(manifest))
        end=min(deadline.timestamp(),time.time()+3600)
        while time.time()<end:
            time.sleep(30)
            try:uploader.run(manifest,'result')
            except Exception as exc:
                status('awaiting_result',last_result_error=str(exc));continue
            m=load(manifest)
            if m['state'] in ['scored','rejected']:
                status(m['state'],score=m.get('score'),result=m.get('official_result'));return
        status('result_pending_timeout')
    except Exception as exc:
        status('error',error=str(exc),traceback=traceback.format_exc());raise
    finally:
        lock.unlink(missing_ok=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('queue');a=p.parse_args();run(a.queue)
