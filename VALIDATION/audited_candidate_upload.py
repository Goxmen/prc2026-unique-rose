"""Upload a prepared, source-hashed candidate under October11 authorization."""
import argparse,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import pyarrow.parquet as pq
import scored_blend_submit as transport
from records import sha256,write_json
from submission_slot_policy import check_slot

def run(manifest_path,stage):
    path=Path(manifest_path).resolve();assert path.is_relative_to(ROOT/'VALIDATION/runs')
    m=json.loads(path.read_text());output=ROOT/m['file'];version=m['version']
    assert output.name==f'unique-rose_v{version}.parquet' and sha256(output)==m['sha256']
    if stage=='upload':
        assert m['state']=='prepared' and m['authorization'] and m['rows']==344841
        assert isinstance(m['jan_jul_cv'],(int,float))
        for p,digest in m['sources'].items():assert sha256(ROOT/p)==digest,p
        table=pq.read_table(output);template=pq.read_table(ROOT/'data/submitting.parquet')
        assert table.schema==template.schema and table.num_rows==344841
        assert table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
        assert table['TAXITIME_SEC_mvt'].null_count==0
        assert (table['TAXITIME_SEC_mvt'].to_numpy()>=0).all()
        public=transport.live.fetch_leaderboard();uploads=transport.live.fetch_bucket(strict=True)
        byfile={r['filename']:r for r in public if r['teamName']==transport.live.TEAM}
        for filename,score in m.get('basis_scores',{}).items():
            assert byfile[filename]['usedPairs']==344841 and abs(byfile[filename]['score']-score)<1e-8
        latest=sorted(public,key=lambda x:x['processedAt'])[-50:]
        assert all(r.get('usedPairs') in [None,344841] for r in latest),'Accepted population changed'
        assert output.name not in {r['file'] for r in uploads},'Remote filename occupied; never re-upload'
        today=datetime.now(timezone.utc).date()
        used=[r['file'] for r in uploads if r.get('uploadedAt') and
            datetime.fromisoformat(r['uploadedAt'].replace('Z','+00:00')).astimezone(timezone.utc).date()==today]
        assert len(used)<5,'Five-upload UTC-day limit reached'
        slot_policy=check_slot(m,used,today)
        write_json(path.parent/'live_preflight.json',dict(at=transport.now(),uploads_today=used,
            uploader_sha256=sha256(__file__),authorization=m['authorization'],slot_policy=slot_policy))
        receipt=transport.mc('cp','--disable-multipart','--custom-header','If-None-Match:*',
            '--attr',f'sha256={m["sha256"]}',output,transport.live.BUCKET+output.name)
        (path.parent/'mc_upload.log').write_text(receipt)
        remote=hashlib.sha256(transport.mc('cat',transport.live.BUCKET+output.name,binary=True)).hexdigest()
        assert remote==m['sha256'];m.update(state='uploaded',uploaded_at=transport.now(),remote_sha256=remote)
        write_json(path,m);print(f'Uploaded v{version}; remote hash exact',flush=True)
    else:
        assert m['state'] in ['uploaded','scored','rejected']
        official=json.loads(transport.mc('cat',transport.live.BUCKET+output.name+'_result.json'))
        write_json(path.parent/'official_result.json',official)
        m.update(official_result=official,checked_at=transport.now())
        if official.get('status')=='Succeeded':
            assert official['used_pairs']==344841;m.update(state='scored',score=official['score'])
        elif official.get('status') not in ['Pending','Running']:m['state']='rejected'
        write_json(path,m);print(json.dumps(official),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['upload','result']);ap.add_argument('manifest')
    a=ap.parse_args();run(a.manifest,a.stage)
