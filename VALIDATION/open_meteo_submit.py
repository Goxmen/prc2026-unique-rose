"""Full-year ERA5 candidates: v60 Rome only, v61 all parts.

Fixed v58 settings and 0.71 network share. Run stages sequentially on the desktop.
Weather provenance and Jan/Jul evidence: reports/OPEN_METEO_JAN_JUL.md.
Both were initially authorized; after v60's upload the user required fresh
approval for further submissions. uploads_paused.json enforces that instruction.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'VALIDATION'), str(ROOT/'pipeline'), str(ROOT/'VALIDATION/tools')]
import hpo_submit as hs
import open_meteo_all as experiment
import rome_hpo_submit as rs
from records import sha256, write_json, runtime_identity

wx, ts, nis = experiment.wx, hs.ts, hs.nis
OUT = hs.RUNS/'v3_20261008_open_meteo_submit'
WORK = OUT/'work'
V58 = hs.RUNS/'v3_20261006_blend_weight_submit'/'submission_v58'
INPUTS = WORK/'inputs.pkl'
COMBO = list(hs.ids.COMBO)


def log(s):
    print(time.strftime('[%H:%M:%S] ')+s, flush=True)


def audit_weather(raw, values):
    """Recompute sampled values directly from API arrays, independently of the join."""
    r, v = raw.reset_index(drop=True), values.reset_index(drop=True)
    r = r.assign(month=pd.to_datetime(r.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m'))
    checked, maximum = 0, 0.
    for ap, group in r.groupby('ADEP_mvt'):
        frames = []
        for path in sorted((wx.CACHE/'weather').glob(f'{ap}_*.json')):
            h = pd.DataFrame(json.loads(path.read_text())['hourly'])
            h.index = pd.to_datetime(h.pop('time'),utc=True)
            frames.append(h)
        hourly = pd.concat(frames)
        hourly = hourly.loc[~hourly.index.duplicated()]
        for _, rows in group.groupby('month'):
            chosen = rows.sample(min(4,len(rows)),random_state=42)
            for i,row in chosen.iterrows():
                t = pd.Timestamp(row.MVT_TIME_UTC_mvt).tz_convert('UTC').floor('h')
                expected = {'om_'+c:float(hourly.loc[t,c]) for c in wx.VARIABLES}
                for c in ['precipitation','snowfall']:
                    for hours in [6,24,48]:
                        window = pd.date_range(t-pd.Timedelta(hours=hours-1),t,freq='h')
                        expected[f'om_{c}_{hours}h'] = float(hourly.loc[window,c].sum())
                for c,x in expected.items():
                    error = abs(float(v.loc[i,c])-x)
                    assert error <= 1e-5*max(1,abs(x)),(ap,t,c,error)
                    maximum = max(maximum,error)
                    checked += 1
    return dict(values_checked=checked,max_error=maximum)


def prepare():
    if (WORK/'prepared.json').exists(): return
    WORK.mkdir(parents=True,exist_ok=True)
    (WORK/'models').mkdir(exist_ok=True)
    (WORK/'parts').mkdir(exist_ok=True)
    for rec in ts.read(wx.CACHE/'weather_manifest.json'):
        assert sha256(ROOT/rec['path']) == rec['sha256']
    data = pd.read_pickle(hs.INPUTS)
    v46 = ts.read(nis.V46/'submission_v46.json')
    expected = ts.read(rs.v43s.PARENT_MANIFEST)['model']['parameters']
    reports = rs.weather.load_reports()
    grid = rs.features.snow_grid(pd.read_parquet(rs.weather.METAR))
    dep, arr = hs.overnight_run.load_inputs(1.)
    X,y,_,raw,params,_ = rs.cv.training_inputs(dep,arr,v46['identity']['config'],reports,grid,expected)
    assert params == data['parameters']
    del dep,arr,reports,grid
    gc.collect()
    table = wx.weather_table()
    lirf = raw.ADEP_mvt.eq('LIRF')
    r = raw.loc[lirf]
    M = rs.signals.classifier_matrix(X.loc[lirf],r,rs.KIND,ts.CLASSIFIER)
    target = rs.rome_hybrid.substituted(X.loc[lirf],np.asarray(y)[lirf],r).astype(float)
    w = wx.weather_join(r,table)
    assert not w.isna().any().any()
    checks = dict(rome=audit_weather(r,w))
    rome = dict(X=pd.concat([M,w],axis=1),y=target,months={})
    for m,d in data['months'].items():
        use = d['raw'].ADEP_mvt.eq('LIRF').to_numpy() & np.isfinite(d['V'][rs.rome_hybrid.GAP].to_numpy(float))
        B = rs.signals.classifier_matrix(d['V'].loc[use],d['raw'].loc[use],rs.KIND,ts.CLASSIFIER)
        b = wx.weather_join(d['raw'].loc[use],table)
        assert not b.isna().any().any()
        rome['months'][m] = dict(use=use,X=pd.concat([B,b],axis=1),ids=d['raw'].mvt_id.to_numpy())
    pd.to_pickle(rome,WORK/'rome.pkl')
    ordinary = raw.loc[data['Xn'].index]
    assert np.array_equal(ordinary.taxi_sec.to_numpy(),data['yn'])
    assert np.array_equal(ordinary.ADEP_mvt.to_numpy(),data['Xn'].airport.astype(str).to_numpy())
    assert set(pd.to_datetime(ordinary.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m')) == {f'2025-{m:02}' for m in range(1,13)}
    assert np.array_equal(pd.to_datetime(ordinary.MVT_TIME_UTC_mvt,utc=True).dt.strftime('%Y-%m').to_numpy(),data['tmonth'])
    w = wx.weather_join(ordinary,table)
    assert not w.isna().any().any()
    checks['ordinary'] = audit_weather(ordinary,w)
    data['Xn'] = pd.concat([data['Xn'],w],axis=1)
    ranking = pd.concat([data['months'][m]['raw'] for m in hs.MONTHS],ignore_index=True)
    assert np.array_equal(ranking.mvt_id.to_numpy(),data['rank_ids'])
    assert not np.intersect1d(ordinary.mvt_id,data['rank_ids']).size
    w = wx.weather_join(ranking,table)
    checks['ranking'] = audit_weather(ranking,w)
    w.index = data['Vn'].index
    assert not w.isna().any().any()
    data['Vn'] = pd.concat([data['Vn'],w],axis=1)
    assert list(data['Xn'][data['names']+COMBO+wx.WEATHER]) == list(experiment.ts.read(experiment.OUT/'inputs/2025-01.json')['columns'])
    pd.to_pickle(data,INPUTS)
    write_json(WORK/'prepared.json',dict(training_rows=len(ordinary),ranking_rows=len(ranking),rome_rows=len(M),
        weather_audit=checks,inputs_sha256=sha256(INPUTS),rome_sha256=sha256(WORK/'rome.pkl'),
        original_inputs_sha256=sha256(hs.INPUTS),weather_manifest_sha256=sha256(wx.CACHE/'weather_manifest.json')))
    log('Full-year matrices prepared and independently checked')


def fit_rome():
    dest = WORK/'parts/rome.parquet'
    if dest.exists(): return
    d = pd.read_pickle(WORK/'rome.pkl')
    model = rs.classifier(d['X'],d['y'],hs.searched()['rome']['overrides'],1700)
    model.save_model(str(WORK/'models/rome.txt'))
    frames = []
    for m in hs.MONTHS:
        item = d['months'][m]
        pred = np.zeros(len(item['ids']))
        pred[item['use']] = model.predict(item['X'],num_threads=8)
        frames.append(pd.DataFrame(dict(mvt_id=item['ids'],pred=pred)))
    pd.concat(frames,ignore_index=True).to_parquet(dest,index=False)
    write_json(dest.with_suffix('.json'),dict(rounds=1700,seed=42,columns=list(d['X']),
        rows=len(d['X']),positives=int(d['y'].sum()),sha256=sha256(dest),model_sha256=sha256(WORK/'models/rome.txt')))


def fit_networks(data):
    import torch
    import nn_improve_jan_jul as nni
    import nn_jan_jul as nnj
    torch.set_num_threads(8)
    cfg = hs.hpo.nn_config(hs.searched()['nn']['overrides'])
    hs.hpo.apply_nn(cfg,nnj,nni)
    cols = data['names']+COMBO+wx.WEATHER
    ts.FOLDS.setdefault(hs.ids.FULL,[f'2025-{m:02}' for m in range(1,13)])
    d = nni.Data(dict(X=data['Xn'][cols],y=data['yn'],Vn=data['Vn'][cols],tmonth=data['tmonth'],
        meta=dict(month=hs.ids.FULL),raw=pd.DataFrame({'taxi_sec':np.full(len(data['Vn']),np.nan)})),'cuda')
    assert d.info['inner_month']=='2025-12'
    dest = WORK/'parts/nn.parquet'
    out = pd.DataFrame(dict(mvt_id=data['rank_ids']))
    for s in hs.ids.GROUPS:
        record = WORK/'parts'/f'nn_group_{s}.json'
        if record.exists(): info = ts.read(record)
        else:
            _,_,curve = nni.train('ple',d,d.inner_train,s,cfg['epochs'],(d.subset(d.inner_val),d.y[d.inner_val]))
            info = dict(config=cfg,stop_epoch=min(curve,key=lambda x:x['rmse'])['epoch'],inner_curve=curve)
            write_json(record,info)
        for k in range(5):
            path = WORK/'parts'/f'nn_{s}_{k}.parquet'
            if not path.exists():
                model,pred,_ = nni.train('ple',d,np.arange(len(d.y)),s+1000*k,info['stop_epoch'],(d.held,d.yv))
                torch.save(dict(state_dict=model.state_dict(),bins=d.bins,cards=d.cards,mu=d.mu,sd=d.sd),WORK/'models'/f'nn_{s}_{k}.pt')
                pd.DataFrame(dict(mvt_id=data['rank_ids'],pred=pred.astype('float32'))).to_parquet(path,index=False)
                del model
                torch.cuda.empty_cache()
                log(f'Full-year network {s}/{k} saved')
            frame = pd.read_parquet(path)
            assert np.array_equal(frame.mvt_id,data['rank_ids'])
            out[f'n{s}_{k}'] = frame.pred.to_numpy()
    out.to_parquet(dest,index=False)
    return dict(config=cfg,columns=cols,network_members=15)


def fit(part):
    dest = WORK/'parts'/f'{part}.parquet'
    if dest.exists(): return
    data = pd.read_pickle(INPUTS)
    started = time.monotonic()
    hs.WORK,hs.THREADS = WORK,8
    if part=='nn': info = fit_networks(data)
    else:
        if part=='catboost':
            data['names'] = data['names']+wx.WEATHER
            hs.hpo.CB_BASE = dict(hs.hpo.CB_BASE,thread_count=8)
            frame,info = hs.catboost_part(data,hs.searched()[part])
        else:
            hs.ids.COMBO = COMBO+wx.WEATHER
            frame,info = hs.gbm_part(data,part,hs.searched()[part])
        frame.to_parquet(dest,index=False)
    write_json(dest.with_suffix('.json'),dict(**info,sha256=sha256(dest),minutes=(time.monotonic()-started)/60))


def build(version):
    assert version in (60,61)
    directory = OUT/f'submission_v{version}'
    manifest_path = directory/f'submission_v{version}.json'
    if manifest_path.exists(): return
    assert not hs.submission(version).exists()
    data = pd.read_pickle(INPUTS)
    new = {}
    for part in ['rome']+([] if version==60 else hs.PARTS):
        f = pd.read_parquet(WORK/'parts'/f'{part}.parquet')
        assert np.array_equal(f.mvt_id.to_numpy(),data['rank_ids'])
        new[part] = f.drop(columns='mvt_id').to_numpy(float).mean(axis=1)
    frames,controls,summary,offset = [],[],{},0
    directory.mkdir(parents=True,exist_ok=True)
    for m in hs.MONTHS:
        d = data['months'][m]
        saved = pd.read_parquet(V58/f'ranking_{m}.parquet')
        assert np.array_equal(saved.mvt_id,d['raw'].mvt_id)
        control,_ = nis.final(saved.blend_base.to_numpy(float),saved.p.to_numpy(float),d['V'],d['raw'],data['parameters'],data['fitted'])
        assert np.max(np.abs(control.pred-saved.pred))<1e-6
        sl = slice(offset,offset+len(saved)); offset += len(saved)
        base = saved.blend_base.to_numpy(float) if version==60 else .71*new['nn'][sl]+.29*np.mean([new[p][sl] for p in hs.GBM],axis=0)
        frame,masks = nis.final(base,new['rome'][sl],d['V'],d['raw'],data['parameters'],data['fitted'])
        frame['blend_base'],frame['month'] = base,m
        delta = frame.pred.to_numpy(float)-saved.pred.to_numpy(float)
        if version==60: assert np.all(delta[d['raw'].ADEP_mvt.to_numpy()!='LIRF']==0)
        assert np.array_equal(frame.rule_triggered,control.rule_triggered)
        summary[m] = dict(rows=len(frame),changed=int((delta!=0).sum()),mean_change=float(delta.mean()),
            mean_abs_change=float(np.abs(delta).mean()),max_abs_change=float(np.abs(delta).max()),over_1h=int((np.abs(delta)>3600).sum()),correction_masks=masks)
        frame.to_parquet(directory/f'ranking_{m}.parquet',index=False)
        frames.append(frame); controls.append(control)
    f = pd.concat(frames,ignore_index=True).set_index('mvt_id').reindex(data['rank_ids']).reset_index()
    c = pd.concat(controls,ignore_index=True).set_index('mvt_id')
    template = pq.read_table(nis.TEMPLATE)
    parent = pq.read_table(hs.submission(58))
    # The submission schema rounds predictions to integer seconds. Reproduce that
    # same formatting before comparing the saved control with the actual upload.
    control_table = hs.fill_template(template,c.index,c.pred.to_numpy())
    assert control_table.equals(parent)
    assert len(f)==344841 and f.mvt_id.is_unique and np.isfinite(f.pred).all() and (f.pred>=0).all()
    assert f.rule_triggered.sum()==43
    table = hs.fill_template(template,f.mvt_id,f.pred.to_numpy())
    nis.transfer.validate_submission(table,template)
    assert table['MVT_ID_mvt'].equals(template['MVT_ID_mvt'])
    pq.write_table(table,hs.submission(version))
    cv = ts.read(experiment.OUT/'all_result.json')
    write_json(manifest_path,dict(version=version,state='prepared',file=str(hs.submission(version).relative_to(ROOT)),
        sha256=sha256(hs.submission(version)),parent='v58',variant='Rome weather only' if version==60 else 'Weather in all five parts',
        jan_jul_cv=dict(control=cv['control'],candidate=cv['parts']['rome' if version==60 else 'all']),
        summary=summary,reproduced_v58=True,network_share=.71,prepared_at=hs.now(),runtime=runtime_identity(),
        source_sha256=sha256(Path(__file__)),weather_manifest_sha256=sha256(wx.CACHE/'weather_manifest.json')))
    log(f'v{version} prepared and validated: '+json.dumps(summary))


def upload(version):
    """Honor the upload pause unless this exact version and hash were approved."""
    approval_path = OUT/f'upload_approval_v{version}.json'
    approval = None
    if (OUT/'uploads_paused.json').exists():
        if approval_path.exists():
            approval = ts.read(approval_path)
        if not approval or approval.get('state') != 'approved' or approval.get('version') != version:
            raise SystemExit('Upload withheld: the user requires fresh approval before any further submission.')
    import hashlib
    path = OUT/f'submission_v{version}'/f'submission_v{version}.json'
    record = ts.read(path)
    local = hs.submission(version)
    assert sha256(local)==record['sha256']
    if approval is not None:
        assert approval['sha256'] == record['sha256'], 'Approval is for a different file'
    mc = shutil.which('mc') or str(Path.home()/'.local/bin/mc')
    remote = 'opensky/prc-2026-unique-rose/'+local.name
    stat = subprocess.run([mc,'stat','--json',remote],capture_output=True,text=True)
    if stat.returncode:
        error = json.loads(stat.stdout or stat.stderr)
        cause = error.get('error',{}).get('cause',{})
        assert cause.get('error',{}).get('Code') in ('NoSuchKey','NotFound') or cause.get('message')=='Object does not exist',error
        subprocess.run([mc,'cp',str(local),remote],check=True)
    proc = subprocess.Popen([mc,'cat',remote],stdout=subprocess.PIPE)
    digest = hashlib.sha256()
    for block in iter(lambda:proc.stdout.read(1024*1024),b''): digest.update(block)
    assert proc.wait()==0 and digest.hexdigest()==record['sha256']
    record.update(state='uploaded',uploaded_at=hs.now(),remote=remote,remote_sha256=digest.hexdigest())
    if approval is not None:
        record['upload_approval'] = dict(path=str(approval_path.relative_to(ROOT)),
                                        approved_at=approval['approved_at'], instruction=approval['instruction'])
    write_json(path,record)
    if approval is not None:
        approval.update(state='consumed', uploaded_at=record['uploaded_at'], remote_sha256=digest.hexdigest())
        write_json(approval_path,approval)
    log(f'v{version} uploaded; remote bytes verified')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','rome','nn','lgb','airport','catboost','build','upload'])
    p.add_argument('--version',type=int,choices=[60,61])
    args=p.parse_args()
    with threadpool_limits(limits=1,user_api='blas'):
        if args.stage=='prepare': prepare()
        elif args.stage=='rome': fit_rome()
        elif args.stage=='build': build(args.version)
        elif args.stage=='upload': upload(args.version)
        else: fit(args.stage)
