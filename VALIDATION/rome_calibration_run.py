"""Run the two frozen Rome calibration experiments, reusing ordinary models.

  .venv/bin/python VALIDATION/rome_calibration_run.py --output-dir <new run>
  Add --resume to continue verified completed work; --smoke is execution-only.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
import glob
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'VALIDATION'),str(ROOT/'pipeline')]
import artifact_control_run as parent_run
import artifact_experiments_run as derived
from common import TRAIN_GLOB, RANKING
import modeling
import overnight_run as shared
from records import RUNS, append_log, data_identity, runtime_identity, sha256, write_json
import reporting
import rome_calibration as recipe
from rome_hybrid_run import save_booster
import rome_signals as signals
import stand_day

PARENT=RUNS/'v3_20260919_artifact_control'
CONTROL='pred_noid_cap'
META=['mvt_id','mvt_time','airport','y','has_flight_plan','arrival_history_available',
      'arrival_history_days_available','raw_flight_id_missing','fold','month','seed']
RAW=['ADEP_mvt','FLIGHT_ID_mvt','MVT_TIME_UTC_mvt','SCHED_TIME_UTC_mvt','LOBT_flt']
READ=list(dict.fromkeys(META+RAW+[CONTROL,'ordinary','p','blended','gap_sec','rule_triggered']))
SOURCE_FILES=sorted(set(parent_run.SOURCES+['VALIDATION/rome_calibration.py',
    'VALIDATION/rome_calibration_run.py','VALIDATION/rome_calibration_report.py','VALIDATION/docs/ROME_CALIBRATION.md']))


def inner_jobs(outer):
    jobs=[]
    for n in (3,4,5):
        months=outer['train'][:n]
        end=str(pd.Period(months[-1],'M').end_time.date())
        jobs.append(dict(id=f"{outer['id']}__inner{n}",scheme='inner',train=months,
                         start=months[0]+'-01',end=end,freeze=end,
                         evals=[dict(fold=f'inner{n}',month=outer['train'][n])]))
    return jobs


def check_inner_scope(outer, inner, train, valid):
    t=pd.to_datetime(train.MVT_TIME_UTC_mvt,utc=True)
    v=pd.to_datetime(valid.MVT_TIME_UTC_mvt,utc=True)
    start=pd.Timestamp(outer['start'],tz='UTC')
    end=pd.Timestamp(outer['end'],tz='UTC')+pd.Timedelta(days=1)
    if t.empty or v.empty or not (t.min()>=start and t.max()<v.min() and v.max()<end):
        raise ValueError('Inner split escapes its outer training window or is not chronological')
    if len(np.intersect1d(train.mvt_id.to_numpy(),valid.mvt_id.to_numpy())):
        raise ValueError('Inner held-out movement entered classifier training')
    if not v.dt.strftime('%Y-%m').eq(inner['evals'][0]['month']).all():
        raise ValueError('Wrong inner validation month')


def extra_inputs(X, raw, dep, arr):
    X=signals.add_signals(X,raw,arr)
    X=stand_day.add_stand_day(X,raw,dep,arr,features=stand_day.STAND_FEATURES)
    return signals.classifier_matrix(X,raw,'clock',recipe.EXTRA)


def training_inputs(td,ta,config):
    X,y,cats,raw=shared.prepare_training(td,ta,config)
    mask=raw.ADEP_mvt.eq('LIRF').to_numpy()
    X,raw,y=X.loc[mask].copy(),raw.loc[mask].copy(),y[mask]
    X=extra_inputs(X,raw,td,ta)
    return X,y,cats,raw


def validation_inputs(td,ta,vd,va,freeze,cats,config):
    raw=vd.loc[vd.ADEP_mvt.eq('LIRF')].copy()
    V=modeling.validation_matrix(td,ta,raw,va,freeze,cats,config)
    V=extra_inputs(V,raw,pd.concat([td,vd],ignore_index=True),pd.concat([ta,va],ignore_index=True))
    return V,raw


def checkpoint(directory,progress):
    write_json(directory/'progress.json',progress)


def entry(directory,path,**kwargs):
    return dict(path=str(path.relative_to(directory)),artifacts=shared.artifact_hashes(directory,path),**kwargs)


def fitted_model(directory,progress,key,X,y,raw,seed,classes,config):
    if key in progress['models']:
        info=progress['models'][key]
        return lgb.Booster(model_file=str(directory/info['path']),params={'num_threads':config['prediction_threads']})
    started=time.monotonic()
    booster,info=recipe.fit_classifier(X,y,X.mvt_minus_sched_sec.to_numpy(),seed,classes,
                                     threads=config['threads'],rounds=config['classifier_rounds'])
    path=directory/'models'/f'{key}.txt'
    save_booster(booster,path)
    ids_path=directory/'models'/f'{key}.train_ids.parquet'
    shared.save_frame(ids_path,pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(),
        mvt_time=raw.MVT_TIME_UTC_mvt.to_numpy(),target=y,
        outcome=recipe.outcomes(y,X.mvt_minus_sched_sec.to_numpy(),classes))))
    progress['models'][key]=entry(directory,path,**info,seed=seed,
        train_start=str(raw.MVT_TIME_UTC_mvt.min()),train_end=str(raw.MVT_TIME_UTC_mvt.max()),
        seconds=round(time.monotonic()-started,3),train_ids=str(ids_path.relative_to(directory)))
    progress['models'][key]['artifacts'].update(shared.artifact_hashes(directory,ids_path))
    checkpoint(directory,progress)
    print(f"Fitted {key}: classes {info['counts']}, {progress['models'][key]['seconds']:.1f}s",flush=True)
    return booster


def inner_predictions(directory,progress,outer,dep,arr,config):
    jobs=inner_jobs(outer)[:1] if config['smoke'] else inner_jobs(outer)
    for job in jobs:
        keys=[f"{job['id']}__{s}" for s in config['seeds']]
        if all(k in progress['inner'] for k in keys):
            continue
        td,ta,validation=shared.split_job(dep,arr,job)
        vd,va=validation[job['evals'][0]['month']]
        check_inner_scope(outer,job,td,vd)
        print(f"Preparing {job['id']}: {job['train']} -> {job['evals'][0]['month']}",flush=True)
        X,y,cats,raw=training_inputs(td,ta,config)
        V,vr=validation_inputs(td,ta,vd,va,job['freeze'],cats,config)
        # These held-out outcomes are calibration TRAINING data; keep the parent's training-label range.
        keep=vr.taxi_sec.between(1,config['max_taxi_sec']).to_numpy()
        V,vr=V.loc[keep],vr.loc[keep]
        for seed,key in zip(config['seeds'],keys):
            if key in progress['inner']:
                continue
            f=pd.DataFrame(dict(mvt_id=vr.mvt_id.to_numpy(),mvt_time=vr.MVT_TIME_UTC_mvt.to_numpy(),
                y=vr.taxi_sec.to_numpy(),gap_sec=V.mvt_minus_sched_sec.to_numpy(),
                noid=vr.FLIGHT_ID_mvt.isna().to_numpy(),seed=seed,inner=job['id']))
            for classes in (2,3):
                model=fitted_model(directory,progress,f"{job['id']}__c{classes}__{seed}",X,y,raw,seed,classes,config)
                p=recipe.probabilities(model,V,classes,config['prediction_threads'])
                if classes==2:p=recipe.binary_schedule_scope(p,f.gap_sec.to_numpy())
                for k in range(classes):f[f'p{classes}_{k}']=p[:,k]
                del model
            path=directory/'inner'/f'{key}.parquet'
            shared.save_frame(path,f)
            progress['inner'][key]=entry(directory,path,rows=len(f),outer=outer['id'],job=job)
            checkpoint(directory,progress)
        del X,y,cats,raw,V,vr,td,ta,vd,va,validation
        gc.collect()


def calibrators(directory,progress,outer,config):
    for seed in config['seeds']:
        key=f"{outer['id']}__{seed}"
        if key in progress['calibrators']:
            continue
        paths=[directory/e['path'] for k,e in progress['inner'].items()
               if e['outer']==outer['id'] and k.endswith('__'+str(seed))]
        f=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
        if f.mvt_id.duplicated().any():raise ValueError('Duplicate inner calibration movement')
        if not (pd.to_datetime(f.mvt_time,utc=True)<pd.Timestamp(outer['end'],tz='UTC')+pd.Timedelta(days=1)).all():
            raise ValueError('Outer held-out data entered calibration')
        fitted={}
        for classes in (2,3):
            p=f[[f'p{classes}_{k}' for k in range(classes)]].to_numpy()
            fitted[str(classes)]=recipe.fit_calibrator(p,recipe.outcomes(f.y,f.gap_sec,classes),f.noid)
        path=directory/'calibrators'/f'{key}.json'
        write_json(path,dict(outer=outer,seed=seed,fitted=fitted,inner_rows=len(f),
                            inner_files=[str(p.relative_to(directory)) for p in paths]))
        progress['calibrators'][key]=entry(directory,path)
        checkpoint(directory,progress)
        print(f"Calibrated {key} from {len(f):,} inner held-out flights",flush=True)


def outer_predictions(directory,progress,source,parent_progress,outer,dep,arr,config):
    if all(f"{e['fold']}__{s}" in progress['predictions'] for e in outer['evals'] for s in config['seeds']):return
    td,ta,validation=shared.split_job(dep,arr,outer)
    X,y,cats,raw=training_inputs(td,ta,config)
    gap=X.mvt_minus_sched_sec.to_numpy()
    day_count=int((recipe.day_eligible(raw,gap)&(recipe.outcomes(y,gap,3)==2)).sum())
    progress['windows'][outer['id']]=dict(job=outer,day_count=day_count,
        classes=np.bincount(recipe.outcomes(y,gap,3),minlength=3).tolist(),
        late_mean=source['parameters'][outer['id']]['late_mean'])
    for seed in config['seeds']:
        model=fitted_model(directory,progress,f"{outer['id']}__c3__{seed}",X,y,raw,seed,3,config)
        del model
    del X,y,raw;gc.collect()
    for ev in outer['evals']:
        fold=ev['fold']
        if all(f'{fold}__{s}' in progress['predictions'] for s in config['seeds']):continue
        vd,va=validation[ev['month']]
        V,vr=validation_inputs(td,ta,vd,va,outer['freeze'],cats,config)
        for seed in config['seeds']:
            key=f'{fold}__{seed}'
            if key in progress['predictions']:continue
            frame=pd.read_parquet(PARENT/source['folds'][fold]['predictions'][str(seed)],columns=READ)
            if config['smoke']:
                frame=frame.set_index('mvt_id').loc[vd.mvt_id.to_numpy()].reset_index()
            if not np.array_equal(frame.mvt_id.to_numpy(),vd.mvt_id.to_numpy()):
                raise ValueError(f'{fold}: parent movement order differs')
            if not np.array_equal(frame.y.to_numpy(),vd.taxi_sec.to_numpy()):raise ValueError('Parent labels differ')
            rome=frame.airport.eq('LIRF').to_numpy()
            if not np.array_equal(frame.loc[rome,'mvt_id'].to_numpy(),vr.mvt_id.to_numpy()):raise ValueError('Rome alignment differs')
            noid=vr.FLIGHT_ID_mvt.isna().to_numpy();g=frame.gap_sec.to_numpy()[rome]
            base=frame.ordinary.to_numpy()[rome];late=source['parameters'][outer['id']]['late_mean']
            p2=np.column_stack([1-frame.p.to_numpy()[rome],frame.p.to_numpy()[rome]])
            bkey=f"{outer['id']}__classifier_stand__{seed}"
            binary=lgb.Booster(model_file=str(PARENT/parent_progress['models'][bkey]['path']),params={'num_threads':config['prediction_threads']})
            replay=recipe.probabilities(binary,V,2,config['prediction_threads'])
            replay=recipe.binary_schedule_scope(replay,g)
            maxdiff=float(np.max(abs(replay-p2)))
            if not config['smoke'] and maxdiff>1e-10:raise ValueError(f'Classifier features no longer reproduce parent: {maxdiff}')
            model=lgb.Booster(model_file=str(directory/progress['models'][f"{outer['id']}__c3__{seed}"]['path']),params={'num_threads':config['prediction_threads']})
            p3=recipe.probabilities(model,V,3,config['prediction_threads'])
            fitted=json.loads((directory/progress['calibrators'][f"{outer['id']}__{seed}"]['path']).read_text())['fitted']
            c2=recipe.calibrate(p2,noid,fitted['2'])
            c3=recipe.calibrate(p3,noid,fitted['3'])
            eligible=recipe.day_eligible(vr,g)
            d3=recipe.constrain_day(c3,eligible,day_count)
            u3=recipe.constrain_day(p3,eligible,day_count)
            predictions=dict(binary=recipe.mixture(base,g,c2,vr,late),
                three=recipe.mixture(base,g,d3,vr,late),binary_uncal=recipe.mixture(base,g,p2,vr,late),
                three_uncal=recipe.mixture(base,g,u3,vr,late))
            if not np.allclose(predictions['binary_uncal'],frame.blended.to_numpy()[rome],atol=1e-8,rtol=0):
                raise ValueError('Uncalibrated mixture differs from saved pre-rule prediction')
            predictions['binary_cal_hard']=np.where(frame.rule_triggered.to_numpy()[rome],g,predictions['binary'])
            for name,p in predictions.items():frame[recipe.COLUMNS[name]]=recipe.postprocess(frame[CONTROL],p,frame,rome)
            for name,p in [('binary_raw',p2),('binary_cal',c2),('three_raw',p3),('three_cal_before_day',c3),('three_cal',d3)]:
                for k in range(p.shape[1]):
                    arrp=np.full(len(frame),np.nan);arrp[rome]=p[:,k];frame[f'{name}_p{k}']=arrp
            frame['day_eligible']=False;frame.loc[rome,'day_eligible']=eligible
            if not np.isfinite(frame[[CONTROL,*recipe.COLUMNS.values()]].to_numpy()).all():raise ValueError('Nonfinite final prediction')
            path=directory/f'oof_{fold}_seed{seed}.parquet'
            shared.save_frame(path,frame)
            progress['predictions'][key]=entry(directory,path,rows=len(frame),binary_replay_max_difference=maxdiff)
            checkpoint(directory,progress)
            del frame,binary,model
        info=copy.deepcopy(source['folds'][fold]);info['predictions']={str(s):f'oof_{fold}_seed{s}.parquet' for s in config['seeds']}
        progress['folds'][fold]=info;checkpoint(directory,progress)
        print(f'Scored {fold}: {len(vd):,} flights, all seeds, both experiments',flush=True)
        del V,vr
        gc.collect()
    del td,ta,validation,cats;gc.collect()


def compare(record,directory,progress,candidate,baseline,block=1):
    key=f'{candidate}_vs_{baseline}_block{block}'
    if key in progress['comparisons']:
        return json.loads((directory/progress['comparisons'][key]['path']).read_text())
    c=copy.deepcopy(record);b=copy.deepcopy(record)
    c['prediction_column']=recipe.COLUMNS[candidate]
    b['prediction_column']=CONTROL if baseline=='control' else recipe.COLUMNS[baseline]
    c['config']['block_days']=b['config']['block_days']=block
    report=reporting.report(c,directory,(b,directory))
    payload=dict(candidate=candidate,baseline=baseline,block_days=block,report=report)
    path=directory/f'comparison_{key}.json';write_json(path,payload)
    progress['comparisons'][key]=entry(directory,path);checkpoint(directory,progress)
    scores=report['scores'];change=report['comparison']['scores']['R_combined']
    print(f"{key}: {scores['R_A']:.4f} / {scores['R_B_long']:.4f} / {scores['R_combined']:.4f}; delta {change['difference']:+.4f}",flush=True)
    return payload


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--resume',action='store_true');ap.add_argument('--smoke',action='store_true')
    ap.add_argument('--threads',type=int,default=6);ap.add_argument('--prediction-threads',type=int,default=20)
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args(argv)
    if args.threads<1 or args.prediction_threads<1:ap.error('Thread counts must be positive')
    directory=args.output_dir.resolve()
    if directory.parent!=RUNS.resolve() or not directory.name.startswith('v3_'):
        ap.error('--output-dir must be VALIDATION/runs/v3_<new-run-name>')
    source=json.loads((PARENT/'result.json').read_text());parent_progress=json.loads((PARENT/'progress.json').read_text())
    config=copy.deepcopy(source['config'])
    config.update(label='rome-binary-and-three-class-calibration',seeds=[42] if args.smoke else [42,142,242],
        sample=.01 if args.smoke else 1.,smoke=args.smoke,threads=args.threads,prediction_threads=args.prediction_threads,
        classifier_rounds=5 if args.smoke else 300,calibration_settings=recipe.SETTINGS,
        candidates=recipe.COLUMNS,bootstrap_reps=2000,bootstrap_seed=0,block_days=1,
        prediction_rule='off',candidate_specs={name:dict(column=column,
            rome_hard_rule=name=='binary_cal_hard',day_component=name in ('three','three_uncal'),
            calibrated=name in ('binary','three','binary_cal_hard')) for name,column in recipe.COLUMNS.items()})
    jobs=[j for j in shared.schedule() if j['scheme']=='v3']
    if args.smoke:
        jobs=copy.deepcopy(jobs[:1]);jobs[0]['evals']=jobs[0]['evals'][:1]
    config['jobs']=jobs
    config=json.loads(json.dumps(config))
    if args.dry_run:
        print(json.dumps(dict(new_boosters=3 if args.smoke else 126,ordinary_fits=0,full_year_fits=0,uploads=0,config=config),indent=2));return
    fingerprints=derived.parent_inputs(source,PARENT)
    # The reused outer classifier and feature implementation must remain exact.
    for name,h in source['signature'].items():
        if sha256(ROOT/name)!=h:raise ValueError(f'Parent feature source changed: {name}')
    for info in parent_progress['models'].values():
        for name,h in info['artifacts'].items():
            if sha256(PARENT/name)!=h:raise ValueError(f'Parent model changed: {name}')
            fingerprints[name]=h
    signature={name:sha256(ROOT/name) for name in SOURCE_FILES}
    identity=data_identity([*glob.glob(TRAIN_GLOB),RANKING,parent_run.weather.METAR])
    if identity!=source['data_identity']:raise ValueError('Input data differs from the matched control')
    runtime=runtime_identity();runtime['scipy']=scipy.__version__
    if args.resume:
        record=json.loads((directory/'result.json').read_text());progress=json.loads((directory/'progress.json').read_text())
        if (record['signature']!=signature or record['config']!=config or record['parent_fingerprints']!=fingerprints
            or record['data_identity']!=identity or record['runtime_identity']['python']!=runtime['python']
            or record['runtime_identity']['versions']!=runtime['versions']
            or record['runtime_identity']['scipy']!=runtime['scipy']):raise ValueError('Resume source/config/data/runtime differs')
        for section in ['models','inner','calibrators','predictions','comparisons']:
            for info in progress[section].values():derived.verify_artifacts(directory,info['artifacts'])
        if record['status'] in ('complete','execution_check_only'):
            print('Already complete:',directory);return
    else:
        directory.mkdir(parents=True,exist_ok=False)
        for sub in ['models','inner','calibrators','source']: (directory/sub).mkdir()
        for name in signature:
            dest=directory/'source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/name).read_bytes())
        record=copy.deepcopy(source)
        for key in ['report','replay_checks','models','parameters','input_coverage','training_audits','reconstruction','minutes']:record.pop(key,None)
        record.update(run_id=directory.name,label=config['label'],config=config,status='running',
            at=datetime.now(timezone.utc).isoformat(timespec='seconds'),signature=signature,
            data_identity=identity,runtime_identity=runtime,parent_fingerprints=fingerprints,
            baseline_id=source['run_id'],prediction_column=CONTROL,official_rmse=None,folds={},
            source=dict(files=signature),fits=dict(new_classifiers=0,reused_outer_binary=1 if args.smoke else 18,ordinary=0,full_year=0),uploads=0)
        progress=dict(models={},inner={},calibrators={},predictions={},folds={},windows={},comparisons={})
        checkpoint(directory,progress)
    started=time.monotonic();record['status']='running';record.pop('error',None);write_json(directory/'result.json',record)
    try:
        pending=any(f"{e['fold']}__{s}" not in progress['predictions'] for j in jobs for e in j['evals'] for s in config['seeds'])
        if pending:
            dep,arr=shared.load_inputs(config['sample'])
            for outer in jobs:
                if all(f"{e['fold']}__{s}" in progress['predictions'] for e in outer['evals'] for s in config['seeds']):continue
                inner_predictions(directory,progress,outer,dep,arr,config)
                calibrators(directory,progress,outer,config)
                outer_predictions(directory,progress,source,parent_progress,outer,dep,arr,config)
            del dep,arr;gc.collect()
        # Recover a fold marker if interruption followed its final prediction write.
        for job in jobs:
            for ev in job['evals']:
                fold=ev['fold']
                if all(f'{fold}__{s}' in progress['predictions'] for s in config['seeds']):
                    info=copy.deepcopy(source['folds'][fold])
                    info['predictions']={str(s):progress['predictions'][f'{fold}__{s}']['path'] for s in config['seeds']}
                    progress['folds'][fold]=info
        checkpoint(directory,progress)
        record.update(folds=progress['folds'],models=progress['models'],windows=progress['windows'],
                      calibrators=progress['calibrators'])
        record['fits']['new_classifiers']=len(progress['models'])
        if args.smoke:
            f=pd.read_parquet(directory/next(iter(progress['predictions'].values()))['path'])
            record['smoke_rmse']={c:float(np.sqrt(np.mean((f[c]-f.y)**2))) for c in recipe.COLUMNS.values()}
            record['status']='execution_check_only'
        else:
            if len(progress['models'])!=126 or len(progress['predictions'])!=48 or len(progress['folds'])!=16:
                raise ValueError('Incomplete frozen workload')
            write_json(directory/'result.json',record)
            for candidate,baseline,block in [('binary','control',1),('three','control',1),
                ('binary','binary_uncal',1),('three','binary',1),('three','three_uncal',1),
                ('binary_cal_hard','control',1),('binary_uncal','control',1),
                ('binary','control',3),('three','control',3)]:
                compare(record,directory,progress,candidate,baseline,block)
            record['candidate_scores']={name:json.loads((directory/progress['comparisons'][f'{name}_vs_control_block1']['path']).read_text())['report']['scores'] for name in ['binary','three']}
            record['comparisons']=progress['comparisons']
            record['report']=copy.deepcopy(source['report'])  # identical saved control, not a candidate score
            from rome_calibration_report import finish
            record['outputs']=finish(record,directory,ROOT)
            record['status']='complete'
    except BaseException as exc:
        record['status']='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed'
        record['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['minutes']=round(record.get('minutes',0)+(time.monotonic()-started)/60,3)
        write_json(directory/'result.json',record)
    if record['status']=='complete':append_log(record,directory.parent,replace_existing=True)
    print('Finished',directory,record.get('candidate_scores',record.get('smoke_rmse')),flush=True)


if __name__=='__main__':main()
