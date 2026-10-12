"""One frozen robust observation screen; inference and scoring are separate."""
import argparse
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_moments_cv as parent
import joint_public_other_runway as model

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_joint_public_other_runway'
DESIGN=dict(settings=model.SETTINGS,parent='exactv78',
    primary='exactv78 plus0.5*(new_joint-old_joint), then unchanged matched/recovered constraints and rounding',
    comparator='original departure-only message unchanged; additional taxi evidence is not subtracted from it',
    scopes='same original touched rows, source support and monthly quality; unchanged local/assigned calendars',
    calibration='all original fold-specific priors/noise/support calibration reused',
    robust='one global capped variance-matched t5 score weight; no per-cell alternatives',
    control_portability='recomputed floating old_joint versus saved GLAPTOP old_joint atol1e-8 seconds rtol0, record max; exact cached integer v78 control; delta against saved old_joint',
    secondary='separately frozen coefficient-one delta transport onto exactv79',
    caveats=['hybrid approximation, not exact Bayesian Student conditioning',
        'uncalibrated cross-runway source-error covariance','prior error may cause downweighting of valid evidence',
        'global q/k/weight can change when disconnected coordinates accompany informative coordinates',
        'reused Jan/Jul development evidence, not independent confirmation'],
    no_fit=True,no_weight_search=True,no_gate_search=True,no_production=True,no_upload=True)
art,read,write,verify=parent.art,parent.read,parent.write,parent.verify


def freeze():
    assert not (OUT/'identity.json').exists(),'Preserve frozen experiment'
    original=parent.guard()
    audit=read(parent.OUT/'audit.json')
    assert audit['status']=='pass' and audit['result']==art(parent.OUT/'result.json')
    inference=read(parent.OUT/'inference.json')
    assert inference['identity']==art(parent.OUT/'identity.json')
    for record in inference['artifacts']:verify(record)
    verify(read(parent.OUT/'result.json')['scored'])
    assert not (OUT/'predictions.parquet').exists()
    import daily_main_refinements_cv as selected
    correction_files=[Path(selected.__file__),Path(selected.daily.__file__),
        Path(selected.daily.constraints.__file__),Path(selected.daily.constraints.exclusion.__file__),
        Path(selected.daily.constraints.monthly.__file__),
        selected.daily.base.OUT/'rows2025.parquet',
        selected.daily.constraints.exclusion.OUT/'matches2025.parquet']
    files=[Path(__file__),Path(model.__file__),Path(parent.__file__),
        Path(model.base.__file__),Path(model.pilot.__file__),
        ROOT/'VALIDATION/other_runway_v79_transport.py',ROOT/'VALIDATION/target205_delta_transport.py',
        parent.OUT/'identity.json',parent.OUT/'inference.json',parent.OUT/'predictions.parquet',
        parent.OUT/'result.json',parent.OUT/'audit.json',parent.OUT/'scored.parquet',
        *correction_files,*[verify(a) for a in original['inputs']],*[verify(a) for a in original['helpers']]]
    checks=read(OUT/'checks.json')
    assert checks['source']==art(model.__file__) and checks['status']=='pass'
    files.append(OUT/'checks.json')
    identity=dict(design=DESIGN,source=art(__file__),model=art(model.__file__),
        sources=[art(p) for p in sorted(set(files))],parent_identity=art(parent.OUT/'identity.json'),
        input_columns=model.pilot.INPUTS,checks=checks,correction_sources=[art(p) for p in correction_files])
    OUT.mkdir(parents=True,exist_ok=True);write(OUT/'identity.json',identity)
    import other_runway_v79_transport as transport
    transport.freeze()
    write(OUT/'ready.json',dict(identity=art(OUT/'identity.json'),transport=art(transport.OUT/'design.json')))
    print(dict(frozen=True,identity=art(OUT/'identity.json')),flush=True)


def guard():
    identity=read(OUT/'identity.json')
    assert identity['design']==DESIGN and identity['source']==art(__file__) and identity['model']==art(model.__file__)
    for record in identity['sources']:verify(record)
    ready=read(OUT/'ready.json');assert ready['identity']==art(OUT/'identity.json');verify(ready['transport'])
    return identity


def infer():
    guard();assert not (OUT/'inference.json').exists() and not (OUT/'predictions.parquet').exists()
    started=time.monotonic()
    rows=pd.read_parquet(parent.OUT/'inputs.parquet',columns=model.pilot.INPUTS)
    ts=pd.read_parquet(parent.OUT/'taxi_source.parquet');ds=pd.read_parquet(parent.OUT/'delay_source.parquet')
    params=read(parent.OUT/'parameters.json')
    old=pd.read_parquet(parent.OUT/'predictions.parquet',columns=['mvt_id','joint','touched']).set_index('mvt_id')
    assert rows.mvt_id.is_unique and old.index.is_unique
    frames=[];cells=[];checks=[]
    for month,prm in params.items():
        source=rows[rows.month.eq(month)].reset_index(drop=True)
        result,summary=model.apply(source,ts,ds,prm['taxi'],prm['delay'],prm['support'])
        cached=old.loc[result.mvt_id]
        np.testing.assert_array_equal(result.touched,cached.touched)
        # Frozen before inference: cross-CPU floating replay only. The integer
        # v78 scoring control stays exact and the delta uses the saved posterior.
        difference=np.abs(result.old_joint.to_numpy()-cached.joint.to_numpy())
        np.testing.assert_allclose(result.old_joint,cached.joint,atol=1e-8,rtol=0)
        result['recomputed_old_joint']=result.old_joint
        result['old_joint']=cached.joint.to_numpy()
        result['new_joint']=np.where(result.extra_cells.gt(0),result.new_joint,result.old_joint)
        result['delta']=result.new_joint-result.old_joint
        assert not result.loc[result.extra_cells.eq(0),'delta'].any()
        frames.append(result.assign(month=month));cells.append(summary.assign(month=month))
        checks.append(dict(month=month,rows=len(result),touched=int(result.touched.sum()),
            extra_observation_rows=int(result.extra_cells.gt(0).sum()),
            old_joint_replay_max_absolute_sec=float(difference.max()),old_joint_replay_atol_sec=1e-8,
            delta_uses_saved_old_joint=True,no_extra_rows_saved_exact=True))
        print(checks[-1],flush=True)
    pd.concat(frames,ignore_index=True).to_parquet(OUT/'predictions.parquet',index=False)
    pd.concat(cells,ignore_index=True).to_parquet(OUT/'cells.parquet',index=False)
    write(OUT/'inference.json',dict(identity=art(OUT/'identity.json'),host=socket.gethostname(),
        seconds=time.monotonic()-started,checks=checks,no_label_or_score_values_loaded_or_used=True,
        provenance_scope='Guards hash frozen source/score artifacts; numerical inference loads only input allowlists, public sources, frozen calibration and cached posterior controls.',
        artifacts=[art(OUT/p) for p in ['predictions.parquet','cells.parquet']]))


def score():
    guard();receipt=read(OUT/'inference.json');assert receipt['identity']==art(OUT/'identity.json')
    for record in receipt['artifacts']:verify(record)
    assert not (OUT/'result.json').exists(),'Preserve completed comparison'
    import duckdb
    import daily_main_refinements_cv as selected
    import tuning_seasonal as validation
    old=pd.read_parquet(parent.OUT/'scored.parquet')
    f=old[['mvt_id','month','day','y']].copy();f['control']=old.candidate
    inference=pd.read_parquet(OUT/'predictions.parquet').set_index('mvt_id').reindex(f.mvt_id)
    f['delta']=inference.delta.fillna(0).to_numpy()
    f['active']=inference.extra_cells.fillna(0).gt(0).to_numpy()
    raw=pd.read_parquet(selected.daily.base.OUT/'rows2025.parquet').set_index('mvt_id').loc[f.mvt_id].reset_index()
    matches=pd.read_parquet(selected.daily.constraints.exclusion.OUT/'matches2025.parquet')
    constrain=selected.daily.constrain
    np.testing.assert_array_equal(constrain(raw,f.mvt_id.to_numpy(),f.month,f.control.to_numpy(),matches),f.control)
    f['candidate']=constrain(raw,f.mvt_id.to_numpy(),f.month,f.control.to_numpy()+.5*f.delta.to_numpy(),matches)
    np.testing.assert_array_equal(f.loc[~f.active,'candidate'],f.loc[~f.active,'control'])
    assert len(f)==344419 and f.mvt_id.is_unique
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ['control','candidate']} for m,g in f.groupby('month')}
    paired=validation.paired_bootstrap(f.day,f.month,(f.control-f.y)**2,(f.candidate-f.y)**2)
    f.to_parquet(OUT/'scored.parquet',index=False)
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),inference=art(OUT/'inference.json'),
        scored=art(OUT/'scored.parquet'),scores=scores,
        cv={c:float(np.mean([s[c] for s in scores.values()])) for c in ['control','candidate']},paired=paired,
        exact_v78_replayed=True,active=int(f.active.sum()),changed=int(f.candidate.ne(f.control).sum()),
        sources=[art(p) for p in [parent.OUT/'scored.parquet',selected.daily.base.OUT/'rows2025.parquet',
            selected.daily.constraints.exclusion.OUT/'matches2025.parquet',Path(selected.daily.__file__)]])
    write(OUT/'result.json',result)
    with duckdb.connect(config={'threads':1}) as con:
        for name,path in [('f',OUT/'scored.parquet'),('o',parent.OUT/'scored.parquet'),
            ('p',OUT/'predictions.parquet'),('raw',selected.daily.base.OUT/'rows2025.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''SELECT count(*),count(distinct f.mvt_id),sum(CASE WHEN
            f.y IS DISTINCT FROM raw.y OR f.month IS DISTINCT FROM raw.month OR f.day IS DISTINCT FROM raw.day OR
            f.control IS DISTINCT FROM o.candidate OR f.delta IS DISTINCT FROM coalesce(p.delta,0) OR
            (NOT f.active AND f.candidate IS DISTINCT FROM f.control) OR
            f.candidate<0 OR NOT isfinite(f.candidate) OR f.candidate<>round_even(f.candidate,0)
            THEN 1 ELSE 0 END) FROM f JOIN o USING(mvt_id) JOIN raw USING(mvt_id) LEFT JOIN p USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for c in ['control','candidate']:
            for m,value in con.execute(f'SELECT month,sqrt(avg(pow({c}-y,2))) FROM f GROUP BY month').fetchall():
                assert abs(value-scores[m][c])<1e-8
    write(OUT/'audit.json',dict(status='pass',rows=n,result=art(OUT/'result.json'),
        exact_v78_truth_calendar_scores_and_unchanged_rows=True,
        scope='Inherited corrections reuse original function; SQL independently verifies truth, identity and scores.'))
    print(dict(cv=result['cv'],paired=paired,active=result['active']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['checks','freeze','infer','score'])
    args=parser.parse_args()
    with threadpool_limits(1):
        if args.stage=='checks':
            path=OUT/'checks.json';assert not path.exists()
            write(path,dict(status='pass',source=art(model.__file__),checks=model.checks()))
        else:globals()[args.stage]()
