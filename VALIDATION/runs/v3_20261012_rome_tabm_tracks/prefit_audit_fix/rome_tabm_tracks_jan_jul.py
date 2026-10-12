"""Fixed matched Rome TabM classifiers: tabular versus learned physical tracks.

The original R20 rows and copy labels are retained. Both arms exclude the five
departure-label history columns and receive identical nine track-quality fields.
Only the physical arm can learn from the raw sequence. No outer-label stopping.
"""
import argparse
import gc
import hashlib
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import rome_neural_jan_jul as original
import rome_tabm_tracks_model as model
import nn_sparse_track_jan_jul as tracks
from records import write_json

ROOT=original.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_rome_tabm_tracks'
MANIFEST=OUT/'identity.json'
PILOT=ROOT/'VALIDATION/runs/v3_20261012_rome_tabm_track_pilot'
art,read,verify=original.art,original.read,original.verify
ARMS=['tabular','physical']
DESIGN=dict(parent='exact v78 and separately coefficient-one transport onto v79',
    change='one-third TabM Rome copy probability, with matched tabular and learned-physical arms',
    primary='physical versus exact selected parent; tabular diagnoses family versus track benefit',
    model=model.SETTINGS,rows='original R20 Rome membership and binary abs(gap-y)<=60 target',
    inputs='75 history-free original columns and identical9 track-quality inputs',
    inner='last allowed training month; encoder/bins fit on remaining training months only',
    stopping='minimum inner-month probability logloss; first epoch on a tie; full refit at selected epoch',
    probability='(2/3)*(.5*publicCat3+.5*originalR20)+(1/3)*new; unchanged outside Rome',
    arms=ARMS,seed=42,machine='desktop RTX3080',
    fixed_tracks='original full2025 sparse physical cache; no source rebuild',
    evaluation='reused Jan/Jul development; one fixed seed, no independent confirmation',
    no_blend_search=True,no_architecture_search=True,no_label_weighting=True,no_upload=True)


def guard():
    assert read(PILOT/'result.json')['enough_observed_examples']
    import daily_main_refinements_cv as parent
    import joint_public_moments_cv as moment
    files=[Path(__file__),Path(model.__file__),Path(original.__file__),Path(model.encoding.__file__),
        Path(model.encoding.original.__file__),Path(model.physical.__file__),Path(model.physical.parent.__file__),
        Path(tracks.__file__),Path(tracks.tensor_seq.__file__),Path(tracks.seq.__file__),
        ROOT/'VALIDATION/rome_tabm_tracks_stack.py',ROOT/'VALIDATION/rome_tabm_tracks_v79_transport.py',
        ROOT/'VALIDATION/flight_id_clock_lgb.py',ROOT/'VALIDATION/target205_delta_transport.py',
        PILOT/'identity.json',PILOT/'result.json',tracks.seq.OUT/'full.json',
        tracks.seq.OUT/'full_index.parquet',tracks.seq.OUT/'full_points.bin',
        original.cat.r20.features.base.OUT/'rows2025.parquet']
    files += [original.cat.r20.OUT/'inputs'/f'{m}.pkl' for m in original.cat.parent.ROUNDS]
    # Freeze the already selected complete correction stack before any fit.
    for directory in [parent.OUT,moment.OUT]:
        assert read(directory/'audit.json')['status']=='pass'
        files.extend([directory/'identity.json',directory/'audit.json',directory/'scored.parquet'])
        files.extend(verify(a) for a in read(directory/'identity.json').get('sources',[]))
    value=dict(design=DESIGN,sources=[art(p) for p in dict.fromkeys(files)])
    OUT.mkdir(parents=True,exist_ok=True)
    if MANIFEST.exists():assert read(MANIFEST)==value
    else:
        assert not list(OUT.glob('**/*.pt')) and not list(OUT.glob('**/*predictions.parquet'))
        write_json(MANIFEST,value)
    return value


def bank():
    manifest=read(tracks.seq.OUT/'full.json')
    for name in ['full_index.parquet','full_points.bin']:
        assert art(tracks.seq.OUT/name)['sha256']==manifest['outputs'][name]
    index=pd.read_parquet(tracks.seq.OUT/'full_index.parquet')
    points=np.memmap(tracks.seq.OUT/'full_points.bin',mode='r',dtype=np.int32).reshape(-1,6)
    return tracks.tensor_seq.RaggedTracks(index,points,'cuda')


def preflight():
    guard();assert not (OUT/'preflight.json').exists()
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080'
    checks=[model.checks('cpu'),model.checks('cuda')]
    b=bank();folds=[]
    # Separate Pandas reconstruction of the source-only feasibility gate.
    raw=pd.read_parquet(original.cat.r20.features.base.OUT/'rows2025.parquet')
    allowed=['2025-03','2025-09','2025-10','2025-11']
    r=raw.loc[raw.airport.eq('LIRF') & raw.month.isin(allowed)].set_index('mvt_id')
    i=b.index.set_index('mvt_id').loc[r.index]
    saved=read(PILOT/'result.json')
    for rec in saved['records']:
        use=(r.gap-r.y).abs().le(60).eq(rec['copy'])
        if rec['month']!='ALL':use &= r.month.eq(rec['month'])
        assert int(use.sum())==rec['rows']
        assert int((use & i['count'].ge(2)).sum())==rec['usable']
        assert int((use & i.raw_ground_count.gt(0)).sum())==rec['ground']
        assert int((use & ~r.has_id).sum())==rec['no_id']
        assert int((use & ~r.has_id & r.gap.between(3600,14400)).sum())==rec['no_id_long']
    for month in original.cat.parent.ROUNDS:
        d=original.inputs(month);last=max(d['months']);hold=d['months']==last
        a,brows=b.align(d['train_ids']),b.align(d['ids'][d['use']])
        assert len(d['A'])==len(a) and len(d['B'])==len(brows)
        np.testing.assert_array_equal(b.index.mvt_id.to_numpy()[a],d['train_ids'])
        np.testing.assert_array_equal(b.index.mvt_id.to_numpy()[brows],d['ids'][d['use']])
        for ids,rows in [(d['train_ids'],a),(d['ids'][d['use']],brows)]:
            sample=np.random.default_rng(42).choice(len(ids),min(80,len(ids)),replace=False)
            expected=tracks.seq.encoded_batch(b.index.iloc[rows[sample]],b.points)
            actual=b.batch(rows[sample])
            for got,want in zip(actual,expected):
                np.testing.assert_allclose(got.cpu().numpy(),want,atol=2e-7,rtol=2e-6)
        folds.append(dict(month=month,train=len(a),held=len(brows),inner_train=int((~hold).sum()),
            inner_held=int(hold.sum()),inner_month=last,train_months=sorted(set(d['months'])),
            original_rows_and_labels=True,columns=list(d['A']),scalar_track_checks=160,
            usable_copy=int(((d['target']==1)&(b.counts[a]>=2)).sum()),
            usable_noncopy=int(((d['target']==0)&(b.counts[a]>=2)).sum())))
    write_json(OUT/'preflight.json',dict(status='pass',identity=art(MANIFEST),checks=checks,folds=folds,
        feasibility_counts_independently_reproduced=True))
    print('Rome population/labels, inner calendar,320 raw-track transforms and matched-model checks pass',flush=True)


def paths(month,arm,phase):
    stem=OUT/f'{month}_{arm}_{phase}'
    return stem.with_suffix('.pt'),stem.with_suffix('.pkl'),stem.with_suffix('.json')


def trained(month,arm,phase):
    mp,pp,rp=paths(month,arm,phase)
    if not rp.exists():
        assert not mp.exists() and not pp.exists(),'Inspect unrecorded checkpoint; never silently refit'
        return None
    rec=read(rp)
    assert rec['identity']==art(MANIFEST) and rec['month']==month and rec['arm']==arm and rec['phase']==phase
    for key in ['model','preprocessing','training_ids']:verify(rec[key])
    return rec


def retain(month,arm,phase,network,state,curve,ids,epochs,months):
    mp,pp,rp=paths(month,arm,phase)
    assert not any(p.exists() for p in [mp,pp,rp])
    with pp.with_suffix('.pkl.tmp').open('wb') as f:pickle.dump(state,f,protocol=5)
    pp.with_suffix('.pkl.tmp').replace(pp)
    torch.save(dict(state_dict=network.state_dict(),design=DESIGN,arm=arm),mp.with_suffix('.pt.tmp'))
    mp.with_suffix('.pt.tmp').replace(mp)
    ip=OUT/f'{month}_{arm}_{phase}_ids.parquet'
    pd.DataFrame(dict(mvt_id=ids)).to_parquet(ip,index=False)
    rec=dict(identity=art(MANIFEST),month=month,arm=arm,phase=phase,model=art(mp),preprocessing=art(pp),
        training_ids=art(ip),rows=len(ids),epochs=epochs,months=sorted(set(months)),curve=curve,
        seed=42,device=torch.cuda.get_device_name(),columns=state['columns'],saved_before_held_inference=True)
    write_json(rp,rec)
    return rec


def load_model(rec):
    with verify(rec['preprocessing']).open('rb') as f:state=pickle.load(f)
    saved=torch.load(verify(rec['model']),map_location='cuda',weights_only=False)
    assert saved['design']==DESIGN and saved['arm']==rec['arm']
    network=model.Classifier(state,rec['arm']).to('cuda')
    network.load_state_dict(saved['state_dict'],strict=True)
    return network,state


def fit():
    guard();pre=read(OUT/'preflight.json')
    assert pre['status']=='pass' and pre['identity']==art(MANIFEST)
    assert torch.cuda.get_device_name()=='NVIDIA GeForce RTX 3080'
    b=bank()
    for month in original.cat.parent.ROUNDS:
        d=original.inputs(month);last=max(d['months']);hold=d['months']==last
        a,brows=b.align(d['train_ids']),b.align(d['ids'][d['use']])
        for arm in ARMS:
            inner=trained(month,arm,'inner')
            if inner is None:
                state=model.encoding.preprocessing(d['A'].iloc[~hold])
                network,curve=model.fit(state,d['A'].iloc[~hold],d['target'][~hold],b,a[~hold],arm,
                    model.SETTINGS['max_epochs'],monitor=(d['A'].iloc[hold],d['target'][hold],a[hold]))
                inner=retain(month,arm,'inner',network,state,curve,d['train_ids'][~hold],
                    model.SETTINGS['max_epochs'],d['months'][~hold])
                del network,state;gc.collect();torch.cuda.empty_cache()
            assert last not in inner['months']
            selected=min(inner['curve'],key=lambda p:p['inner_logloss'])['epoch']
            full=trained(month,arm,'full');network=None
            if full is None:
                state=model.encoding.preprocessing(d['A'])
                network,curve=model.fit(state,d['A'],d['target'],b,a,arm,selected)
                full=retain(month,arm,'full',network,state,curve,d['train_ids'],selected,d['months'])
            assert full['epochs']==selected
            path=OUT/f'{month}_{arm}_predictions.parquet';rp=path.with_suffix('.json')
            if rp.exists():
                rec=read(rp)
                assert rec['full_model']==art(paths(month,arm,'full')[2]) and rec['prediction']==art(path)
            else:
                assert not path.exists(),'Inspect unrecorded predictions'
                recovered=network is None
                if recovered:network,state=load_model(full)
                pred=model.probabilities(network,model.encoding.encode(state,d['B'],'cuda'),b,brows)
                restored,rs=load_model(full)
                check=model.probabilities(restored,model.encoding.encode(rs,d['B'],'cuda'),b,brows)
                np.testing.assert_array_equal(pred,check)
                p=np.zeros(len(d['ids']));p[d['use']]=pred
                pd.DataFrame(dict(mvt_id=d['ids'],p=p)).to_parquet(path,index=False)
                write_json(rp,dict(full_model=art(paths(month,arm,'full')[2]),prediction=art(path),
                    rows=len(p),rome_rows=len(pred),all_rome_reload_exact=True,
                    recovered_saved_checkpoint=recovered,selected_epoch=selected,selection_month=last,
                    inner_model=art(paths(month,arm,'inner')[2]),probability_sha256=hashlib.sha256(pred.tobytes()).hexdigest()))
                del restored,rs
            del network;gc.collect();torch.cuda.empty_cache()
        del d;gc.collect();torch.cuda.empty_cache()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['guard','preflight','fit'])
    torch.set_num_threads(2)
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
