"""Portable fixed Rome mixture-coefficient screen, not probability calibration.

Only training no-ID Rome rows with 1h<schedule gap<4h receive continuous
coefficients (y-other)/(gap-other). Other rows retain the original binary
target. The original late-mean component determines other. Coefficients are
not clipped for training; predictions are clipped to [0,1] inside the same
input-only gate. Matched binary RMSE and original Logloss controls separate
the target and objective changes. No production, search or upload.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'VALIDATION/runs/v3_20261011_rome_continuous_mixture'
MONTHS = ['2025-01', '2025-07']
ARMS = ['logloss', 'binary_rmse', 'continuous_rmse']
DESIGN = dict(parent='public Cat3/R20 in selected public stack',
    change='continuous coefficient target only for original training no-ID Rome 1h<gap<4h rows',
    target='(y-min(late_mean,max(gap-3600,60)))/(gap-min(late_mean,max(gap-3600,60)))',
    training_clip=False, prediction_clip=[0, 1], inference_gate='same no-ID Rome 1h<gap<4h inputs',
    retained='all original80 inputs, original population, seed42,1182/1401 rounds, all explicit tree settings',
    arms=ARMS, controls='all three arms fitted on same laptop GPU',
    primary='replace one of three public Cat members only inside gate; unchanged half R20',
    comparisons='continuous versus matched logloss; binary RMSE separates objective from target',
    interpretation='bounded mixture coefficient, not a calibrated schedule-copy probability',
    no_cost_weights=True, no_stopping_search=True, no_production=True, no_upload=True,
    development='extensively reused January/July development data')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(2**20), b''): h.update(part)
    return h.hexdigest()


def art(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha(path))


def read(path): return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n'); temp.replace(path)


def verify(record):
    path = ROOT / record['path']
    assert sha(path) == record['sha256'], path
    return path


def gate(rows):
    return (rows.airport.eq('LIRF') & ~rows.has_id & rows.gap.gt(3600) & rows.gap.lt(14400)).to_numpy()


def coefficient(rows, binary, tau):
    target = np.asarray(binary, float).copy(); use = gate(rows)
    gap = rows.gap.to_numpy(float)[use]
    other = np.minimum(tau, np.maximum(gap-3600, 60))
    assert np.all(gap-other >= 3540)
    target[use] = (rows.y.to_numpy(float)[use]-other)/(gap-other)
    np.testing.assert_allclose(other + target[use]*(gap-other), rows.y.to_numpy(float)[use], atol=1e-9, rtol=0)
    assert np.isfinite(target).all()
    return target, use


def prepare():
    import rome_public_delay_cat as parent
    assert not (OUT/'identity.json').exists(), 'Preserve prepared experiment'
    raw_path = parent.r20.features.base.OUT/'rows2025.parquet'
    raw = pd.read_parquet(raw_path).set_index('mvt_id')
    folds = {}
    for month in MONTHS:
        source = parent.r20.OUT/'inputs'/f'{month}.pkl'
        d = pd.read_pickle(source)
        rec_path = parent.OUT/'predictions'/f'{month}_s42.json'
        rec = read(rec_path)
        original, old = parent.parent.validated_parent(month)
        for key in ['target', 'train_ids', 'ids', 'use']:
            np.testing.assert_array_equal(d[key], old[key])
        for key in ['A', 'B']:
            pd.testing.assert_frame_equal(d[key][original['columns']], old[key][original['columns']])
        assert list(d['A']) == list(d['B']) == rec['columns'] and len(rec['columns']) == 80
        train = raw.loc[d['train_ids']]
        allowed = parent.r20.features.base.FOLDS[month]
        assert set(train.month) == set(allowed) and month not in allowed
        assert train.airport.eq('LIRF').all() and not np.intersect1d(d['train_ids'], d['ids']).size
        np.testing.assert_array_equal(d['target'], ((train.gap-train.y).abs() <= 60).astype(int))
        meta_path = ROOT/'VALIDATION/runs/v3_20261002_adsblol_jan_jul/splits'/month/'meta.json'
        tau = read(meta_path)['parameters']['late_mean']
        target, use = coefficient(train, d['target'], tau)
        packet = dict(A=d['A'], B=d['B'], binary=np.asarray(d['target'], float), continuous=target,
            train_ids=d['train_ids'], ids=d['ids'], use=d['use'], held_gate=gate(raw.loc[d['ids']]),
            train_gate=use, train_metadata=train[['airport','has_id','gap','y','month']].reset_index(),
            categoricals=rec['categoricals'], params=rec['params'], rounds=rec['rounds'], tau=tau)
        path = OUT/'inputs'/f'{month}.pkl'; path.parent.mkdir(parents=True, exist_ok=True)
        pd.to_pickle(packet, path)
        folds[month] = dict(packet=art(path), original=art(source), original_model_record=art(rec_path),
            meta=art(meta_path), training_rows=len(train), changed_targets=int(use.sum()),
            inference_rows=int(packet['held_gate'].sum()), tau=tau,
            coefficient_quantiles=np.quantile(target[use], [0, .01, .5, .99, 1]).tolist(),
            training_months=sorted(allowed), columns=rec['columns'])
        del packet,d,old; gc.collect()
    write(OUT/'identity.json', dict(design=DESIGN, source=art(__file__), raw=art(raw_path), folds=folds,
        documentation='https://catboost.ai/docs/en/concepts/loss-functions-regression'))
    audit_inputs()


def guard():
    identity = read(OUT/'identity.json')
    assert identity['design'] == DESIGN and identity['source'] == art(__file__)
    for fold in identity['folds'].values(): verify(fold['packet'])
    return identity


def audit_inputs():
    import duckdb
    identity = guard(); checks = []
    with duckdb.connect(config={'threads': 1}) as con:
        con.read_parquet(str(verify(identity['raw']))).create_view('r')
        for month, fold in identity['folds'].items():
            d = pd.read_pickle(verify(fold['packet']))
            f = d['train_metadata'].assign(binary=d['binary'], continuous=d['continuous'], selected=d['train_gate'])
            con.register('f', f)
            n,u,errors,delta = con.execute('''select count(*),count(distinct f.mvt_id),
                sum(case when f.y is distinct from r.y or f.gap is distinct from r.gap
                    or f.has_id is distinct from r.has_id or f.month is distinct from r.month
                    or f.binary != cast(abs(r.gap-r.y)<=60 as int)
                    or f.selected != (r.airport='LIRF' and not r.has_id and r.gap>3600 and r.gap<14400)
                    then 1 else 0 end),
                max(abs(f.continuous - case when f.selected then
                    (r.y-least(?,greatest(r.gap-3600,60)))/(r.gap-least(?,greatest(r.gap-3600,60)))
                    else f.binary end)) from f join r using(mvt_id)''', [d['tau'],d['tau']]).fetchone()
            assert n == u == fold['training_rows'] and errors == 0 and delta < 1e-12
            h = pd.DataFrame(dict(mvt_id=d['ids'], selected=d['held_gate']))
            con.register('h', h)
            wrong = con.execute('''select count(*) from h join r using(mvt_id) where h.selected !=
                (r.airport='LIRF' and not r.has_id and r.gap>3600 and r.gap<14400)''').fetchone()[0]
            assert wrong == 0
            checks.append(dict(month=month, training_rows=n, target_max_error=delta,
                selected_train=int(d['train_gate'].sum()), selected_held=int(d['held_gate'].sum())))
    write(OUT/'input_audit.json', dict(status='pass', identity=art(OUT/'identity.json'),
        all_original_inputs_and_membership_exact=True, independent_sql_targets_and_gates=True, checks=checks))
    print(checks, flush=True)


def fit():
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    import torch
    identity = guard()
    assert read(OUT/'input_audit.json')['identity'] == art(OUT/'identity.json')
    assert read(OUT/'input_audit.json')['status'] == 'pass'
    for month, fold in identity['folds'].items():
        d = pd.read_pickle(verify(fold['packet']))
        frames = []
        for part in ['A', 'B']:
            frame = d[part].copy()
            for name in d['categoricals']:
                frame[name] = frame[name].astype(object).where(frame[name].notna(), '__missing__').astype(str)
            frames.append(frame)
        X,V = frames; inference = Pool(V, cat_features=d['categoricals'])
        for arm in ARMS:
            mp=OUT/'models'/f'{month}_{arm}.cbm'; rp=OUT/'models'/f'{month}_{arm}.json'
            pp=OUT/'predictions'/f'{month}_{arm}.parquet'; pr=pp.with_suffix('.json')
            params = dict(d['params'])
            if arm != 'logloss': params.update(loss_function='RMSE', eval_metric='RMSE')
            cls = CatBoostClassifier if arm == 'logloss' else CatBoostRegressor
            if rp.exists():
                receipt=read(rp)
                assert receipt['identity']==art(OUT/'identity.json') and receipt['params']==params
                assert receipt['host']==socket.gethostname() and receipt['gpu']==torch.cuda.get_device_name()
                model=cls(); model.load_model(str(verify(receipt['model'])))
            else:
                assert not mp.exists(), 'Unrecorded completed model; inspect before any retry'
                target=d['continuous'] if arm=='continuous_rmse' else d['binary']
                model=cls(iterations=d['rounds'], **params); started=time.monotonic()
                print(f'{month} {arm}: {len(X)} rows, {d["rounds"]} rounds', flush=True)
                model.fit(Pool(X, target, cat_features=d['categoricals']))
                mp.parent.mkdir(parents=True,exist_ok=True); temp=mp.with_suffix('.cbm.tmp')
                model.save_model(str(temp)); temp.replace(mp)
                receipt=dict(identity=art(OUT/'identity.json'), model=art(mp), arm=arm, month=month,
                    params=params, rounds=d['rounds'], host=socket.gethostname(), gpu=torch.cuda.get_device_name(),
                    minutes=(time.monotonic()-started)/60)
                write(rp, receipt)
            if pr.exists():
                previous=read(pr); assert previous['model_record']==art(rp); verify(previous['prediction'])
                continue
            assert not pp.exists(), 'Unrecorded prediction; inspect rather than overwrite'
            prediction=model.predict_proba(inference)[:,1] if arm=='logloss' else model.predict(inference)
            restored=cls(); restored.load_model(str(verify(receipt['model'])))
            replay=restored.predict_proba(inference)[:,1] if arm=='logloss' else restored.predict(inference)
            np.testing.assert_array_equal(prediction,replay)
            p=np.zeros(len(d['ids'])); p[d['use']]=np.clip(prediction,0,1)
            assert np.isfinite(p).all()
            pp.parent.mkdir(parents=True,exist_ok=True)
            pd.DataFrame(dict(mvt_id=d['ids'],p=p)).to_parquet(pp,index=False)
            write(pr,dict(model_record=art(rp), prediction=art(pp), all_rome_reload_exact=True,
                unclipped_min=float(prediction.min()), unclipped_max=float(prediction.max()),
                clipped=int(((prediction<0)|(prediction>1)).sum())))
            del model,restored; gc.collect()
        del d,X,V,frames; gc.collect()
    write(OUT/'fit_complete.json',dict(status='complete',identity=art(OUT/'identity.json'),host=socket.gethostname(),
        records=[art(OUT/'predictions'/f'{m}_{a}.json') for m in MONTHS for a in ARMS]))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','audit_inputs','fit'])
    args=p.parse_args()
    with threadpool_limits(1): globals()[args.stage]()
