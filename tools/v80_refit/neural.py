"""Full-2025 refit of missing original v80 neural components, with fixed recipes.

This is a new fit, not recovery of historical GPU arithmetic. Original selected
epochs and ensemble arithmetic remain fixed. No validation, tuning or uploading.
Every epoch has a resumable latest state; selected checkpoints are immutable.
"""
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import nn_improve_jan_jul as nni
import nn_jan_jul as nnj

RUNS = ROOT / 'VALIDATION/runs'
IDEAS = RUNS / 'v3_20261003_ideas_submit'
FINAL = RUNS / 'v3_20261008_final_submit'
INPUTS = IDEAS / 'work/inputs.pkl'
DEFAULT_OUT = RUNS / 'v3_20261013_v80_refit_final'
EXPECTED = {'old': 344841, 'final': 670790, 'train': 2051327}
VERSIONS = [47, 48, 49, 50, 51, 53]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def artifact(path):
    p = Path(path).resolve()
    return {'path': str(p.relative_to(ROOT)), 'sha256': sha(p), 'bytes': p.stat().st_size}


def verify(a):
    path = ROOT / a['path']
    if sha(path) != a['sha256']:
        raise ValueError(f"Changed artifact: {a['path']}")
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def save_torch(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def restore_original_configuration():
    receipt = RUNS / 'v3_20261003_nn_improve_submit/submission_v48/submission_v48.json'
    recipe = read(receipt)['identity']['recipe']
    # Historical HPO helpers modify these module globals. Always restore original.
    nni.NEW = copy.deepcopy(recipe['new'])
    nnj.RECIPE = copy.deepcopy(recipe['v47'])
    if nni.NEW['batch'] != 4096 or nni.NEW['epochs'] != 24 or nnj.RECIPE['epochs'] != 8:
        raise ValueError('Original recipe contract changed')
    return recipe


def trajectories():
    result = [dict(family='plain134', group=42, k=0, seed=42, stop=8, checkpoints=[8])]
    for family in ['ple134', 'combo156']:
        for group in [42, 142, 242]:
            stop = 19 if family == 'combo156' and group == 142 else 17
            epochs = [14, 15, 17] if family == 'ple134' and group == 42 else [stop]
            for k in range(5):
                result.append(dict(family=family, group=group, k=k, seed=group + 1000*k,
                                   stop=stop, checkpoints=epochs))
    return result


def required_paths():
    paths = [INPUTS, FINAL / 'work/inputs_final.pkl', FINAL / 'inputs_final.json',
             ROOT / 'data/submitting.parquet', ROOT / 'data/final_submitting.parquet',
             RUNS / 'v3_20261012_final_missing_ple_plan/plan.json',
             RUNS / 'v3_20261003_nn_improve_submit/submission_v48/submission_v48.json',
             IDEAS / 'work/parts/nn_combo.json',
             RUNS / 'v3_20261005_rome_hpo_submit/submission_v53/submission_v53.json',
             FINAL / 'work/parts/rome.parquet', FINAL / 'work/parts/rome.json']
    model_dir = IDEAS / 'work/models'
    paths += sorted(model_dir.glob('*.txt')) + [model_dir / 'catboost.cbm']
    folder = RUNS / 'v3_20261002_adsblol_submit/submission_v46'
    progress = read(folder / 'progress.json')
    paths.append(folder / 'progress.json')
    for name in ['refit2025__ordinary_adsblol_stand__42', 'refit2025__classifier__42']:
        if name not in progress['models']:
            # Read-only discovery handles historical ordinary key spelling.
            matches = [k for k in progress['models'] if ('ordinary' if 'ordinary' in name else 'classifier') in k]
            if len(matches) != 1:
                raise ValueError('Ambiguous v46 model provenance')
            name = matches[0]
        paths.append(folder / progress['models'][name]['path'])
    for name in ['lgb_combo', 'lgb_combo_airport', 'catboost']:
        paths += [IDEAS / f'work/parts/{name}.parquet', IDEAS / f'work/parts/{name}.json']
    for version in VERSIONS:
        paths.append(ROOT / f'submissions/unique-rose_v{version}.parquet')
    for month in ['2026-01','2026-07']:
        paths.append(RUNS / f'v3_20261005_rome_hpo_submit/submission_v53/ranking_{month}.parquet')
    return list(dict.fromkeys(paths))


def plan():
    paths = required_paths()
    return dict(stage='planned', neural_fits=31, tuning_fits=0, trajectories=trajectories(),
                expected_rows=EXPECTED, required_files=[str(p.relative_to(ROOT)) for p in paths],
                missing=[str(p.relative_to(ROOT)) for p in paths if not p.is_file()],
                bytes=sum(p.stat().st_size for p in paths if p.is_file()),
                retained_pilot_exclusion=('The desktop seed42 recovery pilot remains untouched; all 31 missing '
                    'production trajectories use one consistent execution host and full optimizer/RNG checkpoints.'),
                versions=VERSIONS, no_upload=True)


def encode(encoder, frame, directory, name, chunk=65536):
    """Original transform in independent row chunks, atomic output arrays."""
    directory.mkdir(parents=True, exist_ok=True)
    cp, np_ = directory / f'{name}_cat.npy', directory / f'{name}_num.npy'
    if cp.exists() and np_.exists():
        return np.load(cp, mmap_mode='r'), np.load(np_, mmap_mode='r')
    arrays = None
    for start in range(0, len(frame), chunk):
        cat, num = encoder.transform(frame.iloc[start:start+chunk])
        if arrays is None:
            arrays = [np.lib.format.open_memmap(p.with_suffix('.tmp'), mode='w+', dtype=v.dtype,
                                                shape=(len(frame), v.shape[1]))
                      for p, v in [(cp, cat), (np_, num)]]
        arrays[0][start:start+len(cat)] = cat
        arrays[1][start:start+len(num)] = num
    for a, p in zip(arrays, [cp, np_]):
        a.flush()
        a._mmap.close()
        p.with_suffix('.tmp').replace(p)
    return np.load(cp, mmap_mode='r'), np.load(np_, mmap_mode='r')


def layout(encoder, numeric):
    gauss, plain, offset = [], [], 0
    for name in encoder.nums:
        gauss.append(offset)
        plain.append(offset + 1)
        offset += 2
        if name in encoder.flags:
            plain.append(offset)
            offset += 1
    if offset != numeric.shape[1]:
        raise ValueError('Encoder layout changed')
    bins, ple = [], []
    for col in gauss:
        edges = np.unique(np.quantile(numeric[:, col], np.linspace(0, 1, nni.NEW['bins']+1)).astype(np.float32))
        if len(edges) >= 3:
            ple.append(col)
            bins.append(torch.from_numpy(edges))
        else:
            plain.append(col)
    return bins, ple, plain


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    recipe = restore_original_configuration()
    identity_path = out / 'identity.json'
    if identity_path.exists():
        identity = read(identity_path)
        for item in identity['inputs'] + identity['sources']:
            verify(item)
        if recipe != identity['recipe']:
            raise ValueError('Neural configuration changed')
    else:
        schedule = plan()
        if schedule['missing']:
            raise FileNotFoundError(str(schedule['missing']))
        identity = dict(recipe=recipe, trajectories=trajectories(), fresh_refit=True,
                        inputs=[artifact(p) for p in required_paths()],
                        sources=[artifact(Path(__file__)), artifact(Path(nni.__file__)), artifact(Path(nnj.__file__))],
                        host=os.uname().nodename, original_weights_not_recoverable=True,
                        prior_pilot_reused=False, prior_pilot_reason=schedule['retained_pilot_exclusion'])
        original = read(RUNS / 'v3_20261012_final_missing_ple_plan/plan.json')
        if sha(INPUTS) != original['input_source']['sha256']:
            raise ValueError('Original training cache changed')
        if sha(FINAL / 'work/inputs_final.pkl') != read(FINAL / 'inputs_final.json')['sha256']:
            raise ValueError('Audited final feature cache changed')
        combo = read(IDEAS / 'work/parts/nn_combo.json')['groups']
        if {k:v['stop_epoch'] for k,v in combo.items()} != {'42':17, '142':19, '242':17}:
            raise ValueError('Original combo stops changed')
        write(identity_path, identity)
    marker = out / 'prepared.json'
    if marker.exists():
        d = read(marker)
        for a in d['artifacts']:
            verify(a)
        return
    data = pd.read_pickle(INPUTS)
    if len(data['Xn']) != EXPECTED['train'] or len(data['names']) != 134:
        raise ValueError('Training rows/feature count changed')
    if sorted(set(data['tmonth'])) != [f'2025-{m:02d}' for m in range(1,13)]:
        raise ValueError('Expected all 2025 training months')
    clipped = np.minimum(data['yn'], nnj.RECIPE['label_clip'])
    mu, sd = float(clipped.mean()), float(clipped.std())
    if not np.isfinite(clipped).all() or sd <= 0:
        raise ValueError('Invalid training labels')
    np.save(out / 'target.npy', ((clipped-mu)/sd).astype(np.float32))
    names = list(data['names'])
    from ideas_submit import COMBO
    for family, cols in [('base134', names), ('combo156', names+COMBO)]:
        folder = out / 'encoding' / family
        prep = folder / 'preprocessing.pkl'
        if prep.exists():
            pre = pickle.loads(prep.read_bytes())
            if pre['columns'] != cols:
                raise ValueError('Encoded feature schema changed')
        else:
            encoder = nnj.Encoder(data['Xn'][cols])
            cat, num = encode(encoder, data['Xn'][cols], folder, 'train')
            bins, ple, plain = layout(encoder, num)
            pre = dict(encoder=encoder, bins=bins, ple=ple, plain=plain, cards=encoder.cards,
                       mu=mu, sd=sd, columns=cols, num_width=num.shape[1])
            cat._mmap.close(); num._mmap.close()
            temp = prep.with_suffix('.tmp')
            temp.write_bytes(pickle.dumps(pre, protocol=pickle.HIGHEST_PROTOCOL)); temp.replace(prep)
        cat, num = encode(pre['encoder'], data['Vn'][cols], folder, 'old')
        cat._mmap.close(); num._mmap.close()
    np.save(out / 'old_ids.npy', data['rank_ids'])
    if len(data['rank_ids']) != EXPECTED['old']:
        raise ValueError('Old input rows changed')
    del data
    gc.collect()
    final = pd.read_pickle(FINAL / 'work/inputs_final.pkl')
    if len(final['rank_ids']) != EXPECTED['final']:
        raise ValueError('Final input rows changed')
    np.save(out / 'final_ids.npy', final['rank_ids'])
    for family in ['base134', 'combo156']:
        folder = out / 'encoding' / family
        pre = pickle.loads((folder / 'preprocessing.pkl').read_bytes())
        cat, num = encode(pre['encoder'], final['Vn'][pre['columns']], folder, 'final')
        cat._mmap.close(); num._mmap.close()
    del final
    gc.collect()
    files = sorted((out/'encoding').rglob('*.npy')) + sorted((out/'encoding').rglob('*.pkl')) + sorted(out.glob('*.npy'))
    write(marker, dict(identity=artifact(identity_path), artifacts=[artifact(p) for p in files],
                       target_mean=mu, target_sd=sd, training_rows=EXPECTED['train']))


def to_device(array, columns, device, chunk=65536):
    width = array.shape[1] if columns is None else len(columns)
    result = torch.empty((len(array), width), dtype=torch.int64 if array.dtype == np.int64 else torch.float32,
                         device=device)
    for start in range(0, len(array), chunk):
        part = array[start:start+chunk] if columns is None else array[start:start+chunk, columns]
        result[start:start+len(part)].copy_(torch.from_numpy(np.array(part, copy=True)))
    return result


def model_for(family, pre, device):
    if family == 'plain134':
        return nnj.Net(pre['cards'], pre['num_width']).to(device)
    return nni.PleNet(pre['cards'], pre['bins'], len(pre['plain'])).to(device)


def tensors(folder, view, pre, family, device):
    cat = np.load(folder / f'{view}_cat.npy', mmap_mode='r')
    num = np.load(folder / f'{view}_num.npy', mmap_mode='r')
    result = (to_device(cat, None, device), to_device(num, None, device)) if family == 'plain134' else (
        to_device(cat, None, device), to_device(num, pre['ple'], device), to_device(num, pre['plain'], device))
    cat._mmap.close(); num._mmap.close()
    return result


def snapshot(model, optimizer, scheduler, gen, epoch, curve):
    return dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                generator=gen.get_state(), torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                curve=curve)


def train_one(spec, pre, data, target, directory, device='cuda', stop_override=None):
    """Original training updates/schedule; only checkpointing and monitors differ."""
    family, seed, stop = spec['family'], spec['seed'], spec['stop']
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = model_for(family, pre, device)
    cfg = nnj.RECIPE if family == 'plain134' else nni.NEW['ple']
    batch = nnj.RECIPE['batch'] if family == 'plain134' else nni.NEW['batch']
    schedule_epochs = nnj.RECIPE['epochs'] if family == 'plain134' else nni.NEW['epochs']
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['lr'], total_steps=schedule_epochs*math.ceil(len(target)/batch))
    gen = torch.Generator(device=device).manual_seed(seed)
    directory.mkdir(parents=True, exist_ok=True)
    latest, first, curve = directory / 'latest.pt', 0, []
    state = torch.load(latest, map_location=device, weights_only=False) if latest.exists() else None
    # A kill between the immutable checkpoint and latest-state rename must not
    # repeat an epoch that is already saved in full.
    selected = sorted(directory.glob('epoch*.pt'), key=lambda p:int(p.stem[5:]))
    if selected and (state is None or int(selected[-1].stem[5:]) > state['epoch']):
        state = torch.load(selected[-1], map_location=device, weights_only=False)
    if state is not None:
        if state['spec'] != spec:
            raise ValueError('Checkpoint recipe changed')
        model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler'])
        gen.set_state(state['generator'].cpu()); torch.set_rng_state(state['torch_rng'].cpu())
        np.random.set_state(state['numpy_rng']); random.setstate(state['python_rng'])
        if state['cuda_rng']:
            torch.cuda.set_rng_state_all([r.cpu() for r in state['cuda_rng']])
        first, curve = state['epoch'], state['curve']
        del state
    run_stop = stop if stop_override is None else stop_override
    for epoch in range(first, run_stop):
        started, total = time.monotonic(), 0.0
        perm = torch.randperm(len(target), device=device, generator=gen)
        for i in range(0, len(target), batch):
            b = perm[i:i+batch]
            pred = model(*(a[b] for a in data))
            expected = target[b] if family == 'plain134' else target[b, None].expand_as(pred)
            loss = torch.nn.functional.mse_loss(pred, expected)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            total += float(loss.detach()) * len(b)
        curve.append(dict(epoch=epoch+1, train_mse_std=total/len(target), seconds=time.monotonic()-started))
        if not np.isfinite(curve[-1]['train_mse_std']):
            raise ValueError('Nonfinite training loss')
        state = snapshot(model, opt, sched, gen, epoch+1, curve)
        state['spec'] = spec
        # Save selected full restart states before latest; a crash cannot lose a completed checkpoint.
        if epoch+1 in spec['checkpoints']:
            save_torch(directory / f'epoch{epoch+1}.pt', state)
        save_torch(latest, state)
        print(f"{family} seed {seed} epoch {epoch+1}/{stop}, {curve[-1]['seconds']:.1f}s", flush=True)
    return model, curve


def infer(model, data, pre, chunk=65536):
    model.eval()
    with torch.no_grad():
        values = []
        for start in range(0, len(data[0]), chunk):
            pred = model(*(a[start:start+chunk] for a in data))
            if pred.ndim == 2:
                pred = pred.mean(1)
            values.append(pred.cpu().numpy())
    model.train()
    return np.maximum(np.concatenate(values).astype(float)*pre['sd'] + pre['mu'], 0)


def fit_and_predict(out):
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for the full-year neural refit')
    torch.backends.cuda.matmul.allow_tf32 = False
    for family in ['plain134', 'ple134', 'combo156']:
        folder = out / 'encoding' / ('combo156' if family == 'combo156' else 'base134')
        pre = pickle.loads((folder/'preprocessing.pkl').read_bytes())
        specs = [s for s in trajectories() if s['family'] == family]
        unfinished = [s for s in specs if not (out/'models'/f'{family}_s{s["seed"]}'/'complete.json').exists()]
        if unfinished:
            train = tensors(folder, 'train', pre, family, 'cuda')
            target = torch.from_numpy(np.load(out/'target.npy')).cuda()
            for spec in unfinished:
                directory = out/'models'/f'{family}_s{spec["seed"]}'
                model, curve = train_one(spec, pre, train, target, directory)
                states = [artifact(directory/f'epoch{e}.pt') for e in spec['checkpoints']]
                write(directory/'complete.json', dict(spec=spec, checkpoints=states, curve=curve,
                      prepared_sha256=sha(out/'prepared.json'), host=os.uname().nodename,
                      torch=str(torch.__version__), device=torch.cuda.get_device_name()))
                del model
                gc.collect(); torch.cuda.empty_cache()
            del train, target
            gc.collect(); torch.cuda.empty_cache()
        for view in ['old', 'final']:
            ids = np.load(out/f'{view}_ids.npy')
            held = None
            for spec in specs:
                directory = out/'models'/f'{family}_s{spec["seed"]}'
                record = read(directory/'complete.json')
                if record['prepared_sha256'] != sha(out/'prepared.json') or record['spec'] != spec:
                    raise ValueError('Completed model input identity changed')
                for state_art, epoch in zip(record['checkpoints'], spec['checkpoints']):
                    state_path = verify(state_art)
                    prediction_path = out/'members'/f'{view}_{family}_s{spec["seed"]}_e{epoch}.parquet'
                    receipt = prediction_path.with_suffix('.json')
                    if receipt.exists():
                        rec = read(receipt)
                        verify(rec['prediction'])
                        if rec['state'] != state_art or not rec['reload_exact']:
                            raise ValueError('Prediction/model identity changed')
                        continue
                    if held is None:
                        held = tensors(folder, view, pre, family, 'cuda')
                    state = torch.load(state_path, map_location='cpu', weights_only=False)
                    model = model_for(family, pre, 'cuda'); model.load_state_dict(state['model'])
                    pred = infer(model, held, pre)
                    del model
                    # Fresh instance proves exported checkpoint reproduces every stored value.
                    model = model_for(family, pre, 'cuda'); model.load_state_dict(state['model'])
                    np.testing.assert_array_equal(pred, infer(model, held, pre))
                    if len(pred) != EXPECTED[view] or not np.isfinite(pred).all() or pd.Index(ids).has_duplicates:
                        raise ValueError('Invalid inference output')
                    prediction_path.parent.mkdir(parents=True, exist_ok=True)
                    temp = prediction_path.with_suffix('.tmp')
                    pd.DataFrame(dict(mvt_id=ids, pred=pred)).to_parquet(temp, index=False); temp.replace(prediction_path)
                    write(receipt, dict(state=state_art, prediction=artifact(prediction_path), rows=len(ids), reload_exact=True))
                    del model, state, pred
                    gc.collect(); torch.cuda.empty_cache()
            del held
            gc.collect(); torch.cuda.empty_cache()


def pools(out, view):
    ids = np.load(out/f'{view}_ids.npy')
    def member(family, seed, epoch, cast=False):
        p = out/'members'/f'{view}_{family}_s{seed}_e{epoch}.parquet'
        receipt = read(p.with_suffix('.json')); verify(receipt['prediction'])
        f = pd.read_parquet(p); np.testing.assert_array_equal(f.mvt_id, ids)
        pred = f.pred.to_numpy(float)
        return pred.astype(np.float32).astype(float) if cast else pred
    frame = pd.DataFrame(dict(mvt_id=ids, nn_plain47=member('plain134',42,8)))
    for version, epoch in [(48,15),(49,17)]:
        frame[f'nn_ple{version}'] = np.mean([member('ple134',42+1000*k,epoch) for k in range(5)],axis=0)
    # Original ideas_submit.part_values averages a row-major float64 matrix.
    frame['nn_ple50'] = np.column_stack([member('ple134',g+1000*k,14 if g==42 else 17,True)
                                        for g in [42,142,242] for k in range(5)]).mean(axis=1)
    frame['nn_combo51'] = np.column_stack([member('combo156',g+1000*k,19 if g==142 else 17,True)
                                         for g in [42,142,242] for k in range(5)]).mean(axis=1)
    return frame


def tree_parts(data, view, threads):
    """Unchanged saved original v46 and pre-HPO combo models, never HPO trees."""
    import lightgbm as lgb
    import modeling
    import adsblol_submit as recipe
    import catboost_jan_jul as cbj
    from catboost import CatBoostRegressor, Pool
    from ideas_submit import COMBO
    recipe.configure(46)
    folder = recipe.OUT / 'submission_v46'
    progress = read(folder/'progress.json')
    models = {}
    for part, key in recipe.MODEL_KEYS.items():
        record = progress['models'][key]; path = folder/record['path']
        if sha(path) != record['artifacts'][record['path']]:
            raise ValueError('Retained v46 model changed')
        models[part] = lgb.Booster(model_file=str(path))
    ordinary, probability = [], []
    for d in data['months'].values():
        ordinary.append(recipe.signals.ordinary_prediction(models['ordinary'],d['V'],recipe.KIND,threads,recipe.ORDINARY))
        probability.append(recipe.signals.substitution_probability(models['classifier'],d['V'],d['raw'],recipe.KIND,
                                                                   threads,recipe.cv.CLASSIFIER))
    modeldir = IDEAS/'work/models'
    V = data['Vn'][data['names']+COMBO]
    booster = lgb.Booster(model_file=str(modeldir/'lgb_combo.txt'))
    predictions = [modeling.predict(booster,V,threads=threads).astype(np.float32).astype(float)]
    airport = V.airport.astype(str).to_numpy(); ap_pred = np.full(len(V),np.nan)
    for ap in sorted(set(airport)):
        booster = lgb.Booster(model_file=str(modeldir/f'lgb_combo_airport_{ap}.txt'))
        mask = airport==ap
        ap_pred[mask] = modeling.predict(booster,V.loc[mask],threads=threads)
    predictions.append(ap_pred.astype(np.float32).astype(float))
    cats = recipe.signals._categorical(data['names'])
    cat = CatBoostRegressor(); cat.load_model(str(modeldir/'catboost.cbm'))
    predictions.append(np.maximum(cat.predict(Pool(cbj.frame(data['Vn'][data['names']],cats),cat_features=cats),
                                              thread_count=threads),0).astype(np.float32).astype(float))
    if not all(np.isfinite(p).all() for p in predictions):
        raise ValueError('Nonfinite retained tree predictions')
    # Original model reload must still reproduce retained old bases after serialization.
    if view == 'old':
        for name, p in zip(['lgb_combo','lgb_combo_airport','catboost'],predictions):
            saved = pd.read_parquet(IDEAS/f'work/parts/{name}.parquet').set_index('mvt_id').loc[data['rank_ids'],'pred'].to_numpy(float)
            np.testing.assert_allclose(p,saved,rtol=0,atol=0.001)
    return np.concatenate(ordinary), np.concatenate(probability), np.mean(predictions,axis=0)


def finish(out, threads):
    import nn_improve_submit as nis
    import pyarrow.parquet as pq
    from submit import fill_template
    predictions, diagnostics = {}, {}
    for view in ['old','final']:
        destination = out/'components'/f'{view}.parquet'
        frame = pools(out,view)
        data = pd.read_pickle(INPUTS if view=='old' else FINAL/'work/inputs_final.pkl')
        np.testing.assert_array_equal(frame.mvt_id,data['rank_ids'])
        ordinary, probability, combo = tree_parts(data,view,threads)
        if view=='old':
            probabilities=[]
            for m,item in data['months'].items():
                saved=pd.read_parquet(RUNS/f'v3_20261005_rome_hpo_submit/submission_v53/ranking_{m}.parquet')
                np.testing.assert_array_equal(saved.mvt_id,item['raw'].mvt_id)
                probabilities.append(saved.p.to_numpy(float))
            new_p=np.concatenate(probabilities)
        else:
            rp = pd.read_parquet(FINAL/'work/parts/rome.parquet').set_index('mvt_id')
            new_p = rp.loc[data['rank_ids'],'p'].to_numpy(float)
        template = pq.read_table(ROOT/'data'/('submitting.parquet' if view=='old' else 'final_submitting.parquet'))
        values = {47:('nn_plain47',.3,ordinary),48:('nn_ple48',.6,ordinary),
                  49:('nn_ple49',.3,ordinary),50:('nn_ple50',.6,ordinary),
                  51:('nn_combo51',.6,combo),53:('nn_combo51',.6,combo)}
        previous51 = None
        hold = read(RUNS/'v3_20261005_rome_hpo_submit/submission_v53/submission_v53.json')['held_at_v51']
        for version,(col,weight,tree) in values.items():
            base = weight*frame[col].to_numpy(float)+(1-weight)*tree
            parts,offset = [],0
            for item in data['months'].values():
                end=offset+len(item['raw']); sl=slice(offset,end)
                f,_=nis.final(base[sl],(new_p if version==53 else probability)[sl],item['V'],item['raw'],data['parameters'],data['fitted'])
                np.testing.assert_array_equal(f.mvt_id,frame.mvt_id.iloc[sl])
                parts.append(f.pred.to_numpy(float)); offset=end
            pred=np.concatenate(parts)
            if version==51:
                previous51=pred.copy()
            if version==53:
                mask=frame.mvt_id.isin(hold).to_numpy(); pred[mask]=previous51[mask]
            if not np.isfinite(pred).all() or (pred<0).any():
                raise ValueError('Invalid finished neural component')
            table=fill_template(template,frame.mvt_id,pred)
            series=pd.Series(table['TAXITIME_SEC_mvt'].to_numpy(),index=table['MVT_ID_mvt'].to_numpy())
            frame[f'v{version}']=series.loc[frame.mvt_id].to_numpy()
            if view=='old':
                saved=pd.read_parquet(ROOT/f'submissions/unique-rose_v{version}.parquet').set_index('MVT_ID_mvt').loc[frame.mvt_id,'TAXITIME_SEC_mvt'].to_numpy(float)
                delta=frame[f'v{version}'].to_numpy(float)-saved
                diagnostics[str(version)]=dict(rms_difference=float(np.sqrt(np.mean(delta**2))),
                                              changed_rows=int(np.count_nonzero(delta)),max_abs=float(np.max(abs(delta))))
        destination.parent.mkdir(parents=True,exist_ok=True)
        temp=destination.with_suffix('.tmp'); frame.to_parquet(temp,index=False);temp.replace(destination)
        predictions[view]=artifact(destination)
        del data,frame
        gc.collect()
    models=[artifact(p) for p in sorted((out/'models').glob('*/complete.json'))]
    exports = (sorted((out/'models').glob('*/epoch*.pt')) + sorted((out/'models').glob('*/complete.json'))
               + sorted((out/'encoding').glob('*/preprocessing.pkl')) + sorted((out/'members').glob('*'))
               + [out/'identity.json', out/'prepared.json', out/'old_ids.npy', out/'final_ids.npy'])
    write(out/'result.json',dict(status='pass',predictions=predictions,versions=VERSIONS,
          identity=artifact(out/'identity.json'),prepared=artifact(out/'prepared.json'),models=models,
          artifacts=[artifact(p) for p in exports] + list(predictions.values()),
          fresh_refit_historical_difference_diagnostics=diagnostics,
          exact_historical_prediction_recovery_claimed=False,no_upload=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['plan','prepare','fit','finish','run'])
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if not 1 <= args.threads <= 16:
        parser.error('--threads must be between 1 and 16')
    if args.command=='plan':
        print(json.dumps(plan(),indent=2));return
    torch.set_num_threads(args.threads)
    out=args.out.resolve()/'neural'
    if args.command=='run' and (out/'result.json').exists():
        result=read(out/'result.json')
        if result['status']!='pass':
            raise ValueError('Existing result did not pass')
        identity=read(verify(result['identity']))
        for a in identity['sources'] + identity['inputs'] + result['artifacts']:
            verify(a)
        print('neural: completed artifacts verified; no fit or inference repeated',flush=True)
        return
    with threadpool_limits(args.threads):
        if args.command in ['prepare','run']:
            prepare(out)
        else:
            restore_original_configuration()
        if args.command in ['fit','run']:
            fit_and_predict(out)
        if args.command in ['finish','run']:
            finish(out,args.threads)


if __name__=='__main__':
    main()
