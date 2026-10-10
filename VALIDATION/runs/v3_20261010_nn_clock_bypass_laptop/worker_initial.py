"""Portable, matched-laptop four-fit screen of the frozen physical-clock gate.

Reuse exact fingerprinted156 matrices and compact audited month/label metadata.
Both direct N09 controls and bypass models use seed42 and fixed21/17 epochs.
The frozen update function is copied into an isolated namespace: only model
constructor and eval-only prediction chunk size differ. No original mutation.
"""
import argparse
import functools
import gc
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import types

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_clock_bypass_jan_jul as original
import chunked_direct_data as adapter
import missing_plan_view as metadata
import arrival_plan_jan_jul as inputs
from records import sha256, write_json

ROOT = original.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261010_nn_clock_bypass_laptop'
MANIFEST = OUT / 'portable.json'
PREDICTION_CHUNK = 16384
ARMS = ['direct', 'bypass']
DESIGN = dict(original=original.DESIGN, machine='same laptop for candidate and direct control',
    arms=ARMS, fixed_epochs=original.STOP, seed=42, encoder=adapter.SETTINGS,
    prediction_chunk=PREDICTION_CHUNK, update_function='unchanged frozen original.train bytecode',
    namespace_changes=['model constructor for direct control', 'eval-only prediction chunk16384'],
    data='exact fingerprinted original156 plus audited compact month/y/airport/missing metadata',
    no_full_year_fit_or_upload=True)


def artifact(path):
    path = Path(path)
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def read(path):
    return original.shared.ts.read(path)


class DirectControl(torch.nn.Module):
    def __init__(self, cards, bins, plain, mu, sd):
        super().__init__()
        self.tabular = original.shared.nni.PleNet(cards, bins, plain)

    def forward(self, cat, g, plain, clocks, masks):
        return self.tabular(cat, g, plain)


def training_function(arm):
    if arm not in ARMS:
        raise ValueError('Unknown screen arm')
    namespace = dict(original.train.__globals__)
    namespace['ClockBypass'] = original.ClockBypass if arm == 'bypass' else DirectControl
    namespace['shared'] = types.SimpleNamespace(**{
        **vars(original.shared),
        'predict': functools.partial(original.shared.predict, chunk=PREDICTION_CHUNK)})
    return types.FunctionType(original.train.__code__, namespace, 'portable_' + arm,
                              original.train.__defaults__, original.train.__closure__)


class ClockData(adapter.ChunkedDirectData):
    def __init__(self, item, device, scratch):
        super().__init__(item, device, scratch)
        for attr, frame in [('train', item['X']), ('held', item['Vn'])]:
            clocks, masks = original.physical_clocks(frame)
            setattr(self, attr, (*getattr(self, attr), torch.from_numpy(clocks).to(device),
                                torch.from_numpy(masks).to(device)))

    def preprocessing(self):
        value = super().preprocessing()
        value['design'] = DESIGN
        return value


def prepare():
    """Small desktop packets only; no matrices or raw sources are loaded."""
    names = [ROOT / f'tmp/research_20261010/{host}_input_fingerprints.json' for host in ['desktop', 'laptop']]
    if read(names[0]) != read(names[1]):
        raise ValueError('Desktop/laptop original input fingerprints differ')
    control = original.parent_contract.control_identity()
    matrix_records = {}
    for month in original.STOP:
        path = ROOT / 'VALIDATION/runs/v3_20261010_callsign_lgb_jan_jul/predictions' / f'{month}_s42.json'
        item = read(path)
        matrix_records[month] = dict(path=f'VALIDATION/runs/v3_20261010_arrival_plan_jan_jul/inputs/{month}.pkl',
            sha256=item['inputs_sha256'], audited_candidate_record=artifact(path))
    files = {Path(__file__), *names, metadata.DEFAULT / 'metadata.parquet', metadata.DEFAULT / 'manifest.json',
             ROOT / control['record_path'], ROOT / control['predictions_path']}
    files.update(ROOT / item['audited_candidate_record']['path'] for item in matrix_records.values())
    for module in list(sys.modules.values()):
        value = getattr(module, '__file__', None)
        if value:
            path = Path(value).resolve()
            if path.is_file() and path.is_relative_to(ROOT) and not path.is_relative_to(ROOT / '.venv'):
                files.add(path)
    files.add(ROOT / 'VALIDATION/tests/test_nn_clock_bypass_portable.py')
    files.add(ROOT / 'VALIDATION/nn_sparse_track_prototype.py')
    result = dict(design=DESIGN, original_identity=original.identity(), source=artifact(Path(__file__)),
        matrix_records=matrix_records, files=[artifact(path) for path in sorted(files)])
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists() and read(MANIFEST) != result:
        raise ValueError('Portable screen source/provenance changed')
    write_json(MANIFEST, result)
    (OUT / 'transfer_files.txt').write_text(''.join(item['path']+'\n' for item in result['files'])
                                          +str(MANIFEST.relative_to(ROOT))+'\n')
    return result


def guard():
    manifest = read(MANIFEST)
    if (manifest['design'] != DESIGN or manifest['source'] != artifact(Path(__file__))
            or manifest['original_identity'] != original.identity()):
        raise ValueError('Portable recipe, source or historical control changed')
    for item in manifest['files']:
        if artifact(ROOT / item['path']) != item:
            raise ValueError('Portable input/helper packet changed: ' + item['path'])
    return manifest


def paths(month, arm):
    return OUT / 'networks' / f'{month}_{arm}.parquet', OUT / 'models' / f'{month}_{arm}.pt'


def completed(month, arm):
    prediction, model = paths(month, arm)
    record = prediction.with_suffix('.json')
    if not record.exists():
        if prediction.exists() or model.exists():
            raise ValueError('Unrecorded fit artifact; do not repeat blindly')
        return None
    value = read(record)
    manifest = read(MANIFEST)
    config = original.shared.hpo.nn_config(original.shared.hpo.trial_spec('nn', 'N09')['overrides'])
    if (value['manifest'] != artifact(MANIFEST) or value['design'] != DESIGN
            or value['prediction'] != artifact(prediction) or value['model'] != artifact(model)
            or value['preprocessing'] != artifact(OUT / 'preprocessing' / f'{month}.pkl')
            or value['stop_epoch'] != original.STOP[month] or value['arm'] != arm
            or value['month'] != month or value['seed'] != 42 or value['config'] != config
            or value['input_matrix'] != manifest['matrix_records'][month]):
        raise ValueError('Completed portable screen changed')
    return value


def fit():
    manifest = guard()
    cfg = original.shared.recipe()
    meta, _ = metadata.load()
    for month in original.STOP:
        if all(completed(month, arm) is not None for arm in ARMS):
            continue
        source = ROOT / manifest['matrix_records'][month]['path']
        if sha256(source) != manifest['matrix_records'][month]['sha256']:
            raise ValueError('Exact audited laptop matrix changed')
        data = pd.read_pickle(source)
        names = data['baseline']
        tm, vm = meta.loc[data['train_ids']], meta.loc[data['ids']]
        if (len(names) != 156 or list(data['X'][names]) != list(data['V'][names])
                or set(tm.month.astype(str)) != set(original.shared.ts.FOLDS[month])
                or set(vm.month.astype(str)) != {month}
                or np.intersect1d(data['train_ids'], data['ids']).size):
            raise ValueError('Original schema or fold membership changed')
        np.testing.assert_array_equal(data['y'], tm.taxi_sec)
        for frame, rows in [(data['X'], tm), (data['V'], vm)]:
            np.testing.assert_array_equal(frame.airport.astype(str), rows.airport.astype(str))
            np.testing.assert_array_equal(frame.clock_raw_aobt3_missing.astype(bool), rows.raw_aobt3_missing)
        item = dict(X=data['X'][names], Vn=data['V'][names], y=data['y'], raw=vm.reset_index(),
                    tmonth=tm.month.astype(str).to_numpy(), meta=dict(month=month))
        ids = np.asarray(data['ids']).copy()
        with tempfile.TemporaryDirectory(prefix='encoding_', dir=OUT) as scratch:
            d = ClockData(item, 'cuda', scratch)
        pre = OUT / 'preprocessing' / f'{month}.pkl'
        if pre.exists():
            original.preprocess_guard.retain_saved_preprocessing(pre, d.preprocessing())
        else:
            original.shared.atomic_pickle(pre, d.preprocessing())
        del data, item, tm, vm
        gc.collect()
        initial = original.self_check(d.cards, d.bins, d.train[2].shape[1], 'cuda')
        for arm in ARMS:
            if completed(month, arm) is not None:
                continue
            started = time.monotonic()
            model, prediction, curve = training_function(arm)(d, original.STOP[month])
            dest, model_path = paths(month, arm)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = model_path.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), config=cfg, design=DESIGN,
                            preprocessing=str(pre.relative_to(ROOT))), temporary)
            temporary.replace(model_path)
            original.shared.nni.save_network(dest, ids, prediction)
            write_json(dest.with_suffix('.json'), dict(manifest=artifact(MANIFEST), design=DESIGN,
                arm=arm, month=month, seed=42, stop_epoch=original.STOP[month], config=cfg,
                input_matrix=manifest['matrix_records'][month], preprocessing=artifact(pre),
                prediction=artifact(dest), model=artifact(model_path), initial_check=initial,
                curve=curve, minutes=(time.monotonic()-started)/60, host=platform.node(),
                gpu=torch.cuda.get_device_name(), torch=str(torch.__version__)))
            original.shared.log(f'{month}: same-laptop {arm} saved')
            del model
            torch.cuda.empty_cache()
        del d
        gc.collect(); torch.cuda.empty_cache()


def score():
    guard()
    sc, rome, parent_pool, trees = original.shared.baseline.baseline()
    arms = ['v58', 'historical_direct', 'laptop_direct', 'bypass', 'half_bypass', 'half_laptop_direct']
    errors = {name: {} for name in arms}
    rows = []
    for month in original.STOP:
        pred = {}
        for arm in ARMS:
            if completed(month, arm) is None:
                raise ValueError('All four matched fits are required')
            frame = pd.read_parquet(paths(month, arm)[0])
            np.testing.assert_array_equal(frame.mvt_id, sc.d[month]['ids'])
            pred[arm] = frame.base.to_numpy(float)
        historical = original.shared.direct_member(sc, month, 42, 0)
        neural = dict(v58=parent_pool[month], historical_direct=historical,
            laptop_direct=pred['direct'], bypass=pred['bypass'],
            half_bypass=.5*(parent_pool[month]+pred['bypass']),
            half_laptop_direct=.5*(parent_pool[month]+pred['direct']))
        per_month = {name: [] for name in arms}
        for seed in original.shared.hpo.SEEDS:
            members = {original.shared.hpo.MEMBER_OF[key]: value[seed][month] for key, value in trees.items()}
            for name, value in neural.items():
                per_month[name].append(sc.se(month, seed, nn=value, p=rome[month], **members))
        for name in arms:
            errors[name][month] = np.mean(per_month[name], axis=0)
        rows.append(pd.DataFrame(dict(mvt_id=sc.d[month]['ids'], month=month, day=sc.d[month]['day'],
            **{name: errors[name][month] for name in arms})))
    frame = pd.concat(rows, ignore_index=True)
    scores = {name: sc.cv(value) for name, value in errors.items()}
    if abs(scores['v58']['cv']-295.7414415741871) > 1e-7:
        raise ValueError('Saved v58 control did not reproduce')
    comparisons = {}
    for a, b in [('bypass', 'laptop_direct'), ('half_bypass', 'half_laptop_direct'),
                 ('half_bypass', 'v58'), ('laptop_direct', 'historical_direct')]:
        comparisons[a+'_vs_'+b] = original.shared.ts.paired_bootstrap(frame.day, frame.month, frame[b], frame[a])
    frame.to_parquet(OUT / 'scored.parquet', index=False)
    write_json(OUT / 'score.json', dict(design=DESIGN, scores=scores, comparisons=comparisons,
        caveat='Four fits on same laptop; reused development months, not independent confirmation.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'check', 'fit', 'score'])
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(2):
        {'prepare': prepare, 'check': guard, 'fit': fit, 'score': score}[args.stage]()
