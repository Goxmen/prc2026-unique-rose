"""Recover one completed laptop control after prediction-directory save failure.

No completed updates repeat. Frozen v2 worker/model/preprocessor/log bytes remain
unchanged. Recompute predictions from the saved Jan direct weights, record the
unavailable full-precision training curve honestly, then resume remaining fits.
"""
import gc
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

import nn_clock_bypass_portable as worker

MONTH = '2025-01'
ARM = 'direct'
FAILED_LOG = worker.ROOT / 'tmp/research_20261010/model_diversity/clock_bypass_laptop_v2_fit.log'
EXPECTED = {
    'model': '4d233871b1821cfb6126d5ee106e196b4db65102df621c5f17548f863fc17821',
    'preprocessing': '9dcf8c4d959d846995a5a56625624a8863ad3ad8ae0a5b5a2170b9b3e777db07',
    'failed_log': 'ed31b70a6f7092f8695b56205969755064efc5d53d72d2cff856893f65226482',
}


def recover():
    manifest = worker.guard()
    destination, model_path = worker.paths(MONTH, ARM)
    pre = worker.OUT / 'preprocessing' / f'{MONTH}.pkl'
    for name, path in [('model', model_path), ('preprocessing', pre), ('failed_log', FAILED_LOG)]:
        if worker.sha256(path) != EXPECTED[name]:
            raise ValueError('Frozen failed-stage evidence changed: ' + name)
    if destination.with_suffix('.json').exists():
        worker.completed(MONTH, ARM)
        return
    if destination.exists():
        raise ValueError('Unrecorded prediction requires inspection')
    log = FAILED_LOG.read_text()
    epochs = re.findall(r'epoch (\d+)/21: train ([0-9.]+); monitored ([0-9.]+)', log)
    if [int(row[0]) for row in epochs] != list(range(1, 22)) or "Cannot save file into a non-existent directory" not in log:
        raise ValueError('Expected exactly21 completed updates and the known save failure')
    cfg = worker.original.shared.recipe()
    source = worker.ROOT / manifest['matrix_records'][MONTH]['path']
    if worker.sha256(source) != manifest['matrix_records'][MONTH]['sha256']:
        raise ValueError('Exact input matrix changed')
    data = pd.read_pickle(source)
    meta, _ = worker.metadata.load()
    names = data['baseline']
    tm, vm = meta.loc[data['train_ids']], meta.loc[data['ids']]
    if (len(names) != 156 or list(data['X'][names]) != list(data['V'][names])
            or set(tm.month.astype(str)) != set(worker.original.shared.ts.FOLDS[MONTH])
            or set(vm.month.astype(str)) != {MONTH}
            or np.intersect1d(data['train_ids'], data['ids']).size):
        raise ValueError('Original schema or fold membership changed')
    np.testing.assert_array_equal(data['y'], tm.taxi_sec)
    for frame, rows in [(data['X'], tm), (data['V'], vm)]:
        np.testing.assert_array_equal(frame.airport.astype(str), rows.airport.astype(str))
        np.testing.assert_array_equal(frame.clock_raw_aobt3_missing.astype(bool), rows.raw_aobt3_missing)
    item = dict(X=data['X'][names], Vn=data['V'][names], y=data['y'], raw=vm.reset_index(),
                tmonth=tm.month.astype(str).to_numpy(), meta=dict(month=MONTH))
    ids = np.asarray(data['ids']).copy()
    with tempfile.TemporaryDirectory(prefix='recovery_encoding_', dir=worker.OUT) as scratch:
        d = worker.ClockData(item, 'cuda', scratch)
    worker.original.preprocess_guard.retain_saved_preprocessing(pre, d.preprocessing())
    del data, meta, tm, vm, item
    gc.collect()
    initial = worker.checks.self_check(d.cards, d.bins, d.train[2].shape[1], 'cuda')
    saved = torch.load(model_path, map_location='cuda', weights_only=False)
    if (saved['config'] != cfg or saved['design'] != worker.DESIGN
            or saved['preprocessing'] != str(pre.relative_to(worker.ROOT))):
        raise ValueError('Saved completed model recipe changed')
    model = worker.DirectControl(d.cards, d.bins, d.train[2].shape[1], d.mu, d.sd).cuda()
    model.load_state_dict(saved['state_dict'])
    prediction = worker.original.shared.predict(model, d.held, d.sd, d.mu, d.val_offset,
                                                  chunk=worker.PREDICTION_CHUNK)
    rmse = worker.original.shared.nni.rmse(d.yv, prediction)
    if abs(rmse-float(epochs[-1][2])) > .000051:
        raise ValueError('Reloaded model does not reproduce final logged monitored RMSE')
    destination.parent.mkdir(parents=True, exist_ok=True)
    worker.original.shared.nni.save_network(destination, ids, prediction)
    if worker.sha256(model_path) != EXPECTED['model'] or worker.sha256(pre) != EXPECTED['preprocessing']:
        raise ValueError('Recovery must preserve original weights and preprocessing bytes')
    worker.write_json(destination.with_suffix('.json'), dict(manifest=worker.artifact(worker.MANIFEST),
        design=worker.DESIGN, arm=ARM, month=MONTH, seed=42, stop_epoch=worker.original.STOP[MONTH],
        config=cfg, input_matrix=manifest['matrix_records'][MONTH], preprocessing=worker.artifact(pre),
        prediction=worker.artifact(destination), model=worker.artifact(model_path), initial_check=initial,
        curve=None, minutes=None, host=worker.platform.node(), gpu=torch.cuda.get_device_name(),
        torch=str(torch.__version__), recovery=dict(source=worker.artifact(Path(__file__)),
            failed_log=worker.artifact(FAILED_LOG), completed_training_epochs=21, repeated_training_updates=0,
            full_precision_curve='unavailable after prediction-save failure; retained rounded original log',
            logged_last_rmse=float(epochs[-1][2]), reloaded_rmse=rmse)))
    worker.completed(MONTH, ARM)
    worker.original.shared.log('Recovered January direct predictions from completed saved weights; zero repeated updates')
    del d, model, saved
    gc.collect(); torch.cuda.empty_cache()


if __name__ == '__main__':
    torch.set_num_threads(2)
    with threadpool_limits(2):
        recover()
        worker.fit()
