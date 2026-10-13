"""Audit failed exact-fidelity recovery without repeating fitting or inference."""
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_ple_pilot'
TEACHER = ROOT / 'VALIDATION/runs/v3_20261003_ideas_submit/work/parts/nn_v46.parquet'


def read(path):
    return json.loads(path.read_text())


def art(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            h.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=h.hexdigest())


def verify(a):
    path = ROOT / a['path']
    assert art(path) == {k: a[k] for k in ['path', 'sha256']}, path
    return path


def main():
    start = time.monotonic()
    target = OUT / 'independent_audit.json'
    assert not target.exists()
    identity, prepared, trained, result = [read(OUT / f'{x}.json') for x in ['identity', 'prepared', 'trained', 'result']]
    for a in identity['sources']:
        verify(a)
    for a in prepared['arrays']:
        verify(a)
    verify(prepared['preprocessing'])
    expected_identity = art(OUT / 'identity.json')
    assert all(x['identity'] == expected_identity for x in [prepared, trained, result])
    assert result['trained'] == art(OUT / 'trained.json')
    design = identity['design']
    assert design['seed'] == trained['seed'] == 42 and design['stop'] == trained['epochs'] == 17
    assert design['checkpoints'] == [14, 15, 17] and design['schedule_epochs'] == 24 and design['batch'] == 4096
    assert design['features'] == len(prepared['columns']) == 134 and prepared['rows'] == 2051327
    assert prepared['old_rows'] == result['rows'] == 344841
    assert prepared['controls']['original_and_chunked_encoder_tensors_bins_exact']
    assert prepared['controls']['original_and_observed_training_outputs_weights_losses_RNG_exact']
    assert prepared['independent_SQL_raw_membership_labels_and_exclusions_exact']
    assert trained['saved_before_diagnostics'] and not trained['held_labels_used']
    assert [x['epoch'] for x in trained['curve']] == list(range(1, 18))
    assert all(np.isfinite(x['train_mse_std']) and x['train_mse_std'] >= 0 and x['seconds'] > 0 for x in trained['curve'])
    started = read(OUT / 'fit_started.json')
    assert started['identity'] == expected_identity and started['preflight'] == art(OUT / 'prepared.json')
    assert started['automatic_restart_forbidden'] and not started['inner_fit']
    saved = read(OUT / 'checkpoints.json')
    assert saved['identity'] == expected_identity and saved['states'] == trained['checkpoints']
    assert set(trained['checkpoints']) == set(trained['predictions']) == set(result['reloads']) == {'14', '15', '17'}
    ids = np.load(OUT / 'encoding/old_ids.npy')
    template_ids = pd.read_parquet(ROOT / 'data/submitting.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
    assert len(ids) == len(np.unique(ids)) == 344841 and set(ids) == set(template_ids)
    teacher = pd.read_parquet(TEACHER, columns=['mvt_id', 'n42_0'])
    assert teacher.mvt_id.is_unique and len(teacher) == len(ids) and set(teacher.mvt_id) == set(ids)
    y = teacher.set_index('mvt_id').loc[ids, 'n42_0'].to_numpy()
    assert y.dtype == np.float32 and np.isfinite(y).all()
    assert art(TEACHER) in identity['sources']
    import nn_improve_jan_jul as original
    assert art(Path(original.__file__)) in identity['sources']
    torch.set_num_threads(1)
    states = []
    for epoch in [14, 15, 17]:
        record = trained['checkpoints'][str(epoch)]
        ckpt = torch.load(verify(record), map_location='cpu', weights_only=False)
        assert ckpt['seed'] == 42 and ckpt['epoch'] == epoch and ckpt['schedule_epochs'] == 24
        assert ckpt['preprocessing'] == prepared['preprocessing'] and ckpt['original_source'] == art(Path(original.__file__))
        assert ckpt['saved_before_monitor'] and not ckpt['resume_training_supported']
        assert ckpt['mu'] == identity['mu'] and ckpt['sd'] == identity['sd']
        expected = original.PleNet(ckpt['cards'], ckpt['bins'], ckpt['n_plain']).state_dict()
        assert set(ckpt['state_dict']) == set(expected)
        for key, value in ckpt['state_dict'].items():
            assert value.shape == expected[key].shape and value.dtype == expected[key].dtype
            assert torch.isfinite(value).all(), key
        frame = pd.read_parquet(verify(trained['predictions'][str(epoch)]))
        np.testing.assert_array_equal(frame.mvt_id, ids)
        assert np.isfinite(frame.pred).all() and frame.pred.ge(0).all()
        assert result['reloads'][str(epoch)] == dict(all_old_float_predictions_exact=True, rows=344841)
        states.append(dict(epoch=epoch, checkpoint=record, state_tensors=len(expected),
            scalar_entries=sum(x.numel() for x in ckpt['state_dict'].values()),
            every_state_finite_and_original_architecture_exact=True,
            full_old_reload='Verified immutable original GPU execution receipt; not repeated.'))
        del expected, ckpt
        if epoch == 14:
            pred = frame.pred.to_numpy().astype(np.float32)
    delta = pred.astype(float)-y.astype(float)
    metrics = dict(original_n42_0_exact=bool(np.array_equal(pred, y)),
        changed_float32_rows=int(np.count_nonzero(delta)), rms_difference=float(np.sqrt(np.mean(delta**2))),
        maximum_absolute_difference=float(np.max(abs(delta))),
        changed_single_member_rounded_rows=int(np.count_nonzero(np.rint(pred.astype(float))-np.rint(y.astype(float)))))
    for name, value in metrics.items():
        assert value == result[name], (name, value, result[name])
    assert result['state'] == 'closed_nonexact_pilot' and not metrics['original_n42_0_exact']
    assert result['acceptance_tolerance_fixed'] == 0 and all(result[k] for k in ['no_expansion', 'no_final_inference', 'no_upload'])
    receipt = dict(status='pass', source=art(Path(__file__)), identity=expected_identity,
        result=art(OUT / 'result.json'), prepared=art(OUT / 'prepared.json'), trained=art(OUT / 'trained.json'),
        teacher=art(TEACHER), rows=len(ids), metrics=metrics, states=states,
        original_sources_preprocessing_arrays_and_training_curve_authenticated=True,
        all_old_IDs_teacher_alignment_and_fidelity_arithmetic_exact=True,
        no_repeated_model_inference_or_fit=True, selection='Failed zero-tolerance fidelity gate; preserve and close without expansion.',
        scope='Finite full checkpoint state/architecture and receipt integrity; original GPU reload claims are retained execution evidence, not independent re-inference.',
        seconds=time.monotonic()-start)
    target.write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass', metrics=metrics, seconds=receipt['seconds'])))


if __name__ == '__main__':
    main()
