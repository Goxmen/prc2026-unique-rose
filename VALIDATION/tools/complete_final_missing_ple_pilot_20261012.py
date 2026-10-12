"""Wait for weather audit and explicit root allocation, then run one PLE pilot.

    .venv/bin/python VALIDATION/tools/complete_final_missing_ple_pilot_20261012.py

The frozen plan is mandatory. Before any preparation, completion/allocation.json
must name /root, desktop_exclusive=true, state=released_to_ple_pilot and the
plan SHA256. The weather independent audit must also pass with its frozen
source/producer identities. Neither condition alone starts work. Existing
execution receipts, encodings and fit markers prevent automatic restarts.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_ple_pilot'
OUT = RUN / 'completion'
WEATHER = ROOT / 'VALIDATION/runs/v3_20261012_final_weather_inference'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8388608), b''):
            digest.update(block)
    return digest.hexdigest()


def art(path, with_bytes=False):
    record = dict(path=str(path.relative_to(ROOT)), sha256=sha(path))
    if with_bytes:
        record['bytes'] = path.stat().st_size
    return record


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    temp.replace(path)


def verify(plan):
    for record in plan['sources']:
        assert art(ROOT / record['path']) == record, record['path']


def upstream():
    target = WEATHER / 'independent_audit.json'
    if not target.exists():
        return None
    audit = read(target)
    assert audit['status'] == 'pass', 'Weather audit did not pass; no PLE work'
    freeze = read(WEATHER / 'output_auditor_identity.json')
    assert audit['source'] == freeze['source']
    assert audit['identity'] == art(WEATHER / 'identity.json', True)
    assert audit['auditor_identity'] == art(WEATHER / 'output_auditor_identity.json', True)
    assert audit['prediction_rows_checked'] == 2031262
    assert len(audit['checks']) == 4 and audit['models'] == 28
    assert audit['no_fit'] and audit['no_inference']
    return art(target)


def allocation(plan_sha):
    path = OUT / 'allocation.json'
    if not path.exists():
        return None
    record = read(path)
    assert record['state'] == 'released_to_ple_pilot', 'No valid desktop allocation'
    assert record['plan_sha256'] == plan_sha and record['authority'] == '/root'
    assert record['desktop_exclusive'] is True
    return art(path)


def preflight():
    prepared = read(RUN / 'prepared.json')
    assert prepared['identity'] == art(RUN / 'identity.json')
    assert prepared['rows'] == 2051327 and prepared['old_rows'] == 344841
    assert len(prepared['columns']) == 134 and prepared['fit_started'] is False
    for key in ['original_order_labels_airport_stand_runway_month_exact',
                'original_cards_bins_counts_scales_exact',
                'independent_SQL_raw_membership_labels_and_exclusions_exact']:
        assert prepared[key] is True, key
    controls = prepared['controls']
    assert controls['original_and_chunked_encoder_tensors_bins_exact'] is True
    assert controls['original_and_observed_training_outputs_weights_losses_RNG_exact'] is True
    assert controls['synthetic_epochs'] == 3
    assert prepared['inputs'] == read(RUN / 'identity.json')['expected_inputs']
    for record in prepared['arrays'] + [prepared['preprocessing']]:
        assert art(ROOT / record['path']) == record, record['path']
    assert not (RUN / 'fit_started.json').exists()
    return art(RUN / 'prepared.json')


def completion():
    trained = read(RUN / 'trained.json')
    assert trained['identity'] == art(RUN / 'identity.json')
    assert trained['seed'] == 42 and trained['epochs'] == 17
    assert trained['saved_before_diagnostics'] and trained['held_labels_used'] is False
    assert set(trained['checkpoints']) == set(trained['predictions']) == {'14', '15', '17'}
    for record in list(trained['checkpoints'].values()) + list(trained['predictions'].values()):
        assert art(ROOT / record['path']) == record, record['path']
    result = read(RUN / 'result.json')
    assert result['identity'] == art(RUN / 'identity.json')
    assert result['trained'] == art(RUN / 'trained.json') and result['rows'] == 344841
    assert result['state'] in ['exact_old_pilot_no_expansion', 'closed_nonexact_pilot']
    assert result['acceptance_tolerance_fixed'] == 0
    assert result['no_expansion'] and result['no_final_inference'] and result['no_upload']
    assert result['original_n42_0_exact'] == (result['state'] == 'exact_old_pilot_no_expansion')
    assert set(result['reloads']) == {'14', '15', '17'}
    for check in result['reloads'].values():
        assert check['all_old_float_predictions_exact'] and check['rows'] == 344841
    return art(RUN / 'result.json'), result['state']


def main():
    plan_path = OUT / 'plan.json'
    plan = read(plan_path)
    verify(plan)
    target = OUT / 'execution.json'
    assert not target.exists(), 'Preserve prior execution; no automatic restart'
    for name in ['prepared.json', 'encoding', 'fit_started.json', 'trained.json', 'result.json']:
        assert not (RUN / name).exists(), f'Preserve existing {name}'
    state = dict(state='waiting', pid=os.getpid(), started_at=now(), steps=[],
                 plan_sha256=sha(plan_path), no_expansion=True, no_upload=True,
                 waiting_for=['weather independent audit pass', 'explicit root desktop allocation'])
    save(target, state)
    try:
        while True:
            verify(plan)
            assert sha(plan_path) == state['plan_sha256']
            prior = upstream()
            release = allocation(state['plan_sha256'])
            if prior is not None and release is not None:
                break
            time.sleep(15)
        state.update(state='running', weather_audit=prior, allocation=release, waiting_for=[])
        save(target, state)
        for stage in ['prepare', 'pilot']:
            verify(plan)
            assert sha(plan_path) == state['plan_sha256']
            assert upstream() == prior and allocation(state['plan_sha256']) == release
            state.update(active_stage=stage)
            save(target, state)
            started = now()
            with (OUT / f'{stage}.log').open('x') as log:
                child = subprocess.Popen([str(ROOT / '.venv/bin/python'), *plan['commands'][stage]],
                                         cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                state['child_pid'] = child.pid
                save(target, state)
                code = child.wait()
            assert code == 0, f'{stage} exited {code}; preserve all checkpoints and logs'
            if stage == 'prepare':
                evidence = preflight()
            else:
                evidence, result_state = completion()
            state['steps'].append(dict(stage=stage, started_at=started, completed_at=now(), evidence=evidence))
            save(target, state)
        state.update(state='complete', result_state=result_state, active_stage=None,
                     child_pid=None, finished_at=now())
        save(target, state)
        print(json.dumps(dict(state='complete', result_state=result_state,
                              receipt=str(target.relative_to(ROOT)))), flush=True)
    except BaseException as exc:
        state.update(state='failed', finished_at=now(), failure=repr(exc), traceback=traceback.format_exc())
        save(target, state)
        raise


if __name__ == '__main__':
    main()
