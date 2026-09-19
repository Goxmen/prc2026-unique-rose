"""Refit and submit Artifact-combined (former row44, current row37) as v38.

Run: .venv/bin/python VALIDATION/artifact_submit.py prepare|upload|result
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import artifact_control_run as control
import overnight_run
import plan_window
import record_fixes
import recording_artifacts as corrections
import rome_signals as signals
import stand_day_run as row30
import stand_day_submit as original
from common import TEMPLATE
from overnight_submit import bucket_listing, uploads_today, EXPECTED_ROWS
from records import RUNS, sha256, write_json
from rome_hybrid_run import save_booster
from rome_hybrid_submit import KIND, SEED, MONTHS, islands, now, submission
from submit import fill_template
import weather

VERSION = 38
RUN = RUNS / 'v3_20260919_artifact_experiments'
PARENT = RUNS / 'v3_20260919_artifact_control'
MANIFEST = RUN / 'submission_v38.json'
CHECKPOINT = RUN / 'models' / 'refit_2025_v38.json'
SOURCES = ['VALIDATION/artifact_submit.py', 'VALIDATION/stand_day_submit.py',
           'VALIDATION/rome_hybrid_submit.py', 'VALIDATION/overnight_submit.py',
           'VALIDATION/submit.py', 'VALIDATION/tools/sweep_taxi_thresholds.py']


def validated_run():
    historical, _ = original.validated_run()  # Original model sources and all data hashes.
    parent = json.loads((PARENT / 'result.json').read_text())
    record = json.loads((RUN / 'result.json').read_text())
    cfg = parent['config']
    if (record['status'] != 'complete' or parent['status'] != 'complete'
            or not record['full_benchmark'] or not parent['full_benchmark']
            or cfg['sample'] != 1 or cfg['rounds'] != 800 or cfg['smoke']
            or cfg['seeds'] != [42, 142, 242] or not cfg['late'] or cfg['dayshift']
            or cfg['ordinaries']['stand'] != list(control.ORDINARY)
            or cfg['classifiers']['stand'] != list(control.CLASSIFIER)
            or record['derived_from'] != {'run': PARENT.name, 'column': 'pred_noid_cap'}
            or record['config']['candidates']['combined'] != 'pred_combined'
            or record['config']['recording_artifact_settings'] != corrections.SETTINGS
            or len(record['folds']) != 16):
        raise ValueError('Selected validation recipe is incomplete or changed')
    if record['source_signature'] != parent['signature']:
        raise ValueError('Correction/control source link changed')
    for source in (parent, record):
        for name, digest in source['signature'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Validated source changed: {name}')
        if source['data_identity'] != historical['data_identity']:
            raise ValueError('Correction/control data differ from historical data')
    if record['parent_input_fingerprints']['result.json'] != sha256(PARENT / 'result.json'):
        raise ValueError('Validated control record changed')
    return record, cfg


def apply_corrections(base, raw, X, parameters, probability):
    if not raw.index.equals(X.index):
        raise ValueError('Raw movements and correction features are misaligned')
    observable = raw.loc[:, list(dict.fromkeys(['mvt_id', 'FLIGHT_mvt', *corrections.RAW]))].copy()
    for name in corrections.SIGNALS:
        observable[name] = X[name].to_numpy()
    window = plan_window.apply(base, observable)
    capped = record_fixes.noid_cap(window, observable)
    predictions, masks = corrections.predict(capped, observable, parameters, probability,
                                               twin_mask=record_fixes.twin_rows(observable))
    frame = pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), pred_stand=base,
                              pred_window=window, pred_control=capped,
                              pred_combined=predictions['pred_combined'], p=probability))
    for name, mask in masks.items():
        frame['mask_' + name] = mask
    return frame, masks


def source_fingerprints():
    return {name: sha256(ROOT / name) for name in SOURCES}


def configure_transfer():
    # Reuse the existing conditional upload/result implementation with new paths only.
    original.VERSION, original.NAME = VERSION, 'Artifact-combined'
    original.RUN, original.MANIFEST = RUN, MANIFEST


def upload():
    manifest = json.loads(MANIFEST.read_text())
    if manifest['preparation_source'] != source_fingerprints():
        raise ValueError('Preparation source changed')
    for name, digest in manifest['validation_records'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError('Validation record changed')
    if sha256(TEMPLATE) != manifest['template_sha256']:
        raise ValueError('Template changed')
    configure_transfer()
    original.upload()


def result():
    configure_transfer()
    original.result()


def prepare():
    record, config = validated_run()
    if MANIFEST.exists() or submission(VERSION).exists():
        raise ValueError('v38 is already prepared; use upload/result')
    remote = bucket_listing()
    if f'unique-rose_v{VERSION}.parquet' in remote or len(uploads_today(remote, now())) >= 5:
        raise ValueError('Filename is occupied or daily quota is exhausted')
    bindings = {str(p.relative_to(ROOT)): sha256(p)
                for p in [PARENT / 'result.json', RUN / 'result.json']}
    guard = dict(validation_records=bindings, preparation_source=source_fingerprints(),
                 template_sha256=sha256(TEMPLATE))
    state = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else dict(guard=guard, models={})
    if state['guard'] != guard:
        raise ValueError('Checkpoint sources, template or validation records changed')
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    write_json(CHECKPOINT, state)
    started = time.monotonic()
    print('Loading full 2025 training data and preparing features', flush=True)
    reports = weather.load_reports()
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1.0, ranking=True)
    if len(rank) != EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Ranking movements are missing or duplicated')
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS:
        raise ValueError('Ranking months changed')
    X, y, cats, raw = overnight_run.prepare_training(dep, arr, config)
    X = row30.add_inputs(X, raw, reports, dep, arr)
    parameters = signals.fit_parameters(X, y, raw)
    audit = control.training_audit(raw, X, y)
    fitted = corrections.fit(audit, late_mean=parameters['late_mean'])
    calibration = dict(parameters=parameters, corrections=fitted, training_rows=len(X))
    if 'calibration' in state and state['calibration'] != calibration:
        raise ValueError('Full-year calibration changed since checkpoint')
    state['calibration'] = calibration
    audit_path = RUN / 'refit' / 'training_audit_v38.parquet'
    audit_path.parent.mkdir(exist_ok=True)
    audit.to_parquet(audit_path, index=False)
    write_json(CHECKPOINT, state)
    print(f"Fitting seed42 on {len(X):,} departures; late mean={parameters['late_mean']:.6f}; "
          f"day blend={fitted['day']['pooled_weight']:.6f}", flush=True)
    boosters = {}
    for part in ('ordinary', 'classifier'):
        path = RUN / 'models' / f'refit_2025__v38__42.{part}.txt'
        if part in state['models']:
            if not path.exists() or sha256(path) != state['models'][part]['sha256']:
                raise ValueError(f'Checkpoint {part} model changed')
            boosters[part] = lgb.Booster(model_file=str(path), params={'num_threads': 20})
            print(f'Reusing checkpoint {part}', flush=True)
            continue
        at = time.monotonic()
        if part == 'ordinary':
            booster = signals.fit_ordinary(X, y, raw, KIND, SEED, rounds=800,
                                           threads=6, extra=control.ORDINARY)
        else:
            booster = signals.fit_classifier(X, y, raw, KIND, SEED, threads=6,
                                             extra=control.CLASSIFIER)
        save_booster(booster, path)
        state['models'][part] = dict(path=str(path.relative_to(RUN)), sha256=sha256(path),
                                    seconds=round(time.monotonic() - at, 3))
        write_json(CHECKPOINT, state)
        boosters[part] = booster
        print(f"Saved {part}: {state['models'][part]['seconds']} seconds", flush=True)
    del X, y, raw, audit
    gc.collect()
    ranking = islands(dep, arr, rank, rank_arr, cats, config)
    pieces = []
    for month, (part, part_arr, V) in ranking.items():
        print(f'Predicting {month}: {len(part):,} departures', flush=True)
        V = row30.add_inputs(V, part, reports, pd.concat([dep, part], ignore_index=True),
                            pd.concat([arr, part_arr], ignore_index=True))
        out = signals.predict(boosters['ordinary'], boosters['classifier'], V, part, KIND,
                              threads=20, extra=control.ORDINARY, classifier_extra=control.CLASSIFIER,
                              late_mean=parameters['late_mean'], dayshift_weight=None)
        frame, masks = apply_corrections(out['final'], part, V, fitted, out['p'])
        frame['month'] = month
        pieces.append(frame)
        print('Correction masks: ' + json.dumps({k:int(v.sum()) for k,v in masks.items()}), flush=True)
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(rank.mvt_id).reset_index()
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred_combined.to_numpy())
    original.validate_submission(table, template)
    prediction_path = RUN / 'refit' / 'v38.parquet'
    frame.to_parquet(prediction_path, index=False)
    submission(VERSION).parent.mkdir(exist_ok=True)
    pq.write_table(table, submission(VERSION))
    previous = ROOT / 'tmp/recording_artifacts_20260919/unique-rose_v34.parquet'
    comparison = None
    if previous.exists():
        old = pq.read_table(previous)
        base = fill_template(template, frame.mvt_id, frame.pred_control.to_numpy())
        original.validate_submission(old, template)
        delta = base['TAXITIME_SEC_mvt'].to_numpy().astype(float) - old['TAXITIME_SEC_mvt'].to_numpy()
        comparison = dict(file='unique-rose_v34.parquet', sha256=sha256(previous),
                          rows_changed=int((delta != 0).sum()), max_abs_sec=float(np.abs(delta).max()),
                          mean_abs_sec=float(np.abs(delta).mean()))
    manifest = dict(version=VERSION, row=37, former_row=44, file=submission(VERSION).name,
        recipe='Artifact-combined', candidate='combined', column='pred_combined',
        validation_run=RUN.name, control_run=PARENT.name, seed=42, rounds=800,
        classifier_rounds=300, freeze='2025-12-31', ranking_months=MONTHS,
        rows=table.num_rows, model=state['models'], calibration=calibration,
        validation_signature=record['signature'], control_signature=record['source_signature'],
        data_identity=record['data_identity'], validation_scores=record['candidate_scores']['combined']['scores'],
        predictions=str(prediction_path.relative_to(RUN)), predictions_sha256=sha256(prediction_path),
        sha256=sha256(submission(VERSION)), **guard,
        masks={c:int(frame[c].sum()) for c in frame if c.startswith('mask_')},
        correction_rows_changed=int((frame.pred_combined != frame.pred_control).sum()),
        prediction_summary=frame.pred_combined.describe().to_dict(),
        control_against_scored_v34=comparison, minutes=round((time.monotonic()-started)/60,3),
        state='prepared', prepared_at=now())
    write_json(MANIFEST, manifest)
    print('Prepared ' + str(submission(VERSION)) + ': ' + json.dumps(manifest['masks']), flush=True)
    print('Control replay comparison: ' + json.dumps(comparison), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'upload', 'result'])
    args = parser.parse_args()
    globals()[args.action]()
