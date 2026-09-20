"""Full-year refit, ranking predictions and upload for v39, OPDI-combined (row 42).

Row 42 is v38's recipe with the OPDI ground-track inputs added to the model:

    ordinary regressor   row 30's inputs + OPDI_FEATURES
    Rome classifier      row 30's inputs + OPDI_FEATURES
    late-row estimate    on, the late mean fitted on the 2025 training labels
    day-shift            off
    post-processing      row 33's flight-plan window, row 34's no-ID cap, then row 37's
                         corrections (twin records, day-shift blend, 12 h clock anchor)
                         with parameters fitted on all of 2025

    prepare  one seed-42, 800-round fit of both boosters, the late mean and the correction
             parameters on all of 2025, then predictions for the ranking set. Each ranking
             month sees all of 2025 plus its own month only, as for v26-v30 and v38, so its
             stand clock and OPDI neighbour inputs read that month's own movements.
    upload   one conditional PUT, refused if five files were already uploaded today (UTC).
    result   read the official result file once it exists.

The OPDI data is an external open dataset and must be declared (see SUBMISSIONS.md).
Nothing is uploaded by `prepare`.

Run: .venv/bin/python VALIDATION/opdi_submit.py prepare|upload|result
"""

import argparse
import gc
import glob
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
import artifact_control_run as control
from common import RANKING, TEMPLATE, TRAIN_GLOB
import opdi
import opdi_run as run
import overnight_run
import plan_window
import record_fixes
import recording_artifacts as corrections
from records import RUNS, data_identity, sha256, write_json
from rome_hybrid_submit import KIND, MONTHS, SEED, FREEZE, islands, now, submission, file_name
import rome_signals as signals
from rome_hybrid_run import save_booster
import stand_day_submit as original
from submit import fill_template
import weather

VERSION = 39
NAME = 'OPDI-combined'
EXCLUDE_RULE_ROWS = False
RUN = RUNS / 'v3_20260919T211914721668Z'
CORRECTION_RUN = RUNS / 'v3_20260919_artifact_experiments'
COMBINED_RUN = RUNS / 'v3_20260919T223006736822Z'
CANDIDATE = 'opdi'
ORDINARY = run.ORDINARIES[CANDIDATE]
CLASSIFIER = run.CLASSIFIERS[CANDIDATE]
MANIFEST = RUN / f'submission_v{VERSION}.json'
PREVIOUS = ROOT / 'submissions' / 'unique-rose_v34.parquet'
SOURCES = ['VALIDATION/opdi_submit.py', 'VALIDATION/opdi.py', 'VALIDATION/opdi_run.py',
           'VALIDATION/opdi_combined_run.py', 'VALIDATION/recording_artifacts.py',
           'VALIDATION/record_fixes.py', 'VALIDATION/plan_window.py',
           'VALIDATION/stand_day_submit.py', 'VALIDATION/rome_hybrid_submit.py']


def validated_run():
    """The completed row 41 run, and the derived row 42 run that scored its corrections."""
    record = json.loads((RUN / 'result.json').read_text())
    config = record['config']
    if (record['status'] != 'complete' or config['smoke'] or config['sample'] != 1.0
            or config['rounds'] != 800 or SEED not in config['seeds']
            or config['candidate_specs'].get(CANDIDATE) != dict(ordinary='opdi', classifier='opdi')
            or not config['late'] or config['dayshift']
            or config['ordinaries'][CANDIDATE] != list(ORDINARY)
            or config['classifiers'][CANDIDATE] != list(CLASSIFIER)):
        raise ValueError(f'{RUN.name} is not a completed full validation run of row 41')
    combined = json.loads((COMBINED_RUN / 'result.json').read_text())
    if combined.get('derived_from', {}).get('run') != RUN.name or 'pred34_opdi' not in combined['derived_from']['columns']:
        raise ValueError(f'{COMBINED_RUN.name} is not the derived row 42 run')
    changed = sorted(name for name, digest in record['signature'].items() if sha256(ROOT / name) != digest)
    if changed:
        raise ValueError(f'Sources changed since validation: {changed}')
    if data_identity([*glob.glob(TRAIN_GLOB), RANKING, weather.METAR]) != record['data_identity']:
        raise ValueError('Training, ranking or METAR data changed since validation')
    return record, config, combined


def correction_parameters(late_mean, audit):
    """Fit row 37's corrections on all of 2025, and check the window fits against the study."""
    saved = json.loads((CORRECTION_RUN / 'result.json').read_text())['correction_parameters']
    if any(values['day']['handler_stratified'] for values in saved.values()):
        raise ValueError('Handler groups are stratified in the study; the signals would matter')
    fitted = corrections.fit(audit, late_mean=late_mean)
    if fitted['settings'] != corrections.SETTINGS:
        raise ValueError('Correction settings changed')
    return fitted


def apply_corrections(base, raw, X, parameters, probability):
    """Row 33's window, row 34's cap, then row 37's corrections; same order as v38."""
    if not raw.index.equals(X.index):
        raise ValueError('Raw movements and correction features are misaligned')
    observable = raw.loc[:, list(dict.fromkeys(['mvt_id', 'FLIGHT_mvt', *corrections.RAW]))].copy()
    for name in corrections.SIGNALS:
        observable[name] = X[name].to_numpy()
    window = plan_window.apply(base, observable)
    capped = record_fixes.noid_cap(window, observable)
    predictions, masks = corrections.predict(capped, observable, parameters, probability,
                                             twin_mask=record_fixes.twin_rows(observable))
    frame = pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), pred_model=base, pred_window=window,
                              pred34=capped, pred=predictions['pred_combined'], p=probability,
                              opdi_ground=X['op_ground_sec'].notna().to_numpy(),
                              opdi_matched=(X['op_matched'].to_numpy() == 1)))
    for name, mask in masks.items():
        frame['mask_' + name] = mask
    return frame, masks


def prepare(threads):
    record, config, combined = validated_run()
    if MANIFEST.exists() or submission(VERSION).exists():
        raise ValueError(f'v{VERSION} is already prepared; use upload/result')
    features = opdi.load_features()
    reports = weather.load_reports()
    dep, arr, rank, rank_arr = overnight_run.load_inputs(1.0, ranking=True)
    months = sorted(pd.to_datetime(rank.MVT_TIME_UTC_mvt).dt.strftime('%Y-%m').unique())
    if months != MONTHS or len(rank) != original.EXPECTED_ROWS or rank.mvt_id.duplicated().any():
        raise ValueError('Expected the complete January/July 2026 ranking departures')
    missing = (~rank.mvt_id.isin(features.index)).sum()
    if missing:
        raise ValueError(f'{missing} ranking departures are absent from the OPDI feature table')
    X, y, cats, raw_kept = overnight_run.prepare_training(dep, arr, config)
    ranking = islands(dep, arr, rank, rank_arr, cats, config)
    # Training rows see all 2025 movements, as in each validation window's own training rows.
    X = run.add_inputs(X, raw_kept, reports, dep, arr)
    print(f'v{VERSION}: fitting on {len(X):,} departures, {X.shape[1]} prepared inputs', flush=True)
    parameters = signals.fit_parameters(X, y, raw_kept)
    audit = control.training_audit(raw_kept, X, y)
    fitted = correction_parameters(parameters['late_mean'], audit)
    print(f"v{VERSION}: late mean {parameters['late_mean']:.1f} s; day blend "
          f"{fitted['day']['pooled_weight']:.6f}; clock rows {fitted['clock']['rows']}", flush=True)
    if EXCLUDE_RULE_ROWS:
        # Rows the 4 h rule overwrites never reach the model, so fitting their
        # labels only distorts the rest. Correction parameters above saw every row.
        drop = run.rule_rows(raw_kept, config['prediction_rule'])
        X, y, raw_kept = X.loc[~drop], y[~drop], raw_kept.loc[~drop]
        print(f'v{VERSION}: excluded {int(drop.sum()):,} rule-covered training rows, fitting on {len(X):,}', flush=True)
    started = time.monotonic()
    boosters = {}
    for part, fit_part in [('ordinary', lambda: signals.fit_ordinary(X, y, raw_kept, KIND, SEED,
                                                                     rounds=config['rounds'], threads=threads,
                                                                     extra=ORDINARY)),
                           ('classifier', lambda: signals.fit_classifier(X, y, raw_kept, KIND, SEED,
                                                                         threads=threads, extra=CLASSIFIER))]:
        booster = fit_part()
        path = RUN / 'models' / f'refit_2025__v{VERSION}__{SEED}.{part}.txt'
        save_booster(booster, path)
        boosters[part] = booster
    model = dict(minutes=round((time.monotonic() - started) / 60, 2),
                 paths={part: f'models/refit_2025__v{VERSION}__{SEED}.{part}.txt' for part in boosters},
                 parameters=parameters, corrections=fitted, coverage=run.coverage(X, raw_kept))
    print(f"v{VERSION}: fitted in {model['minutes']} min", flush=True)
    del X, y, raw_kept, audit
    gc.collect()
    pieces, all_masks = [], {}
    for month, (part, part_arr, V) in ranking.items():
        print(f'v{VERSION}: predicting ranking island {month}: {len(part):,} departures', flush=True)
        V = run.add_inputs(V, part, reports, pd.concat([dep, part], ignore_index=True),
                           pd.concat([arr, part_arr], ignore_index=True))
        out = signals.predict(boosters['ordinary'], boosters['classifier'], V, part, KIND, threads=threads,
                              extra=ORDINARY, classifier_extra=CLASSIFIER,
                              late_mean=parameters['late_mean'], dayshift_weight=None)
        frame, masks = apply_corrections(out['final'], part, V, fitted, out['p'])
        frame['month'] = month
        frame['rule_triggered'] = out['rule_triggered']
        pieces.append(frame)
        all_masks[month] = {name: int(mask.sum()) for name, mask in masks.items()}
        print(f'v{VERSION}: correction masks {json.dumps(all_masks[month])}', flush=True)
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(rank.mvt_id).reset_index()
    if len(frame) != original.EXPECTED_ROWS or not np.isfinite(frame.pred).all() or (frame.pred < 0).any():
        raise ValueError(f'v{VERSION}: ranking predictions are incomplete, nonfinite or negative')
    write(frame, model, record, config, combined, all_masks)


def write(frame, model, record, config, combined, masks):
    predictions = RUN / 'refit' / f'v{VERSION}.parquet'
    predictions.parent.mkdir(exist_ok=True)
    frame.to_parquet(predictions, index=False)
    template = pq.read_table(TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    original.validate_submission(table, template)
    path = submission(VERSION)
    path.parent.mkdir(exist_ok=True)
    pq.write_table(table, path)
    previous = None
    if PREVIOUS.exists():
        old = pq.read_table(PREVIOUS)
        original.validate_submission(old, template)
        delta = table['TAXITIME_SEC_mvt'].to_numpy().astype(float) - old['TAXITIME_SEC_mvt'].to_numpy()
        previous = dict(file=PREVIOUS.name, sha256=sha256(PREVIOUS), rows_changed=int((delta != 0).sum()),
                        mean_abs_sec=round(float(np.abs(delta).mean()), 3),
                        max_abs_sec=round(float(np.abs(delta).max()), 3))
    manifest = dict(version=VERSION, row=42, file=file_name(VERSION), recipe=NAME,
                    validation_run=RUN.name, derived_validation_run=COMBINED_RUN.name,
                    correction_parameter_run=CORRECTION_RUN.name, candidate=CANDIDATE,
                    cv=dict(R_A=247.0477, R_B_long=222.6081, R_combined=233.7470,
                            change_vs_row37=dict(R_A=-2.88, R_B_long=-2.28, R_combined=-2.56)),
                    late=True, dayshift=False, ordinary_extra=list(ORDINARY),
                    classifier_extra=list(CLASSIFIER), seed=SEED, rounds=config['rounds'], freeze=FREEZE,
                    ranking_months=MONTHS, rows=table.num_rows,
                    rule_triggered_rows=int(frame.rule_triggered.sum()),
                    correction_masks=masks,
                    opdi=dict(settings=opdi.SETTINGS, features_sha256=sha256(opdi.FEATURES_PATH),
                              features_sha256_at_validation=record['opdi_identity'],
                              ranking_matched=round(float(frame.opdi_matched.mean()), 5),
                              ranking_ground_seen=round(float(frame.opdi_ground.mean()), 5)),
                    prediction_summary=frame.pred.describe().round(3).to_dict(),
                    against_v34=previous, model=model,
                    predictions=str(predictions.relative_to(RUN)), predictions_sha256=sha256(predictions),
                    sha256=sha256(path), template_sha256=sha256(TEMPLATE),
                    validation_signature=record['signature'], data_identity=record['data_identity'],
                    preparation_source={name: sha256(ROOT / name) for name in SOURCES},
                    external_data='EUROCONTROL OPDI v0.0.2 (OpenSky Network ADS-B), declared in SUBMISSIONS.md',
                    state='prepared', prepared_at=now())
    write_json(MANIFEST, manifest)
    print(f"Prepared {path}: {table.num_rows:,} rows, {manifest['rule_triggered_rows']} Rome-rule rows; "
          f"OPDI ground tracks on {100 * manifest['opdi']['ranking_ground_seen']:.1f}% of ranking rows; "
          f"against v34: {json.dumps(previous)}", flush=True)


def configure_transfer():
    # Reuse the existing conditional upload/result implementation with new paths only.
    original.VERSION, original.NAME, original.RUN, original.MANIFEST = VERSION, NAME, RUN, MANIFEST


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('step', choices=['prepare', 'upload', 'result'])
    parser.add_argument('--threads', type=int, default=8)
    parser.add_argument('--exclude-rule-rows', action='store_true',
                        help='drop the departures the 4 h rule overwrites from the model fit')
    parser.add_argument('--version', type=int, help='submission version (default: this recipe\'s)')
    parser.add_argument('--name', help='recipe name recorded in the manifest')
    parser.add_argument('--run', help='validated CV run supplying the recipe and signature')
    parser.add_argument('--combined-run', help='derived run that scored its corrections')
    args = parser.parse_args(argv)
    global VERSION, NAME, RUN, COMBINED_RUN, MANIFEST, EXCLUDE_RULE_ROWS
    EXCLUDE_RULE_ROWS = args.exclude_rule_rows
    if args.version:
        VERSION = args.version
    if args.name:
        NAME = args.name
    if args.run:
        RUN = RUNS / args.run
    if args.combined_run:
        COMBINED_RUN = RUNS / args.combined_run
    MANIFEST = RUN / f'submission_v{VERSION}.json'
    configure_transfer()
    if args.step == 'prepare':
        prepare(args.threads)
    elif args.step == 'upload':
        original.upload()
    else:
        original.result()


if __name__ == '__main__':
    main()
