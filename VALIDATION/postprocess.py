"""Register a fixed-rule candidate from a completed run's saved predictions; no fits."""

import argparse
import copy
from datetime import datetime, timezone
import glob
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from records import (ROOT, RUNS, read_run, write_json, append_log, source_snapshot,
                     data_identity, sha256, runtime_identity)
from modeling import PARAMS  # Also establishes the shared pipeline import path.
from common import connect, TRAIN_GLOB, RANKING
from registered_rules import RULE_FILES, apply_rule
from protocol import VERSION, SCORE_VERSION
from reporting import report, print_report

RULE_FILE = 'pipeline/prediction_rules.py'


def transform_predictions(frame, raw, rule):
    if frame.mvt_id.isna().any() or frame.mvt_id.duplicated().any():
        raise ValueError('Saved predictions require unique nonmissing movement IDs')
    if raw.index.hasnans or not raw.index.is_unique:
        raise ValueError('Raw movements require unique nonmissing movement IDs')
    if not frame.mvt_id.isin(raw.index).all():
        raise ValueError('Missing raw movements for saved predictions')
    inputs = raw.loc[frame.mvt_id].set_axis(frame.index)
    if not np.array_equal(pd.to_datetime(frame.mvt_time, utc=True).to_numpy(),
                          pd.to_datetime(inputs.MVT_TIME_UTC_mvt, utc=True).to_numpy()):
        raise ValueError('Saved and raw movement times differ')
    if not frame.airport.equals(inputs.ADEP_mvt.rename('airport')):
        raise ValueError('Saved and raw airports differ')
    if 'raw_flight_id_missing' in frame and not np.array_equal(
            frame.raw_flight_id_missing.to_numpy(), inputs.FLIGHT_ID_mvt.isna().to_numpy()):
        raise ValueError('Original flight-ID missingness differs')
    pred, audit = apply_rule(frame.pred.to_numpy(), inputs, rule)
    out = frame.assign(model_pred=frame.pred, pred=pred)
    for name in audit:
        out[name] = audit[name]
    changed = out.loc[audit.rule_triggered].copy()
    changed['parent_squared_error'] = (changed.model_pred-changed.y)**2
    changed['rule_squared_error'] = (changed.pred-changed.y)**2
    changed['delta_squared_error'] = changed.rule_squared_error-changed.parent_squared_error
    changed['parent_abs_error'] = (changed.model_pred-changed.y).abs()
    changed['rule_abs_error'] = (changed.pred-changed.y).abs()
    changed['delta_abs_error'] = changed.rule_abs_error-changed.parent_abs_error
    return out, changed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--parent', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--rule', choices=['rome_6h', 'rome_4h'], default='rome_6h')
    ap.add_argument('--output-dir', type=Path, default=RUNS)
    args = ap.parse_args(argv)
    parent, parent_dir = read_run(args.parent)
    if (parent.get('status') != 'complete' or parent.get('protocol') != VERSION
            or parent.get('score_version') != SCORE_VERSION):
        raise ValueError('Expected a completed v3 parent')
    if parent['config'].get('prediction_rule', 'off') != 'off' or parent.get('prediction_column', 'pred') != 'pred':
        raise ValueError('Use an unmodified model-prediction parent')
    if parent['config']['params'] != PARAMS:
        raise ValueError('Parent model parameters differ from the shared recipe')
    if parent['data_identity'] != data_identity([*glob.glob(TRAIN_GLOB), RANKING]):
        raise ValueError('Raw datasets differ from the parent validation run')
    run_id = datetime.now(timezone.utc).strftime('v3_%Y%m%dT%H%M%S%fZ')
    outdir = args.output_dir.resolve()/run_id
    outdir.mkdir(parents=True)
    cfg = copy.deepcopy(parent['config'])
    cfg.update(label=args.label, prediction_rule=args.rule)
    record = {k: copy.deepcopy(parent[k]) for k in
              ['protocol', 'score_version', 'data_identity', 'ranking_weights', 'full_benchmark', 'models', 'folds']}
    record.update(run_id=run_id, label=args.label, kind='postprocessed_predictions',
                  at=datetime.now(timezone.utc).isoformat(timespec='seconds'), config=cfg,
                  baseline_id=parent['run_id'], model_parent_id=parent['run_id'],
                  parent_record_sha256=sha256(parent_dir/'result.json'),
                  model_source=copy.deepcopy(parent['source']),
                  rule_source={name: sha256(ROOT/name) for name in RULE_FILES[args.rule]},
                  runtime=runtime_identity(), status='running', official_rmse=None,
                  new_model_fits=0, prediction_provenance={})
    relative = lambda path: os.path.relpath(parent_dir/path, outdir)
    for model in record['models'].values():
        for details in model['seeds'].values():
            for key in ['model', 'training_loss']:
                details[key] = relative(details[key])
        if 'recovery' in model:
            model['recovery']['audit'] = relative(model['recovery']['audit'])
    start = time.monotonic()
    try:
        record['source'] = source_snapshot(outdir)
        write_json(outdir/'result.json', record)
        con = connect()
        con.execute('SET threads=2')
        raw = con.sql('''SELECT MVT_ID_mvt mvt_id, ADEP_mvt, FLIGHT_ID_mvt,
            MVT_TIME_UTC_mvt, SCHED_TIME_UTC_mvt FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',
            params=[TRAIN_GLOB]).df().set_index('mvt_id')
        con.close()
        changes = []
        for name, fold in record['folds'].items():
            if 'recovery' in fold:
                fold['recovery']['audit'] = relative(fold['recovery']['audit'])
            for seed, filename in fold['predictions'].items():
                original = parent_dir/filename
                frame = pd.read_parquet(original)
                output, changed = transform_predictions(frame, raw, args.rule)
                destination = f'oof_{name}_seed{seed}.parquet'
                output.to_parquet(outdir/destination, index=False)
                fold['predictions'][seed] = destination
                record['prediction_provenance'][f'{name}/{seed}'] = {
                    'parent_file': relative(filename), 'parent_sha256': sha256(original),
                    'candidate_sha256': sha256(outdir/destination),
                    'triggered': int(output.rule_triggered.sum()), 'changed': int(output.rule_changed.sum())}
                changes.append(changed)
            print(f'{name}: saved rule predictions for {len(fold["predictions"])} seeds', flush=True)
        changed = pd.concat(changes, ignore_index=True)
        columns = ['mvt_id', 'mvt_time', 'fold', 'seed', 'airport', 'y', 'model_pred', 'pred',
                   'rule_gap_sec', 'rule_triggered', 'rule_changed', 'parent_squared_error',
                   'rule_squared_error', 'delta_squared_error', 'parent_abs_error',
                   'rule_abs_error', 'delta_abs_error']
        changed[columns].sort_values(['mvt_id', 'fold', 'seed']).to_csv(outdir/'changed_rows.csv', index=False)
        record['changed_rows'] = 'changed_rows.csv'
        record['triggered_unique_flights'] = int(changed.mvt_id.nunique())
        del raw, changes, changed, frame, output
        record['report'] = report(record, outdir, (parent, parent_dir))
        print_report(record['report'])
        record['status'] = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        record['status'] = 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'failed'
        record['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        record['minutes'] = round((time.monotonic()-start)/60, 3)
        write_json(outdir/'result.json', record)
        append_log(record, outdir.parent)
    print(f'Completed rule candidate: {outdir}', flush=True)


if __name__ == '__main__':
    main()
