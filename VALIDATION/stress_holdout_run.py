"""Predict additional analogue months with saved June-November models, without fitting."""
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import opdi_disruption_run as run
import prepare_stress_analogues as prepare
from records import sha256, write_json

DEFAULT = run.DEFAULT / 'stress_holdouts'


def read(path):
    return json.loads(Path(path).read_text())


def main():
    selected = read(prepare.OUT / 'selection.json')
    months = selected['supplemental_months']
    if not set(months) <= {'2025-02', '2025-03', '2025-04', '2025-05'}:
        raise ValueError('Supplemental months must be outside June-November training')
    job = next(j for j in run.overnight_run.schedule() if j['id'] == 'v3_2025-11')
    configs, records, progresses = {}, {}, {}
    for variant in run.HISTORICAL:
        records[variant] = read(run.DEFAULT / variant / 'result.json')
        progresses[variant] = read(run.DEFAULT / variant / 'progress.json')
        if records[variant]['status'] != 'complete':
            raise ValueError('Original comparisons must finish first')
        for name, digest in records[variant]['signature'].items():
            if sha256(ROOT / name) != digest:
                raise ValueError(f'Fitted model source changed: {name}')
        configs[variant] = records[variant]['config']
    identity = dict(selection_sha256=sha256(prepare.OUT / 'selection.json'),
        sources={str(path.relative_to(ROOT)): sha256(path) for path in
                 [Path(__file__), prepare.DESIGN, Path(run.__file__), Path(run.features.__file__)]},
        model_records={v: sha256(run.DEFAULT / v / 'result.json') for v in run.HISTORICAL}, months=months)
    DEFAULT.mkdir(parents=True, exist_ok=True)
    state_path = DEFAULT / 'result.json'
    if state_path.exists():
        state = read(state_path)
        if state['identity'] != identity:
            raise ValueError('Cannot resume changed supplemental predictions')
        for variant, progress in state['variants'].items():
            run.overnight_run.check_artifacts(DEFAULT / variant, progress)
        if state['status'] == 'complete':
            print('Supplemental holdouts already complete', flush=True)
            return
    else:
        state = dict(status='running', at=datetime.now(timezone.utc).isoformat(), identity=identity,
            fits=0, variants={v: dict(predictions={}, folds={}) for v in run.HISTORICAL})
        write_json(state_path, state)
    if not months:
        state['status'] = 'complete'
        write_json(state_path, state)
        print('No additional months selected by observable-input matching', flush=True)
        return
    started = time.monotonic()
    try:
        dep, arr = run.overnight_run.load_inputs(1.)
        td, ta, _ = run.overnight_run.split_job(dep, arr, job)
        # Both recipes fit categories before exclusion on the same raw window.
        X, y, cats, raw = run.overnight_run.prepare_training(td, ta, configs['v39'])
        train_ids = set(raw.mvt_id)
        del X, y, raw
        gc.collect()
        validation = run.seasonal.split_frames(dep, arr, dict(train=job['train'], val=months))[2]
        reports = run.weather.load_reports()
        grid = run.features.snow_grid(pd.read_parquet(run.weather.METAR))
        for month in months:
            name = 'RETRO_' + month.replace('-', '_')
            ev = dict(fold=name, month=month, role='retrospective', gap=None)
            raw, V = run.recipe.validation_inputs(job, ev, td, ta, validation, cats, configs['v39'], reports)
            if set(raw.mvt_id) & train_ids:
                raise ValueError('Supplemental evaluation overlaps model training')
            donors = pd.concat([ta, validation[month][1]], ignore_index=True)
            V = run.features.add_features(V, raw, donors, grid)
            for variant in run.HISTORICAL:
                directory = DEFAULT / variant
                directory.mkdir(exist_ok=True)
                progress = state['variants'][variant]
                if name in progress['folds']:
                    continue
                paths = {}
                cp = records[variant]['correction_parameters'][job['id']]
                params = records[variant]['parameters'][job['id']]
                for seed in configs[variant]['seeds']:
                    key = f'{name}__{seed}'
                    if key in progress['predictions']:
                        paths[str(seed)] = progress['predictions'][key]['path']
                        continue
                    classifier = run.load_model(run.DEFAULT / variant, progresses[variant], f'{job["id"]}__shared__{seed}')
                    p = run.signals.substitution_probability(classifier, V, raw, run.recipe.KIND, 20, run.CLASSIFIER)
                    frames, ordinary_values = {}, {}
                    for arm, extra in [('control', run.ORDINARY), ('candidate', run.EXTRA)]:
                        model = run.load_model(run.DEFAULT / variant, progresses[variant], f'{job["id"]}__{arm}__{seed}')
                        ordinary = run.signals.ordinary_prediction(model, V, run.recipe.KIND, 20, extra)
                        ordinary_values[arm] = ordinary
                        frames[arm] = run.prior.frame_for(raw, V, job, ev, seed, ordinary, p, configs[variant], params, cp)
                        del model
                    out = frames['control'][run.KEEP].copy()
                    out['pred_control'] = frames['control'].pred_combined.to_numpy()
                    out['pred_disruption'] = frames['candidate'].pred_combined.to_numpy()
                    out['ordinary_control'] = ordinary_values['control']
                    out['ordinary_disruption'] = ordinary_values['candidate']
                    out['p_shared'] = p
                    out['rule_triggered'] = frames['control'].rule_triggered.to_numpy()
                    for col in run.features.FEATURES:
                        out[col] = V[col].to_numpy()
                    if not np.isfinite(out[['y', 'pred_control', 'pred_disruption']]).all().all():
                        raise ValueError('Nonfinite supplemental predictions')
                    path = directory / f'oof_{name}_seed{seed}.parquet'
                    run.overnight_run.save_frame(path, out)
                    progress['predictions'][key] = dict(path=path.name, rows=len(out),
                        artifacts=run.overnight_run.artifact_hashes(directory, path))
                    paths[str(seed)] = path.name
                    write_json(state_path, state)
                    del classifier, frames, ordinary_values, out
                progress['folds'][name] = dict(name=name, val=month, model_id=job['id'], predictions=paths,
                    retrospective=True, training_overlap_rows=0)
                write_json(state_path, state)
                print(f'Saved {variant}/{month}: {len(raw):,} held-out rows x three seeds; no fits', flush=True)
            del raw, V, donors
            gc.collect()
        state.update(status='complete')
        state.pop('error', None)
    except BaseException:
        state.update(status='interrupted', error=traceback.format_exc())
        raise
    finally:
        state['minutes'] = state.get('minutes', 0) + (time.monotonic() - started) / 60
        write_json(state_path, state)


if __name__ == '__main__':
    with threadpool_limits(limits=1, user_api='blas'):
        main()
