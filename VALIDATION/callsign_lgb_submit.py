"""Full-year callsign main LightGBM and the single fixed combined candidate.

Fit original LB1 seed42/10000 on the unchanged ordinary population and156
inputs plus callsign6. Replace only that member in the equal three-tree mean;
combine full15 callsign networks and fixed half CatBoost/R20 Rome probability.
All corrections, templates and old/final identities use the audited parent.
No upload entry point. Fits and builds require an allocated machine slot.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import callsign_submit as parent
import callsign_lgb_jan_jul as cv
import callsign_lgb_control_jan_jul as control
import callsign_lgb_combo_score as combo
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261010_callsign_lgb_submit'
DESIGN = dict(parent='v58', ordinary_tree_seed=42, rounds=10000, trial='LB1',
    new_features=list(cv.FEATURES), original_feature_count=156,
    population='unchanged original2051327 ordinary training rows',
    tree_mean='equal mean of new main LightGBM and original airport LightGBM and CatBoost',
    neural='full15 callsign PLE', network_share=.71,
    rome='fixed half three-seed CatBoost plus half original R20',
    corrections='unchanged callsign_submit.final_frame', views=['old', 'final'], upload=False)


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return parent.artifact(path)


def verify(record):
    return parent.features.verify_artifact(record)


def log(message):
    print(time.strftime('[%H:%M:%S] ') + message, flush=True)


def specification():
    records = [read(cv.OUT / 'predictions' / f'{month}_s42.json') for month in ['2025-01', '2025-07']]
    first = records[0]
    for rec in records:
        if any(rec[name] != first[name] for name in ['params', 'columns', 'categorical', 'rounds', 'seed']):
            raise ValueError('Completed CV LightGBM settings differ')
        if rec['runner_sha256'] != sha256(Path(cv.__file__)):
            raise ValueError('Completed callsign LightGBM source changed')
    if first['seed'] != 42 or first['rounds'] != 10000 or first['params']['num_threads'] != 8:
        raise ValueError('Frozen CV seed/round/thread specification changed')
    expected = cv.ts.lgb_params(dict(cv.hpo.V44, **cv.hpo.trial_spec('lgb', 'LB1')['overrides']), 42, .02)
    expected['num_threads'] = 8
    if first['params'] != expected or len(first['columns']) != 162 or first['columns'][-6:] != list(cv.FEATURES):
        raise ValueError('Validated LB1/156+6 configuration changed')
    if read(control.OUT / 'result.json')['comparisons']['callsign_vs_laptop_control']['decision'] != 'improvement':
        raise ValueError('Matched laptop control does not support callsign feature gain')
    if read(combo.OUT / 'result.json')['comparisons']['combined_calls_lgb_vs_combined_matched']['decision'] != 'improvement':
        raise ValueError('Fixed incremental combination did not pass')
    return first['params'], first['columns'], first['categorical']


def guard():
    params, names, cats = specification()
    current = dict(design=DESIGN, params=params, columns=names, categoricals=cats,
        source_sha256=sha256(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)): sha256(Path(m.__file__))
                 for m in [parent, cv, cv.modeling, cv.clocks]},
        parent_prepared=artifact(parent.OUT / 'prepared.json'),
        parent_design=artifact(parent.OUT / 'design.json'),
        evidence=[artifact(control.OUT / 'result.json'), artifact(combo.OUT / 'result.json')]
                 + [artifact(cv.OUT / 'predictions' / f'{m}_s42.json') for m in ['2025-01', '2025-07']])
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'design.json'
    if path.exists():
        if read(path) != current:
            raise ValueError('Full-year callsign main-LightGBM design changed')
    else:
        if list(OUT.glob('**/*.parquet')) or list(OUT.glob('**/*.txt')):
            raise ValueError('Unrecorded full-year artifacts require inspection')
        write_json(path, current)
    return current


def completed_model():
    record_path, path = OUT / 'model.json', OUT / 'models/lgb.txt'
    if not record_path.exists():
        if path.exists():
            raise ValueError('Unrecorded saved model; inspect before any repeated fit')
        return None
    rec = read(record_path)
    if rec['design_sha256'] != sha256(OUT / 'design.json') or rec['prepared_sha256'] != sha256(parent.OUT / 'prepared.json'):
        raise ValueError('Saved full-year model provenance changed')
    recipe = read(OUT / 'design.json')
    if rec['params'] != recipe['params'] or rec['columns'] != recipe['columns'] or rec['categoricals'] != recipe['categoricals']:
        raise ValueError('Saved full-year model settings changed')
    verify(rec['model'])
    return rec


def ranking_inputs(view, manifest):
    """Reuse the compact portable-worker export when present; never create it."""
    packet_manifest = parent.OUT / 'portable_worker' / 'manifest.json'
    if packet_manifest.exists():
        import callsign_submit_portable as portable
        packet = portable.ranking_packet(view)
        provenance = dict(kind='shared stripped ranking packet', manifest=artifact(packet_manifest),
                          source=artifact(Path(portable.__file__)))
        return packet, provenance
    data = parent.ranking_data(view)
    packet = dict(Vn=data['Vn'][manifest['original_columns']], rank_ids=data['rank_ids'], view=view)
    return packet, dict(kind='audited original ranking cache', source=manifest['views'][view]['inputs'])


def fit():
    recipe = guard()
    manifest = parent.validate_prepared()
    params, names, cats = recipe['params'], recipe['columns'], recipe['categoricals']
    rec = completed_model()
    if rec is None:
        data = pd.read_pickle(parent.hs.INPUTS)
        membership = pd.read_parquet(verify(manifest['training_ids']))
        np.testing.assert_array_equal(membership.original_index, data['Xn'].index)
        ids = membership.mvt_id.to_numpy(np.int64)
        if len(ids) != manifest['training_rows'] or len(ids) != 2051327 or len(data['yn']) != len(ids):
            raise ValueError('Full-year ordinary population changed')
        base = data['Xn'][manifest['original_columns']]
        table, _ = parent.features.load_features('train')
        X = cv.clocks.add_features(base, ids, table)
        pd.testing.assert_frame_equal(X[manifest['original_columns']], base, check_exact=True)
        actual_cats = cv.ideas.signals._categorical(X.columns) + [c for c in cv.ideas.ef.CATEGORICAL if c in X.columns]
        if list(X) != names or actual_cats != cats:
            raise ValueError('Training feature/category order differs from completed CV')
        if any(not pd.api.types.is_numeric_dtype(X[c]) for c in cv.FEATURES):
            raise ValueError('Calls metadata must remain numeric')
        y = np.asarray(data['yn'], float)
        if not np.isfinite(y).all():
            raise ValueError('Nonfinite original training labels')
        started = time.monotonic()
        log(f'Fit {len(X):,} unchanged ordinary rows, original156+6, LB1 seed42/10000, eight threads')
        model = lgb.train(params, lgb.Dataset(X, y, categorical_feature=cats), num_boost_round=10000)
        if model.current_iteration() != 10000:
            raise ValueError('Model stopped before the frozen iteration count')
        path = OUT / 'models/lgb.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.txt.tmp')
        model.save_model(str(temp))
        temp.replace(path)
        rec = dict(design_sha256=sha256(OUT / 'design.json'), prepared_sha256=sha256(parent.OUT / 'prepared.json'),
            model=artifact(path), params=params, columns=names, categoricals=cats, rows=len(X),
            training_ids=manifest['training_ids'], original_inputs=manifest['original_inputs'],
            train_features=manifest['train_features']['artifact'], rounds=10000, seed=42,
            minutes=(time.monotonic() - started) / 60, host=platform.node(), lightgbm=lgb.__version__)
        write_json(OUT / 'model.json', rec)
        del data, base, X, y, table, membership
        gc.collect()
    else:
        model = lgb.Booster(model_file=str(verify(rec['model'])))
        log('Verified completed full-year model; prediction-only resume')
    if model.feature_name() != names:
        raise ValueError('Saved model feature order changed')
    for view in parent.VIEWS:
        dest = OUT / 'predictions' / f'{view}.parquet'
        metadata = dest.with_suffix('.json')
        if metadata.exists():
            old = read(metadata)
            if (old['model_sha256'] != rec['model']['sha256']
                    or old['design_sha256'] != sha256(OUT / 'design.json')
                    or old['prepared_sha256'] != sha256(parent.OUT / 'prepared.json')
                    or old['view'] != view
                    or old['feature_table'] != manifest['views'][view]['features']['artifact']):
                raise ValueError('Saved prediction provenance changed')
            verify(old['prediction'])
            continue
        if dest.exists():
            raise ValueError('Unrecorded prediction requires inspection')
        packet, provenance = ranking_inputs(view, manifest)
        ids = np.asarray(packet['rank_ids'])
        if packet['view'] != view or len(ids) != parent.features.EXPECTED_ROWS[view] or pd.Index(ids).has_duplicates:
            raise ValueError('Ranking packet view or IDs changed')
        table, _ = parent.features.load_features(view)
        V = cv.clocks.add_features(packet['Vn'][manifest['original_columns']], ids, table)
        pd.testing.assert_frame_equal(V[manifest['original_columns']], packet['Vn'][manifest['original_columns']], check_exact=True)
        if list(V) != names:
            raise ValueError('Ranking feature order changed')
        values = cv.modeling.predict(model, V, threads=params['num_threads'])
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError('Invalid main-LightGBM prediction')
        parent.save_frame(dest, pd.DataFrame(dict(mvt_id=ids, pred=values.astype(np.float32))))
        write_json(metadata, dict(view=view, prediction=artifact(dest), design_sha256=sha256(OUT / 'design.json'),
            model_sha256=rec['model']['sha256'], prepared_sha256=sha256(parent.OUT / 'prepared.json'),
            feature_table=manifest['views'][view]['features']['artifact'], ranking_inputs=provenance, rows=len(ids)))
        del packet, table, V
        gc.collect()
        log(f'{view}: full-year main-LightGBM predictions saved')


def replace_main(components, ids, replacement):
    """Pure fixed equal-mean replacement with ordered IDs and finite values."""
    np.testing.assert_array_equal(components.mvt_id.to_numpy(), np.asarray(ids))
    np.testing.assert_array_equal(replacement.mvt_id.to_numpy(), np.asarray(ids))
    if pd.Index(ids).has_duplicates:
        raise ValueError('Duplicate IDs within a ranking view')
    values = components[parent.hs.GBM].to_numpy(float).copy()
    if values.shape != (len(ids), 3) or not np.isfinite(values).all():
        raise ValueError('Invalid parent component matrix')
    new = replacement.pred.to_numpy(float)
    if not np.isfinite(new).all() or (new < 0).any():
        raise ValueError('Invalid replacement values')
    parent_mean = values.mean(axis=1)
    values[:, parent.hs.GBM.index('lgb')] = new
    return parent_mean, values.mean(axis=1)


def parent_components(view, ids):
    if view == 'old':
        folder = parent.hs.OUT / 'submission_v54'
        paths = [folder / f'ranking_{m}.parquet' for m in parent.features.RANK_MONTHS[view]]
        cols = ['mvt_id'] + [f'{name}_part' for name in parent.hs.GBM]
        frame = pd.concat([pd.read_parquet(path, columns=cols) for path in paths], ignore_index=True)
        frame = frame.rename(columns={f'{name}_part': name for name in parent.hs.GBM})
    else:
        paths = [parent.fs.WORK / 'parts/trees.parquet']
        frame = pd.read_parquet(paths[0], columns=['mvt_id'] + parent.hs.GBM)
    np.testing.assert_array_equal(frame.mvt_id, ids)
    return frame, [artifact(p) for p in paths]


def build(view, catboost_record):
    guard()
    manifest = parent.validate_prepared()
    model = completed_model()
    if model is None:
        raise ValueError('Full-year main LightGBM is not fitted')
    data = parent.ranking_data(view)
    control_result = parent.parent_control(view, data)
    _, original_tree, original_p, parent_sources = parent.parent_parts(view, data)
    pred_record = read(OUT / 'predictions' / f'{view}.json')
    if (pred_record['model_sha256'] != model['model']['sha256'] or pred_record['view'] != view
            or pred_record['design_sha256'] != sha256(OUT / 'design.json')
            or pred_record['prepared_sha256'] != sha256(parent.OUT / 'prepared.json')
            or pred_record['feature_table'] != manifest['views'][view]['features']['artifact']):
        raise ValueError('Main-LightGBM prediction belongs to another model/view')
    new = pd.read_parquet(verify(pred_record['prediction']))
    components, tree_sources = parent_components(view, data['rank_ids'])
    reproduced, tree = replace_main(components, data['rank_ids'], new)
    np.testing.assert_array_equal(reproduced, original_tree)
    values, networks = [], []
    for group in parent.GROUPS:
        for k in parent.NETS:
            rec = parent.completed_member(group, k)
            if rec is None:
                raise ValueError('All15 callsign networks must finish first')
            frame = pd.read_parquet(verify(rec['predictions'][view]))
            np.testing.assert_array_equal(frame.mvt_id, data['rank_ids'])
            values.append(frame.base.to_numpy(float))
            networks.append(artifact(parent.member_paths(group, k)[1]))
    neural = np.mean(values, axis=0)
    airports = np.concatenate([data['months'][m]['raw'].ADEP_mvt.to_numpy() for m in parent.features.RANK_MONTHS[view]])
    cat, cat_record = parent.catboost_probability(catboost_record, view, data['rank_ids'], airports)
    probability = .5 * original_p + .5 * cat
    frame, masks = parent.final_frame(data, neural, tree, probability)
    frame['main_lgb_original'] = components.lgb.to_numpy(float)
    frame['main_lgb_callsign'] = new.pred.to_numpy(float)
    template_path = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
    template = pq.read_table(template_path)
    table = parent.hs.fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    if table.schema != template.schema or table.column_names != template.column_names:
        raise ValueError('Candidate template schema changed')
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt' and not table[name].equals(template[name]):
            raise ValueError('A protected template column changed')
    filled = table['TAXITIME_SEC_mvt'].to_numpy()
    if len(filled) != parent.features.EXPECTED_ROWS[view] or not np.isfinite(filled).all() or (filled < 0).any():
        raise ValueError('Invalid serialized candidate values')
    directory = OUT / 'candidates' / view
    if directory.exists():
        raise FileExistsError('Preserve the already built combined candidate')
    directory.mkdir(parents=True)
    path = directory / 'candidate.parquet'
    pq.write_table(table, path)
    parent.save_frame(directory / 'parts.parquet', frame)
    write_json(directory / 'manifest.json', dict(state='prepared_not_uploaded', view=view, design=DESIGN,
        design_sha256=sha256(OUT / 'design.json'), output=artifact(path), template=artifact(template_path),
        source_sha256=sha256(Path(__file__)), parent_prepared=artifact(parent.OUT / 'prepared.json'),
        exact_parent_control=control_result, exact_parent_tree_mean=True,
        model_record=artifact(OUT / 'model.json'), main_prediction=artifact(OUT / 'predictions' / f'{view}.json'),
        network_records=networks, parent_parts=parent_sources, parent_tree_components=tree_sources,
        rome=cat_record, correction_masks=masks, rule_triggered=int(frame.rule_triggered.sum()),
        cv_evidence=artifact(combo.OUT / 'result.json'), rows=len(frame)))
    log(f'{view}: fixed combined candidate prepared, no upload')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'fit', 'build'])
    parser.add_argument('--view', choices=parent.VIEWS)
    parser.add_argument('--catboost-record', type=Path)
    args = parser.parse_args()
    if args.stage == 'build' and (args.view is None or args.catboost_record is None):
        parser.error('build requires --view and --catboost-record for the fixed selected combination')
    with threadpool_limits(limits=1, user_api='blas'):
        if args.stage == 'fit':
            fit()
        elif args.stage == 'build':
            build(args.view, args.catboost_record)
        else:
            specification()
            log('Frozen LB1/seed42/10000 and matched/incremental CV evidence pass; no fit or data build')


if __name__ == '__main__':
    main()
