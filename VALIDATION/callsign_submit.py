"""Prepare/fill the validated callsign15 candidate; deliberately no upload stage.

Original full-year ordinary rows and156 inputs are reused. Six audited callsign
inputs are appended by movement ID. The original N09 recipe has fixed full-year
epochs15/21/16 for groups42/142/242, five networks each, with no new stopping fit.
Original and final ranking views remain separate despite overlapping IDs.

Stages: check; prepare; fit; build --view old|final. Optional --catboost-record
accepts the reviewed three-seed mean Rome probability artifact, applying only
the fixed half-CatBoost/half-R20 blend. Fits/builds require explicit scheduling.
Files stay in this experiment directory until a later upload decision.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import pickle
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import callsign_submit_features as features
import final_submit as fs
import hpo_submit as hs
import nn_callsign_clocks_jan_jul as cv
import nn_callsign_pool_resume as resume
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261010_callsign_submit'
EPOCHS = {42: 15, 142: 21, 242: 16}
GROUPS = [42, 142, 242]
NETS = list(range(5))
VIEWS = ['old', 'final']
W_NN = .71
PARENT_NN = hs.WORK / 'parts/nn.json'
V58_MANIFEST = ROOT / 'VALIDATION/runs/v3_20261006_blend_weight_submit/submission_v58/submission_v58.json'
MEMBERSHIP = hs.nis.V46 / 'training_membership.parquet'
DESIGN = dict(parent='v58', change='original156 plus six callsign inputs in all15 N09 networks',
    columns=features.FEATURE_NAMES, groups=GROUPS, nets=NETS, epochs={str(k): v for k, v in EPOCHS.items()},
    seed_formula='group+1000*k', stopping='frozen original full-year15/21/16; no new inner fit',
    target='direct min(taxi_sec,10800), original ordinary population', network_share=W_NN,
    views=VIEWS, tree_models='unchanged saved parent predictions', rules='unchanged nis.final',
    arms=['callsign15', 'callsign15_half_catboost_R20'], upload_stage=False)


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    return features.artifact(path)


def log(message):
    print(time.strftime('[%H:%M:%S] ') + str(message), flush=True)


def guard():
    parent = read(PARENT_NN)
    cfg = cv.shared.recipe()
    if parent['config'] != cfg or {int(g): d['stop_epoch'] for g, d in parent['groups'].items()} != EPOCHS:
        raise ValueError('Original full-year N09 configuration/epochs changed')
    current = dict(design=DESIGN, source_sha256=sha256(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)): sha256(Path(m.__file__))
                 for m in [features, cv, resume, cv.shared, cv.sensor, cv.shared.nnj, cv.shared.nni, fs, hs, hs.nis]},
        evidence=features.evidence(), parent_nn=artifact(PARENT_NN), v58_manifest=artifact(V58_MANIFEST))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'design.json'
    if path.exists():
        if read(path) != current:
            raise ValueError('Prepared source/design/evidence changed; use a separate run directory')
    else:
        if list(OUT.glob('**/*.parquet')) or list(OUT.glob('**/*.pt')):
            raise ValueError('Unrecorded production artifacts require inspection')
        write_json(path, current)
    return current


def save_frame(path, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(temp, index=False)
    temp.replace(path)


def training_ids(data):
    """Recover original positional IDs and independently check saved membership."""
    record = read(hs.nis.V46 / 'submission_v46.json')
    if sha256(MEMBERSHIP) != record['training']['artifacts']['training_membership.parquet']:
        raise ValueError('Saved ordinary training population changed')
    membership = pd.read_parquet(MEMBERSHIP)
    expected = membership.loc[membership.used_by_ordinary, 'mvt_id'].to_numpy(np.int64)
    with duckdb.connect(config={'threads': 2}) as con:
        con.execute("SET TimeZone='UTC'")
        # Same ordered source scan as movement_data.load_phase; no hidden block column.
        raw = con.execute('''SELECT MVT_ID_mvt::BIGINT mvt_id,TAXITIME_SEC_mvt taxi_sec,
            ADEP_mvt airport,MVT_TIME_UTC_mvt t,STAND_mvt stand,RUNWAY_mvt runway
            FROM read_parquet(?) WHERE PHASE_mvt='DEP' ''',
            [str(ROOT / 'data/training_2025-*.parquet')]).df()
    selected = raw.loc[data['Xn'].index]
    ids = selected.mvt_id.to_numpy(np.int64)
    np.testing.assert_array_equal(ids, expected)
    np.testing.assert_array_equal(selected.taxi_sec.to_numpy(float), data['yn'])
    np.testing.assert_array_equal(selected.airport.to_numpy(), data['Xn'].airport.astype(object).to_numpy())
    np.testing.assert_array_equal(selected.t.dt.strftime('%Y-%m').to_numpy(), data['tmonth'])
    for name in ['stand', 'runway']:
        key = selected.airport.astype('string') + ':' + selected[name].astype('string')
        actual = data['Xn'][name].astype('string')
        np.testing.assert_array_equal(key.fillna('<MISSING>').to_numpy(), actual.fillna('<MISSING>').to_numpy())
    if len(ids) != data['ordinary_rows'] or len(ids) != record['training']['ordinary_rows']:
        raise ValueError('Full-year ordinary row count changed')
    return ids


def ranking_data(view):
    path = fs.WORK / f'inputs_{view}.pkl'
    manifest = read(fs.OUT / f'inputs_{view}.json')
    if sha256(path) != manifest['sha256']:
        raise ValueError('Original prepared ranking inputs changed')
    data = pd.read_pickle(path)
    if len(data['rank_ids']) != features.EXPECTED_ROWS[view] or list(data['months']) != features.RANK_MONTHS[view]:
        raise ValueError('Ranking IDs/months changed')
    ids = np.concatenate([data['months'][m]['raw'].mvt_id.to_numpy() for m in features.RANK_MONTHS[view]])
    np.testing.assert_array_equal(ids, data['rank_ids'])
    if pd.Index(ids).has_duplicates:
        raise ValueError('Duplicate IDs within a ranking view')
    return data


def parent_parts(view, data):
    """Reuse the exact parent bases, avoiding any inference/refit numerical drift."""
    if view == 'old':
        directory = hs.OUT / 'submission_v54'
        frames = [pd.read_parquet(directory / f'ranking_{m}.parquet') for m in features.RANK_MONTHS[view]]
        frame = pd.concat(frames, ignore_index=True)
        np.testing.assert_array_equal(frame.mvt_id, data['rank_ids'])
        sources = [artifact(directory / f'ranking_{m}.parquet') for m in features.RANK_MONTHS[view]]
        return frame.nn_base.to_numpy(float), frame.gbm_base.to_numpy(float), frame.p.to_numpy(float), sources
    values, sources = {}, []
    for name in ['nn', 'trees', 'rome']:
        path = fs.WORK / 'parts' / f'{name}.parquet'
        frame = pd.read_parquet(path)
        np.testing.assert_array_equal(frame.mvt_id, data['rank_ids'])
        values[name] = frame
        sources.append(artifact(path))
    return (values['nn'].pred.to_numpy(float), values['trees'][hs.GBM].to_numpy(float).mean(axis=1),
            values['rome'].p.to_numpy(float), sources)


def final_frame(data, neural, tree, probability):
    if not all(len(v) == len(data['rank_ids']) for v in [neural, tree, probability]):
        raise ValueError('Part length mismatch')
    frames, masks, offset = [], {}, 0
    for month, item in data['months'].items():
        n = len(item['raw'])
        sl = slice(offset, offset + n)
        base = W_NN * neural[sl] + (1 - W_NN) * tree[sl]
        frame, mask = hs.nis.final(base, probability[sl], item['V'], item['raw'], data['parameters'], data['fitted'])
        frame['nn_base'], frame['gbm_base'], frame['p'], frame['month'] = neural[sl], tree[sl], probability[sl], month
        frames.append(frame)
        masks[month] = mask
        offset += n
    result = pd.concat(frames, ignore_index=True)
    np.testing.assert_array_equal(result.mvt_id, data['rank_ids'])
    if result.mvt_id.duplicated().any() or not np.isfinite(result.pred).all() or result.pred.lt(0).any():
        raise ValueError('Invalid final predictions')
    return result, masks


def assert_parent_output(frame, template, expected, raw_parent):
    """Check the float prediction and the historical integer serialization."""
    np.testing.assert_array_equal(frame.mvt_id.to_numpy(), raw_parent.mvt_id.to_numpy())
    np.testing.assert_array_equal(frame.pred.to_numpy(), raw_parent.pred.to_numpy())
    table = hs.fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    if table.schema != expected.schema or table.column_names != expected.column_names:
        raise ValueError('Replayed parent submission schema differs')
    for name in table.column_names:
        if not table[name].equals(expected[name]):
            raise ValueError(f'Replayed parent submission column differs: {name}')


def parent_control(view, data):
    nn, tree, p, sources = parent_parts(view, data)
    frame, masks = final_frame(data, nn, tree, p)
    expected_path = ROOT / 'submissions' / ('unique-rose_v58.parquet' if view == 'old' else 'unique-rose_v63.parquet')
    if view == 'old' and sha256(expected_path) != read(V58_MANIFEST)['sha256']:
        raise ValueError('Scored v58 submission changed')
    if view == 'final':
        manifest = read(fs.OUT / 'submission_v63/submission_v63.json')
        if sha256(expected_path) != manifest['sha256']:
            raise ValueError('Prepared final parent file changed')
    raw_directory = V58_MANIFEST.parent if view == 'old' else fs.OUT / 'submission_v63'
    raw_paths = [raw_directory / f'ranking_{m}.parquet' for m in features.RANK_MONTHS[view]]
    raw_parent = pd.concat([pd.read_parquet(path, columns=['mvt_id', 'pred']) for path in raw_paths],
                           ignore_index=True)
    template_path = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
    assert_parent_output(frame, pq.read_table(template_path), pq.read_table(expected_path), raw_parent)
    return dict(exact_parent_reproduced=True, rows=len(frame), parent_submission=artifact(expected_path),
                exact_raw_predictions_reproduced=True, parent_raw_predictions=[artifact(p) for p in raw_paths],
                parent_template=artifact(template_path), parent_parts=sources,
                correction_masks=masks, rule_triggered=int(frame.rule_triggered.sum()))


def prepare():
    identity = guard()
    if (OUT / 'prepared.json').exists():
        validate_prepared()
        log('Existing prepared artifacts verified; skipped')
        return
    features.guard()
    data = pd.read_pickle(hs.INPUTS)
    names = data['names'] + list(hs.ids.COMBO)
    if len(names) != 156 or len(set(names)) != 156:
        raise ValueError('Original156 input contract changed')
    ids = training_ids(data)
    table, train_record = features.load_features('train')
    augmented = cv.add_features(data['Xn'][names], ids, table)
    pd.testing.assert_frame_equal(augmented[names], data['Xn'][names], check_exact=True)
    del augmented, table
    train_path = OUT / 'training_ids.parquet'
    if train_path.exists():
        raise ValueError('Unrecorded prepared training IDs require inspection')
    save_frame(train_path, pd.DataFrame(dict(mvt_id=ids, original_index=data['Xn'].index.to_numpy())))
    parts = {}
    for view in VIEWS:
        ranking = ranking_data(view)
        if ranking['names'] != data['names'] or ranking['parameters'] != data['parameters'] or ranking['fitted'] != data['fitted']:
            raise ValueError('Ranking feature/final correction recipe changed')
        if view == 'old':
            np.testing.assert_array_equal(ranking['rank_ids'], data['rank_ids'])
            if not fs.same_frame(ranking['Vn'][names], data['Vn'][names]):
                raise ValueError('Old ranking no longer matches original full-year input cache')
        table, record = features.load_features(view)
        augmented = cv.add_features(ranking['Vn'][names], ranking['rank_ids'], table)
        pd.testing.assert_frame_equal(augmented[names], ranking['Vn'][names], check_exact=True)
        if np.intersect1d(ids, ranking['rank_ids']).size:
            raise ValueError('Training/ranking IDs overlap')
        parts[view] = dict(inputs=artifact(fs.WORK / f'inputs_{view}.pkl'), features=record,
                           control=parent_control(view, ranking))
        del table, ranking, augmented
        gc.collect()
    manifest = dict(identity=identity, design_sha256=sha256(OUT / 'design.json'),
        original_inputs=artifact(hs.INPUTS), original_columns=names, train_features=train_record,
        membership=artifact(MEMBERSHIP), training_ids=artifact(train_path), training_rows=len(ids),
        views=parts, feature_design=artifact(features.DEFAULT / 'design.json'))
    write_json(OUT / 'prepared.json', manifest)
    log('Original156 columns, training IDs/labels and both exact parent controls verified')


def validate_prepared():
    identity = guard()
    manifest = read(OUT / 'prepared.json')
    if manifest['identity'] != identity or manifest['design_sha256'] != sha256(OUT / 'design.json'):
        raise ValueError('Prepared recipe/implementation changed')
    for key in ['original_inputs', 'membership', 'training_ids', 'feature_design']:
        features.verify_artifact(manifest[key])
    features.verify_artifact(manifest['train_features']['artifact'])
    for view in VIEWS:
        item = manifest['views'][view]
        features.verify_artifact(item['inputs'])
        features.verify_artifact(item['features']['artifact'])
        features.verify_artifact(item['control']['parent_submission'])
        features.verify_artifact(item['control']['parent_template'])
        for part in item['control']['parent_parts'] + item['control']['parent_raw_predictions']:
            features.verify_artifact(part)
    return manifest


def member_paths(group, k):
    stem = f's{group}_k{k}'
    return OUT / 'models' / f'{stem}.pt', OUT / 'networks' / f'{stem}.json'


def completed_member(group, k):
    model, path = member_paths(group, k)
    if not path.exists():
        if model.exists() or any((OUT / 'networks' / f'{view}_s{group}_k{k}.parquet').exists() for view in VIEWS):
            raise ValueError('Unrecorded network artifact; inspect interrupted fit')
        return None
    record = read(path)
    if (record['design_sha256'] != sha256(OUT / 'design.json')
            or record['prepared_sha256'] != sha256(OUT / 'prepared.json')
            or record['seed'] != group + 1000 * k or record['epochs'] != EPOCHS[group]):
        raise ValueError('Existing full-year network provenance changed')
    for item in [record['model'], record['preprocessing'], *record['predictions'].values()]:
        features.verify_artifact(item)
    return record


def fit():
    manifest = validate_prepared()
    if not torch.cuda.is_available():
        raise RuntimeError('Full-year N09 fits require the allocated GPU')
    if all(completed_member(g, k) is not None for g in GROUPS for k in NETS):
        log('All15 full-year networks already complete; skipped')
        return
    data = pd.read_pickle(hs.INPUTS)
    ids = pd.read_parquet(ROOT / manifest['training_ids']['path']).mvt_id.to_numpy()
    names = manifest['original_columns']
    table, _ = features.load_features('train')
    X = cv.add_features(data['Xn'][names], ids, table)
    pd.testing.assert_frame_equal(X[names], data['Xn'][names], check_exact=True)
    held, view_ids, view_sizes = [], {}, {}
    for view in VIEWS:
        ranking = ranking_data(view)
        table, _ = features.load_features(view)
        V = cv.add_features(ranking['Vn'][names], ranking['rank_ids'], table)
        pd.testing.assert_frame_equal(V[names], ranking['Vn'][names], check_exact=True)
        held.append(V)
        view_ids[view] = np.asarray(ranking['rank_ids']).copy()
        view_sizes[view] = len(V)
        del ranking
    # Inference rows are concatenated positionally, never joined by overlapping old/final IDs.
    V = pd.concat(fs.widen(held), ignore_index=True)
    cv.ts.FOLDS.setdefault(hs.ids.FULL, features.TRAIN_MONTHS)
    item = dict(X=X, y=data['yn'], Vn=V, tmonth=data['tmonth'], meta=dict(month=hs.ids.FULL),
                raw=pd.DataFrame({'taxi_sec': np.full(len(V), np.nan)}))
    cfg = cv.shared.recipe()
    d = cv.DirectData(item, 'cuda')
    if np.any(d.offset) or np.any(d.val_offset) or d.info['inner_month'] != '2025-12':
        raise ValueError('Direct-target/full-year preprocessing contract changed')
    pre = OUT / 'preprocessing/full_year.pkl'
    pre.parent.mkdir(parents=True, exist_ok=True)
    state = d.preprocessing()
    state['design'] = DESIGN
    if pre.exists():
        resume.retain_saved_preprocessing(pre, state)
    else:
        cv.shared.atomic_pickle(pre, state)
    record = dict(design=DESIGN, config=cfg, encoding=d.info, columns=list(X),
        preprocessing=artifact(pre), prepared_sha256=sha256(OUT / 'prepared.json'),
        view_rows=view_sizes, view_order=VIEWS, label_monitor='ranking labels absent; no epoch selection')
    pre_meta = pre.with_suffix('.json')
    if pre_meta.exists() and read(pre_meta) != record:
        raise ValueError('Saved preprocessing metadata changed')
    write_json(pre_meta, record)
    del data, table, X, V, item, held
    gc.collect()
    for group in GROUPS:
        for k in NETS:
            if completed_member(group, k) is not None:
                continue
            started = time.monotonic()
            model, pred, curve = cv.shared.train(d, np.arange(len(d.y)), group + 1000 * k, EPOCHS[group],
                                                (d.held, d.yv, d.val_offset))
            if not np.isfinite(pred).all() or (pred < 0).any():
                raise ValueError('Invalid full-year network predictions')
            model_path, record_path = member_paths(group, k)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            temp = model_path.with_suffix('.pt.tmp')
            torch.save(dict(state_dict=model.state_dict(), config=cfg, design=DESIGN,
                            preprocessing=str(pre.relative_to(ROOT))), temp)
            temp.replace(model_path)
            predictions, offset = {}, 0
            for view in VIEWS:
                n = view_sizes[view]
                dest = OUT / 'networks' / f'{view}_s{group}_k{k}.parquet'
                save_frame(dest, pd.DataFrame(dict(mvt_id=view_ids[view], base=pred[offset:offset+n].astype(np.float32))))
                predictions[view] = artifact(dest)
                offset += n
            if offset != len(pred):
                raise ValueError('Ranking prediction slicing did not cover both views')
            # There are no ranking labels; persist only actual training diagnostics.
            curve = [{name: value for name, value in row.items() if name != 'rmse'} for row in curve]
            write_json(record_path, dict(design_sha256=sha256(OUT / 'design.json'),
                prepared_sha256=sha256(OUT / 'prepared.json'), seed=group+1000*k, epochs=EPOCHS[group],
                model=artifact(model_path), preprocessing=artifact(pre), predictions=predictions,
                config=cfg, training_curve=curve, minutes=(time.monotonic()-started)/60,
                device=torch.cuda.get_device_name(), torch=str(torch.__version__)))
            log(f'Full-year callsign group{group} k{k}, epoch{EPOCHS[group]} saved for both views')
            del model
            torch.cuda.empty_cache()


def catboost_probability(path, view, ids, raw_airport):
    record = read(path)
    if record['view'] != view or record['seeds'] != GROUPS or record['probability_column'] != 'p':
        raise ValueError('Optional Rome part is not the fixed reviewed three-seed mean')
    source = features.verify_artifact(record['source'])
    if source.name != 'rome_catboost_submit.py':
        raise ValueError('Optional Rome part must come from the separately reviewed production runner')
    frame = pd.read_parquet(features.verify_artifact(record['prediction']))
    np.testing.assert_array_equal(frame.mvt_id, ids)
    probability = frame.p.to_numpy(float)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError('Invalid optional Rome probability')
    if np.any(probability[np.asarray(raw_airport) != 'LIRF'] != 0):
        raise ValueError('Optional Rome probability is nonzero outside Rome')
    combo = ROOT / 'VALIDATION/runs/v3_20261010_research_combo_jan_jul/result.json'
    evidence = read(combo)
    if evidence['comparisons']['combined_vs_control']['decision'] != 'improvement':
        raise ValueError('Combined CV evidence does not pass')
    return probability, dict(record=artifact(path), contents=record, combined_cv=artifact(combo))


def build(view, catboost_record=None):
    manifest = validate_prepared()
    data = ranking_data(view)
    control = parent_control(view, data)
    _, tree, probability, sources = parent_parts(view, data)
    values, network_records = [], []
    for group in GROUPS:
        for k in NETS:
            record = completed_member(group, k)
            if record is None:
                raise ValueError('All15 full-year members must finish before building a candidate')
            frame = pd.read_parquet(ROOT / record['predictions'][view]['path'])
            np.testing.assert_array_equal(frame.mvt_id, data['rank_ids'])
            values.append(frame.base.to_numpy(float))
            network_records.append(artifact(member_paths(group, k)[1]))
    neural = np.mean(values, axis=0)
    extra = None
    arm = 'callsign15'
    if catboost_record is not None:
        airport = np.concatenate([data['months'][m]['raw'].ADEP_mvt.to_numpy() for m in features.RANK_MONTHS[view]])
        candidate, extra = catboost_probability(catboost_record, view, data['rank_ids'], airport)
        probability = .5 * probability + .5 * candidate
        arm = 'callsign15_half_catboost_R20'
    directory = OUT / 'candidates' / f'{view}_{arm}'
    if directory.exists():
        raise FileExistsError('Candidate directory already exists; preserve and inspect it')
    frame, masks = final_frame(data, neural, tree, probability)
    template_path = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
    template = pq.read_table(template_path)
    if template.num_rows != features.EXPECTED_ROWS[view]:
        raise ValueError('Submission template row count changed')
    table = hs.fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    if table.schema != template.schema or not table['MVT_ID_mvt'].equals(template['MVT_ID_mvt']):
        raise ValueError('Submission schema/ID order changed')
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt' and not table[name].equals(template[name]):
            raise ValueError(f'Unexpected template column modification: {name}')
    filled = table['TAXITIME_SEC_mvt'].to_numpy()
    if not np.isfinite(filled).all() or (filled < 0).any():
        raise ValueError('Candidate file has invalid predictions')
    directory.mkdir(parents=True)
    output = directory / 'candidate.parquet'
    pq.write_table(table, output)
    save_frame(directory / 'parts.parquet', frame)
    write_json(directory / 'manifest.json', dict(state='prepared_not_uploaded', view=view, arm=arm,
        design_sha256=sha256(OUT / 'design.json'), prepared_sha256=sha256(OUT / 'prepared.json'),
        source_sha256=sha256(Path(__file__)), output=artifact(output), template=artifact(template_path),
        ranking=artifact(features.RANKING[view]), rows=table.num_rows, exact_parent_control=control,
        network_records=network_records, parent_parts=sources, optional_rome=extra,
        correction_masks=masks, rule_triggered=int(frame.rule_triggered.sum()),
        mean_prediction=float(frame.pred.mean()), cv_evidence=features.evidence()))
    log(f'Prepared {output}; no upload capability exists in this runner')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'fit', 'build'])
    parser.add_argument('--view', choices=VIEWS, default='old')
    parser.add_argument('--catboost-record', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    with threadpool_limits(limits=1, user_api='blas'):
        if args.stage == 'check':
            parent = read(PARENT_NN)
            assert {int(g): v['stop_epoch'] for g, v in parent['groups'].items()} == EPOCHS
            features.evidence()
            log('Full15 passing CV and frozen original full-year epochs verified; no build/fit run')
        elif args.stage == 'prepare':
            prepare()
        elif args.stage == 'fit':
            fit()
        else:
            build(args.view, args.catboost_record)


if __name__ == '__main__':
    main()
