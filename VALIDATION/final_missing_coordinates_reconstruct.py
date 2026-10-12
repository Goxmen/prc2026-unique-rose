"""Recover absent v43/v44 ordinary boosters, reusing the exact saved classifier.

The original full-year training packet retains these112 inputs unchanged. The
two fits use recorded original parameters, with separate exact old-vector gates.
No classifier, neural network, calibration or retained model is fitted again.
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import callsign_submit as parent
import stand_coords_submit as original
import tuned_submit
from records import runtime_identity, sha256, write_json

OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_missing_coordinates_reconstruct'
FOLDERS = {43: original.run.DEFAULT / 'submission_v43', 44: tuned_submit.ts.DEFAULT / 'submission_v44'}
CLASSIFIER = parent.hs.nis.V46 / 'models/refit2025__classifier__42.txt'
CLASSIFIER_SHA = '9501e34314f919425cc74cc345eef9562e895f1516002847fba1583afa97d31e'
DESIGN = dict(versions=[43, 44], rounds={'43': 800, '44': 5000}, seed=42, threads=8,
    ordinary_rows=2051327, ordinary_features=112, classifier_features=68,
    training_source='Retained exact ideas full-year Xn: select original112 columns, preserving categorical metadata and row order',
    missing_fits='Two original ordinary boosters only; retained classifier hash is exact for both',
    success='For each version all344841 original integers exact before its final inference',
    failure='Preserve saved model and mismatch; close affected version without final inference or sweep',
    no_classifier_fit=True, no_neural_fit=True, no_calibration_fit=True, no_upload=True)


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha256(path))


def originals():
    return {v: read(folder / f'submission_v{v}.json') for v, folder in FOLDERS.items()}


def freeze():
    assert not (OUT / 'identity.json').exists()
    prior = originals(); refs = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')
    assert sha256(CLASSIFIER) == CLASSIFIER_SHA
    paths = {Path(__file__), Path(parent.__file__), parent.hs.INPUTS, parent.MEMBERSHIP, CLASSIFIER,
        parent.hs.nis.V46 / 'submission_v46.json', ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json',
        ROOT / 'VALIDATION/runs/v3_20261008_open_meteo_submit/work/prepared.json'}
    assert sha256(parent.hs.INPUTS) == read(ROOT / 'VALIDATION/runs/v3_20261008_open_meteo_submit/work/prepared.json')['original_inputs_sha256']
    for v, manifest in prior.items():
        assert manifest['training']['ordinary_rows'] == 2051327
        assert manifest['classifier_rounds'] == 300
        assert manifest['model']['refit2025__classifier__42']['artifacts']['models/refit2025__classifier__42.txt'] == CLASSIFIER_SHA
        for rel, digest in manifest['identity']['signature'].items():
            assert sha256(ROOT / rel) == digest, rel
            paths.add(ROOT / rel)
        paths.update([FOLDERS[v] / f'submission_v{v}.json', FOLDERS[v] / 'categories.json', FOLDERS[v] / 'refit.json'])
        ref = refs[str(v)]; assert sha256(ROOT / ref['file']) == ref['sha256'] == manifest['sha256']
        paths.add(ROOT / ref['file'])
    features = prior[43]['model']['refit2025__ordinary__42']['features']
    assert len(features) == 112 and features == prior[44]['model']['refit2025__ordinary__42']['features']
    params = {43: prior[44]['model']['refit2025__ordinary_v43__42']['params'], 44: prior[44]['params']}
    for p in params.values():
        assert p['num_threads'] == 8 and p['seed'] == 42
    for view in ['old', 'final']:
        packet = parent.fs.WORK / f'inputs_{view}.pkl'
        receipt = parent.fs.OUT / f'inputs_{view}.json'
        assert sha256(packet) == read(receipt)['sha256']; paths.update([packet, receipt])
        paths.add(ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet'))
    import glob
    from common import TRAIN_GLOB
    paths.update(Path(p) for p in glob.glob(TRAIN_GLOB))
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'identity.json', dict(design=DESIGN, sources=[art(p) for p in sorted(paths)],
        runtime=runtime_identity(), params=params, ordinary_features=features,
        classifier_features=prior[43]['model']['refit2025__classifier__42']['features'],
        original_source_bytes_exact=True, retained_classifier_hash_exact=True,
        original_hashes={v: m['model']['refit2025__ordinary__42']['artifacts']['models/refit2025__ordinary__42.txt'] for v, m in prior.items()}))
    print(json.dumps(dict(stage='frozen', sources=len(paths), ordinary_features=len(features), fits_started=0)))


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['design'] == DESIGN
    for item in identity['sources']:
        assert sha256(ROOT / item['path']) == item['sha256'], item['path']
    assert identity['runtime']['versions'] == runtime_identity()['versions']
    return identity


def prepare():
    identity = guard(); assert not (OUT / 'preflight.json').exists()
    assert not (OUT / 'training.pkl').exists()
    data = pd.read_pickle(parent.hs.INPUTS)
    ids = parent.training_ids(data)
    X = data['Xn'][identity['ordinary_features']].copy(); y = np.asarray(data['yn'], float)
    assert len(X) == len(ids) == 2051327
    cats = read(FOLDERS[43] / 'categories.json')
    assert cats == read(FOLDERS[44] / 'categories.json')
    for name, values in cats.items():
        if name in X:
            assert isinstance(X[name].dtype, pd.CategoricalDtype)
            assert X[name].cat.categories.tolist() == values, name
    # Independent membership includes both the four-hour-rule exclusion and the
    # schedule-copy exclusion; no classifier labels or weights are re-estimated.
    from common import TRAIN_GLOB
    check = pd.DataFrame(dict(mvt_id=ids, y=y))
    with duckdb.connect(config={'threads': 1, 'memory_limit': '512MB'}) as con:
        con.register('check_rows', check)
        con.execute("""CREATE TEMP TABLE raw AS SELECT MVT_ID_mvt mvt_id,TAXITIME_SEC_mvt y,
            ADEP_mvt airport,FLIGHT_ID_mvt plan_id,
            epoch(MVT_TIME_UTC_mvt-SCHED_TIME_UTC_mvt) gap FROM read_parquet(?) WHERE PHASE_mvt='DEP'""", [TRAIN_GLOB])
        n, bad = con.execute("""WITH eligible AS (SELECT * FROM raw WHERE y BETWEEN 1 AND 200000
            AND NOT (airport='LIRF' AND plan_id IS NULL AND gap>=14400)
            AND NOT (airport='LIRF' AND abs(gap-y)<=60))
            SELECT count(*),sum(CASE WHEN e.mvt_id IS NULL OR c.mvt_id IS NULL OR e.y IS DISTINCT FROM c.y THEN 1 ELSE 0 END)
            FROM eligible e FULL JOIN check_rows c USING(mvt_id)""").fetchone()
        assert n == len(X) and bad == 0, (n, bad)
    old_p = pd.concat([pd.DataFrame(dict(mvt_id=d['raw'].mvt_id.to_numpy(), p=d['p'])) for d in data['months'].values()], ignore_index=True)
    assert old_p.mvt_id.is_unique and len(old_p) == 344841
    old_p.to_parquet(OUT / 'original_classifier_old.parquet', index=False)
    pd.to_pickle(dict(X=X, y=y), OUT / 'training.pkl')
    pd.DataFrame(dict(mvt_id=ids, original_index=X.index.to_numpy())).to_parquet(OUT / 'training_ids.parquet', index=False)
    write_json(OUT / 'preflight.json', dict(identity=art(OUT / 'identity.json'), training=art(OUT / 'training.pkl'),
        training_ids=art(OUT / 'training_ids.parquet'), old_classifier=art(OUT / 'original_classifier_old.parquet'),
        rows=len(X), columns=list(X), original_order_labels_airport_stand_runway_month_exact=True,
        independent_SQL_rule_and_copy_exclusion_membership_exact=True,
        original_category_lists_exact=True, original112_values_reused_without_change=True, fits_started=0))
    print(json.dumps(dict(stage='preflight_pass', rows=len(X), features=len(X.columns))))


def fit(version):
    identity = guard(); preflight = read(OUT / 'preflight.json')
    assert preflight['identity'] == art(OUT / 'identity.json') and preflight['training'] == art(OUT / 'training.pkl')
    folder = OUT / f'v{version}'; folder.mkdir(exist_ok=True)
    assert not (folder / 'model.txt').exists() and not (folder / 'trained.json').exists()
    data = pd.read_pickle(OUT / 'training.pkl'); started = time.monotonic()
    model = original.signals._train(identity['params'][str(version)], data['X'], data['y'],
        DESIGN['rounds'][str(version)], f'missing v{version}')
    model.save_model(str(folder / 'model.txt'))
    write_json(folder / 'trained.json', dict(identity=art(OUT / 'identity.json'), preflight=art(OUT / 'preflight.json'),
        model=art(folder / 'model.txt'), columns=model.feature_name(), trees=model.num_trees(),
        original_model_hash_exact=sha256(folder / 'model.txt')==identity['original_hashes'][str(version)],
        seconds=time.monotonic()-started, runtime=runtime_identity(), saved_before_inference=True))
    print(json.dumps(dict(stage='saved_model', version=version, seconds=time.monotonic()-started)))


def infer(version):
    identity = guard(); folder = OUT / f'v{version}'
    assert not (folder / 'result.json').exists()
    trained = read(folder / 'trained.json')
    assert trained['identity'] == art(OUT / 'identity.json') and trained['model'] == art(folder / 'model.txt')
    ordinary = lgb.Booster(model_file=str(folder / 'model.txt'))
    reload = lgb.Booster(model_file=str(folder / 'model.txt'))
    classifier = lgb.Booster(model_file=str(CLASSIFIER))
    assert classifier.feature_name() == identity['classifier_features']
    prior = originals()[version]
    ref = read(ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json')[str(version)]
    results = {}
    for view in ['old', 'final']:
        assert not (folder / f'{view}.json').exists(), 'Preserve completed inference; inspect before resume'
        data = parent.ranking_data(view); cached = OUT / f'original_classifier_{view}.parquet'
        probability = pd.read_parquet(cached).set_index('mvt_id').p if cached.exists() else None
        frames, probability_parts = [], []
        for month, item in data['months'].items():
            V, raw = item['V'], item['raw']
            base = original.signals.ordinary_prediction(ordinary, V, 'clock', 8, original.ORDINARY)
            np.testing.assert_array_equal(base, original.signals.ordinary_prediction(reload, V, 'clock', 8, original.ORDINARY))
            if probability is None:
                p = original.signals.substitution_probability(classifier, V, raw, 'clock', 8, original.cv.CLASSIFIER)
                probability_parts.append(pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(), p=p)))
            else:
                p = probability.loc[raw.mvt_id].to_numpy()
            out = original.signals.combine(base, p, V, raw, late_mean=prior['training']['parameters']['late_mean'], dayshift_weight=None)
            frame, _ = original.v42.parent.apply_corrections(out['final'], raw, V, prior['training']['corrections'], p)
            frame['month'], frame['ordinary'], frame['rule_triggered'] = month, base, out['rule_triggered']
            frames.append(frame)
        if probability is None:
            pd.concat(probability_parts, ignore_index=True).to_parquet(cached, index=False)
            write_json(cached.with_suffix('.json'), dict(model=art(CLASSIFIER), inputs=art(parent.fs.WORK / f'inputs_{view}.pkl'),
                prediction=art(cached), reason='Prior final legacy exporter retained only final predictions, not this classifier component'))
        template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
        ids = pq.read_table(template, columns=['MVT_ID_mvt'])['MVT_ID_mvt'].to_numpy()
        frame = pd.concat(frames, ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
        frame['submitted'] = np.rint(np.maximum(frame.pred, 0))
        assert frame.mvt_id.is_unique and np.isfinite(frame[['pred', 'submitted']]).all().all()
        path = folder / f'{view}_v{version}.parquet'; frame.to_parquet(path, index=False)
        report = dict(prediction=art(path), rows=len(frame), months=frame.month.value_counts().sort_index().to_dict(),
            all_ordinary_model_reload_predictions_exact=True, classifier=art(cached), template_order_exact=True)
        if view == 'old':
            old = pd.read_parquet(ROOT / ref['file']).set_index('MVT_ID_mvt').loc[ids, 'TAXITIME_SEC_mvt'].to_numpy()
            delta = frame.submitted.to_numpy()-old
            report.update(exact_original_integer_replay=bool(np.all(delta == 0)), changed_rows=int(np.count_nonzero(delta)),
                rms_difference=float(np.sqrt(np.mean(delta**2))), maximum_absolute_difference=float(np.max(abs(delta))))
        results[view] = report; write_json(folder / f'{view}.json', report)
        del data, frames, frame; gc.collect()
        if view == 'old' and not report['exact_original_integer_replay']:
            write_json(folder / 'result.json', dict(state='closed_approximation_old_replay_failed', views=results,
                identity=art(OUT / 'identity.json'), no_final_inference=True, no_sweep=True, no_upload=True))
            print(json.dumps(dict(stage='closed_approximation', version=version, result=report)), flush=True); return
        print(json.dumps(dict(stage=view+'_complete', version=version, rows=report['rows'])), flush=True)
    write_json(folder / 'result.json', dict(state='exact_old_replay_final_inference_complete', views=results,
        identity=art(OUT / 'identity.json'), no_upload=True, independent_final_audit_pending=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'prepare', 'fit', 'infer'])
    parser.add_argument('--version', type=int, choices=[43, 44])
    args = parser.parse_args()
    with threadpool_limits(8):
        if args.stage in ['fit', 'infer']:
            assert args.version
            globals()[args.stage](args.version)
        else:
            globals()[args.stage]()
