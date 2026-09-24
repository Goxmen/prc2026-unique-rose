"""Step 2: date-held-out propensities and exact within-airport kNN coverage.

Run .venv/bin/python VALIDATION/tools/check_train_ranking_similarity.py.
Resumable outputs live in tmp/train_ranking_step2; only summaries are tracked.
No taxi-out fitting, leaderboard outcomes, losses or validation weights are used.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import inspect
import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).parent))
import train_ranking_similarity as sim

SOURCE = ROOT/'tmp/train_ranking_step1'
CACHE = ROOT/'tmp/train_ranking_step2'
OUT = ROOT/'VALIDATION/reports/train_ranking_step2'
CASES = {'january_year': ('2025-01', '2026-01'),
         'july_year': ('2025-07', '2026-07'),
         'january_february_control': ('2025-01', '2025-02'),
         'july_august_control': ('2025-07', '2025-08')}
VIEWS = {'raw': sim.RAW_GROUPS, 'rich': list(sim.GROUPS)}
PERMUTATIONS = 2
K_VALUES = [1, 5, 10]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def frame_for(months):
    f = pd.concat([pd.read_parquet(SOURCE/f'{m}.parquet').assign(label=i)
                   for i, m in enumerate(months)], ignore_index=True)
    assert f.mvt_id.is_unique and f.mvt_id.notna().all()
    return f.sort_values('mvt_id', kind='stable').reset_index(drop=True)


def verify_checkpoint(path, signature):
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    assert data['design_sha256'] == signature, f'Checkpoint design mismatch: {path}'
    for name, digest in data['files'].items():
        assert sha(CACHE/name) == digest, f'Checkpoint file changed: {name}'
    return data


def _classify_fold(frame, case, fold, view, signature, permute=True):
    stem = f'{case}_fold{fold}_{view}'
    checkpoint = CACHE/f'{stem}.json'
    if (existing := verify_checkpoint(checkpoint, signature)) is not None:
        return existing
    started = time.time()
    role = sim.roles(frame.date, fold)
    fit, cal, test = [frame.loc[role == name].reset_index(drop=True) for name in ['fit','cal','test']]
    columns = ['airport'] + sum([sim.GROUPS[g] for g in VIEWS[view]], [])
    X, encoding = sim.model_matrix(fit, columns)
    Xcal, _ = sim.model_matrix(cal, columns, encoding)
    Xtest, _ = sim.model_matrix(test, columns, encoding)
    weights = sim.balanced_weights(fit.airport, fit.label)
    data = lgb.Dataset(X, label=fit.label.to_numpy(), weight=weights,
                       categorical_feature=[c for c in columns if c in sim.CATEGORICAL])
    model = lgb.train(sim.MODEL_PARAMS, data, num_boost_round=sim.ROUNDS)
    cal_logits = model.predict(Xcal, raw_score=True, num_threads=6)
    calibration = sim.fit_calibrator(cal_logits, cal.label, sim.balanced_weights(cal.airport, cal.label))
    logits = model.predict(Xtest, raw_score=True, num_threads=6)
    probabilities = sim.calibrated(logits, calibration)
    prediction = test[['mvt_id','date','airport','label']].copy()
    prediction['fold'] = fold
    prediction['logit'] = logits
    prediction['propensity'] = probabilities
    predictions_path = CACHE/f'{stem}.parquet'
    prediction.to_parquet(predictions_path, index=False)
    model_path = CACHE/f'{stem}.txt'
    model.save_model(str(model_path))
    scores = []
    for airport, rows in test.groupby('airport').indices.items():
        y, p = test.label.to_numpy()[rows], probabilities[rows]
        metric = sim.classification_metrics(y, p, sim.balanced_weights(test.airport.iloc[rows], y))
        scores.append(dict(airport=airport, n_source=int((y==0).sum()), n_target=int((y==1).sum()),
                           raw_auc=float(roc_auc_score(y, logits[rows])), **metric))
    importance = []
    if permute and view == 'rich':
        for repeat in range(PERMUTATIONS):
            rng = np.random.default_rng(sim.SEED + 100*fold + repeat)
            order = np.arange(len(test))
            for _, rows in test.groupby('airport').indices.items():
                order[rows] = rng.permutation(rows)
            for group, group_columns in sim.GROUPS.items():
                shuffled = Xtest.copy()
                for c in group_columns:
                    shuffled[c] = Xtest[c].iloc[order].array
                perm_logits = model.predict(shuffled, raw_score=True, num_threads=6)
                for metric in scores:
                    rows = test.index[test.airport.eq(metric['airport'])].to_numpy()
                    auc = float(roc_auc_score(test.label.to_numpy()[rows], perm_logits[rows]))
                    importance.append(dict(airport=metric['airport'], group=group, repeat=repeat,
                                           auc_drop=metric['raw_auc']-auc))
    date_roles = {name: sorted(str(x.date()) for x in part.date.unique())
                  for name, part in [('fit', fit), ('cal', cal), ('test', test)]}
    result = dict(case=case, fold=fold, view=view, design_sha256=signature,
                  fit_rows=len(fit), cal_rows=len(cal), test_rows=len(test),
                  date_roles=date_roles, calibration=calibration, scores=scores,
                  importance=importance, seconds=time.time()-started,
                  files={p.name: sha(p) for p in [predictions_path, model_path]})
    write_json(checkpoint, result)
    print(f'classifier {stem}: {len(test):,} held-out rows; {result["seconds"]:.1f}s', flush=True)
    return result


def _knn_fold(frame, case, airport, fold, signature):
    stem = f'knn_{case}_{airport}_fold{fold}'
    checkpoint = CACHE/f'{stem}.json'
    if (existing := verify_checkpoint(checkpoint, signature)) is not None:
        return existing
    started = time.time()
    # Apply the date policy globally before selecting an airport.
    role = sim.roles(frame.date, fold)
    local = frame.assign(role=role).loc[frame.airport.eq(airport)].copy()
    reference = local[local.role.eq('fit') & local.label.eq(0)].sort_values('mvt_id').reset_index(drop=True)
    cal = local[local.role.eq('cal') & local.label.eq(0)]
    test = local[local.role.eq('test')]
    query = pd.concat([cal, test], ignore_index=True)
    assert len(reference) >= 10 and len(cal) and len(test)
    assert not set(reference.mvt_id) & set(query.mvt_id)
    assert reference.airport.nunique() == query.airport.nunique() == 1
    neighbours = sim.nearest(reference, query, k=max(K_VALUES))
    cal_mask = query.role.eq('cal').to_numpy()
    result_frame = query.loc[~cal_mask, ['mvt_id','date','airport','label']].copy().reset_index(drop=True)
    result_frame['fold'] = fold
    thresholds = {}
    for view, nearest in neighbours.items():
        for k in K_VALUES:
            d = nearest['distance'][:,k-1]
            threshold = float(np.quantile(d[cal_mask], .95))
            thresholds[f'{view}_k{k}'] = threshold
            result_frame[f'{view}_d{k}'] = d[~cal_mask]
            result_frame[f'{view}_far{k}'] = d[~cal_mask] > threshold
        for k in range(max(K_VALUES)):
            result_frame[f'{view}_neighbour{k+1}'] = reference.mvt_id.to_numpy()[nearest['index'][~cal_mask,k]]
    output_path = CACHE/f'{stem}.parquet'
    result_frame.to_parquet(output_path, index=False)
    result = dict(case=case, airport=airport, fold=fold, design_sha256=signature,
                  n_reference=len(reference), n_cal=len(cal), n_test=len(test),
                  reference_dates=sorted(str(x.date()) for x in reference.date.unique()),
                  calibration_dates=sorted(str(x.date()) for x in cal.date.unique()),
                  thresholds=thresholds, seconds=time.time()-started,
                  files={output_path.name: sha(output_path)})
    write_json(checkpoint, result)
    print(f'kNN {case} {airport} fold {fold}: {len(query):,} queries/{len(reference):,} donors; {result["seconds"]:.1f}s', flush=True)
    return result


def block_rate_difference(frame, flag, seed=42, repeats=1000):
    """Fixed-score sensitivity: resample three-day blocks within each population."""
    rng = np.random.default_rng(seed)
    values = []
    for label in [0,1]:
        f = frame[frame.label.eq(label)].copy()
        f['block'] = (f.date.dt.day-1)//3
        g = f.groupby('block')[flag].agg(['sum','count'])
        draw = rng.integers(0, len(g), size=(repeats, len(g)))
        values.append(100*g['sum'].to_numpy()[draw].sum(axis=1)/g['count'].to_numpy()[draw].sum(axis=1))
    return list(map(float, np.quantile(values[1]-values[0], [.025,.975])))


def summarize(signature):
    fold_rows, importance_rows = [], []
    for case in CASES:
        for fold in range(4):
            for view in VIEWS:
                result = verify_checkpoint(CACHE/f'{case}_fold{fold}_{view}.json', signature)
                assert result is not None, (case, fold, view)
                for metric in result['scores']:
                    fold_rows.append(dict(case=case, fold=fold, view=view,
                                          **{k:v for k,v in metric.items() if k != 'reliability'}))
                for row in result['importance']:
                    importance_rows.append(dict(case=case, fold=fold, view=view, **row))
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT/'classifier_folds.csv', index=False)
    metrics = folds.groupby(['case','view','airport']).agg(
        mean_auc=('raw_auc','mean'), min_fold_auc=('raw_auc','min'), max_fold_auc=('raw_auc','max'),
        mean_calibrated_auc=('auc','mean'), mean_balanced_brier=('brier','mean'),
        mean_ece=('ece','mean'), n_source=('n_source','sum'), n_target=('n_target','sum')).reset_index()
    metrics.to_csv(OUT/'classifiers.csv', index=False)
    importance = pd.DataFrame(importance_rows)
    importance.to_csv(OUT/'permutation_folds.csv', index=False)
    importance.groupby(['case','view','airport','group']).agg(mean_auc_drop=('auc_drop','mean'),
        min_auc_drop=('auc_drop','min'), max_auc_drop=('auc_drop','max')).reset_index().to_csv(OUT/'permutation.csv', index=False)
    knn_rows, propensity_rows, slices = [], [], []
    for case in ['january_year','july_year']:
        original = frame_for(CASES[case])
        propensities = {}
        for view in VIEWS:
            pieces = [pd.read_parquet(CACHE/f'{case}_fold{fold}_{view}.parquet') for fold in range(4)]
            propensities[view] = pd.concat(pieces, ignore_index=True)
            assert propensities[view].mvt_id.is_unique and set(propensities[view].mvt_id) == set(original.mvt_id)
        for airport in sorted(original.airport.unique()):
            for view, propensity in propensities.items():
                airport_p = propensity[propensity.airport.eq(airport)]
                for label, g in airport_p.groupby('label'):
                    propensity_rows.append(dict(case=case, view=view, airport=airport, label=int(label), n=len(g),
                        p10=float(g.propensity.quantile(.1)), p50=float(g.propensity.median()),
                        p90=float(g.propensity.quantile(.9)), above_08_pct=float(100*g.propensity.gt(.8).mean())))
            pieces = []
            for fold in range(4):
                checkpoint = verify_checkpoint(CACHE/f'knn_{case}_{airport}_fold{fold}.json', signature)
                assert checkpoint is not None
                pieces.append(pd.read_parquet(CACHE/next(iter(checkpoint['files']))))
            knn = pd.concat(pieces, ignore_index=True)
            assert knn.mvt_id.is_unique
            assert set(knn.mvt_id) == set(original.loc[original.airport.eq(airport), 'mvt_id'])
            assert knn.label.value_counts().to_dict() == airport_p.label.value_counts().to_dict()
            for view in VIEWS:
                for k in K_VALUES:
                    flag = f'{view}_far{k}'
                    rates = knn.groupby('label')[flag].mean()*100
                    interval = block_rate_difference(knn, flag)
                    knn_rows.append(dict(case=case, airport=airport, view=view, k=k,
                        n_source=int(knn.label.eq(0).sum()), n_target=int(knn.label.eq(1).sum()),
                        source_far_pct=float(rates[0]), target_far_pct=float(rates[1]),
                        delta_pp=float(rates[1]-rates[0]), delta_p025=interval[0], delta_p975=interval[1]))
            # Save the step-1 leads as slices without accessing any model errors.
            merged = original.merge(knn, on=['mvt_id','date','airport','label'], how='inner', validate='one_to_one')
            for name, mask in [('all', np.ones(len(merged), bool)),
                               ('no_flight_id', merged.no_flight_id.eq(1)),
                               ('schedule_gap_over4h', merged.takeoff_minus_sched_min.gt(240)),
                               ('snow', merged.wx_snow.eq(1))]:
                for label in [0,1]:
                    g = merged.loc[mask & merged.label.eq(label)]
                    slices.append(dict(case=case, airport=airport, slice=name, label=label, n=len(g),
                        raw_far_pct=float(100*g.raw_far5.mean()) if len(g) else None,
                        rich_far_pct=float(100*g.rich_far5.mean()) if len(g) else None))
    pd.DataFrame(knn_rows).to_csv(OUT/'knn.csv', index=False)
    pd.DataFrame(propensity_rows).to_csv(OUT/'propensities.csv', index=False)
    pd.DataFrame(slices).to_csv(OUT/'slices.csv', index=False)
    null = verify_checkpoint(CACHE/'january_null_fold0_rich.json', signature)
    assert null is not None
    write_json(OUT/'checks.json', dict(design_sha256=signature,
        null_mean_auc=float(np.mean([s['raw_auc'] for s in null['scores']])),
        null_airports={s['airport']:s['raw_auc'] for s in null['scores']},
        null_limit='Random labels are a sanity check, not proof of absence of calendar leakage.',
        script_sha256=sha(Path(__file__)), utility_sha256=sha(Path(sim.__file__)),
        outputs={p.name:sha(p) for p in sorted(OUT.glob('*.csv'))}))
    print('Summary complete', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['all','classifiers','knn','summary'], default='all')
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    source_manifest = json.loads((SOURCE/'manifest.json').read_text())
    months = sorted(set(sum([list(p) for p in CASES.values()], [])))
    for month in months:
        assert sha(SOURCE/f'{month}.parquet') == source_manifest['files'][month]
    design = dict(cases=CASES, model=sim.MODEL_PARAMS, rounds=sim.ROUNDS, views=VIEWS,
        feature_groups=sim.GROUPS, categoricals=sim.CATEGORICAL, excluded=sorted(sim.EXCLUDED),
        day_blocks='1-8 / 9-16 / 17-24 / 25-end; test block i, calibration i+1 modulo 4, others fit; one-day embargo',
        priors='Equal airport weight; equal 2025/target class weight within each airport. Diagnostic propensities, not new validation weights.',
        permutation='Two held-out joint-group shuffles within airport; association, not causal attribution.',
        knn='Exact, same-airport/matched-month source-fit donors; median/IQR scaling (fallback p90-p10, then 1), signed-log clock gaps, explicit missing indicators, categorical mismatch, equal group weights. Primary fifth-neighbour distance; first/tenth sensitivity.',
        threshold='Each airport/fold/view/k uses the source calibration dates 95th distance percentile. Report both untouched source-test and target-test exceedances.',
        bootstrap='1000 resamples of three-day blocks within each year; models, donors and thresholds fixed. Descriptive intervals, no multiple-testing correction.',
        k_values=K_VALUES, primary_k=5, seed=sim.SEED, calibration='Nonnegative-slope Platt scaling on separate dates; report held-out Brier and five-bin ECE.',
        input_sha256={m:source_manifest['files'][m] for m in months},
        utility_sha256=sha(Path(sim.__file__)),
        classifier_code_sha256=hashlib.sha256(inspect.getsource(_classify_fold).encode()).hexdigest(),
        knn_code_sha256=hashlib.sha256(inspect.getsource(_knn_fold).encode()).hexdigest(),
        constraints='No taxi-out labels, prediction errors, leaderboard scores or weighting modifications. Reference comparisons are not independent replications.')
    signature = hashlib.sha256(json.dumps(design, sort_keys=True).encode()).hexdigest()
    path = OUT/'design.json'
    if path.exists():
        assert json.loads(path.read_text()) == json.loads(json.dumps(design)), 'Design changed: retain previous run and create a new output directory.'
    else:
        write_json(path, design)
    if args.mode in ['all','classifiers']:
        for case, months in CASES.items():
            frame = frame_for(months)
            for fold in range(4):
                for view in VIEWS:
                    _classify_fold(frame, case, fold, view, signature)
        frame = frame_for(CASES['january_year'])
        frame['label'] = np.random.default_rng(142).integers(0,2,len(frame))
        _classify_fold(frame, 'january_null', 0, 'rich', signature, permute=False)
    if args.mode in ['all','knn']:
        for case in ['january_year','july_year']:
            frame = frame_for(CASES[case])
            def run_airport(airport):
                for fold in range(4):
                    _knn_fold(frame, case, airport, fold, signature)
            # Exact distance kernels release the GIL; three airports fit easily
            # in memory. This runs after the classifiers, never alongside fits.
            with ThreadPoolExecutor(max_workers=3) as executor:
                jobs = [executor.submit(run_airport, airport) for airport in sorted(frame.airport.unique())]
                for job in as_completed(jobs):
                    job.result()
    if args.mode in ['all','summary']:
        summarize(signature)


if __name__ == '__main__':
    main()
