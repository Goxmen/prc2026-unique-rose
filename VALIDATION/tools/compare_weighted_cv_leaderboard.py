"""Compare the existing raw-feature weights with recorded, actually scored submissions.

No taxi-out training or leaderboard tuning. Run from any directory with the project
venv. Exact prediction artifacts are required; aggregate-only runs are excluded.
"""

import gc
import hashlib
import json
from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION/tools'), str(ROOT / 'VALIDATION')]
import analyze_adversarial_shift as adv
from reporting import group_loss, load_losses

REPORTS = ROOT / 'VALIDATION/reports'
CACHE = ROOT / 'tmp/weighted_cv_leaderboard_20260920'
PLAN = REPORTS / 'weighted_cv_leaderboard_design.json'
OUTPUT = REPORTS / 'weighted_cv_leaderboard.json'
REPORT = REPORTS / 'WEIGHTED_CV_LEADERBOARD.md'
EARLY = REPORTS / 'leaderboard_selected_20260916.json'
LATEST = ROOT / 'VALIDATION/runs/v3_20260919_artifact_experiments/submission_v38.json'
PAIRS = [(13, 12), (14, 13), (15, 13), (16, 14), (16, 15)]
METHODS = ['unweighted', 'weighted', 'weighted_clipped_p99']


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def freeze(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing design/cache differs: {path}')
    else:
        write(path, value)


def roster():
    rows = []
    for item in read(EARLY)['submissions']:
        directory = ROOT / 'VALIDATION/runs' / item['validation_run']
        assert digest(directory / 'result.json') == item['validation_record_sha256']
        record = read(directory / 'result.json')
        assert item['state'] == 'uploaded'
        assert item['official_result']['status'] == 'Succeeded'
        assert item['official_result']['used_pairs'] == 344841
        assert item['score'] == item['official_result']['score']
        rows.append(dict(version=item['version'], name=item['codename'],
                         official=float(item['score']), run=item['validation_run'],
                         column=record.get('prediction_column', 'pred'),
                         expected_scores=record['report']['scores']))
    item = read(LATEST)
    for name, expected in item['validation_records'].items():
        assert digest(ROOT / name) == expected
    assert item['official_result']['status'] == 'Succeeded'
    assert item['official_result']['used_pairs'] == 344841
    assert item['score'] == item['official_result']['score']
    rows.append(dict(version=item['version'], name='artifact combined',
                     official=float(item['score']), run=item['validation_run'],
                     column=item['column'], expected_scores=item['validation_scores']))
    assert [r['version'] for r in rows] == [12, 13, 14, 15, 16, 38]
    for row in rows:
        directory = ROOT / 'VALIDATION/runs' / row['run']
        record = read(directory / 'result.json')
        for fold in sum(adv.GROUP_FOLDS.values(), []):
            for seed in record['config']['seeds']:
                assert (directory / record['folds'][fold]['predictions'][str(seed)]).is_file()
    return rows


def weights(data_identity):
    for entry in data_identity:
        assert digest(ROOT / 'data' / entry['file']) == entry['sha256']
    signature = dict(implementation_sha256=digest(Path(adv.__file__)),
                     features=adv.FEATURES, params=adv.PARAMS, rounds=adv.ROUNDS,
                     seed=adv.SEED, sources=adv.GROUP_SOURCE, data_identity=data_identity)
    freeze(CACHE / 'weight_spec.json', signature)
    paths = {name: CACHE / f'{name}.parquet' for name in adv.GROUP_SOURCE}
    metadata_path = CACHE / 'weights.json'
    if metadata_path.exists():
        metadata = read(metadata_path)
        assert all(digest(path) == metadata[name]['sha256'] for name, path in paths.items())
        return {name: pd.read_parquet(path) for name, path in paths.items()}, metadata
    print('Loading raw training/ranking features', flush=True)
    frame = adv.load()
    target = frame.year.eq('2026')
    metadata, result = {}, {}
    # Preserve the four fitted discriminators as well as the reusable weights.
    original_fit = adv.fit
    for name, months in adv.GROUP_SOURCE.items():
        models = []

        def fit_and_save(train, label, seed=adv.SEED):
            model = original_fit(train, label, seed)
            path = CACHE / f'{name}_half{len(models)}.txt'
            model.save_model(str(path))
            models.append(dict(path=str(path.relative_to(ROOT)), sha256=digest(path)))
            return model

        adv.fit = fit_and_save
        try:
            ratios, counts, calibration = adv.cross_fitted(
                frame, frame.ym.isin(months), target, name)
        finally:
            adv.fit = original_fit
        assert ratios.index.is_unique and ratios.index.notna().all()
        assert np.isfinite(ratios.ratio).all() and (ratios.ratio > 0).all()
        ratios.to_parquet(paths[name])
        result[name] = ratios
        metadata[name] = dict(path=str(paths[name].relative_to(ROOT)),
                              sha256=digest(paths[name]), rows=len(ratios),
                              class_counts=counts, models=models, calibration=calibration,
                              ess_fraction=float(ratios.ratio.sum()**2 /
                                                 (ratios.ratio.pow(2).sum() * len(ratios))))
    write(metadata_path, metadata)
    return result, metadata


def score(row, ratios, populations, reference_record):
    directory = ROOT / 'VALIDATION/runs' / row['run']
    record = read(directory / 'result.json')
    # Later models also record external weather; the underlying flight data
    # must still match exactly, while additional feature sources are allowed.
    actual_data = {item['file']: item for item in record['data_identity']}
    assert all(actual_data.get(item['file']) == item
               for item in reference_record['data_identity'])
    assert record['ranking_weights'] == reference_record['ranking_weights']
    assert record['config']['seeds'] == [42, 142, 242]
    record['prediction_column'] = row['column']
    wanted = sum(adv.GROUP_FOLDS.values(), [])
    record['folds'] = {f: record['folds'][f] for f in wanted}
    files = {p: digest(directory / p) for fold in record['folds'].values()
             for p in fold['predictions'].values()}
    losses = load_losses(record, directory)
    modes = {mode: {method: {} for method in METHODS}
             for mode in ['standard', '42', '142', '242']}
    groups = {}
    for name, folds in adv.GROUP_FOLDS.items():
        loss = group_loss(losses, folds)
        identity = loss.meta[['mvt_time', 'airport', 'y']].copy()
        identity['airport'] = identity.airport.astype(str)
        identity_hash = hashlib.sha256(
            pd.util.hash_pandas_object(identity, index=True).values.tobytes()).hexdigest()
        if name in populations:
            assert populations[name] == identity_hash, 'Submission flight IDs/labels differ'
        populations[name] = identity_hash
        assert len(ratios[name]) == len(loss.meta)
        assert loss.meta.index.is_unique and loss.meta.index.isin(ratios[name].index).all()
        w = ratios[name].loc[loss.meta.index, 'ratio'].to_numpy()
        wc = np.minimum(w, np.quantile(w, .99))
        for mode in modes:
            sq = loss.sq.mean(axis=0) if mode == 'standard' else loss.sq[loss.seeds.index(int(mode))]
            for method, weight in zip(METHODS, [np.ones(len(w)), w, wc]):
                modes[mode][method][name] = float(np.sqrt(np.average(sq, weights=weight)))
        np.testing.assert_allclose(modes['standard']['unweighted'][name],
                                   row['expected_scores'][name], rtol=0, atol=1e-8)
        groups[name] = dict(rows=len(w), identity_sha256=identity_hash)
    jan = record['ranking_weights']['jan_fraction']
    for mode in modes.values():
        for scores in mode.values():
            scores['R_combined'] = float(np.sqrt(
                jan * scores['R_A']**2 + (1 - jan) * scores['R_B_long']**2))
    np.testing.assert_allclose(modes['standard']['unweighted']['R_combined'],
                               row['expected_scores']['R_combined'], rtol=0, atol=1e-8)
    if row['version'] == 38:
        prior = read(REPORTS / 'adversarial_shift.json')['accounting']
        for method, key in [('weighted', 'cv_reweighted'),
                            ('weighted_clipped_p99', 'cv_reweighted_clipped_p99')]:
            np.testing.assert_allclose(modes['standard'][method]['R_combined'],
                                       prior[key], rtol=0, atol=1e-8)
    print(f"v{row['version']}: " + ', '.join(
        f'{method}={scores["R_combined"]:.4f}' for method, scores in modes['standard'].items())
        + f", official={row['official']:.4f}", flush=True)
    return {**row, 'modes': modes, 'groups': groups, 'prediction_sha256': files,
            'record_sha256': digest(directory / 'result.json')}


def compare(rows, mode='standard', versions=None, pairs=PAIRS):
    by_version = {r['version']: r for r in rows}
    versions = list(by_version) if versions is None else versions
    result = {}
    for method in METHODS:
        cv = {v: by_version[v]['modes'][mode][method]['R_combined'] for v in versions}
        gaps = {v: abs(cv[v] - by_version[v]['official']) for v in versions}
        differences = []
        for candidate, parent in pairs:
            delta_lb = by_version[candidate]['official'] - by_version[parent]['official']
            delta_cv = cv[candidate] - cv[parent]
            differences.append(dict(candidate=candidate, parent=parent, delta_lb=delta_lb,
                                    delta_cv=delta_cv, direction_correct=bool(np.sign(delta_cv) == np.sign(delta_lb)),
                                    absolute_change_error=abs(delta_cv - delta_lb)))
        material = [r for r in differences if abs(r['delta_lb']) >= 1.0]
        result[method] = dict(level_mae=float(np.mean(list(gaps.values()))),
                              level_max_error=max(gaps.values()), absolute_gaps=gaps,
                              direction_correct=sum(r['direction_correct'] for r in differences),
                              direction_n=len(differences),
                              material_direction_correct=sum(r['direction_correct'] for r in material),
                              material_direction_n=len(material),
                              change_mae=float(np.mean([r['absolute_change_error'] for r in differences])),
                              comparisons=differences)
    plain, weighted = result['unweighted'], result['weighted']
    result['user_criterion_passes'] = bool(
        weighted['level_mae'] <= plain['level_mae']
        and weighted['direction_correct'] >= plain['direction_correct']
        and (weighted['level_mae'] < plain['level_mae']
             or weighted['direction_correct'] > plain['direction_correct']))
    return result


def markdown(result):
    main = result['comparison']['standard']
    a, b = main['unweighted'], main['weighted']
    verdict = ('Weighted CV meets your two criteria on these submissions.'
               if main['user_criterion_passes'] else
               'Weighted CV does not meet both of your criteria on these submissions.')
    lines = ['# Weighted CV versus actual leaderboard scores', '', f'**{verdict}**', '',
             '| Check | Current CV | Weighted CV |', '|---|---:|---:|',
             f"| Average absolute CV–LB gap, six submissions | {a['level_mae']:.2f} s | {b['level_mae']:.2f} s |",
             f"| Correct improvement/regression calls | {a['direction_correct']}/{a['direction_n']} | {b['direction_correct']}/{b['direction_n']} |",
             f"| Correct calls when LB changes by at least 1 s | {a['material_direction_correct']}/{a['material_direction_n']} | {b['material_direction_correct']}/{b['material_direction_n']} |",
             '', '| Submission | Current CV | Weighted CV | Actual LB |', '|---|---:|---:|---:|']
    for row in result['submissions']:
        scores = row['modes']['standard']
        lines.append(f"| v{row['version']} — {row['name']} | {scores['unweighted']['R_combined']:.2f} | "
                     f"{scores['weighted']['R_combined']:.2f} | {row['official']:.2f} |")
    lines += ['', 'All scores are RMSE in seconds; lower is better.', '',
              '| Change | Actual LB change | Current CV change | Weighted CV change |',
              '|---|---:|---:|---:|']
    for plain, weighted in zip(a['comparisons'], b['comparisons']):
        lines.append(f"| v{plain['parent']} → v{plain['candidate']} | {plain['delta_lb']:+.3f} | "
                     f"{plain['delta_cv']:+.3f} | {weighted['delta_cv']:+.3f} |")
    clipped = main['weighted_clipped_p99']
    lines += ['', f"Secondary check: average error in the size of a change is {a['change_mae']:.2f} → "
              f"{b['change_mae']:.2f} s. This is separate from your two deciding criteria.", '',
              f"Capping weights at p99 gives a {clipped['level_mae']:.2f} s average gap and "
              f"{clipped['direction_correct']}/{clipped['direction_n']} correct directions. "
              'Individual training-seed results are saved in the JSON.', '',
              '**Scope:** these are six historical submissions, with five related rule/recovery comparisons. '
              'v38 contributes to score gaps; its exact scored parent v34 has no local prediction files. '
              'The overnight series v17–v21 and later v26–v30/v33–v34 also lack their original prediction files here. '
              'They were excluded rather than replaced with model replays. These results do not establish '
              'performance on those feature changes or future submissions.', '',
              '**Checks:** original submission manifests and record hashes match; all six unweighted scores '
              'and the existing v38 weighted result reproduce. Flights and labels match across submissions. '
              'The same fixed weights are used for every model. No taxi-out refits, uploads, fold removals '
              'or default-score changes.', '',
              'Reproduce: `.venv/bin/python VALIDATION/tools/compare_weighted_cv_leaderboard.py`. '
              'Weights and four discriminator models are retained in `tmp/weighted_cv_leaderboard_20260920/`. '
              'Full results: [weighted_cv_leaderboard.json](weighted_cv_leaderboard.json).']
    REPORT.write_text('\n'.join(lines) + '\n')


def main():
    rows = roster()
    design = dict(
        scope='Retrospective comparison against already-known leaderboard scores; no tuning or independent confirmation claim.',
        submissions=rows, pairs=PAIRS,
        criterion='Mean absolute CV–LB gap decreases with direction accuracy no worse, or conversely.',
        primary='Existing combined CV, equal average of squared errors across seeds, existing per-source raw-feature weights.',
        sensitivity=['Each seed separately', 'p99 weight clipping already in original analysis',
                     'LB changes of at least one second', 'All 15 pairs, descriptive only',
                     'Exclude v38, which informed development of these weights'],
        exclusions='Original predictions absent for v17–v21, v26–v30, v33–v34; v34 replay is not its scored model. Other submissions have no matched local v3 prediction record.',
        weight_implementation_sha256=digest(Path(adv.__file__)),
        manifests={str(p.relative_to(ROOT)): digest(p) for p in [EARLY, LATEST]})
    # JSON roundtrip turns pair tuples into lists for stable resume comparisons.
    freeze(PLAN, json.loads(json.dumps(design)))
    reference = read(ROOT / 'VALIDATION/runs' / rows[0]['run'] / 'result.json')
    ratios, metadata = weights(reference['data_identity'])
    populations, scored = {}, []
    for row in rows:
        scored.append(score(row, ratios, populations, reference))
        write(CACHE / f"v{row['version']}_scores.json", scored[-1])
        gc.collect()
    result = dict(design_sha256=digest(PLAN), implementation_sha256=digest(Path(__file__)),
                  submissions=scored, weights=metadata,
                  comparison={mode: compare(scored, mode) for mode in ['standard', '42', '142', '242']},
                  without_v38=compare(scored, versions=[12, 13, 14, 15, 16]),
                  all_pairs_descriptive=compare(scored, pairs=[(b, a) for a, b in combinations(
                      [r['version'] for r in rows], 2)]))
    write(OUTPUT, result)
    markdown(result)
    print(json.dumps({m: {k: v for k, v in r.items() if k not in ['comparisons', 'absolute_gaps']}
                      for m, r in result['comparison']['standard'].items() if isinstance(r, dict)}, indent=2))
    print(f'Wrote {REPORT.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
