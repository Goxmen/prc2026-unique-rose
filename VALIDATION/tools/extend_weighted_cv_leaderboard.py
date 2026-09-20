"""Add the five diverse overnight submissions to the fixed weighted-CV check.

Uses hash-verified original predictions recovered from the user's desktop.
The initial six-submission design/results and existing weights stay unchanged.
"""

import gc
import hashlib
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

import compare_weighted_cv_leaderboard as base
from reporting import COVERAGE, IDENTITY, Loss, group_loss, same_flights

ROOT = base.ROOT
RUN = ROOT / 'VALIDATION/runs/overnight_20260917'
PLAN = base.REPORTS / 'weighted_cv_leaderboard_extended_design.json'
OUTPUT = base.REPORTS / 'weighted_cv_leaderboard_extended.json'
REPORT = base.REPORTS / 'WEIGHTED_CV_LEADERBOARD_EXTENDED.md'
CACHE = ROOT / 'tmp/weighted_cv_leaderboard_extension_20260920'
RECIPES = {
    17: ('no_month', 'Remove month'),
    18: ('no_history', 'Remove departure history'),
    19: ('clock_features', 'Add clock features'),
    20: ('small_trees', 'Use smaller trees'),
    21: ('rule_routing', 'Exclude rule-handled rows from training'),
}
NEW_PAIRS = [(version, 15) for version in RECIPES]
ALL_PAIRS = base.PAIRS + NEW_PAIRS


def prepare_roster():
    manifest = base.read(RUN / 'submission_manifest.json')
    cv = base.read(RUN / 'cv_scores.json')
    record = base.read(RUN / 'result.json')
    assert record['status'] == 'complete'
    assert record['config']['seeds'] == [42, 142, 242]
    assert record['config'] == cv['runner_config']
    assert record['jobs'] == record['signature']['jobs']
    entries = {s['version']: s for s in manifest['submissions']}
    rows = []
    for version, (recipe, name) in RECIPES.items():
        item = entries[version]
        assert item['recipe'] == recipe
        assert item['official_result']['status'] == 'Succeeded'
        assert item['official_result']['used_pairs'] == 344841
        assert item['score'] == item['official_result']['score']
        rows.append(dict(version=version, name=name, recipe=recipe,
                         official=float(item['score']), run=RUN.name, column='pred',
                         expected_scores={k: cv['new'][recipe]['standard'][k]
                                          for k in ['R_A', 'R_B_long', 'R_combined']}))
    return rows, record, cv


def adapted_record(recipe, record, cv, wait_seconds=0):
    """Map the overnight manifest to the existing loss reader, without changing files."""
    folds = {}
    expected_files = {}
    deadline = time.monotonic() + wait_seconds
    wanted = set(sum(base.adv.GROUP_FOLDS.values(), []))
    for job in record['jobs']:
        if job['scheme'] != 'v3':
            continue
        for entry in job['evals']:
            fold = entry['fold']
            if fold not in wanted:
                continue
            assert fold not in folds
            predictions = {}
            for seed in record['config']['seeds']:
                path = f"predictions/{job['id']}__{recipe}__{seed}__{fold}.parquet"
                expected = cv['prediction_sha256'][path]
                if not (RUN / path).is_file() and wait_seconds:
                    print(f'Waiting for transfer: {path}', flush=True)
                    while not (RUN / path).is_file() and time.monotonic() < deadline:
                        time.sleep(1)
                assert base.digest(RUN / path) == expected, f'Original artifact changed: {path}'
                expected_files[path] = expected
                predictions[str(seed)] = path
            folds[fold] = dict(val=entry['month'], predictions=predictions)
    assert set(folds) == wanted
    return dict(config=record['config'], prediction_column='pred', folds=folds), expected_files


def load_original_losses(record):
    """Read only scoring columns; check monthly bounds without formatting every timestamp."""
    losses = {}
    seeds = tuple(record['config']['seeds'])
    columns = ['mvt_id', 'fold', 'seed', 'pred', *IDENTITY, *COVERAGE]
    for fold, info in record['folds'].items():
        meta, squares = None, []
        start = pd.Timestamp(info['val'] + '-01', tz='UTC')
        end = start + pd.offsets.MonthBegin(1)
        for seed in seeds:
            frame = pd.read_parquet(RUN / info['predictions'][str(seed)], columns=columns)
            assert frame.mvt_id.notna().all() and frame.mvt_id.is_unique
            assert frame.fold.eq(fold).all() and frame.seed.eq(seed).all()
            assert np.isfinite(frame[['y', 'pred']].to_numpy()).all()
            frame = frame.set_index('mvt_id').sort_index()
            frame['mvt_time'] = pd.to_datetime(frame.mvt_time, utc=True)
            assert frame.mvt_time.notna().all()
            assert frame.mvt_time.ge(start).all() and frame.mvt_time.lt(end).all()
            if meta is None:
                meta = frame[IDENTITY + COVERAGE].copy()
            else:
                same_flights(meta, frame)
                assert meta[COVERAGE].equals(frame[COVERAGE])
            squares.append((frame.y.to_numpy() - frame.pred.to_numpy())**2)
        losses[fold] = Loss(meta, np.stack(squares), seeds)
    return losses


def score(row, record, cv, ratios, populations, signature, wait_seconds=0):
    recipe = row['recipe']
    adapted, files = adapted_record(recipe, record, cv, wait_seconds)
    cached_path = CACHE / f'{recipe}_scores.json'
    if cached_path.exists():
        cached = base.read(cached_path)
        if cached['cache_signature'] == signature:
            assert cached['prediction_sha256'] == files
            for name, group in cached['groups'].items():
                assert group['identity_sha256'] == populations[name]
            return cached
    losses = load_original_losses(adapted)
    # Reproduce each stored fold and seed, not just a single aggregate number.
    for fold, loss in losses.items():
        assert len(loss.meta) == cv['folds'][recipe][fold]['n']
        for i, seed in enumerate(loss.seeds):
            np.testing.assert_allclose(loss.sq[i].mean(),
                cv['folds'][recipe][fold]['mse_by_seed'][str(seed)], rtol=1e-12, atol=1e-8)
    modes = {mode: {method: {} for method in base.METHODS}
             for mode in ['standard', '42', '142', '242']}
    groups = {}
    for name, folds in base.adv.GROUP_FOLDS.items():
        loss = group_loss(losses, folds)
        identity = loss.meta[['mvt_time', 'airport', 'y']].copy()
        identity['airport'] = identity.airport.astype(str)
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(identity, index=True).values.tobytes()).hexdigest()
        assert fingerprint == populations[name], 'Flight IDs or labels differ from the initial six'
        assert loss.meta.index.is_unique and len(loss.meta) == len(ratios[name])
        assert loss.meta.index.isin(ratios[name].index).all()
        w = ratios[name].loc[loss.meta.index, 'ratio'].to_numpy()
        assert np.isfinite(w).all() and (w > 0).all()
        for method, weight in zip(base.METHODS,
                                  [np.ones(len(w)), w, np.minimum(w, np.quantile(w, .99))]):
            seed_mse = np.average(loss.sq, axis=1, weights=weight)
            modes['standard'][method][name] = float(np.sqrt(seed_mse.mean()))
            for i, seed in enumerate(loss.seeds):
                modes[str(seed)][method][name] = float(np.sqrt(seed_mse[i]))
        groups[name] = dict(rows=len(w), identity_sha256=fingerprint)
    jan = cv['jan_fraction']
    for mode, methods in modes.items():
        for scores in methods.values():
            scores['R_combined'] = float(np.sqrt(
                jan * scores['R_A']**2 + (1 - jan) * scores['R_B_long']**2))
        for metric, actual in methods['unweighted'].items():
            np.testing.assert_allclose(actual, cv['new'][recipe][mode][metric], rtol=0, atol=1e-8)
    result = {**row, 'modes': modes, 'groups': groups, 'prediction_sha256': files,
              'cache_signature': signature, 'record_sha256': base.digest(RUN / 'result.json')}
    base.write(cached_path, result)
    print(f"{recipe}: " + ', '.join(
        f'{method}={scores["R_combined"]:.4f}' for method, scores in modes['standard'].items()), flush=True)
    return result


def describe(result):
    overall = result['comparison']['standard']
    added = result['added_five']['standard']
    old, weighted = overall['unweighted'], overall['weighted']
    new_old, new_weighted = added['unweighted'], added['weighted']
    verdict = ('Weighted CV still meets your two criteria across all eleven submissions.'
               if overall['user_criterion_passes'] else
               'With these five added cases, weighted CV does not meet both of your criteria.')
    lines = ['# Weighted CV: five more approaches', '', f'**{verdict}**', '',
             '| Check | Current CV | Weighted CV |', '|---|---:|---:|',
             f"| Average absolute gap, all 11 submissions | {old['level_mae']:.2f} s | {weighted['level_mae']:.2f} s |",
             f"| Correct directions, all 10 comparisons | {old['direction_correct']}/10 | {weighted['direction_correct']}/10 |",
             f"| Average absolute gap, five additions only | {new_old['level_mae']:.2f} s | {new_weighted['level_mae']:.2f} s |",
             f"| Correct directions, five additions only | {new_old['direction_correct']}/5 | {new_weighted['direction_correct']}/5 |",
             '', 'The five additions each change a different part of v15:', '',
             '| Submission / approach | Current CV | Weighted CV | Actual LB |', '|---|---:|---:|---:|']
    for row in result['submissions']:
        if row['version'] not in RECIPES:
            continue
        scores = row['modes']['standard']
        lines.append(f"| v{row['version']} — {row['name']} | {scores['unweighted']['R_combined']:.2f} | "
                     f"{scores['weighted']['R_combined']:.2f} | {row['official']:.2f} |")
    lines += ['', '| Change from v15 | Actual LB change | Current CV change | Weighted CV change |',
              '|---|---:|---:|---:|']
    for a, b in zip(new_old['comparisons'], new_weighted['comparisons']):
        lines.append(f"| v{a['candidate']} | {a['delta_lb']:+.3f} | {a['delta_cv']:+.3f} | {b['delta_cv']:+.3f} |")
    lines += ['', 'All values are seconds of RMSE. Negative changes mean improvement.', '',
              f"For LB changes of at least 1 s, overall direction accuracy is "
              f"{old['material_direction_correct']}/{old['material_direction_n']} → "
              f"{weighted['material_direction_correct']}/{weighted['material_direction_n']}. "
              'v20 changes LB by 0.9999 s, so it is included in the main count but falls just below this threshold.', '',
              f"Error in predicting the size of a change is {old['change_mae']:.2f} → "
              f"{weighted['change_mae']:.2f} s overall. This is separate from your two criteria.", '',
              '**Checks:** original predictions were recovered from the desktop and verified against all '
              '162 saved file hashes, including the overnight control. Every contributing fold/seed score and every '
              'original combined score reproduces. Flight IDs and labels match the earlier six cases. '
              'The weights, folds and scoring formula are unchanged. No model training or submissions ran.', '',
              '**Limits:** this is a historical comparison. All five added approaches share v15 as their '
              'parent; they are not five independent datasets. The JSON also reports each seed, p99 clipping, '
              'and differences measured against the overnight control replay. Primary comparisons use the '
              'original scored v15, since the replay was not submitted.', '',
              'Reproduce: `.venv/bin/python VALIDATION/tools/extend_weighted_cv_leaderboard.py`. '
              '[Full results](weighted_cv_leaderboard_extended.json); '
              '[original six-case comparison](WEIGHTED_CV_LEADERBOARD.md).']
    REPORT.write_text('\n'.join(lines) + '\n')


def main(wait_seconds=0):
    rows, record, cv = prepare_roster()
    original = base.read(base.OUTPUT)
    assert base.digest(Path(base.__file__)) == original['implementation_sha256']
    assert base.digest(base.PLAN) == original['design_sha256']
    reference = base.read(ROOT / 'VALIDATION/runs' / original['submissions'][0]['run'] / 'result.json')
    assert record['signature']['data_identity'] == reference['data_identity']
    assert cv['jan_fraction'] == reference['ranking_weights']['jan_fraction']
    design = dict(
        original_result_sha256=base.digest(base.OUTPUT), submissions=rows,
        selection='All five actual overnight submissions: calendar ablation, history ablation, clock features, capacity, training routing.',
        new_pairs=NEW_PAIRS, all_pairs=ALL_PAIRS,
        primary='Same fixed weights and user criteria as the initial six-case comparison; every new candidate compared to original scored v15.',
        scope='Retrospective; official scores already known. No tuning or independent-confirmation claim.',
        sensitivity=['Each seed', 'Existing p99 clipping', 'Changes >= 1 s',
                     'Overnight control replay for CV differences only; same recorded v15 LB anchor',
                     'All pairs descriptive only'],
        manifests={str(p.relative_to(ROOT)): base.digest(p) for p in
                   [RUN / 'cv_scores.json', RUN / 'result.json', RUN / 'submission_manifest.json']},
        weight_files={k: v['sha256'] for k, v in original['weights'].items()})
    # Keep the first design intact; freeze an explicit extension before scoring.
    import json
    base.freeze(PLAN, json.loads(json.dumps(design)))
    assert (base.CACHE / 'weights.json').is_file(), 'Use existing weights; do not fit replacements'
    ratios, metadata = base.weights(reference['data_identity'])
    assert metadata == original['weights']
    populations = {k: v['identity_sha256'] for k, v in original['submissions'][0]['groups'].items()}
    signature = dict(design_sha256=base.digest(PLAN), implementation_sha256=base.digest(Path(__file__)))
    new = []
    for row in rows:
        new.append(score(row, record, cv, ratios, populations, signature, wait_seconds))
        gc.collect()
    control = score(dict(recipe='control', name='Unsubmitted overnight control replay'),
                    record, cv, ratios, populations, signature, wait_seconds)
    all_rows = sorted(original['submissions'] + new, key=lambda r: r['version'])
    versions = [r['version'] for r in all_rows]
    result = dict(**signature, original_result_sha256=base.digest(base.OUTPUT),
                  submissions=all_rows, weights=metadata, overnight_control=control,
                  comparison={mode: base.compare(all_rows, mode, pairs=ALL_PAIRS)
                              for mode in ['standard', '42', '142', '242']},
                  added_five={}, overnight_control_sensitivity={},
                  all_pairs_descriptive=base.compare(all_rows,
                      pairs=[(b, a) for a, b in combinations(versions, 2)]))
    original_parent = next(r for r in all_rows if r['version'] == 15)
    replay_parent = {**original_parent, 'modes': control['modes']}
    for mode in ['standard', '42', '142', '242']:
        result['added_five'][mode] = base.compare([original_parent] + new, mode, pairs=NEW_PAIRS)
        result['overnight_control_sensitivity'][mode] = base.compare([replay_parent] + new, mode, pairs=NEW_PAIRS)
        # Exclude the anchor from the five-addition level summary, but keep it
        # for candidate-minus-parent differences in the comparison helper.
        for summary in [result['added_five'][mode], result['overnight_control_sensitivity'][mode]]:
            for method in base.METHODS:
                gaps = {str(v): abs(next(r for r in new if r['version'] == v)['modes'][mode][method]['R_combined']
                                   - next(r for r in new if r['version'] == v)['official']) for v in RECIPES}
                summary[method].update(absolute_gaps=gaps, level_mae=float(np.mean(list(gaps.values()))),
                                       level_max_error=max(gaps.values()))
            a, b = summary['unweighted'], summary['weighted']
            summary['user_criterion_passes'] = bool(b['level_mae'] <= a['level_mae']
                and b['direction_correct'] >= a['direction_correct']
                and (b['level_mae'] < a['level_mae'] or b['direction_correct'] > a['direction_correct']))
    base.write(OUTPUT, result)
    describe(result)
    for scope in ['comparison', 'added_five', 'overnight_control_sensitivity']:
        print(scope, {m: {k: v for k, v in x.items() if k not in ['comparisons', 'absolute_gaps']}
                      for m, x in result[scope]['standard'].items() if isinstance(x, dict)}, flush=True)
    print(f'Wrote {REPORT.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-artifacts', type=int, default=0,
                        help='Maximum seconds per recipe to wait for an active file transfer')
    main(parser.parse_args().wait_for_artifacts)
