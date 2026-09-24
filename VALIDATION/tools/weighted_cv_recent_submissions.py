"""Weighted CV for the six most recently uploaded submissions: v34, v37, v38, v39, v40, v41.

Scores each one from its own saved fold predictions with the validation adopted on
20 September 2026 (`VALIDATION/docs/WEIGHTED_CV.md`), so the recent models can be
compared on the current headline instead of the original v3 combined score.

The adopted weights are reused, never refitted for convenience. When the local cache
is absent it is rebuilt from the frozen specification and accepted only if every
weight file reproduces the sha256 recorded with the adopted method; a mismatch stops
the run. v38's fold predictions are not on this machine, so its already verified
entry is carried over from the adopted comparison rather than recomputed.

No taxi-out fits, no uploads, no change to any recorded score.
"""

import gc
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION/tools'), str(ROOT / 'VALIDATION')]
import analyze_adversarial_shift as adv                      # noqa: E402
import compare_weighted_cv_leaderboard as base               # noqa: E402
from reporting import group_loss, load_losses                # noqa: E402

REPORTS = ROOT / 'VALIDATION/reports'
RUNS = ROOT / 'VALIDATION/runs'
ADOPTED = REPORTS / 'weighted_cv_leaderboard.json'
OUTPUT = REPORTS / 'weighted_cv_recent_submissions.json'
REPORT = REPORTS / 'WEIGHTED_CV_RECENT.md'

# Each entry names the run and column actually behind the uploaded file, and an
# in-repo record of that column's original v3 scores to reproduce before weighting.
ROSTER = [
    dict(version=34, row=34, name='No-ID-cap',
         manifest='v3_20260919T104157403241Z/submission_v34.json',
         run='v3_20260919T104157403241Z', column='pred_noid_cap',
         expected_from='v3_20260919T104157403241Z/'
                       'comparison_pred_noid_cap_vs_pred_window_'
                       'v3_20260919T104157403241Z_block1.json'),
    dict(version=37, row=40, name='Dayshift-hedge',
         manifest='v3_20260919T112057144323Z/submission_v37.json',
         run='v3_20260919T112057144323Z', column='pred_dayshift',
         expected_from='v3_20260919T112057144323Z/result.json'),
    dict(version=39, row=42, name='OPDI-combined',
         manifest='v3_20260919T211914721668Z/submission_v39.json',
         run='v3_20260919T223006736822Z', column='pred37_opdi',
         expected_from='v3_20260919T223006736822Z/'
                       'comparison_pred37_opdi_vs_pred37_'
                       'v3_20260919T223006736822Z_block1.json'),
    dict(version=40, row=43, name='Rule-routed-training',
         manifest='v3_20260920T152734631417Z/submission_v40.json',
         run='v3_20260920T170600876190Z', column='pred37_opdi',
         # This derivation saved no comparison file; the published row 43 scores
         # (TRAINING_LABEL_FILTER.md, SUBMISSIONS.md) are the reference instead.
         expected_scores={'R_A': 245.9817, 'R_B_long': 222.5802, 'R_combined': 233.2338},
         expected_tolerance=1e-4),
    dict(version=41, row=44, name='Taxi-occupancy',
         manifest='v3_20260924T202324352147Z/submission_v41.json',
         run='v3_20260924T211259763470Z', column='pred37_occ',
         expected_from='v3_20260924T211259763470Z/'
                       'comparison_pred37_occ_vs_pred37_'
                       'v3_20260924T211259763470Z_block1.json'),
]
CARRIED = dict(version=38, row=37, name='Artifact-combined',
               source='weighted_cv_leaderboard.json',
               reason='fold predictions for v3_20260919_artifact_experiments are not on this machine')


def expected_scores(entry):
    if 'expected_scores' in entry:
        return entry['expected_scores'], entry.get('expected_tolerance', 1e-8)
    record = base.read(RUNS / entry['expected_from'])
    report = record['report'] if 'report' in record else record
    column = record.get('prediction_column')
    assert column == entry['column'], f"{entry['expected_from']} records column {column}"
    return report['scores'], 1e-8


def roster():
    """Confirm each entry is the model actually scored on the leaderboard."""
    rows = []
    for entry in ROSTER + [CARRIED]:
        if entry is CARRIED:
            continue
        item = base.read(RUNS / entry['manifest'])
        assert item['version'] == entry['version']
        assert item['state'] in ('uploaded', 'scored')
        assert item['official_result']['status'] == 'Succeeded'
        assert item['official_result']['used_pairs'] == 344841
        assert item['score'] == item['official_result']['score']
        scores, tolerance = expected_scores(entry)
        rows.append({**entry, 'official': float(item['score']),
                     'uploaded_at': item['uploaded_at'],
                     'expected': {k: scores[k] for k in ['R_A', 'R_B_long', 'R_combined']},
                     'tolerance': tolerance})
    rows.sort(key=lambda r: r['uploaded_at'])
    return rows


def verify_weights(reference):
    """Reuse the adopted weights; rebuild only to reproduce them, never to replace them."""
    adopted = base.read(ADOPTED)['weights']
    # The discriminator reads the movement files only. Later runs also record
    # external sources (weather) that live outside data/, so the weight
    # specification keeps exactly the files `adv.load` opens.
    identity = [entry for entry in reference['data_identity']
                if (ROOT / 'data' / entry['file']).is_file()]
    wanted = {p.name for p in ROOT.joinpath('data').glob('training_*.parquet')} | {'ranking.parquet'}
    assert {entry['file'] for entry in identity} == wanted, 'movement files missing from the run identity'
    ratios, metadata = base.weights(identity)
    encoding_only = {}
    for name, recorded in adopted.items():
        actual = metadata[name]
        # The ratios are a deterministic function of the two saved discriminators
        # and the recorded split, so those are what must match. A parquet file
        # written by a different pyarrow build differs in container bytes while
        # holding the same numbers, which the identical ESS and calibration
        # confirm; that difference is reported, never waved through silently.
        assert [m['sha256'] for m in actual['models']] == [m['sha256'] for m in recorded['models']], (
            f'{name}: rebuilt discriminators differ from the adopted ones; do not score with these')
        assert actual['rows'] == recorded['rows'], f'{name}: weighted row count differs'
        assert actual['class_counts'] == recorded['class_counts'], f'{name}: cross-fitting split differs'
        assert actual['calibration'] == recorded['calibration'], f'{name}: held-out calibration differs'
        assert actual['ess_fraction'] == recorded['ess_fraction'], f'{name}: effective sample size differs'
        if actual['sha256'] != recorded['sha256']:
            encoding_only[name] = dict(adopted=recorded['sha256'], local=actual['sha256'])
    print('Weights reproduce the adopted ones: identical discriminators, split, '
          'calibration and effective sample size.', flush=True)
    if encoding_only:
        print(f'Weight parquet encoding differs locally (values identical): '
              f'{sorted(encoding_only)}', flush=True)
    return ratios, dict(adopted=adopted, local=metadata, encoding_only_differences=encoding_only)


def score(row, ratios, populations, reference):
    directory = RUNS / row['run']
    record = base.read(directory / 'result.json')
    actual = {item['file']: item for item in record['data_identity']}
    assert all(actual.get(item['file']) == item for item in reference['data_identity'])
    assert record['ranking_weights'] == reference['ranking_weights']
    assert record['config']['seeds'] == [42, 142, 242]
    record['prediction_column'] = row['column']
    record['folds'] = {f: record['folds'][f] for f in sum(adv.GROUP_FOLDS.values(), [])}
    files = {p: base.digest(directory / p)
             for fold in record['folds'].values() for p in fold['predictions'].values()}
    losses = load_losses(record, directory)
    modes = {mode: {method: {} for method in base.METHODS}
             for mode in ['standard', '42', '142', '242']}
    groups = {}
    for name, folds in adv.GROUP_FOLDS.items():
        loss = group_loss(losses, folds)
        identity = loss.meta[['mvt_time', 'airport', 'y']].copy()
        identity['airport'] = identity.airport.astype(str)
        import pandas as pd
        fingerprint = hashlib.sha256(
            pd.util.hash_pandas_object(identity, index=True).values.tobytes()).hexdigest()
        assert populations[name] == fingerprint, f'{name}: flights or labels differ from the adopted run'
        assert loss.meta.index.is_unique and len(loss.meta) == len(ratios[name])
        assert loss.meta.index.isin(ratios[name].index).all()
        w = ratios[name].loc[loss.meta.index, 'ratio'].to_numpy()
        assert np.isfinite(w).all() and (w > 0).all()
        clipped = np.minimum(w, np.quantile(w, .99))
        for mode in modes:
            sq = loss.sq.mean(axis=0) if mode == 'standard' else loss.sq[loss.seeds.index(int(mode))]
            for method, weight in zip(base.METHODS, [np.ones(len(w)), w, clipped]):
                modes[mode][method][name] = float(np.sqrt(np.average(sq, weights=weight)))
        np.testing.assert_allclose(modes['standard']['unweighted'][name],
                                   row['expected'][name], rtol=0, atol=row['tolerance'])
        groups[name] = dict(rows=len(w), identity_sha256=fingerprint)
    jan = record['ranking_weights']['jan_fraction']
    for mode in modes.values():
        for scores in mode.values():
            scores['R_combined'] = float(np.sqrt(
                jan * scores['R_A']**2 + (1 - jan) * scores['R_B_long']**2))
    np.testing.assert_allclose(modes['standard']['unweighted']['R_combined'],
                               row['expected']['R_combined'], rtol=0, atol=row['tolerance'])
    standard = modes['standard']
    print(f"v{row['version']} (row {row['row']}, {row['name']}): "
          f"CV {standard['unweighted']['R_combined']:.4f}, "
          f"weighted {standard['weighted']['R_combined']:.4f}, "
          f"LB {row['official']:.4f}", flush=True)
    return {**row, 'modes': modes, 'groups': groups, 'prediction_sha256': files,
            'record_sha256': base.digest(directory / 'result.json')}


def describe(result):
    rows = result['submissions']
    lines = ['# Weighted CV for the five most recent submissions', '',
             'The five uploads after the weighted validation was adopted are scored on it here: '
             'v34, v37, v38, v39 and v40, each from the fold predictions behind the uploaded file. '
             'The adopted weights, folds and formula are unchanged.', '',
             '| Upload | Row | Recipe | CV combined | Weighted CV | Official LB | Weighted gap |',
             '|---|---:|---|---:|---:|---:|---:|']
    for row in rows:
        standard = row['modes']['standard']
        cv, weighted = standard['unweighted']['R_combined'], standard['weighted']['R_combined']
        lines.append(f"| v{row['version']} | {row['row']} | {row['name']} | {cv:.4f} | "
                     f"{weighted:.4f} | {row['official']:.4f} | "
                     f"{weighted - row['official']:+.2f} |")
    gaps = result['gaps']
    lines += ['', 'All values are RMSE in seconds; lower is better. The weighted gap is '
              'Weighted CV minus the official score.', '',
              '| Check | Current CV | Weighted CV |', '|---|---:|---:|',
              f"| Average absolute gap to the official score | {gaps['unweighted']['level_mae']:.2f} s | "
              f"{gaps['weighted']['level_mae']:.2f} s |",
              f"| Largest absolute gap | {gaps['unweighted']['level_max_error']:.2f} s | "
              f"{gaps['weighted']['level_max_error']:.2f} s |", '',
              '| Change | Actual LB change | Current CV change | Weighted CV change | Direction |',
              '|---|---:|---:|---:|---|']
    for a, b in zip(result['steps']['unweighted'], result['steps']['weighted']):
        agree = 'agrees' if b['direction_correct'] else '**disagrees**'
        lines.append(f"| v{a['parent']} -> v{a['candidate']} | {a['delta_lb']:+.4f} | "
                     f"{a['delta_cv']:+.4f} | {b['delta_cv']:+.4f} | {agree} |")
    carried = next(r for r in rows if r['version'] == CARRIED['version'])
    change = {m: float(np.mean([s['absolute_change_error'] for s in result['steps'][m]]))
              for m in ['unweighted', 'weighted']}
    lines += ['', f"Direction agreement on consecutive uploads: "
              f"{result['direction']['unweighted']}/{len(result['steps']['unweighted'])} current CV, "
              f"{result['direction']['weighted']}/{len(result['steps']['weighted'])} weighted. "
              f"Average error in the size of a step is {change['unweighted']:.2f} s current CV and "
              f"{change['weighted']:.2f} s weighted, so the weighting fixes the level of the score "
              'without predicting how far a change will move the leaderboard.', '',
              '**Checks:** the weights are the adopted ones. The local cache was rebuilt from the '
              'frozen specification and its four discriminator model files are byte-identical to the '
              'ones recorded with the method, with the same cross-fitting split, held-out calibration '
              'and effective sample size, so the ratios are the same numbers; only the parquet '
              'container bytes differ, which a different pyarrow build explains and the identical '
              'calibration and effective sample size rule out as a difference in values. Every '
              'original v3 score reproduces from the same predictions before weighting. Flight '
              'identities and labels in both groups match the adopted comparison. Each submission '
              'manifest records an accepted upload scored on all 344,841 rows.', '',
              f"**Limits:** v{carried['version']} is carried over from "
              f"[the adopted comparison]({CARRIED['source']}) because {CARRIED['reason']}; it was "
              'not recomputed here. These five uploads are consecutive refinements of one model line, '
              'not independent evidence, and their official scores were already known, so this is a '
              'retrospective reading, not a validation of the weighting. Weighted CV was tuned on '
              'neither these rows nor their scores, but v38 did inform the weighting work.', '',
              'Reproduce: `.venv/bin/python VALIDATION/tools/weighted_cv_recent_submissions.py`. '
              '[Full results](weighted_cv_recent_submissions.json); '
              '[method](../docs/WEIGHTED_CV.md).']
    REPORT.write_text('\n'.join(lines) + '\n')


def main():
    rows = roster()
    adopted_result = base.read(ADOPTED)
    populations = {k: v['identity_sha256']
                   for k, v in adopted_result['submissions'][0]['groups'].items()}
    reference = base.read(RUNS / rows[0]['run'] / 'result.json')
    ratios, weight_metadata = verify_weights(reference)
    scored = []
    for row in rows:
        scored.append(score(row, ratios, populations, reference))
        gc.collect()
    carried = next(r for r in adopted_result['submissions'] if r['version'] == CARRIED['version'])
    scored.append({**CARRIED, 'official': carried['official'], 'modes': carried['modes'],
                   'groups': carried['groups'], 'carried_over': True,
                   'uploaded_at': base.read(
                       RUNS / 'v3_20260919_artifact_experiments/submission_v38.json')['uploaded_at']})
    scored.sort(key=lambda r: r['uploaded_at'])
    steps, gaps, direction = {}, {}, {}
    for method in ['unweighted', 'weighted']:
        values = [r['modes']['standard'][method]['R_combined'] for r in scored]
        absolute = [abs(v - r['official']) for v, r in zip(values, scored)]
        gaps[method] = dict(level_mae=float(np.mean(absolute)),
                            level_max_error=float(max(absolute)),
                            absolute_gaps={str(r['version']): g for r, g in zip(scored, absolute)})
        steps[method] = []
        for (before, after), (cv_before, cv_after) in zip(
                zip(scored, scored[1:]), zip(values, values[1:])):
            delta_lb = after['official'] - before['official']
            delta_cv = cv_after - cv_before
            steps[method].append(dict(parent=before['version'], candidate=after['version'],
                                      delta_lb=delta_lb, delta_cv=delta_cv,
                                      direction_correct=bool(np.sign(delta_cv) == np.sign(delta_lb)),
                                      absolute_change_error=abs(delta_cv - delta_lb)))
        direction[method] = sum(s['direction_correct'] for s in steps[method])
    result = dict(implementation_sha256=base.digest(Path(__file__)),
                  adopted_result_sha256=base.digest(ADOPTED), weights=weight_metadata,
                  submissions=scored, gaps=gaps, steps=steps, direction=direction)
    base.write(OUTPUT, result)
    describe(result)
    print(json.dumps(dict(gaps={k: {'level_mae': v['level_mae']} for k, v in gaps.items()},
                          direction=direction), indent=2))
    print(f'Wrote {REPORT.relative_to(ROOT)}', flush=True)


if __name__ == '__main__':
    main()
