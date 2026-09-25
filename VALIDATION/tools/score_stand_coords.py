"""Score the stand-coordinates experiment: Weighted CV for the main run, and the
hidden-stand test. Design: VALIDATION/docs/STAND_COORDINATES_EXPERIMENT.md.

Run: .venv/bin/python VALIDATION/tools/score_stand_coords.py --mode main|hidden
Writes VALIDATION/reports/stand_coords_experiment/<mode>.json. Reads saved
predictions only; no fits.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import reporting
import score_disruption_experiment as scoring
import stand_coords_run as run
from records import sha256, write_json

OUT = ROOT / 'VALIDATION/reports/stand_coords_experiment'
V42_CV = ROOT / 'VALIDATION/reports/opdi_disruption_experiment/v40/results.json'
ARMS = [('control', 'pred_control'), ('candidate', 'pred_stand')]


def read(path):
    return json.loads(Path(path).read_text())


def checked_record(mode):
    directory = run.DEFAULT / mode
    record, progress = read(directory / 'result.json'), read(directory / 'progress.json')
    want = run.EXPECTED[mode]
    if record['status'] != 'complete' or record['fits'] != want['models'] or len(record['folds']) != want['folds']:
        raise ValueError(f'Expected a complete {mode} run')
    for name, digest in record['signature'].items():
        if sha256(ROOT / name) != digest:
            raise ValueError(f'Changed experiment source: {name}')
    run.overnight_run.check_artifacts(directory, progress)
    return directory, record


def weights():
    """The adopted Weighted CV weights. This Mac's copy was written by a separate run
    of the same discriminators, so its parquet bytes differ from gunes's; accept it
    only if the four discriminator files are byte-identical to the adopted record and
    the weights reproduce its row count and effective sample size to 1e-12."""
    adopted = read(ROOT / 'VALIDATION/reports/weighted_cv_leaderboard.json')['weights']
    out, hashes = {}, {}
    for name in ('R_A', 'R_B_long'):
        spec = adopted[name]
        path = ROOT / spec['path']
        for model in spec['models']:
            if sha256(ROOT / model['path']) != model['sha256']:
                raise ValueError(f'Weight discriminator changed: {model["path"]}')
        ratio = pd.read_parquet(path).ratio
        w = ratio.to_numpy(float)
        ess = w.sum() ** 2 / (w ** 2).sum() / len(w)
        if len(w) != spec['rows'] or abs(ess - spec['ess_fraction']) > 1e-12:
            raise ValueError(f'{name} weights do not reproduce the adopted weights')
        hashes[name] = dict(local=sha256(path), adopted=spec['sha256'], ess_fraction=float(ess))
        out[name] = ratio
    return out, hashes


def airport_changes(losses):
    """Unweighted RMSE per airport, all 16 folds pooled equally per evaluated row."""
    rows = []
    for ap in sorted(losses['control'][next(iter(losses['control']))].meta.airport.unique()):
        sq = {arm: np.concatenate([l.sq.mean(axis=0)[l.meta.airport.eq(ap).to_numpy()]
                                   for l in losses[arm].values()]) for arm in losses}
        a, b = (float(np.sqrt(sq[arm].mean())) for arm in ('control', 'candidate'))
        rows.append(dict(airport=ap, rows=len(sq['control']), control=a, candidate=b, change=b - a))
    return rows


def ageing(losses):
    """Row 44's warning sign: does the gain shrink when an older model predicts the
    same month? (candidate - control on the B fold) - (the same on its A fold);
    positive means the gain fades with model age."""
    calendar, draws = reporting.bootstrap_calendar(losses['control'], 2000, 0, block_days=1)
    boot = {arm: {k: reporting.boot_scores(v, calendar, draws) for k, v in losses[arm].items()} for arm in losses}
    c, k = losses['candidate'], losses['control']
    out = {}
    for name, (old, fresh) in reporting.aging_pairs(k).items():
        estimate = ((reporting.rmse(c[old]) - reporting.rmse(k[old]))
                    - (reporting.rmse(c[fresh]) - reporting.rmse(k[fresh])))
        draws_ = ((boot['candidate'][old] - boot['control'][old])
                  - (boot['candidate'][fresh] - boot['control'][fresh]))
        out[name] = reporting.interval(estimate, draws_)
    return out


def score_main():
    directory, record = checked_record('main')
    w, weight_hashes = weights()
    losses = {arm: reporting.load_losses(dict(record, prediction_column=col), directory) for arm, col in ARMS}
    groups = {arm: reporting.score_groups(value) for arm, value in losses.items()}
    ratios = {}
    for name in ('R_A', 'R_B_long'):
        reporting.same_flights(groups['control'][name].meta, groups['candidate'][name].meta)
        ids = groups['control'][name].meta.index
        if len(ids) != len(w[name]) or len(ids.difference(w[name].index)):
            raise ValueError('Weight/scoring population differs')
        ratios[name] = w[name].loc[ids].to_numpy(float)
    scores = {arm: scoring.levels(value, ratios) for arm, value in groups.items()}
    changes = scoring.compare(groups, ratios, scores)
    winter_record = dict(record, folds={'JAN_RETRO': record['january_holdout']})
    winter = {arm: reporting.load_losses(dict(winter_record, prediction_column=col), directory)['JAN_RETRO']
              for arm, col in ARMS}
    winter_scores = {arm: reporting.rmse(loss) for arm, loss in winter.items()}
    winter_changes = {}
    for block in (1, 3):
        calendar, draws = reporting.bootstrap_calendar(winter, 2000, 0, block_days=block)
        winter_changes[str(block)] = reporting.interval(winter_scores['candidate'] - winter_scores['control'],
            reporting.boot_scores(winter['candidate'], calendar, draws)
            - reporting.boot_scores(winter['control'], calendar, draws))
    v42 = read(V42_CV)['scores']['candidate']
    reproduction = {mode: {name: dict(local_control=scores['control'][mode][name]['score'],
                                      v42_cv=v42[mode][name]['score'],
                                      difference=scores['control'][mode][name]['score'] - v42[mode][name]['score'])
                           for name in ('R_A', 'R_B_long', 'R_combined')} for mode in ('weighted', 'unweighted')}
    result = dict(mode='main', run=str(directory.relative_to(ROOT)), design_sha256=record['design_sha256'],
        source_sha256=sha256(Path(__file__)), run_record_sha256=sha256(directory / 'result.json'),
        weights_sha256=weight_hashes, scores=scores, changes=changes, winter_scores=winter_scores,
        winter_changes=winter_changes, airports=airport_changes(losses), ageing=ageing(losses),
        v42_reproduction=reproduction,
        v42_reproduction_note='gunes ran v42 CV with the rebuilt OPDI cache on x86; this control uses the '
                              'original cache on Apple M3, so levels differ; the paired change is what counts.',
        minutes=record['minutes'])
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'main.json', result)
    w1 = changes['1']['weighted']['R_combined']
    print(json.dumps(dict(scores={arm: {m: {n: round(v['score'], 4) for n, v in s[m].items()} for m in s}
                                  for arm, s in scores.items()},
                          weighted_combined_change=w1, three_day=changes['3']['weighted']['R_combined']['ci95'],
                          january=winter_changes['1'], reproduction=reproduction['weighted']['R_combined'],
                          ageing={n: [round(v['difference'], 3)] + [round(x, 3) for x in v['ci95']]
                                  for n, v in result['ageing'].items()}), indent=1))


def hidden_losses(directory, record):
    """Per fold, the evaluated rows split into hidden and visible stands."""
    out = {arm: {} for arm, _ in ARMS}
    extra = {}
    for arm, col in ARMS:
        losses = reporting.load_losses(dict(record, prediction_column=col), directory)
        for fold, loss in losses.items():
            if fold not in extra:
                frame = pd.read_parquet(directory / record['folds'][fold]['predictions']['42'],
                                        columns=['mvt_id', 'hidden_stand', 'sc_x_m', 'stand_key'])
                extra[fold] = frame.set_index('mvt_id').loc[loss.meta.index]
            out[arm][fold] = loss
    return out, extra


def subset(losses, extra, pick):
    return {fold: reporting.Loss(loss.meta[m], loss.sq[:, m], loss.seeds) for fold, loss in losses.items()
            for m in [pick(extra[fold]).to_numpy(bool)] if m.any()}


def pooled(parts):
    meta = pd.concat([p.meta for p in parts.values()])
    return reporting.Loss(meta, np.concatenate([p.sq for p in parts.values()], axis=1), next(iter(parts.values())).seeds)


def paired(losses_by_arm):
    a, b = (pooled(losses_by_arm[arm]) for arm in ('control', 'candidate'))
    out = dict(rows=len(a.meta), control=reporting.rmse(a), candidate=reporting.rmse(b))
    out['change'] = out['candidate'] - out['control']
    for block in (1, 3):
        calendar, draws = reporting.bootstrap_calendar(losses_by_arm['control'], 2000, 0, block_days=block)
        out[f'{block}_day'] = reporting.interval(out['change'], reporting.boot_scores(b, calendar, draws)
                                                 - reporting.boot_scores(a, calendar, draws))
    return out


def score_hidden():
    directory, record = checked_record('hidden')
    losses, extra = hidden_losses(directory, record)
    views = {'hidden': lambda e: e.hidden_stand,
             'hidden_located': lambda e: e.hidden_stand & e.sc_x_m.notna(),
             'hidden_unlocated': lambda e: e.hidden_stand & e.sc_x_m.isna(),
             'visible': lambda e: ~e.hidden_stand}
    result = dict(mode='hidden', run=str(directory.relative_to(ROOT)), design_sha256=record['design_sha256'],
                  source_sha256=sha256(Path(__file__)), run_record_sha256=sha256(directory / 'result.json'),
                  hidden_stands=record['hidden_stands'], minutes=record['minutes'], views={})
    for name, pick in views.items():
        parts = {arm: subset(losses[arm], extra, pick) for arm in losses}
        result['views'][name] = paired(parts) if parts['control'] else None
    hidden = {arm: pooled(subset(losses[arm], extra, views['hidden'])) for arm in losses}
    result['hidden_airports'] = []
    for ap in sorted(hidden['control'].meta.airport.unique()):
        m = hidden['control'].meta.airport.eq(ap).to_numpy()
        a, b = (float(np.sqrt(hidden[arm].sq[:, m].mean())) for arm in ('control', 'candidate'))
        result['hidden_airports'].append(dict(airport=ap, rows=int(m.sum()), control=a, candidate=b, change=b - a))
    stands = pd.concat([extra[f].loc[extra[f].hidden_stand, 'stand_key'] for f in extra])
    result['hidden_distinct_stands'] = int(stands.nunique())
    OUT.mkdir(parents=True, exist_ok=True)
    write_json(OUT / 'hidden.json', result)
    print(json.dumps(dict(views=result['views'], airports=result['hidden_airports']), indent=1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['main', 'hidden'])
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        score_main() if args.mode == 'main' else score_hidden()
