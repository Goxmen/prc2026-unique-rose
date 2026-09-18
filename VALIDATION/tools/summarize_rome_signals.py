"""A/B table for a completed rome_signals_run.py run: each candidate against the row-28 control.

For every candidate column the paired day-bootstrap comparison against `pred` is
computed with compare.py (reused when its file already exists). Then each
candidate's change in squared error is split over four disjoint populations, as a
share of the control's pooled squared error (16 folds, all seeds):

    Rome rule rows                 LIRF, no flight ID, gap >= 4 h
    LIRF no ID, gap 1-4 h          the rows the late estimate and classifier inputs target
    LIRF, everything else          flight ID present, or no ID with gap <= 1 h
    other airports

One more comparison is derived rather than fitted: `all_no_dayshift`, the `all`
candidate without the day-shift blend. The day-shift only rewrites rule rows,
which otherwise keep the four-hour rule's gap prediction, so this column is exact:

    pred_all_no_dayshift = gap_sec where dayshift_rows else pred_all

It is scored on slim temporary copies of the prediction files (the run directory
is not modified) and compared with `pred` by the same paired day bootstrap.

Run: .venv/bin/python VALIDATION/tools/summarize_rome_signals.py --run <run id>
Writes rome_signals_summary.json and the derived comparison file in the run
directory and prints markdown tables.
"""

import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import compare
from records import read_run, write_json
from reporting import IDENTITY, COVERAGE, report

SCORES = ['R_A', 'R_B_long', 'R_combined']
DERIVED = 'pred_all_no_dayshift'
POPULATIONS = ['Rome rule rows', 'LIRF no ID, gap 1-4 h', 'LIRF, everything else', 'other airports']


def comparison(run_id, directory, column):
    path = directory / f'comparison_{column}_vs_pred_{run_id}_block1.json'
    if not path.exists():
        compare.main(['--run', run_id, '--prediction-column', column, '--baseline-prediction-column', 'pred'])
    return json.loads(path.read_text())['report']['comparison']['scores']


def derived_comparison(record, directory):
    """Paired comparison of `all` without the day-shift against the control."""
    keep = ['mvt_id', *IDENTITY, *COVERAGE, 'fold', 'seed', 'pred']
    with tempfile.TemporaryDirectory() as tmp:
        for info in record['folds'].values():
            for file in info['predictions'].values():
                d = pd.read_parquet(directory / file)
                if not np.array_equal(np.where(d.dayshift_rows, d.gap_sec, d.pred_dayshift), d.pred):
                    raise ValueError(f'{file}: day-shift rows do not fall back to the rule prediction')
                d[DERIVED] = np.where(d.dayshift_rows, d.gap_sec, d.pred_all)
                d.loc[:, [*keep, DERIVED]].to_parquet(Path(tmp) / file, index=False)
        candidate = dict(copy.deepcopy(record), prediction_column=DERIVED)
        result = report(candidate, tmp, (dict(copy.deepcopy(record), prediction_column='pred'), tmp))
    write_json(directory / f"comparison_{DERIVED}_vs_pred_{record['run_id']}_block1.json",
               {'run_id': record['run_id'], 'baseline_id': record['run_id'], 'prediction_column': DERIVED,
                'baseline_prediction_column': 'pred', 'derived': 'gap_sec where dayshift_rows else pred_all',
                'bootstrap': {k: record['config'][k] for k in ['block_days', 'bootstrap_reps', 'bootstrap_seed']},
                'report': result})
    return result


def populations(record, directory, columns):
    frames = []
    for info in record['folds'].values():
        for file in info['predictions'].values():
            frames.append(pd.read_parquet(directory / file, columns=[
                'airport', 'y', 'gap_sec', 'raw_flight_id_missing', 'rule_triggered', *columns]))
    d = pd.concat(frames, ignore_index=True)
    lirf = d.airport.eq('LIRF')
    no_id_late = lirf & d.raw_flight_id_missing & d.gap_sec.gt(3600) & d.gap_sec.lt(14400)
    d['population'] = np.select([d.rule_triggered, no_id_late, lirf], POPULATIONS[:3], POPULATIONS[3])
    control = (d.pred - d.y) ** 2
    total = control.sum()
    out = {'rows': d.population.value_counts().reindex(POPULATIONS, fill_value=0).astype(int).to_dict(),
           'control_share': (control.groupby(d.population).sum() / total).reindex(POPULATIONS, fill_value=0.0).to_dict()}
    for column in columns:
        if column == 'pred':
            continue
        change = ((d[column] - d.y) ** 2 - control).groupby(d.population).sum() / total
        out[column] = change.reindex(POPULATIONS).fillna(0.0).to_dict()
    return out


def fmt(item):
    lo, hi = item['ci95']
    signs = ''.join('-' if v < 0 else '+' for v in item['difference_by_seed'].values())
    return f"{item['difference']:+.2f} [{lo:+.2f}, {hi:+.2f}] {signs}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', required=True)
    args = ap.parse_args(argv)
    record, directory = read_run(args.run)
    if record.get('status') != 'complete':
        raise ValueError('Summarize a completed run')
    columns = record['config']['candidates']
    scores = {name: comparison(record['run_id'], directory, column)
              for name, column in columns.items() if column != 'pred'}
    derived = derived_comparison(record, directory)
    scores['all_no_dayshift'] = derived['comparison']['scores']
    split = populations(record, directory, list(columns.values()))
    summary = dict(run_id=record['run_id'], control=record['report']['scores'], comparisons=scores,
                   all_no_dayshift_scores=derived['scores'], populations=split, parameters=record['parameters'])
    write_json(directory / 'rome_signals_summary.json', summary)

    print('| Candidate | ' + ' | '.join(f'{s} change [95%] seeds' for s in SCORES) + ' | Decision (combined) |')
    print('|---|' + '---|' * (len(SCORES) + 1))
    for name, item in scores.items():
        print(f'| `{name}` | ' + ' | '.join(fmt(item[s]) for s in SCORES) + f" | {item['R_combined']['decision']} |")
    print('\n| Candidate | ' + ' | '.join(POPULATIONS) + ' |')
    print('|---|' + '---|' * len(POPULATIONS))
    print('| control share of squared error | ' + ' | '.join(
        f"{split['control_share'][p]:.2%}" for p in POPULATIONS) + ' |')
    for name, column in columns.items():
        if column != 'pred':
            print(f'| `{name}` | ' + ' | '.join(f'{split[column][p]:+.3%}' for p in POPULATIONS) + ' |')
    print(f"\nRows per population (pooled): {split['rows']}")
    print(f"Wrote {directory / 'rome_signals_summary.json'}")


if __name__ == '__main__':
    main()
