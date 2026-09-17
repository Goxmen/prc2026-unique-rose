"""Explain a residual regression using saved prediction errors, without new fits."""

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from records import ROOT, read_run, write_json


def flight_errors(run, directory, names):
    parts = []
    columns = ['mvt_id', 'airport', 'y', 'clock_baseline_sec', 'clock_baseline_source', 'rule_triggered']
    for name in names:
        base = None
        for seed in run['config']['seeds']:
            frame = pd.read_parquet(directory/run['folds'][name]['predictions'][str(seed)],
                                    columns=[*columns, 'pred', 'model_pred']).set_index('mvt_id').sort_index()
            if base is None:
                base = frame.drop(columns=['pred', 'model_pred']).copy()
                base[['mean_squared_error', 'mean_prediction', 'mean_model_prediction']] = 0.
            else:
                pd.testing.assert_frame_equal(base[columns[1:]], frame[columns[1:]])
            base['mean_squared_error'] += (frame.y-frame.pred)**2/len(run['config']['seeds'])
            base['mean_prediction'] += frame.pred/len(run['config']['seeds'])
            base['mean_model_prediction'] += frame.model_pred/len(run['config']['seeds'])
        parts.append(base)
    values = pd.concat(parts)
    groups = values.groupby(level=0, sort=True)
    out = groups[columns[1:]].first()
    out[['mean_squared_error', 'mean_prediction', 'mean_model_prediction']] = groups[
        ['mean_squared_error', 'mean_prediction', 'mean_model_prediction']].mean()
    return out


def stats(frame):
    return {'n': len(frame), 'direct_rmse': float(np.sqrt(frame.direct_squared_error.mean())),
            'residual_rmse': float(np.sqrt(frame.mean_squared_error.mean())),
            'delta_sse': float(frame.delta_squared_error.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--t1', required=True)
    parser.add_argument('--t2', required=True)
    args = parser.parse_args()
    t1, d1 = read_run(args.t1)
    t2, d2 = read_run(args.t2)
    out = {'direct': t1['run_id'], 'residual': t2['run_id'], 'groups': {},
           'scope': 'Post-result error diagnosis only; no new fit, threshold or model selection sweep.'}
    for group, names in [('R_A', [f'A{i}' for i in range(1, 7)]), ('R_B_long', ['B3', 'B4', 'B7'])]:
        direct = flight_errors(t1, d1, names)
        residual = flight_errors(t2, d2, names)
        pd.testing.assert_frame_equal(direct[['airport', 'y', 'clock_baseline_sec', 'clock_baseline_source', 'rule_triggered']],
                                      residual[['airport', 'y', 'clock_baseline_sec', 'clock_baseline_source', 'rule_triggered']])
        x = residual.assign(direct_squared_error=direct.mean_squared_error,
                            direct_prediction=direct.mean_prediction)
        x['delta_squared_error'] = x.mean_squared_error-x.direct_squared_error
        np.testing.assert_allclose(np.sqrt(x.mean_squared_error.mean()), t2['report']['scores'][group], rtol=0, atol=1e-9)
        np.testing.assert_allclose(np.sqrt(x.direct_squared_error.mean()), t1['report']['scores'][group], rtol=0, atol=1e-9)
        masks = {'all': np.ones(len(x), bool),
                 'baseline_at_most_2h': x.clock_baseline_sec.le(7200),
                 'baseline_2h_to_24h': x.clock_baseline_sec.gt(7200) & x.clock_baseline_sec.le(86400),
                 'baseline_over_24h': x.clock_baseline_sec.gt(86400),
                 'ordinary_baseline_over_24h': x.y.le(7200) & x.clock_baseline_sec.gt(86400)}
        out['groups'][group] = {'slices': {name: stats(x.loc[mask]) for name, mask in masks.items()},
            'baseline_sources': {int(source): stats(rows) for source, rows in x.groupby('clock_baseline_source')},
            'worst_changes': x.sort_values(['delta_squared_error', 'mvt_id'], ascending=[False, True]).head(10).reset_index().to_dict('records')}
    path = ROOT/'VALIDATION/reports/stage3_residual_errors.json'
    write_json(path, out)
    print(path)


if __name__ == '__main__':
    main()
