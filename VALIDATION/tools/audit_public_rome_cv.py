"""Reload every held probability and independently recompute Rome CV scores."""
import argparse
import gc
import importlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import duckdb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from threadpoolctl import threadpool_limits
from records import write_json


def main(name):
    run = importlib.import_module({'peers': 'rome_public_peers_jan_jul',
        'punctuality': 'rome_punctuality_jan_jul'}[name])
    run.guard()
    checks = []
    for month in run.cat.parent.ROUNDS:
        data = pd.read_pickle(run.OUT / 'inputs' / f'{month}.pkl')
        original = pd.read_pickle(run.r20.OUT / 'inputs' / f'{month}.pkl')
        for column in ['target', 'train_ids', 'ids', 'use']:
            np.testing.assert_array_equal(data[column], original[column])
        for part in ['A', 'B']:
            pd.testing.assert_frame_equal(data[part][original[part].columns], original[part])
        for seed in run.cat.parent.SEEDS:
            path = run.OUT / 'predictions' / f'{month}_s{seed}.parquet'
            rec = run.cat.parent.shared.ts.read(path.with_suffix('.json'))
            assert rec['identity'] == run.artifact(run.OUT / 'identity.json')
            assert rec['prediction'] == run.artifact(path)
            assert rec['model'] == run.artifact(ROOT / rec['model']['path'])
            model = CatBoostClassifier()
            model.load_model(str(ROOT / rec['model']['path']))
            assert model.feature_names_ == list(data['B']) == rec['columns']
            frame = run.cat.parent.parent.cbj.frame(data['B'], rec['categoricals'])
            p = np.zeros(len(data['ids']))
            p[data['use']] = model.predict_proba(Pool(frame, cat_features=rec['categoricals']), thread_count=2)[:, 1]
            stored = pd.read_parquet(path)
            np.testing.assert_array_equal(stored.mvt_id, data['ids'])
            np.testing.assert_array_equal(p, stored.p)
            checks.append(dict(month=month, seed=seed, rows=len(p),
                model_rows=len(frame), all_probabilities_exact=True))
            del model, frame
        del data, original
        gc.collect()
    result = run.cat.parent.shared.ts.read(run.OUT / 'result.json')
    with duckdb.connect(config={'threads': 1}) as con:
        rows = con.execute('''select month, sqrt(avg(pow(control-y,2))) control,
            sqrt(avg(pow(pred-y,2))) pred, sqrt(avg(pow(v75-y,2))) v75,
            sqrt(avg(pow(transferred-y,2))) transferred from read_parquet(?)
            group by month order by month''', [str(run.OUT / 'scored.parquet')]).fetchdf()
    for row in rows.to_dict('records'):
        for column in ['control', 'pred', 'v75', 'transferred']:
            assert abs(row[column]-result['scores'][row['month']][column]) < 1e-8
    audit = dict(status='pass', source=run.artifact(__file__), identity=run.artifact(run.OUT / 'identity.json'),
        result=run.artifact(run.OUT / 'result.json'), model_reload=checks,
        independent_sql_scores=True, original_rows_targets_and_columns_exact=True)
    write_json(run.OUT / 'audit.json', audit)
    print(audit, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', choices=['peers', 'punctuality'], required=True)
    args = parser.parse_args()
    with threadpool_limits(2):
        main(args.run)
