"""Jan/Jul CV of a CatBoost ordinary regressor beside v48's LightGBM + network blend (idea 3).

Same training rows, v46's 134 inputs and held-out rows as `ideas_jan_jul.py`; categorical inputs
(the 11 LightGBM categoricals) are CatBoost categoricals, missing values as their own category.

Frozen recipe (fixed before any run): GPU, RMSE loss, depth 8, learning rate 0.05, l2_leaf_reg 3,
border_count 254, at most 10,000 iterations. The iteration count is chosen as `ple`'s epochs
were: fit on the training months except the last (November / December), keep the iteration with
the lowest RMSE on that last month (patience 300), then refit on all training months for that
many iterations. Seeds 42/142/242. Held-out base predictions go to the ideas run directory as arm
`catboost`; scoring is in `VALIDATION/tools/score_ideas_jan_jul.py`.

Run: .venv/bin/python VALIDATION/catboost_jan_jul.py [--seeds 42 142 242]
"""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import ideas_jan_jul as ideas
import overnight_run
import rome_signals as signals
import seasonal
import tuning_seasonal as ts
from records import sha256, write_json

OUT = ideas.OUT
RECIPE = dict(loss_function='RMSE', depth=8, learning_rate=0.05, l2_leaf_reg=3, border_count=254,
              iterations=10000, od_patience=300)


def frame(X, cats):
    X = X.copy()
    for c in cats:
        X[c] = X[c].astype(object).where(X[c].notna(), '__missing__').astype(str)
    return X


GPU_RAM_PART = 0.95       # CatBoost's default; lower it to share the GPU (memory only, same model)


def model(seed, iterations, **extra):
    return CatBoostRegressor(gpu_ram_part=GPU_RAM_PART, loss_function=RECIPE['loss_function'], depth=RECIPE['depth'],
                             learning_rate=RECIPE['learning_rate'], l2_leaf_reg=RECIPE['l2_leaf_reg'],
                             border_count=RECIPE['border_count'], iterations=iterations, random_seed=seed,
                             task_type='GPU', devices='0', verbose=1000, allow_writing_files=False, **extra)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--seeds', type=int, nargs='+', default=ideas.SEEDS)
    ap.add_argument('--gpu-ram-part', type=float, default=GPU_RAM_PART)
    args = ap.parse_args()
    globals()['GPU_RAM_PART'] = args.gpu_ram_part
    (OUT / 'predictions').mkdir(parents=True, exist_ok=True)
    ideas.ARMS = {'catboost': []}            # load only v46's inputs
    ideas.load()
    record_path = OUT / 'catboost.json'
    record = json.loads(record_path.read_text()) if record_path.exists() else {}
    record['recipe'], record['runner'] = RECIPE, sha256(Path(__file__))
    tmonth = {}                               # calendar month of every training row
    dep, arr = overnight_run.load_inputs(1.)
    for month in ts.SPLITS:
        td, _, _ = seasonal.split_frames(dep, arr, dict(train=ts.FOLDS[month], val=[month]))
        idx = ideas.DATA['items'][month]['X'].index
        tmonth[month] = pd.to_datetime(td.loc[idx, 'MVT_TIME_UTC_mvt'], utc=True).dt.strftime('%Y-%m').to_numpy()
    del dep, arr
    gc.collect()
    for seed in args.seeds:
        path = OUT / 'predictions' / f'catboost_seed{seed}.parquet'
        if path.exists():
            continue
        frames = []
        for month in ts.SPLITS:
            item = ideas.DATA['items'][month]
            cols = item['names']
            cats = signals._categorical(cols)
            X, V = frame(item['X'][cols], cats), frame(item['Vn'][cols], cats)
            last = tmonth[month] == max(ts.FOLDS[month])
            started = time.monotonic()
            inner = model(seed, RECIPE['iterations'], od_type='Iter', od_wait=RECIPE['od_patience'], use_best_model=True)
            inner.fit(Pool(X[~last], item['y'][~last], cat_features=cats),
                      eval_set=Pool(X[last], item['y'][last], cat_features=cats))
            best = int(inner.get_best_iteration()) + 1
            inner_rmse = float(inner.get_best_score()['validation']['RMSE'])
            del inner
            final = model(seed, best)
            final.fit(Pool(X, item['y'], cat_features=cats))
            base = np.maximum(final.predict(Pool(V, cat_features=cats)), 0)
            imp = dict(zip(cols, final.get_feature_importance()))
            record.setdefault(str(seed), {})[month] = dict(
                iterations=best, inner_month=max(ts.FOLDS[month]), inner_rmse=inner_rmse,
                minutes=(time.monotonic() - started) / 60,
                top_importance=dict(sorted(imp.items(), key=lambda kv: -kv[1])[:15]))
            write_json(record_path, record)
            ideas.log(f'catboost seed {seed} {month}: {best} iterations (inner RMSE {inner_rmse:.1f}), '
                      f'{(time.monotonic() - started) / 60:.1f} min')
            frames.append(pd.DataFrame({'mvt_id': item['ids'], 'month': month, 'base': base.astype(np.float32)}))
            del final, X, V
            gc.collect()
        overnight_run.save_frame(path, pd.concat(frames, ignore_index=True))


if __name__ == '__main__':
    main()
