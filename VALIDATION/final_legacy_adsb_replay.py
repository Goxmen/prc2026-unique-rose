"""Saved v45/v46 inference with exact scored controls before final-set replay."""
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import adsblol_submit as recipe
from records import sha256, write_json

ROOT = recipe.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_legacy_adsb_replay'


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    cache_path = ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json'
    cache = json.loads(cache_path.read_text())
    records = {}
    sources = [Path(__file__), Path(recipe.__file__), Path(recipe.signals.__file__),
               Path(recipe.v43s.v42.parent.__file__), cache_path]
    for version in [45, 46]:
        recipe.configure(version)
        folder = recipe.OUT / f'submission_v{version}'
        record = json.loads((folder / 'progress.json').read_text())
        paths = {}
        for part, key in recipe.MODEL_KEYS.items():
            m = record['models'][key]
            path = folder / m['path']
            assert sha256(path) == m['artifacts'][m['path']]
            paths[part] = path
            sources.append(path)
        expected = ROOT / cache[str(version)]['file']
        assert sha256(expected) == cache[str(version)]['sha256']
        sources += [folder / 'progress.json', expected]
        records[version] = dict(models=paths, training=record['training'])
    identity = dict(sources=[art(p) for p in sources], no_fit=True, no_upload=True,
                    gate='Every old integer prediction exact before final inference',
                    versions=[45, 46], final_model_reload=True)
    if (OUT / 'identity.json').exists():
        assert json.loads((OUT / 'identity.json').read_text()) == identity
    else:
        write_json(OUT / 'identity.json', identity)
    for view in ['old', 'final']:
        if (OUT / f'{view}.json').exists():
            continue
        source = ROOT / f'VALIDATION/runs/v3_20261008_final_submit/work/inputs_{view}.pkl'
        data = pd.read_pickle(source)
        template = ROOT / ('data/submitting.parquet' if view == 'old' else 'data/final_submitting.parquet')
        ids = pd.read_parquet(template, columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
        assert len(set(ids)) == len(ids) and set(data['rank_ids']) == set(ids)
        results = []
        for version, record in records.items():
            if view == 'final' and not json.loads((OUT / f'old_v{version}.json').read_text())['all_original_predictions_exact']:
                continue
            recipe.configure(version)
            recipe.run.THREADS = 4
            models = {k: lgb.Booster(model_file=str(p)) for k, p in record['models'].items()}
            reloads = {k: lgb.Booster(model_file=str(p)) for k, p in record['models'].items()} if view == 'final' else None
            frames = []
            for item in data['months'].values():
                V, raw = item['V'], item['raw']
                for model in models.values():
                    assert set(model.feature_name()).issubset(V.columns)
                frame, masks = recipe.predict(models['ordinary'], models['classifier'], V, raw,
                                               record['training']['parameters'], record['training']['corrections'])
                if reloads is not None:
                    check, masks2 = recipe.predict(reloads['ordinary'], reloads['classifier'], V, raw,
                                                   record['training']['parameters'], record['training']['corrections'])
                    pd.testing.assert_frame_equal(frame, check, check_exact=True)
                    assert masks == masks2
                frames.append(frame[['mvt_id', 'pred']])
            frame = pd.concat(frames, ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
            frame['submitted'] = np.rint(np.maximum(frame.pred, 0))
            assert frame.mvt_id.is_unique and np.isfinite(frame.submitted).all()
            path = OUT / f'{view}_v{version}.parquet'
            frame.to_parquet(path, index=False)
            result = dict(version=version, rows=len(frame), identity=art(OUT / 'identity.json'),
                          inputs=art(source), prediction=art(path), template=art(template))
            if view == 'old':
                expected = pd.read_parquet(ROOT / cache[str(version)]['file']).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.loc[ids]
                delta = frame.submitted.to_numpy() - expected.to_numpy()
                result.update(all_original_predictions_exact=bool((delta == 0).all()),
                              changed_rows=int(np.count_nonzero(delta)), rms_difference=float(np.sqrt(np.mean(delta ** 2))),
                              max_abs_difference=float(np.max(np.abs(delta))))
            else:
                result.update(old_control_exact=True, all_final_reloads_exact=True)
            write_json(OUT / f'{view}_v{version}.json', result)
            print({k: v for k, v in result.items() if not isinstance(v, dict)}, flush=True)
            results.append(result)
            del models, reloads, frames
            gc.collect()
        write_json(OUT / f'{view}.json', dict(view=view, results=results, no_fit=True, no_upload=True))
        del data
        gc.collect()


if __name__ == '__main__':
    with threadpool_limits(4):
        run()
