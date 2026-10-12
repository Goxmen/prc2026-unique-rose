"""Replay retained historical boosters; only exact old controls reach final inputs.

No fits or candidate selection. The control model's possible v12/13/15 mappings
are hypotheses, subject to the same all-row integer equality as v17--v21.
"""
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import overnight_models as recipe
from records import sha256, write_json
from registered_rules import apply_rule

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'VALIDATION/runs/overnight_20260917'
OUT = ROOT / 'VALIDATION/runs/v3_20261012_final_legacy_overnight_replay'
MAPPINGS = {
    'control': [(12, 'off'), (13, 'rome_6h'), (15, 'rome_4h')],
    'no_month': [(17, 'rome_4h')],
    'no_history': [(18, 'rome_4h')],
    'clock_features': [(19, 'rome_4h')],
    'small_trees': [(20, 'rome_4h')],
    'rule_routing': [(21, 'rome_4h')],
}


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha256(path))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    progress_path = RUN / 'progress.json'
    progress = json.loads(progress_path.read_text())
    cache_path = ROOT / 'data/derived/scored_submission_cache_20261011/manifest.json'
    cache = json.loads(cache_path.read_text())
    paths = [Path(__file__), Path(recipe.__file__), Path(recipe.modeling.__file__),
             ROOT / 'pipeline/registered_rules.py', ROOT / 'pipeline/prediction_rules.py',
             progress_path, RUN / 'submission_manifest.json', cache_path]
    models = {}
    for name in MAPPINGS:
        record = progress['refit_models'][f'refit_2025__{name}__42']
        path = RUN / record['path']
        assert sha256(path) == record['artifacts'][record['path']]
        models[name] = path
        paths.append(path)
        for version, _ in MAPPINGS[name]:
            ref = cache[str(version)]
            path = ROOT / ref['file']
            assert sha256(path) == ref['sha256']
            paths.append(path)
    identity = dict(sources=[art(p) for p in paths], mappings=MAPPINGS,
                    gate='Every original integer prediction exact before final inference',
                    no_fit=True, no_upload=True, final_reload_required=True)
    # JSON round trip normalizes tuples in the immutable identity.
    identity = json.loads(json.dumps(identity))
    if (OUT / 'identity.json').exists():
        assert json.loads((OUT / 'identity.json').read_text()) == identity
    else:
        write_json(OUT / 'identity.json', identity)
    for view in ['old', 'final']:
        if (OUT / f'{view}.json').exists():
            continue
        source = ROOT / f'VALIDATION/runs/v3_20261008_final_submit/work/inputs_{view}.pkl'
        packet = pd.read_pickle(source)
        template = ROOT / ('data/submitting.parquet' if view == 'old' else 'data/final_submitting.parquet')
        ids = pd.read_parquet(template, columns=['MVT_ID_mvt']).MVT_ID_mvt.to_numpy()
        assert len(set(ids)) == len(ids) and set(packet['rank_ids']) == set(ids)
        results = []
        for name, variants in MAPPINGS.items():
            if view == 'final':
                variants = [(v, rule) for v, rule in variants
                            if json.loads((OUT / f'old_v{v}.json').read_text())['all_original_predictions_exact']]
            if not variants:
                continue
            if all((OUT / f'{view}_v{v}.json').exists() for v, _ in variants):
                results.extend(json.loads((OUT / f'{view}_v{v}.json').read_text()) for v, _ in variants)
                continue
            booster = lgb.Booster(model_file=str(models[name]))
            reload = lgb.Booster(model_file=str(models[name])) if view == 'final' else None
            pieces = {v: [] for v, _ in variants}
            max_reload = 0.0
            for item in packet['months'].values():
                V, raw = item['V'], item['raw']
                pred = recipe.predict_model(booster, V, name, threads=4)
                if reload is not None:
                    check = recipe.predict_model(reload, V, name, threads=4)
                    max_reload = max(max_reload, float(np.max(np.abs(check - pred))))
                    assert np.array_equal(pred, check)
                for version, rule in variants:
                    corrected, _ = apply_rule(pred, raw, rule)
                    pieces[version].append(pd.DataFrame(dict(mvt_id=raw.mvt_id.to_numpy(),
                        pred_raw=pred, pred=corrected, submitted=np.rint(np.maximum(corrected, 0)))))
            for version, rule in variants:
                frame = pd.concat(pieces[version], ignore_index=True).set_index('mvt_id').loc[ids].reset_index()
                assert frame.mvt_id.is_unique and np.isfinite(frame[['pred_raw', 'pred', 'submitted']]).all().all()
                path = OUT / f'{view}_v{version}.parquet'
                frame.to_parquet(path, index=False)
                result = dict(version=version, recipe=name, rule=rule, rows=len(frame),
                              identity=art(OUT / 'identity.json'), inputs=art(source), template=art(template),
                              prediction=art(path), model=art(models[name]))
                if view == 'old':
                    ref = ROOT / cache[str(version)]['file']
                    expected = pd.read_parquet(ref).set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.loc[ids].to_numpy()
                    delta = frame.submitted.to_numpy() - expected
                    result.update(all_original_predictions_exact=bool((delta == 0).all()),
                        changed_rows=int(np.count_nonzero(delta)), rms_difference=float(np.sqrt(np.mean(delta ** 2))),
                        max_abs_difference=float(np.max(np.abs(delta))))
                else:
                    result.update(old_control_exact=True, all_final_reloads_exact=max_reload == 0,
                                  maximum_reload_difference=max_reload)
                write_json(OUT / f'{view}_v{version}.json', result)
                print({k: v for k, v in result.items() if not isinstance(v, dict)}, flush=True)
                results.append(result)
            del booster, reload, pieces
            gc.collect()
        write_json(OUT / f'{view}.json', dict(view=view, results=results, no_fit=True, no_upload=True))
        del packet
        gc.collect()


if __name__ == '__main__':
    with threadpool_limits(4):
        run()
