"""Verify serialized production Rome models against saved probabilities."""
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'VALIDATION'))
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from threadpoolctl import threadpool_limits
import rome_catboost_submit as runner


def main():
    prepared = runner.validate_prepared(portable=True)
    selection = runner.read(runner.OUT / 'selection.json')
    names, cats, params = runner.parent_spec()
    rounds = selection['rounds']
    validation = selection['inner_curve']['validation']['Logloss']
    assert rounds == int(np.argmin(validation)) + 1
    inner = CatBoostClassifier().load_model(str(runner.verify(selection['model'])))
    assert inner.tree_count_ == rounds and inner.feature_names_ == names
    report = dict(prepared_sha256=runner.sha256(runner.OUT / 'prepared.json'),
        source_sha256=runner.sha256(Path(__file__)),
        selection_sha256=runner.sha256(runner.OUT / 'selection.json'),
        selected_rounds=rounds, stop_rows=selection['held_rows'], seed_audits={})
    for seed in runner.SEEDS:
        record = runner.completed_seed(seed, rounds)
        assert record is not None
        model = CatBoostClassifier().load_model(str(runner.verify(record['model'])))
        assert model.tree_count_ == rounds and model.feature_names_ == names
        assert model.random_seed_ == seed
        model_params = model.get_all_params()
        assert model_params['loss_function'] == params['loss_function']
        assert model_params['depth'] == params['depth']
        assert model_params['l2_leaf_reg'] == params['l2_leaf_reg']
        np.testing.assert_allclose(model_params['learning_rate'], params['learning_rate'], rtol=1e-6)
        views = {}
        for view in runner.VIEWS:
            packet = pd.read_pickle(runner.verify(prepared['views'][view]['packet']))
            saved = pd.read_parquet(runner.verify(record['predictions'][view]))
            np.testing.assert_array_equal(saved.mvt_id, packet['ids'])
            p = saved.p.to_numpy(float)
            assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
            assert np.all(p[~packet['use']] == 0)
            chosen = np.unique(np.linspace(0, len(packet['X']) - 1, 512).astype(int))
            query = runner.cbj.frame(packet['X'].iloc[chosen][names], cats)
            reload_p = model.predict_proba(Pool(query, cat_features=cats), thread_count=2)[:, 1]
            expected = p[packet['use']][chosen]
            np.testing.assert_array_equal(reload_p, expected)
            views[view] = dict(total_rows=len(saved), eligible_rows=int(packet['use'].sum()),
                               reload_samples=len(chosen), exact_reload_probability_match=True)
        report['seed_audits'][str(seed)] = dict(model_sha256=record['model']['sha256'], views=views)
    output = runner.OUT / 'portable_model_audit.json'
    if output.exists() and runner.read(output) != report:
        raise ValueError('Prior portable model audit differs; preserve it for inspection')
    runner.write_json(output, report)
    print(json.dumps(dict(all_checks_pass=True, rounds=rounds, seeds=runner.SEEDS,
                         reload_samples=512 * len(runner.VIEWS) * len(runner.SEEDS))))


if __name__ == '__main__':
    with threadpool_limits(2):
        main()
