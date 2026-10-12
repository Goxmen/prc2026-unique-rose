"""Reuse the selected public-input recipe on the separate four-month namespace.

No training, model selection, version assignment or upload. The old ranking
view must replay exactly; the retained independent SQL audit checks the new
view with only its input/output filenames changed.
"""
import argparse
import ast
import inspect
import textwrap
import types
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import public_delay_features as feature
from records import write_json
from tools import audit_public_delay_features as sql_audit

ROOT = feature.base.ROOT
OUT = ROOT / 'data/derived/final_public_20261012'
RECORDS = ROOT / 'VALIDATION/runs/v3_20261012_final_public_features'
art = feature.base.artifact
DESIGN = dict(recipe='Unchanged twelve selected public daily-departure inputs',
    source_year=2026, views=['old', 'final'], final_rows=670790,
    months=['2026-01', '2026-02', '2026-06', '2026-07'],
    controls='All old-view feature values replay exactly against retained production inputs',
    audit='Retained independent SQL equations; only input and output locations changed',
    no_label_reads=True, no_fit=True, no_candidate=True, no_upload=True)


def audit_function():
    tree = ast.parse(textwrap.dedent(inspect.getsource(sql_audit.main)))
    replacements = {
        "from read_parquet('data/training_2025-*.parquet') where PHASE_mvt='DEP' ":
            "from read_parquet('data/final_ranking.parquet') where PHASE_mvt='DEP' ",
        'all_pre_departure_delays_2025.csv': 'all_pre_departure_delays_2026.csv',
        'features2025.parquet': 'features.parquet',
    }
    counts = dict.fromkeys(replacements, 0)

    class Substitute(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                value = node.value
                for old, new in replacements.items():
                    if old in value:
                        counts[old] += value.count(old)
                        value = value.replace(old, new)
                return ast.copy_location(ast.Constant(value=value), node)
            return node

    tree = Substitute().visit(tree)
    assert list(counts.values()) == [1, 1, 1], counts
    scope = dict(sql_audit.main.__globals__,
        feature=types.SimpleNamespace(base=feature.base, FEATURES=feature.FEATURES, OUT=OUT),
        __file__=__file__)
    exec(compile(ast.fix_missing_locations(tree), '<final-public-input-SQL-paths-only>', 'exec'), scope)
    return scope['main']


def prepare():
    assert not (OUT / 'manifest.json').exists() and not (OUT / 'features.parquet').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    RECORDS.mkdir(parents=True, exist_ok=True)
    source = feature.base.public(2026)
    prior = ROOT / 'VALIDATION/runs/v3_20261011_rome_public_delay_submit/ranking_features.parquet'
    old = feature.build(feature.load([ROOT / 'data/ranking.parquet']), source).set_index('mvt_id')
    retained = pd.read_parquet(prior).set_index('mvt_id')
    assert len(old) == len(retained) == 344841 and old.index.is_unique and retained.index.is_unique
    pd.testing.assert_frame_equal(old.loc[retained.index], retained, check_exact=True)
    del old, retained
    raw = feature.load([ROOT / 'data/final_ranking.parquet'])
    assert set(raw.columns) == {'mvt_id', 'airport', 'day', 'gap', 'clock', 'fltrule'}
    assert len(raw) == raw.mvt_id.nunique() == DESIGN['final_rows']
    assert sorted(raw.day.str[:7].unique()) == DESIGN['months']
    result = feature.build(raw, source).set_index('mvt_id')
    ids = pd.read_parquet(ROOT / 'data/final_submitting.parquet', columns=['MVT_ID_mvt']).MVT_ID_mvt
    assert len(result) == len(ids) and result.index.is_unique and ids.is_unique
    result = result.loc[ids].reset_index()
    np.testing.assert_array_equal(result.mvt_id, ids)
    assert list(result) == ['mvt_id', *feature.FEATURES]
    assert np.isfinite(result[feature.FEATURES]).all().all()
    result.to_parquet(OUT / 'features.parquet', index=False)
    raw.to_parquet(OUT / 'inputs.parquet', index=False)
    paths = [Path(__file__), Path(feature.__file__), Path(feature.base.__file__),
        Path(sql_audit.__file__), ROOT / 'data/ranking.parquet', ROOT / 'data/final_ranking.parquet',
        ROOT / 'data/final_submitting.parquet', prior,
        feature.base.SOURCE / 'all_pre_departure_delays_2026.csv']
    value = dict(design=DESIGN, source=art(Path(__file__)), sources=[art(p) for p in paths],
        rows=len(result), months=raw.day.str[:7].value_counts().sort_index().to_dict(),
        old_all_row_exact_replay=True, output=art(OUT / 'features.parquet'), raw=art(OUT / 'inputs.parquet'))
    write_json(OUT / 'manifest.json', value)
    write_json(RECORDS / 'prepared.json', value)
    print(dict(prepared=True, rows=len(result), old_replay_exact=True), flush=True)


def audit():
    import json
    assert not (OUT / 'audit.json').exists()
    manifest = json.loads((OUT / 'manifest.json').read_text())
    assert manifest['design'] == DESIGN
    for record in [*manifest['sources'], manifest['output'], manifest['raw']]:
        assert art(ROOT / record['path']) == record
    audit_function()()
    value = json.loads((OUT / 'audit.json').read_text())
    assert value['status'] == 'pass' and value['rows'] == DESIGN['final_rows']
    value.update(original_auditor=art(Path(sql_audit.__file__)),
        specialization='Three input/output filename substitutions; all SQL equations unchanged',
        old_view_replay_exact=manifest['old_all_row_exact_replay'])
    write_json(OUT / 'audit.json', value)
    write_json(RECORDS / 'audit.json', value)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['check', 'prepare', 'audit'])
    with threadpool_limits(1):
        stage = parser.parse_args().stage
        if stage == 'check':
            audit_function()
            print('Unchanged independent SQL equations; three path substitutions verified')
        else:
            globals()[stage]()
