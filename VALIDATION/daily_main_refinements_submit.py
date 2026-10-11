"""Add the selected daily main tree to the combined production component.

Reuse all other models and full-year daily calibration. Recompute the source
correction after replacing the main tree, with exact parent replay. No version
or upload is assigned here; this small component belongs in the combined file.
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import catboost_refinements_submit as parent
import daily_main_refinements_cv as cv
import daily_main_submit as main
from records import write_json
from submit import fill_template

ROOT, base = parent.ROOT, parent.parent
OUT = ROOT / 'VALIDATION/runs/v3_20261011_daily_main_refinements_submit'
MANIFEST = OUT / 'identity.json'
art, read, verify = parent.art, parent.read, parent.verify
DESIGN = dict(parent='selected full174 CatBoost/airport/t5/monthly/daily-quality component',
    change='main LightGBM184 inputs with ten daily-taxi features',
    calibration='reuse full2025 parameters; recompute fixed source shift',
    cv='daily_main_refinements_cv.combined; reused development and small incremental gain',
    no_fits=True, no_version=True, no_upload=True)


def guard():
    for module in [main, cv, parent]:
        assert read(module.OUT / 'audit.json')['status'] == 'pass'
    parent.guard(); cv.guard(); main.validate()
    result = read(cv.OUT / 'result.json')
    assert result['paired']['primary']['decision'] == 'improvement'
    assert all(x['combined'] < x['full_cat'] for x in result['scores'].values())
    paths = [Path(__file__), Path(main.__file__), Path(cv.__file__), Path(parent.__file__),
        Path(base.__file__), Path(base.daily.__file__), Path(base.aggregate.__file__),
        Path(base.selected.rome.physical.parent.__file__), Path(base.constraints.__file__),
        Path(base.constraints.exclusion.__file__),
        main.MANIFEST, main.OUT / 'audit.json', main.OUT / 'predictions/old.json', main.OUT / 'predictions/old.parquet',
        cv.OUT / 'identity.json', cv.OUT / 'result.json', cv.OUT / 'audit.json', cv.OUT / 'scored.parquet',
        parent.MANIFEST, parent.OUT / 'result.json', parent.OUT / 'audit.json', parent.OUT / 'predictions.parquet',
        parent.OUT / 'candidate.parquet', parent.OUT / 'cv.parquet', base.OUT / 'parameters.json',
        base.OUT / 'predictions.parquet', base.aggregate.OUT / 'predictions.parquet',
        base.aggregate.V76 / 'predictions.parquet', base.daily.SOURCE / 'daily_runway.parquet',
        base.daily.base.OUT / 'ranking_inputs.parquet', base.constraints.exclusion.OUT / 'ranking_matches.parquet',
        ROOT / 'data/submitting.parquet']
    value = dict(design=DESIGN, sources=[art(p) for p in paths])
    OUT.mkdir(parents=True, exist_ok=True)
    if MANIFEST.exists():
        assert read(MANIFEST) == value
    else:
        assert not list(OUT.glob('*.parquet'))
        write_json(MANIFEST, value)


def prepare():
    guard()
    assert not (OUT / 'result.json').exists(), 'Preserve completed component'
    data = base.selected.rome.physical.parent.ranking_data('old'); ids = np.asarray(data['rank_ids'])
    control = pd.read_parquet(parent.OUT / 'predictions.parquet')
    original_base = pd.read_parquet(base.OUT / 'predictions.parquet')
    new_main = pd.read_parquet(main.OUT / 'predictions/old.parquet')
    original = pd.read_parquet(base.aggregate.V76 / 'predictions.parquet')
    aggregate = pd.read_parquet(base.aggregate.OUT / 'predictions.parquet')
    for frame in [control, original_base, new_main, original, aggregate]:
        np.testing.assert_array_equal(frame.mvt_id, ids)
    raw = pd.read_parquet(base.daily.base.OUT / 'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(base.constraints.exclusion.OUT / 'ranking_matches.parquet')
    inputs = control[['mvt_id', 'airport', 'local_day', 'runway', 'eligible', 'month']]
    outputs, cells = {}, []
    for arm, values in [('control', original_base.main.to_numpy(float)), ('candidate', new_main.pred.to_numpy(float))]:
        tree = np.column_stack([values, original_base.airport_new.to_numpy(float),
            control.candidate_catboost.to_numpy(float)]).mean(axis=1)
        direct, _ = base.selected.rome.physical.parent.final_frame(data,
            original.neural_new.to_numpy(), tree, original.p.to_numpy())
        refined_float = base.selected.main_candidate.constrained(ids, raw,
            aggregate.new74_serialized.to_numpy(), direct.pred.to_numpy(), original.control.to_numpy(), matches)
        refined = np.rint(np.maximum(refined_float, 0))
        _, cell, shift = base.daily.apply(inputs, refined, base.daily.public(), read(base.OUT / 'parameters.json'))
        applied = np.where(control.quality_eligible, shift, 0.)
        bounded = base.aggregate.constrain(ids, raw, refined + applied, matches)
        final = np.rint(np.maximum(bounded, 0)).astype('int32')
        if arm == 'control':
            for actual, key in [(tree, 'tree'), (direct.pred, 'direct'), (refined_float, 'refined_float'),
                (refined, 'refined'), (shift, 'shift'), (applied, 'applied'), (bounded, 'bounded'), (final, 'pred')]:
                np.testing.assert_array_equal(actual, control['candidate_' + key])
        outputs[arm] = dict(main=values, tree=tree, direct=direct.pred.to_numpy(), refined_float=refined_float,
            refined=refined, shift=shift, applied=applied, bounded=bounded, pred=final)
        cells.append(cell.assign(arm=arm))
    frame = inputs.copy(); frame['quality_eligible'] = control.quality_eligible.to_numpy()
    for arm, values in outputs.items():
        for key, value in values.items():
            frame[arm + '_' + key] = value
    frame.to_parquet(OUT / 'predictions.parquet', index=False)
    pd.concat(cells, ignore_index=True).to_parquet(OUT / 'cells.parquet', index=False)
    template = pq.read_table(ROOT / 'data/submitting.parquet')
    table = fill_template(template, ids, outputs['candidate']['pred'])
    assert table.schema == template.schema
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt':
            assert table[name].equals(template[name])
    pq.write_table(table, OUT / 'candidate.parquet')
    assert pq.read_table(OUT / 'candidate.parquet').equals(table)
    held = pd.read_parquet(cv.OUT / 'scored.parquet')[['mvt_id', 'month', 'day', 'y', 'full_cat', 'combined']].rename(
        columns={'full_cat': 'control', 'combined': 'pred'})
    held_parent = pd.read_parquet(parent.OUT / 'cv.parquet').set_index('mvt_id')
    np.testing.assert_array_equal(held.control, held_parent.loc[held.mvt_id, 'pred'])
    held.to_parquet(OUT / 'cv.parquet', index=False)
    validation = read(cv.OUT / 'result.json')
    write_json(OUT / 'result.json', dict(design=DESIGN, identity=art(MANIFEST), rows=len(ids),
        jan_jul_cv=validation['cv']['combined'], cv_months={m: x['combined'] for m, x in validation['scores'].items()},
        paired=validation['paired']['primary'], parent_control_all_stages_exact=True,
        full2025_calibration_reused=art(base.OUT / 'parameters.json'),
        artifacts=[art(OUT / p) for p in ['candidate.parquet', 'predictions.parquet', 'cells.parquet', 'cv.parquet']],
        changed_rows=int((frame.candidate_pred != frame.control_pred).sum()),
        prediction_rms_change=float(np.sqrt(np.mean((frame.candidate_pred - frame.control_pred).to_numpy(float)**2))),
        no_version=True, no_upload=True))


def audit():
    guard()
    assert not (OUT / 'audit.json').exists(), 'Preserve completed audit'
    result = read(OUT / 'result.json')
    for entry in result['artifacts']:
        verify(entry)
    with duckdb.connect(config={'threads': 1}) as con:
        for name, path in [('c', OUT / 'predictions.parquet'), ('p', parent.OUT / 'predictions.parquet'),
            ('b', base.OUT / 'predictions.parquet'), ('m', main.OUT / 'predictions/old.parquet'),
            ('s', OUT / 'candidate.parquet'), ('v', OUT / 'cv.parquet'), ('r', cv.OUT / 'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
        control_checks = ' or '.join(f'c.control_{k} is distinct from p.candidate_{k}' for k in
            ['tree', 'direct', 'refined_float', 'refined', 'shift', 'applied', 'bounded', 'pred'])
        n, unique, errors, tree_error = con.execute(f'''select count(*),count(distinct c.mvt_id),sum(case when
            {control_checks} or c.control_main is distinct from b.main or c.candidate_main is distinct from m.pred
            or c.quality_eligible is distinct from p.quality_eligible
            or c.candidate_refined != round_even(greatest(c.candidate_refined_float,0),0)
            or c.candidate_applied != case when c.quality_eligible then c.candidate_shift else 0 end
            or c.candidate_pred != round_even(greatest(c.candidate_bounded,0),0)
            or c.candidate_pred is distinct from s.TAXITIME_SEC_mvt then 1 else 0 end),
            max(abs(c.candidate_tree-(cast(m.pred as double)+cast(b.airport_new as double)+cast(p.candidate_catboost as double))/3.0))
            from c join p using(mvt_id) join b using(mvt_id) join m using(mvt_id) join s on c.mvt_id=s.MVT_ID_mvt''').fetchone()
        assert n == unique == 344841 and errors == 0 and tree_error < 1e-8, (n, unique, errors, tree_error)
        count, unique, wrong = con.execute('''select count(*),count(distinct v.mvt_id),sum(case when
            v.month is distinct from r.month or v.day is distinct from r.day or v.y is distinct from r.y
            or v.control is distinct from r.full_cat or v.pred is distinct from r.combined then 1 else 0 end)
            from v join r using(mvt_id)''').fetchone()
        assert count == unique == 344419 and wrong == 0
        scores = con.execute('select month,sqrt(avg(pow(pred-y,2))) from v group by month').fetchall()
        for month, value in scores:
            np.testing.assert_allclose(value, result['cv_months'][month], atol=1e-8, rtol=0)
        np.testing.assert_allclose(np.mean([value for month, value in scores]), result['jan_jul_cv'], atol=1e-8, rtol=0)
    frame = pd.read_parquet(OUT / 'predictions.parquet'); ids = frame.mvt_id.to_numpy()
    inputs = frame[['mvt_id', 'airport', 'local_day', 'runway', 'eligible', 'month']]
    raw = pd.read_parquet(base.daily.base.OUT / 'ranking_inputs.parquet').set_index('mvt_id').loc[ids].reset_index()
    matches = pd.read_parquet(base.constraints.exclusion.OUT / 'ranking_matches.parquet')
    for arm in ['control', 'candidate']:
        _, _, shift = base.daily.apply(inputs, frame[arm + '_refined'].to_numpy(),
            base.daily.public(), read(base.OUT / 'parameters.json'))
        np.testing.assert_array_equal(shift, frame[arm + '_shift'])
        bounded = base.aggregate.constrain(ids, raw,
            frame[arm + '_refined'].to_numpy() + frame[arm + '_applied'].to_numpy(), matches)
        np.testing.assert_array_equal(bounded, frame[arm + '_bounded'])
    template, table = pq.read_table(ROOT / 'data/submitting.parquet'), pq.read_table(OUT / 'candidate.parquet')
    assert table.schema == template.schema
    for name in template.column_names:
        if name != 'TAXITIME_SEC_mvt':
            assert table[name].equals(template[name])
    write_json(OUT / 'audit.json', dict(status='pass', identity=art(MANIFEST), result=art(OUT / 'result.json'),
        source=art(__file__), ranking_rows=n, cv_rows=count, tree_mean_max_error=tree_error,
        all_sql_parent_stages_tree_means_gates_rounding_scores=True,
        all_daily_shifts_final_bounds_replayed=True, template_schema_and_unmodified_columns_exact=True,
        no_version=True, no_upload=True))
    print('All ranking rows and CV: combined daily-main component audited', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['guard', 'prepare', 'audit'])
    with threadpool_limits(1):
        globals()[parser.parse_args().stage]()
