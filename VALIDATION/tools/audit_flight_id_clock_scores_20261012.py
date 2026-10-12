"""Independent saved-result and model-receipt audit; no repeated inference."""
import json
import time

import duckdb
import audit_public_cohort_scores_20261012 as common

ROOT, RUNS = common.ROOT, common.RUNS
read, art, verify, metrics = common.read, common.artifact, common.verify, common.validate_metrics
RUN = RUNS / 'v3_20261012_flight_id_clock_lgb'
BASE = RUNS / 'v3_20261011_public_daily_taxi_lgb'
TRANSPORT = RUNS / 'v3_20261012_flight_id_clock_delta_transport'
OUT = ROOT / 'VALIDATION/reports/source_clocks_20261012/flight_id_clock_score_audit.json'


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Preserve completed receipt'
    identity, preflight = read(RUN / 'identity.json'), read(RUN / 'preflight.json')
    result, audit = read(RUN / 'result.json'), read(RUN / 'audit.json')
    frozen = read(TRANSPORT / 'design.json')
    transport = read(TRANSPORT / 'flight_id_clock/result.json')
    assert result['design'] == identity['design']
    assert result['identity'] == audit['identity'] == preflight['identity'] == art(RUN / 'identity.json')
    assert preflight['status'] == audit['status'] == 'pass'
    assert audit['result'] == art(RUN / 'result.json')
    assert result['exact_v77_and_v78_controls'] and audit['all_parent_controls_exact']
    assert audit['all_daily_shifts_and_v78_constraints_replayed']
    assert transport['identity'] == art(TRANSPORT / 'design.json') and transport['design'] == frozen['design']
    assert frozen['design']['coefficient'] == 1 and frozen['design']['new_bounds'] is False
    assert frozen['design']['experiments'] == dict(flight_id_clock=dict(directory=RUN.name,
        control='control', arms=['candidate']))
    for item in [*frozen['sources'], *result['artifacts'], *transport['sources'], transport['scored']]:
        verify(item)
    # Full input bytes were verified by retained fit and scorer receipts.
    for item in identity['sources']:
        if item['path'].endswith(('.py', '.json')):
            verify(item)
    members = []
    for month in ['2025-01', '2025-07']:
        original = read(BASE / 'predictions' / f'{month}_s42.json')
        rp = RUN / 'predictions' / f'{month}_s42.json'
        sp = RUN / 'models' / f'{month}_s42.json'
        record, staged = read(rp), read(sp)
        assert record['staged_model'] == art(sp)
        assert record['identity'] == staged['identity'] == art(RUN / 'identity.json')
        assert staged['month'] == month
        assert record['all_control_and_new_held_reload_exact'] and staged['model_saved_before_held_inference']
        for field in ['params', 'categoricals', 'rounds', 'train_rows', 'held_rows']:
            assert record[field] == staged[field] == original[field]
        assert record['columns'] == staged['columns'] == original['columns'] + [
            'idclock_iobt_residual', 'idclock_neighbor_span']
        assert len(record['columns']) == 186 and record['rounds'] == 10000
        assert record['machine'].lower() == staged['machine'].lower() == original['machine'].lower()
        for item in [record['prediction'], record['model']]:
            verify(item)
        assert record['model'] == staged['model']
        members.extend([art(rp), art(sp)])
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        for view, path in [('f', RUN / 'scored.parquet'), ('old', common.PARENT78 / 'scored.parquet'),
                           ('p', common.PARENT79 / 'cv.parquet'),
                           ('t', TRANSPORT / 'flight_id_clock/scored.parquet')]:
            con.read_parquet(str(path)).create_view(view)
            assert con.execute(f'SELECT count(*),count(distinct mvt_id) FROM {view}').fetchone() == (344419, 344419)
        n, bad = con.execute('''SELECT count(*),sum(CASE WHEN
            f.y IS DISTINCT FROM old.y OR f.month IS DISTINCT FROM old.month OR f.day IS DISTINCT FROM old.day OR
            f.control IS DISTINCT FROM old.candidate OR f.increment IS DISTINCT FROM old.increment OR
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day OR
            t.y IS DISTINCT FROM p.y OR t.month IS DISTINCT FROM p.month OR t.day IS DISTINCT FROM p.day OR
            t.control IS DISTINCT FROM p.pred OR
            t.candidate IS DISTINCT FROM round_even(greatest(0,p.pred+f.candidate-f.control),0) OR
            (f.candidate=f.control AND t.candidate IS DISTINCT FROM p.pred)
            THEN 1 ELSE 0 END) FROM f JOIN old USING(mvt_id) JOIN p USING(mvt_id) JOIN t USING(mvt_id)''').fetchone()
        assert n == 344419 and bad == 0
        for view in ['f', 't']:
            terms = ' OR '.join(f'{c} IS NULL OR NOT isfinite({c}) OR {c}<0 OR {c}<>round_even({c},0)'
                                for c in ['control', 'candidate'])
            assert con.execute(f'SELECT count(*) FROM {view} WHERE {terms}').fetchone()[0] == 0
        primary = metrics(con, 'f', result, ['control', 'candidate', 'control_v77', 'candidate_v77'])
        secondary = metrics(con, 't', transport, ['control', 'candidate'])
    effects = {name: value['cv']['candidate'] - value['cv']['control']
               for name, value in [('v78', result), ('v79', transport)]}
    value = dict(status='pass', source=art(__file__), helper=art(common.__file__), rows=n,
        seconds=time.monotonic()-start, members=members, effects=effects,
        sql_scores=primary, sql_transport_scores=secondary,
        inputs=[art(p) for p in [RUN/'identity.json', RUN/'preflight.json', RUN/'result.json',
            RUN/'audit.json', RUN/'scored.parquet', TRANSPORT/'design.json',
            TRANSPORT/'flight_id_clock/result.json', TRANSPORT/'flight_id_clock/scored.parquet',
            common.PARENT78/'scored.parquet', common.PARENT79/'cv.parquet']],
        scope='Independent saved-model recipes and SQL all-row scores, parent values, truth/calendar and serialization. '
              'Large-input, source-feature and complete model-reload checks are inherited from retained receipts.',
        interpretation='Same-desktop fixed two-feature screen on reused development data; no independent holdout.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status='pass', rows=n, effects=effects), indent=2))


if __name__ == '__main__':
    main()
