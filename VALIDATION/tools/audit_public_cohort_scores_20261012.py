"""Independent bounded SQL audit of frozen cohort CV and v79 transport.

Consumes completed predictions/receipts only. It does not import a model,
recompute the inherited correction stack, train, select a recipe, or upload.
"""
import hashlib
import json
from pathlib import Path
import time

import duckdb

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / 'VALIDATION/runs'
TRAIN = RUNS / 'v3_20261012_nn_public_cohort'
STACK = RUNS / 'v3_20261012_nn_public_cohort_stack'
TRANSPORT = RUNS / 'v3_20261012_cohort_delta_transport'
PARENT78 = RUNS / 'v3_20261011_joint_public_moments'
PARENT79 = RUNS / 'v3_20261012_target205_candidate/submission_v79'
OUT = ROOT / 'VALIDATION/reports/cohort_context_20261012/score_audit.json'
ARMS = ['parent_single', 'cohort_single', 'selected', 'replace_cohort']


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def verify(record):
    current = artifact(ROOT / record['path'])
    assert current == {key: record[key] for key in ['path', 'sha256']}, record['path']


def validate_metrics(con, view, result, columns):
    scores = {}
    for column in columns:
        rows = con.execute(f'SELECT month, sqrt(avg(pow({column}-y,2))) '
                           f'FROM {view} GROUP BY month ORDER BY month').fetchall()
        assert [row[0] for row in rows] == ['2025-01', '2025-07']
        scores[column] = dict(rows)
        for month, value in rows:
            assert abs(value - result['scores'][month][column]) <= 1e-8
        assert abs(sum(value for _, value in rows) / 2 - result['cv'][column]) <= 1e-8
    return scores


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve completed independent receipt'
    identity = read(TRAIN / 'identity.json')
    fit_audit = read(TRAIN / 'audit.json')
    stack = read(STACK / 'result.json')
    stack_audit = read(STACK / 'audit.json')
    transport = read(TRANSPORT / 'cohort/result.json')
    frozen = read(TRANSPORT / 'design.json')
    assert fit_audit['status'] == stack_audit['status'] == 'pass'
    assert fit_audit['identity'] == artifact(TRAIN / 'identity.json')
    assert stack_audit['result'] == artifact(STACK / 'result.json')
    assert stack['exact_v78_replayed'] and transport['exact_v78_control']
    assert transport['identity'] == artifact(TRANSPORT / 'design.json')
    assert transport['design'] == frozen['design']
    assert frozen['design']['coefficient'] == 1 and frozen['design']['new_bounds'] is False
    assert frozen['design']['experiments'] == {
        'cohort': dict(directory=STACK.name, control='selected', arms=['replace_cohort'])}
    verify(identity['source'])
    for item in identity['files']:
        verify(item)
    for item in stack['sources'] + transport['sources'] + frozen['sources']:
        verify(item)
    verify(stack['scored'])
    verify(transport['scored'])
    assert identity['selected_v78'] == artifact(PARENT78 / 'result.json')
    members = []
    for month, epochs in [('2025-01', 21), ('2025-07', 17)]:
        path = TRAIN / f'networks/{month}_s42.json'
        record = read(path)
        assert record['identity'] == artifact(TRAIN / 'identity.json')
        assert record['month'] == month and record['seed'] == 42 and record['epochs'] == epochs
        assert record['design'] == identity['design']
        assert record['inputs'] == identity['folds'][month]
        assert record['bank'] == identity['bank'] and record['all_held_reloaded_exact']
        for key in ['model', 'preprocessing', 'prediction', 'checkpoint']:
            verify(record[key])
        verify(identity['folds'][month]['control_prediction'])
        members.append(artifact(path))
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        for name, path in [('f', STACK / 'scored.parquet'),
                           ('old', PARENT78 / 'scored.parquet'),
                           ('p', PARENT79 / 'cv.parquet'),
                           ('t', TRANSPORT / 'cohort/scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
            count, unique = con.execute(f'SELECT count(*),count(distinct mvt_id) FROM {name}').fetchone()
            assert count == unique == 344419
        count, bad = con.execute('''SELECT count(*), sum(CASE WHEN
            f.y IS DISTINCT FROM old.y OR f.month IS DISTINCT FROM old.month OR
            f.day IS DISTINCT FROM old.day OR f.selected IS DISTINCT FROM old.candidate OR
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR
            f.day IS DISTINCT FROM p.day OR t.y IS DISTINCT FROM p.y OR
            t.month IS DISTINCT FROM p.month OR t.day IS DISTINCT FROM p.day OR
            t.control IS DISTINCT FROM p.pred OR
            t.replace_cohort IS DISTINCT FROM
                round_even(greatest(0,p.pred+f.replace_cohort-f.selected),0) OR
            (f.replace_cohort=f.selected AND t.replace_cohort IS DISTINCT FROM p.pred)
            THEN 1 ELSE 0 END)
            FROM f JOIN old USING(mvt_id) JOIN p USING(mvt_id) JOIN t USING(mvt_id)''').fetchone()
        assert count == 344419 and bad == 0
        for view, columns in [('f', ARMS), ('t', ['control', 'replace_cohort'])]:
            terms = ' OR '.join(f'{c} IS NULL OR NOT isfinite({c}) OR {c}<0 OR '
                                f'{c}<>round_even({c},0)' for c in columns)
            assert con.execute(f'SELECT count(*) FROM {view} WHERE {terms}').fetchone()[0] == 0
        metrics = validate_metrics(con, 'f', stack, ARMS)
        transported = validate_metrics(con, 't', transport, ['control', 'replace_cohort'])
        paired_days = {}
        for view, control, arm, name in [
            ('f', 'selected', 'replace_cohort', 'one_of_three'),
            ('f', 'parent_single', 'cohort_single', 'same_desktop_single'),
            ('t', 'control', 'replace_cohort', 'fixed_v79_transport')]:
            rows = con.execute(f'''SELECT month, count(*),sum(delta<0),sum(delta>0),sum(delta=0)
                FROM (SELECT month,day,avg(pow({arm}-y,2)-pow({control}-y,2)) delta
                    FROM {view} GROUP BY month,day) GROUP BY month ORDER BY month''').fetchall()
            paired_days[name] = [dict(month=m,days=n,improve=better,worsen=worse,tie=tie)
                                 for m,n,better,worse,tie in rows]
    receipt = dict(status='pass',rows=count,seconds=time.monotonic()-started,
        source=artifact(__file__),members=members,
        inputs=[artifact(p) for p in [TRAIN/'identity.json',TRAIN/'audit.json',
            STACK/'result.json',STACK/'audit.json',STACK/'scored.parquet',
            TRANSPORT/'design.json',TRANSPORT/'cohort/result.json',
            TRANSPORT/'cohort/scored.parquet',PARENT78/'scored.parquet',PARENT79/'cv.parquet']],
        sql_scores=metrics,sql_transport_scores=transported,paired_day_sse_direction=paired_days,
        source_calendar_truth_ids_exact=True,exact_v78_replay=True,
        all_row_v79_transport_and_zero_delta_exact=True,
        scope='Independent SQL arithmetic/serialization and artifact binding. Inherited correction '
              'stack correctness is covered by reviewed frozen scorer and exact parent replay; '
              'this audit does not independently reimplement those corrections. No new statistical confirmation.')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass',rows=count,seconds=receipt['seconds'],
        cv=stack['cv'],transport_cv=transport['cv']),indent=2))


if __name__ == '__main__':
    main()
