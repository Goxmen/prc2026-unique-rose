"""One frozen own-score geometry update with two genuinely new model vectors.

The previous 52-vector screen stays closed. This adds both portable parents,
uses their actual official scores, and keeps the prior optimizer and gate.
Preparation, the single solve, and independent audit are separate resumable
stages. No model fitting, version assignment, or upload happens here.
"""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import tabm_feedback_geometry as previous

ROOT = previous.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261013_portable_feedback_geometry'
DESIGN = ROOT / 'VALIDATION/reports/portable_feedback_geometry_design_20261013.json'
DESIGN_SHA = '5a837c985d1a268b51a85027915d5a96d16e89dbc40b54ec7d099c586588b047'
art, read, verify = previous.art, previous.read, previous.verify
CV_PATHS = {
    81: ROOT / 'VALIDATION/runs/v3_20261013_portable_direct_reserve/cv/predictions.parquet',
    82: ROOT / 'VALIDATION/runs/v3_20261013_portable_public_source/cv/stages.parquet',
}


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(manifests):
    assert not (OUT / 'identity.json').exists()
    assert art(DESIGN)['sha256'] == DESIGN_SHA
    design = read(DESIGN)
    for source in design['sources']:
        verify(source)
    old = read(previous.OUT / 'result.json')
    assert len(old['versions']) == 52 and not old['clears_fixed_quarter_second_gate']
    sources = [Path(__file__), DESIGN, previous.OUT / 'result.json',
               Path(previous.__file__), Path(previous.parent.transport.__file__),
               Path(previous.parent.transport.live.__file__),
               Path(previous.parent.base.legacy.ts.__file__),
               Path(previous.parent.base.geometry.__file__),
               Path(previous.independent.__file__), Path(previous.parent.__file__)]
    for record in old['outputs']:
        sources.append(verify(record))
    sources += [verify(source) for source in design['sources']]
    sources += [ROOT / f'data/training_{month}-01_{following}-01.parquet'
                for month, following in [('2025-01', '2025-02'), ('2025-07', '2025-08')]]
    specs = {}
    for version, path in zip([81, 82], manifests):
        path = Path(path).resolve()
        assert path.is_relative_to(ROOT / 'VALIDATION/runs')
        m = read(path)
        assert m['version'] == version and m['rows'] == 344841
        assert m['state'] in ['prepared', 'uploaded', 'scored']
        candidate = verify({'path': m['file'], 'sha256': m['sha256']})
        checked_path = verify(m['audit'])
        checked = read(checked_path)
        assert checked['status'] == 'pass'
        result_path = verify(checked['result'])
        result = read(result_path)
        assert result['identity'] == checked['identity']
        if version == 81:
            assert result['views']['old']['candidate'] == checked['views']['old']['candidate'] == m['source_candidate']
            assert result['views']['cv']['prediction'] == checked['views']['cv']['prediction'] == art(CV_PATHS[version])
            assert result['cv'] == m['jan_jul_cv']
        else:
            assert result['candidate'] == checked['candidate'] == m['source_candidate']
            cv_audit_path = verify(m['cv_audit'])
            cv_record_path = verify(m['cv_record'])
            cv_audit, cv_record = read(cv_audit_path), read(cv_record_path)
            assert cv_audit['status'] == 'pass' and cv_audit['metrics']['score'] == m['cv_record']
            assert cv_audit['result'] == cv_record['result']
            cv_result_path = verify(cv_record['result'])
            cv_result = read(cv_result_path)
            assert cv_result['prediction'] == cv_audit['prediction'] == art(CV_PATHS[version])
            assert cv_result['identity'] == cv_audit['identity'] == result['identity']
            assert cv_record['cv']['submitted'] == m['jan_jul_cv']
            sources += [cv_audit_path, cv_record_path, cv_result_path]
        assert m['source_candidate']['sha256'] == art(candidate)['sha256']
        sources += [checked_path, result_path, verify(result['identity']), verify(m['source_candidate'])]
        specs[str(version)] = dict(manifest=str(path.relative_to(ROOT)),
                                  file=art(candidate), cv=art(CV_PATHS[version]),
                                  jan_jul_cv=m['jan_jul_cv'])
        sources += [candidate, CV_PATHS[version]]
    identity = dict(design=art(DESIGN), versions=old['versions'] + [81, 82],
                    prior=art(previous.OUT / 'result.json'), additions=specs,
                    sources=[art(p) for p in sorted(set(sources))],
                    created_at=previous.parent.transport.now())
    write(OUT / 'identity.json', identity)
    print(json.dumps({'identity': art(OUT / 'identity.json'), 'solve_performed': False}))


def guard():
    identity = read(OUT / 'identity.json')
    for source in identity['sources']:
        verify(source)
    assert identity['versions'][-2:] == [81, 82] and len(identity['versions']) == 54
    return identity


def solve():
    identity = guard()
    assert not (OUT / 'solve_started.json').exists(), 'Never repeat the solve'
    prior = read(verify(identity['prior']))
    versions = identity['versions']
    scores = dict(prior['official_scores'])
    live = previous.parent.transport.live.fetch_leaderboard()
    ours = {r['filename']: r for r in live if r['teamName'] == previous.parent.transport.live.TEAM}
    official, manifest_receipts = {}, {}
    for version in [81, 82]:
        spec = identity['additions'][str(version)]
        m = read(ROOT / spec['manifest'])
        assert m['state'] == 'scored' and m['official_result']['status'] == 'Succeeded'
        assert m['version'] == version and m['official_result']['used_pairs'] == 344841
        assert m['sha256'] == m['remote_sha256'] == spec['file']['sha256']
        assert m['file'] == spec['file']['path']
        assert m['score'] == m['official_result']['score']
        scores[str(version)] = m['score']
        official[str(version)] = m['official_result']
        manifest_receipts[str(version)] = art(ROOT / spec['manifest'])
    for version in versions:
        row = ours[f'unique-rose_v{version}.parquet']
        assert row['usedPairs'] == 344841 and row['score'] == scores[str(version)]
    best = min(r['score'] for r in ours.values() if r.get('usedPairs') == 344841)
    ranking = pd.read_parquet(previous.OUT / 'ranking_basis.parquet')
    held = pd.read_parquet(previous.OUT / 'cv_basis.parquet')
    assert len(ranking) == ranking.mvt_id.nunique() == 344841
    assert len(held) == held.mvt_id.nunique() == 344419
    for version in [81, 82]:
        spec = identity['additions'][str(version)]
        candidate = pd.read_parquet(verify(spec['file'])).set_index('MVT_ID_mvt')
        cv = pd.read_parquet(verify(spec['cv'])).set_index('mvt_id')
        assert candidate.index.is_unique and cv.index.is_unique
        assert len(candidate) == len(ranking) and len(cv) == len(held)
        candidate, cv = candidate.loc[ranking.mvt_id], cv.loc[held.mvt_id]
        for field in ['month', 'day', 'airport', 'y']:
            np.testing.assert_array_equal(held[field], cv[field])
        ranking[f'v{version}'] = candidate.TAXITIME_SEC_mvt.to_numpy(float)
        held[f'v{version}'] = cv.submitted.to_numpy(float)
        assert np.array_equal(held[f'v{version}'], np.rint(np.maximum(held[f'v{version}'], 0)))
    cols = [f'v{v}' for v in versions]
    score_vector = np.array([scores[str(v)] for v in versions])
    write(OUT / 'live_scores.json', dict(rows=live, checked_at=previous.parent.transport.now()))
    write(OUT / 'solve_started.json', dict(identity=art(OUT / 'identity.json'),
          official=official, scored_manifests=manifest_receipts,
          live_snapshot=art(OUT / 'live_scores.json'), scores=scores, current_best=best,
          at=previous.parent.transport.now()))
    solution, raw = previous.parent.base.geometry.solve(
        ranking[cols].to_numpy(float), score_vector, versions, -3., 4., 1.)
    # Preserve solved coefficients before any later scoring/audit operation.
    write(OUT / 'solution.json', solution)
    ranking.to_parquet(OUT / 'ranking_basis.parquet', index=False)
    held.to_parquet(OUT / 'cv_basis.parquet', index=False)
    pd.DataFrame(dict(mvt_id=ranking.mvt_id, raw=raw,
                     submitted=np.rint(np.maximum(raw, 0)))).to_parquet(
                         OUT / 'diagnostic_ranking.parquet', index=False)
    weights = np.array([solution['weights'][str(v)] for v in versions])
    frame = held[['mvt_id', 'month', 'day', 'airport', 'y']].copy()
    frame['control'] = held.v80
    frame['pred'] = np.rint(np.maximum(held[cols].to_numpy(float) @ weights, 0))
    frame.to_parquet(OUT / 'diagnostic_cv.parquet', index=False)
    months = {month: {c: float(np.sqrt(np.mean((g[c] - g.y) ** 2)))
                      for c in ['control', 'pred']} for month, g in frame.groupby('month')}
    paired = previous.parent.base.legacy.ts.paired_bootstrap(
        frame.day, frame.month, (frame.control - frame.y) ** 2, (frame.pred - frame.y) ** 2)
    write(OUT / 'result.json', dict(identity=art(OUT / 'identity.json'),
          solution=art(OUT / 'solution.json'), official=art(OUT / 'solve_started.json'),
          months=months, cv=float(np.mean([m['pred'] for m in months.values()])),
          paired=paired, outputs=[art(p) for p in sorted(OUT.glob('*.parquet'))],
          interpretation='Own leaderboard score selection; reused development CV',
          historical_cv_caveat=prior['historical_cv_caveat'],
          independent_audit_pending=True, no_upload=True))
    print(json.dumps({'projected_rmse': solution['projected_rmse'],
                      'cv': read(OUT / 'result.json')['cv'], 'paired': paired}))


def audit():
    identity = guard()
    result = read(OUT / 'result.json')
    assert result['identity'] == art(OUT / 'identity.json')
    for record in result['outputs']:
        verify(record)
    solution, official = read(verify(result['solution'])), read(verify(result['official']))
    assert official['identity'] == result['identity']
    live = read(verify(official['live_snapshot']))['rows']
    ours = {r['filename']:r for r in live if r['teamName'] == previous.parent.transport.live.TEAM}
    assert official['current_best'] == min(r['score'] for r in ours.values() if r.get('usedPairs') == 344841)
    for version, record in official['scored_manifests'].items():
        m = read(verify(record))
        assert m['state'] == 'scored' and m['official_result'] == official['official'][version]
    for version in identity['versions']:
        row = ours[f'unique-rose_v{version}.parquet']
        assert row['usedPairs'] == 344841 and row['score'] == official['scores'][str(version)]
    versions = identity['versions']
    weights = np.array([solution['weights'][str(v)] for v in versions])
    scores = np.array([official['scores'][str(v)] for v in versions])
    assert abs(weights.sum() - 1) < 1e-8 and weights.min() >= -3 - 1e-8 and weights.max() <= 4 + 1e-8
    with duckdb.connect(config={'threads': 1, 'memory_limit': '2GB'}) as con:
        for name, file in [('r','ranking_basis'), ('h','cv_basis'),
                           ('o','diagnostic_ranking'), ('c','diagnostic_cv')]:
            con.read_parquet(str(OUT / (file + '.parquet'))).create_view(name)
        con.read_parquet(str(previous.OUT / 'ranking_basis.parquet')).create_view('old_r')
        con.read_parquet(str(previous.OUT / 'cv_basis.parquet')).create_view('old_h')
        old_difference = ' OR '.join(f'a.v{v} IS DISTINCT FROM b.v{v}' for v in versions[:-2])
        for name in ['r', 'h']:
            assert con.execute(f'''SELECT count(*) FROM {name} a FULL JOIN old_{name} b
                USING(mvt_id) WHERE a.mvt_id IS NULL OR b.mvt_id IS NULL
                OR {old_difference}''').fetchone()[0] == 0
        for version in [81, 82]:
            spec = identity['additions'][str(version)]
            con.read_parquet(str(verify(spec['file']))).create_view('source')
            con.read_parquet(str(verify(spec['cv']))).create_view('cv_source')
            assert con.execute(f'''select count(*) from r full join source
                on r.mvt_id=source.MVT_ID_mvt where r.v{version}
                is distinct from source.TAXITIME_SEC_mvt''').fetchone()[0] == 0
            assert con.execute(f'''select count(*) from h full join cv_source s using(mvt_id)
                where h.v{version} is distinct from s.submitted or h.y is distinct from s.y
                or h.day is distinct from s.day or h.month is distinct from s.month''').fetchone()[0] == 0
        geometry = previous.independent.sql_geometry(con, versions, weights, scores)
        np.testing.assert_allclose(geometry['sql_projected_rmse'], solution['projected_rmse'], atol=1e-7, rtol=0)
        expr = previous.independent.expression(versions, weights, 'h')
        rows = con.execute(f'''select count(*),count(distinct h.mvt_id),
            count(*) FILTER(WHERE c.mvt_id IS NULL OR h.mvt_id IS NULL
                OR c.pred IS DISTINCT FROM round_even(greatest(0,({expr})),0)
                OR c.y IS DISTINCT FROM h.y OR c.month IS DISTINCT FROM h.month
                OR c.day IS DISTINCT FROM h.day OR c.airport IS DISTINCT FROM h.airport
                OR c.control IS DISTINCT FROM h.v80)
            from h full join c using(mvt_id)''').fetchone()
        assert rows == (344419, 344419, 0)
        raw = [str(verify(s)) for s in identity['sources']
               if Path(s['path']).name.startswith(('training_2025-01-', 'training_2025-07-'))]
        assert len(raw) == 2
        con.execute("SET TimeZone='UTC'")
        con.read_parquet(raw).create_view('truth')
        assert con.execute('''SELECT count(*) FROM c LEFT JOIN truth t ON c.mvt_id=t.MVT_ID_mvt
            WHERE t.MVT_ID_mvt IS NULL OR t.PHASE_mvt!='DEP' OR c.y IS DISTINCT FROM t.TAXITIME_SEC_mvt
            OR c.month IS DISTINCT FROM strftime(t.MVT_TIME_UTC_mvt,'%Y-%m')
            OR c.day IS DISTINCT FROM strftime(t.MVT_TIME_UTC_mvt,'%Y-%m-%d')
            OR c.airport IS DISTINCT FROM t.ADEP_mvt''').fetchone()[0] == 0
        for month, score, control in con.execute('select month,sqrt(avg(pow(pred-y,2))),sqrt(avg(pow(control-y,2))) from c group by month').fetchall():
            np.testing.assert_allclose(score, result['months'][month]['pred'], atol=1e-8, rtol=0)
            np.testing.assert_allclose(control, result['months'][month]['control'], atol=1e-8, rtol=0)
        day = con.execute('''WITH ordered AS (SELECT *,row_number() OVER() ordinal FROM c)
            SELECT month,day,count(*) n,sum(pow(control-y,2)) a,sum(pow(pred-y,2)) b,
            min(ordinal) first_position FROM ordered GROUP BY month,day
            ORDER BY month,first_position''').fetchdf()
    rng = np.random.default_rng(0)
    draws = np.zeros(2000)
    for month in sorted(result['months']):
        group = day[day.month.eq(month)]
        indices = rng.integers(0, len(group), size=(2000, len(group)))
        denom = group.n.to_numpy()[indices].sum(1)
        draws += (np.sqrt(group.b.to_numpy()[indices].sum(1)/denom)
                  - np.sqrt(group.a.to_numpy()[indices].sum(1)/denom))/2
    ci = np.percentile(draws, [2.5, 97.5])
    np.testing.assert_allclose(ci, result['paired']['ci95'], atol=1e-8, rtol=0)
    np.testing.assert_allclose(result['cv'], np.mean([r['pred'] for r in result['months'].values()]), atol=1e-8, rtol=0)
    np.testing.assert_allclose(result['paired']['change'], np.mean([r['pred']-r['control'] for r in result['months'].values()]), atol=1e-8, rtol=0)
    upper = geometry['bounds']['conservative_last_decimal']['serialized_rmse_interval'][1]
    receipt = dict(status='pass', identity=art(OUT / 'identity.json'),
                   result=art(OUT / 'result.json'), independent_sql=geometry,
                   current_best=official['current_best'], conservative_gain=official['current_best']-upper,
                   gate_pass=upper <= official['current_best']-.25,
                   no_version_assigned=True, no_upload=True)
    write(OUT / 'audit.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['status','current_best','conservative_gain','gate_pass']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','solve','audit'])
    parser.add_argument('--manifests', nargs=2, type=Path)
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.stage == 'prepare':
            assert args.manifests is not None
            prepare(args.manifests)
        elif args.stage == 'solve':
            solve()
        else:
            audit()
