"""Reusable independent SQL/paired-day audit for fixed control/candidate screens.

This checks provenance and statistical arithmetic. The producing runner must
separately check its model/source transformation and exact parent controls.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def art(path):
    return dict(path=str(path.relative_to(ROOT)), sha256=sha(path))


def main(directory):
    directory = directory.resolve()
    assert directory.is_relative_to(ROOT)
    result_path = directory / 'result.json'
    result = json.loads(result_path.read_text())
    assert set(result['cv']) == {'control', 'candidate'}
    receipt = directory / 'statistical_audit.json'
    assert not receipt.exists()
    sources = [result['identity'], result['scored']]
    identity = json.loads((ROOT / result['identity']['path']).read_text())
    sources += identity['sources']
    for a in sources:
        assert sha(ROOT / a['path']) == a['sha256'], a['path']
    rng = np.random.default_rng(0)
    draws = np.zeros(2000)
    actual_scores = {}
    with duckdb.connect(config={'threads': 1, 'memory_limit': '256MB'}) as con:
        con.read_parquet(str(ROOT / result['scored']['path'])).create_view('f')
        rows, unique = con.execute('select count(*),count(distinct mvt_id) from f').fetchone()
        assert rows == unique == 344419
        assert con.execute('''select count(*) from f where y is null or control is null or candidate is null
            or not isfinite(y) or not isfinite(control) or not isfinite(candidate) or candidate<0
            or candidate<>round_even(candidate,0)''').fetchone()[0] == 0
        months = sorted(result['scores'])
        for month in months:
            actual = con.execute('select sqrt(avg(pow(control-y,2))),sqrt(avg(pow(candidate-y,2))) from f where month=?', [month]).fetchone()
            actual_scores[month] = dict(zip(['control', 'candidate'], actual))
            np.testing.assert_allclose(actual, [result['scores'][month][c] for c in ['control', 'candidate']], atol=1e-8, rtol=0)
            days = np.asarray(con.execute('''select count(*),sum(pow(control-y,2)),sum(pow(candidate-y,2))
                from (select *,row_number() over() pos from f) where month=? group by day order by min(pos)''', [month]).fetchall())
            summed = days[rng.integers(0, len(days), (len(draws), len(days)))].sum(axis=1)
            draws += (np.sqrt(summed[:, 2] / summed[:, 0]) - np.sqrt(summed[:, 1] / summed[:, 0])) / len(months)
        cv = {c: float(np.mean([x[c] for x in actual_scores.values()])) for c in ['control', 'candidate']}
        for c in cv:
            np.testing.assert_allclose(cv[c], result['cv'][c], atol=1e-8, rtol=0)
        ci = np.percentile(draws, [2.5, 97.5])
        change = cv['candidate'] - cv['control']
        np.testing.assert_allclose(ci, result['paired']['ci95'], atol=1e-8, rtol=0)
        np.testing.assert_allclose(change, result['paired']['change'], atol=1e-8, rtol=0)
        decision = 'improvement' if ci[1] < 0 else 'regression' if ci[0] > 0 else 'inconclusive'
        assert decision == result['paired']['decision']
    value = dict(status='pass', source=art(Path(__file__)), result=art(result_path), rows=rows,
                 sources_verified=len(sources), independent_SQL_RMSE_and_day_bootstrap=True,
                 cv=cv, ci95=ci.tolist(), decision=decision,
                 scope='Provenance and statistical arithmetic; transformation/control checks remain in the frozen producing runner.')
    receipt.write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(dict(status='pass', rows=rows, cv=cv, ci95=ci.tolist(), decision=decision)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    main(parser.parse_args().directory)
