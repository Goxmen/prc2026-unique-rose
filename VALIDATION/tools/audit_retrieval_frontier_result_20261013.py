"""Independent raw-label, SQL and paired-day audit of the fixed retrieval pilot.

This auditor is run only after both arms, the prescore audit, and scoring finish.
It does not import the producer or model, fit anything, or decode Jan/Jul data.
"""
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'VALIDATION/runs/v3_20261013_train_label_retrieval'
ARMS = ['labels_off', 'labels_on']
CHECKED = set()


def read(path):
    return json.loads(Path(path).read_text())


def artifact(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest(),
                bytes=path.stat().st_size)


def verify(record):
    path = (ROOT / record['path']).resolve()
    path.relative_to(ROOT)
    key = (str(path), record['sha256'], record['bytes'])
    if key not in CHECKED:
        assert artifact(path) == record, path
        CHECKED.add(key)
    return path


def close(actual, expected, tolerance=2e-9):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=tolerance)


def main():
    destination = OUT / 'result_audit.json'
    assert not destination.exists(), 'Preserve the completed audit'
    identity_record = artifact(OUT / 'identity.json')
    identity = read(OUT / 'identity.json')
    assert identity['query_month'] == '2025-04'
    assert identity['training_months'] == [
        f'2025-{month:02d}' for month in [2, 3, 5, 6, 8, 9, 10, 11, 12]]
    assert identity['settings']['arms'] == ARMS
    assert identity['no_January_July_files']
    assert artifact(__file__) in identity['sources']
    for record in identity['sources']:
        verify(record)

    receipts = {}
    for name in ['prepared', 'preflight', 'fit', 'result']:
        value = read(OUT / f'{name}.json')
        assert value['identity'] == identity_record
        for key in ['artifacts', 'dependencies']:
            for record in value.get(key, []):
                verify(record)
        receipts[name] = value
    assert receipts['fit']['no_query_labels_loaded']
    prescore = read(OUT / 'prescore_audit.json')
    assert prescore['status'] == 'pass'
    assert prescore['identity'] == identity_record
    assert prescore['fit'] == artifact(OUT / 'fit.json')
    result = receipts['result']

    query_meta = np.load(OUT / 'inputs/query_meta.npy')
    query_ids = query_meta[:, 0]
    assert np.all(np.diff(query_ids) > 0)
    predictions = []
    for arm in ARMS:
        inference = read(OUT / arm / 'inference.json')
        training = read(OUT / arm / 'training.json')
        for value in [inference, training]:
            assert value['identity'] == identity_record
            assert value['arm'] == arm and value['no_query_labels_loaded']
            for key in ['artifacts', 'dependencies']:
                for record in value.get(key, []):
                    verify(record)
        assert inference['exact_reload']
        prediction = pd.read_parquet(OUT / arm / 'predictions.parquet')
        np.testing.assert_array_equal(prediction.mvt_id, query_ids)
        assert np.isfinite(prediction.prediction).all()
        expected = np.round(np.clip(prediction.prediction.to_numpy(), 0, None))
        np.testing.assert_array_equal(prediction.serialized, expected)
        predictions.append(expected)

    raw_records = [record for record in identity['sources']
                   if Path(record['path']).name.startswith('training_2025-04-01_')
                   and record['path'].endswith('.parquet')]
    assert len(raw_records) == 1
    raw = pd.read_parquet(verify(raw_records[0]), columns=[
        'MVT_ID_mvt', 'PHASE_mvt', 'MVT_TIME_UTC_mvt', 'TAXITIME_SEC_mvt'],
        filters=[('PHASE_mvt', '==', 'DEP')]).set_index('MVT_ID_mvt')
    assert raw.index.is_unique
    raw = raw.loc[query_ids]
    timestamp = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
    assert timestamp.dt.strftime('%Y-%m').eq('2025-04').all()
    stamps = timestamp.astype('datetime64[ms, UTC]').astype('int64').to_numpy()
    np.testing.assert_array_equal(stamps, query_meta[:, 1])
    day = timestamp.dt.strftime('%Y-%m-%d').to_numpy()
    y = raw.TAXITIME_SEC_mvt.to_numpy(float)
    finite = np.isfinite(y)
    assert finite.any()

    packet = pd.read_parquet(OUT / 'scored.parquet')
    np.testing.assert_array_equal(packet.mvt_id, query_ids)
    np.testing.assert_array_equal(packet.day, day)
    np.testing.assert_array_equal(packet.y, y)
    np.testing.assert_array_equal(packet.finite_target, finite)
    for arm, prediction in zip(ARMS, predictions):
        np.testing.assert_array_equal(packet[arm], prediction)
    assert result['rows'] == len(query_ids)
    assert result['scored_rows'] == int(finite.sum())
    assert result['nonfinite_target_rows'] == int((~finite).sum())

    # A separate SQL engine reconstructs losses from raw y and saved integer
    # predictions. No producer metrics, target clipping, or fitted values enter.
    truth = pd.DataFrame(dict(day=day[finite], y=y[finite],
        control=predictions[0][finite], candidate=predictions[1][finite]))
    connection = duckdb.connect(config={'threads': 1})
    connection.register('truth', truth)
    metrics = connection.execute('''
        SELECT sqrt(avg((control-y)*(control-y))),
               sqrt(avg((candidate-y)*(candidate-y))) FROM truth
    ''').fetchnumpy()
    rmse = np.array([float(value[0]) for value in metrics.values()])
    daily = connection.execute('''
        SELECT day, count(*) AS n,
               sum((control-y)*(control-y)) AS control_sse,
               sum((candidate-y)*(candidate-y)) AS candidate_sse
        FROM truth GROUP BY day ORDER BY day
    ''').df()
    n = daily.n.to_numpy(dtype=np.int64)
    sse = daily[['control_sse', 'candidate_sse']].to_numpy(float)
    rng = np.random.Generator(np.random.PCG64(20261013))
    samples = rng.integers(0, len(daily), size=(2000, len(daily)))
    bootstrap = []
    for sample in samples:
        weights = np.bincount(sample, minlength=len(daily))
        estimate = np.sqrt(weights @ sse / (weights @ n))
        bootstrap.append(float(estimate[1] - estimate[0]))
    bootstrap = np.asarray(bootstrap)
    close(bootstrap, np.load(OUT / 'bootstrap.npy'))
    interval = np.percentile(bootstrap, [2.5, 97.5])
    close(rmse, [result['RMSE'][arm] for arm in ARMS])
    close(rmse[1] - rmse[0], result['delta'])
    close(interval, result['CI95'])
    gate = bool(rmse[1] < rmse[0] and interval[1] < 0)
    assert result['gate'] is gate

    features = pd.read_parquet(OUT / 'inputs/query_features.parquet',
                               columns=['mvt_id', 'airport'])
    np.testing.assert_array_equal(features.mvt_id, query_ids)
    truth['airport'] = features.airport.to_numpy()[finite]
    connection.unregister('truth')
    connection.register('truth', truth)
    airports = connection.execute('''
        SELECT airport, count(*) AS n,
               sqrt(avg((control-y)*(control-y))) AS control_rmse,
               sqrt(avg((candidate-y)*(candidate-y))) AS candidate_rmse
        FROM truth GROUP BY airport ORDER BY airport
    ''').df()
    assert set(airports.airport) == set(result['airports'])
    for row in airports.itertuples(index=False):
        assert result['airports'][row.airport]['rows'] == row.n
        close([row.control_rmse, row.candidate_rmse],
              result['airports'][row.airport]['RMSE'])
    connection.close()

    report = dict(status='pass', identity=identity_record,
        auditor=artifact(__file__), result=artifact(OUT / 'result.json'),
        prescore=artifact(OUT / 'prescore_audit.json'),
        rows=len(query_ids), scored_rows=int(finite.sum()), utc_days=len(daily),
        SQL_RMSE=dict(zip(ARMS, rmse.tolist())), delta=float(rmse[1]-rmse[0]),
        CI95=interval.tolist(), gate=gate,
        checks=['source and receipt hashes', 'raw April labels and UTC dates',
                'all query IDs and target exclusions', 'both saved rounded predictions',
                'independent SQL RMSE and airport diagnostics',
                'independent paired UTC-day weighted bootstrap', 'fixed decision gate'],
        scope='Fixed small retrieval mechanism pilot; not v80 or official validation')
    with destination.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in
        ['status', 'rows', 'SQL_RMSE', 'delta', 'CI95', 'gate']}))


if __name__ == '__main__':
    main()
