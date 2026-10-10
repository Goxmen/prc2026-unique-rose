"""Training-date pilot of time-varying arrival recording lag, without taxi-out labels.

Reuse the audited v2 physical-stop witness on twelve fixed dates belonging to
both outer training folds. A portable packet permits laptop execution without
copying the full identity cache. Judge causal same-day lag context against
leave-day-out airport/airline references, not against a zero-lag baseline.
"""
import argparse
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

import arrival_stand_clock_pilot_v2 as witness
from records import sha256, write_json

ROOT = witness.ROOT
OUT = ROOT / 'VALIDATION/runs/v3_20261010_arrival_clock_context_pilot'
DAYS = [f'2025-{m:02d}-{d:02d}' for m in [3, 9, 10, 11] for d in [5, 15, 25]]
DESIGN = dict(days=DAYS, scope='fixed dates in intersection of original January/July training months',
    target='supplied arrival in-block minus independently witnessed stop; no departure labels',
    witness='unchanged audited v2', airline='leading letters of cached observed/fallback callsign',
    static='airport median on other pilot dates; airport/airline median shrunk K10 toward airport',
    dynamic='median residual of same-airport observations available before landing in preceding3h; K5',
    dynamic_airline='same residual restricted to callsign airline; K3',
    causal='max(stop confirmation, supplied in-block) strictly before query landing',
    boundary='references only within same pilot UTC day; early-day context censored',
    departure_prediction=False, model_fitting=False, upload=False)


def artifact(path):
    return dict(path=str(Path(path).relative_to(ROOT)), sha256=sha256(path))


def verify(record):
    path = ROOT / record['path']
    if sha256(path) != record['sha256']:
        raise ValueError('Pilot artifact changed: ' + str(path))
    return path


def identity():
    return dict(design=DESIGN, runner=artifact(Path(__file__)),
        helpers=[artifact(Path(m.__file__)) for m in [witness, witness.rotation, witness.sf, witness.sg]])


def prepare():
    if (OUT / 'prepared.json').exists():
        load()
        return
    start = time.monotonic()
    con = duckdb.connect()
    con.execute("SET threads=1; SET memory_limit='1GB'; SET TimeZone='UTC'")
    source = witness.rotation.DEFAULT / 'identity.parquet'
    q = con.execute('''SELECT mvt_id,airport,stand,hex,cs,epoch_ms(t)::BIGINT landing_ms,
        epoch_ms(inblock)::BIGINT inblock_ms,strftime(t,'%Y-%m-%d') AS day
        FROM read_parquet(?) WHERE phase='ARR'
        AND strftime(t,'%Y-%m-%d') IN (SELECT unnest(?)) ORDER BY mvt_id''', [str(source), DAYS]).df()
    geometry = pd.read_parquet(witness.sf.COORDS, columns=['airport', 'stand', 'lat', 'lon'])
    geometry = geometry.rename(columns=dict(lat='stand_lat', lon='stand_lon'))
    q = q.merge(geometry, on=['airport', 'stand'], how='left', validate='many_to_one')
    if not q.mvt_id.is_unique or set(q.day) != set(DAYS):
        raise ValueError('Pilot population/date mismatch')
    q['airline'] = q.cs.fillna('').str.extract(r'^([A-Z]+)', expand=False).fillna('<MISSING>')
    (OUT / 'inputs').mkdir(parents=True, exist_ok=True)
    query_path = OUT / 'inputs/queries.parquet'
    q.to_parquet(query_path, index=False)
    days, sources = {}, {str(source.relative_to(ROOT)): sha256(source),
                         str(witness.sf.COORDS.relative_to(ROOT)): sha256(witness.sf.COORDS)}
    for day in DAYS:
        following = (pd.Timestamp(day) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
        paths = [witness.rotation.HEATMAP / 'positions' / f'{date}.parquet' for date in [day, following]]
        if not all(p.exists() for p in paths):
            raise ValueError('Required cached source day missing')
        for path in paths:
            sources[str(path.relative_to(ROOT))] = sha256(path)
        keys = q[q.day.eq(day) & q.hex.notna()][['airport', 'hex']].drop_duplicates()
        con.register('keys', keys)
        points = con.execute('''SELECT p.airport,p.hex,epoch_ms(p.ts)::BIGINT ts_ms,p.lat,p.lon,p.alt,p.gs
            FROM read_parquet(?) p SEMI JOIN keys k ON p.airport=k.airport AND p.hex=k.hex''',
            [[str(p) for p in paths]]).df()
        destination = OUT / 'inputs' / f'{day}.parquet'
        points.to_parquet(destination, index=False)
        days[day] = dict(points=len(points), artifact=artifact(destination))
    write_json(OUT / 'prepared.json', dict(identity=identity(), query=artifact(query_path),
        days=days, sources=sources, rows=len(q), seconds=time.monotonic()-start,
        bytes=sum(p.stat().st_size for p in (OUT / 'inputs').glob('*.parquet'))))
    print('Prepared', len(q), 'arrival rows', flush=True)


def load():
    value = json.loads((OUT / 'prepared.json').read_text())
    if value['identity'] != identity():
        raise ValueError('Frozen pilot source/design changed')
    verify(value['query'])
    for entry in value['days'].values():
        verify(entry['artifact'])
    return value


def run():
    manifest = load()
    if (OUT / 'result.json').exists():
        old = json.loads((OUT / 'result.json').read_text())
        if old['prepared'] != artifact(OUT / 'prepared.json'):
            raise ValueError('Completed pilot contract changed')
        verify(old['witnesses']); verify(old['predictions'])
        print('Completed pilot reused', flush=True)
        return
    start = time.monotonic()
    q = pd.read_parquet(verify(manifest['query']))
    found = []
    for day in DAYS:
        points = pd.read_parquet(verify(manifest['days'][day]['artifact']))
        groups = {(a, int(h)): p for (a, h), p in points.groupby(['airport', 'hex'], sort=False)}
        for row in q[q.day.eq(day)].itertuples():
            if pd.isna(row.hex) or pd.isna(row.stand_lat):
                continue
            p = groups.get((row.airport, int(row.hex)))
            if p is None:
                continue
            result = witness.witness(p, row.landing_ms, row.stand_lat, row.stand_lon)
            if result is not None:
                found.append(dict(mvt_id=row.mvt_id, **result))
        print(day, 'cumulative witnesses', len(found), flush=True)
    matches = pd.DataFrame(found)
    full = q.merge(matches, on='mvt_id', how='left', validate='one_to_one')
    full['lag'] = (full.inblock_ms - full.stop_ms) / 1000
    full['available_ms'] = full[['inblock_ms', 'confirm_ms']].max(axis=1).where(full.lag.notna())
    witness_path = OUT / 'witnesses.parquet'
    full.to_parquet(witness_path, index=False)
    scored, edges = [], []
    usable = full[full.lag.notna()].copy()
    for day in DAYS:
        history = usable[usable.day.ne(day)]
        airport = history.groupby('airport').lag.median()
        airline = history.groupby(['airport', 'airline']).lag.agg(['median', 'size'])

        def prior(a, carrier):
            if a not in airport:
                return np.nan
            if (a, carrier) not in airline.index:
                return float(airport[a])
            row = airline.loc[(a, carrier)]
            return float((row['size'] * row['median'] + 10 * airport[a]) / (row['size'] + 10))

        current = usable[usable.day.eq(day)].copy()
        current['prior'] = [prior(a, c) for a, c in zip(current.airport, current.airline)]
        current['residual'] = current.lag - current.prior
        for row in current.itertuples():
            if not np.isfinite(row.prior):
                continue
            donors = current[current.airport.eq(row.airport) & current.available_ms.lt(row.landing_ms)
                             & current.available_ms.ge(row.landing_ms - 3 * 3600000)
                             & current.residual.notna()]
            same = donors[donors.airline.eq(row.airline)]
            adjustment = float(donors.residual.median() * len(donors) / (len(donors)+5)) if len(donors) else 0.
            own = float(same.residual.median() * len(same) / (len(same)+3)) if len(same) else 0.
            scored.append(dict(mvt_id=row.mvt_id, day=day, airport=row.airport, airline=row.airline,
                target=row.lag, airport_prior=float(airport[row.airport]), static=row.prior,
                dynamic=row.prior+adjustment, dynamic_airline=row.prior+own,
                donors=len(donors), airline_donors=len(same)))
            for donor in donors.itertuples():
                assert donor.mvt_id != row.mvt_id and donor.available_ms < row.landing_ms
                edges.append(dict(query=row.mvt_id, donor=donor.mvt_id,
                                  age_sec=(row.landing_ms-donor.available_ms)/1000))
    scored = pd.DataFrame(scored)
    prediction_path = OUT / 'predictions.parquet'
    scored.to_parquet(prediction_path, index=False)
    pd.DataFrame(edges).to_parquet(OUT / 'donor_edges.parquet', index=False)

    def metrics(frame):
        return {name: dict(rmse=float(np.sqrt(np.mean((frame[name]-frame.target)**2))),
                           mae=float(np.mean(abs(frame[name]-frame.target))))
                for name in ['airport_prior', 'static', 'dynamic', 'dynamic_airline']}

    result = dict(identity=identity(), prepared=artifact(OUT / 'prepared.json'),
        arrivals=len(full), witness_rows=int(full.stop_ms.notna().sum()), scored=len(scored),
        context_rows=int(scored.donors.gt(0).sum()), airline_context_rows=int(scored.airline_donors.gt(0).sum()),
        donor_edges=len(edges), metrics=metrics(scored),
        per_airport={a: dict(rows=len(f), **metrics(f)) for a, f in scored.groupby('airport')},
        per_day={d: dict(rows=len(f), **metrics(f)) for d, f in scored.groupby('day')},
        seconds=time.monotonic()-start, witnesses=artifact(witness_path), predictions=artifact(prediction_path),
        caution='Arrival recording-lag pilot only. No departure labels, taxi-out model, held-out score or gain forecast.')
    write_json(OUT / 'result.json', result)
    print(result['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'run'])
    globals()[parser.parse_args().stage]()
