"""Independent raw-clock/query-speed and SQL checks of the failed source pilot."""
from pathlib import Path
import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/reports/spatial_peer_motion_20261012'


def read(p):return json.loads(Path(p).read_text())


def art(p):
    p=Path(p);return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size)


def main():
    result=read(OUT/'result.json');identity=read(OUT/'identity.json')
    assert art(OUT/'identity.json')==result['identity']
    assert not result['source_gate'] and result['status']=='completed'
    for record in identity['sources']+[result['source_queries']]:assert art(ROOT/record['path'])==record
    q=pd.read_parquet(OUT/'source_queries.parquet');assert len(q)==2839
    assert not q.duplicated(['mvt_id','ts']).any()
    assert q.observed_speed.between(0,60).all()
    checked=0
    for day,frame in q.groupby('day'):
        chunk=ROOT/f'data/derived/sparse_track_sequence_20261010/full_chunks/{day}'
        index=pd.read_parquet(chunk.with_suffix('.parquet')).set_index('mvt_id')
        points=np.fromfile(chunk.with_suffix('.bin'),np.int32).reshape(-1,6)
        for mid,rows in frame.groupby('mvt_id'):
            flight=index.loc[mid];raw=points[int(flight.start):int(flight.start+flight['count'])]
            absolute=int(flight.to_ms)-raw[:,0].astype(np.int64)
            for row in rows.itertuples(index=False):
                take=np.flatnonzero(absolute==row.ts);assert len(take)==1
                source=raw[take[0]]
                assert row.hex==int(flight.hex) and row.airport==flight.airport
                assert row.observed_speed==source[4]/10 and source[3]==-123
                assert 0<=source[0]<=7200000 and pd.Timestamp(row.ts,unit='ms',tz='UTC').strftime('%Y-%m-%d')==day
                checked+=1
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.register('q',q)
        assert con.execute('''select count(*) from q where supported is distinct from (matched_peers>=3)
            or matched_peers<>least(16,past_peers,null_peers,spatial_peers)
            or (supported and (not isfinite(dynamic) or not isfinite("null") or not isfinite(spatial)))''').fetchone()[0]==0
        for day in result['scores']:
            for arm in ['dynamic','null','spatial']:
                actual=con.execute(f'''select avg(abs("{arm}"-observed_speed)),sqrt(avg(pow("{arm}"-observed_speed,2)))
                    from q where supported and day=?''',[day]).fetchone()
                ref=result['scores'][day]['metrics'][arm]
                np.testing.assert_allclose(actual,[ref['mae'],ref['rmse']],atol=1e-12,rtol=0)
            n,s=con.execute('select count(*),sum(supported::int) from q where day=?',[day]).fetchone()
            assert (n,s)==(result['scores'][day]['queries'],result['scores'][day]['supported'])
            assert result['scores'][day]['metrics']['dynamic']['mae']>result['scores'][day]['metrics']['spatial']['mae']
        for arm in ['dynamic','null','spatial']:
            value=con.execute(f'select avg(abs("{arm}"-observed_speed)) from q where supported').fetchone()[0]
            np.testing.assert_allclose(value,result['pooled_MAE'][arm],atol=1e-12,rtol=0)
    value=dict(status='pass_raw_query_and_SQL_arithmetic',source=art(__file__),result=art(OUT/'result.json'),
        queries=checked,supported=int(q.supported.sum()),all_query_aircraft_calendar_raw_speed_exact=True,
        all_matched_counts_and_source_scores_SQL_exact=True,no_taxi_labels=True,no_fits=True,
        scope='Raw query observations and outcome arithmetic only; donor construction is retained frozen pilot code, not independently rerun.',
        decision='Closed: temporal neighbor speed loses to spatial-only control on both frozen training dates.')
    assert not (OUT/'audit.json').exists()
    (OUT/'audit.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))


if __name__=='__main__':main()
