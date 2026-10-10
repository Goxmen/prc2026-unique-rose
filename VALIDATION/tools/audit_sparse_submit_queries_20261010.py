"""Bounded independent raw-input audit of production sequence query packets."""
from pathlib import Path
import hashlib
import json
import sys

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data/derived/sparse_track_submit_20261010'
REPORT = ROOT / 'tmp/research_20261010/error_mechanisms/sparse_submit_queries_audit.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    pa.set_cpu_count(2)
    con = duckdb.connect()
    con.execute("SET threads=2; SET TimeZone='UTC'; SET memory_limit='1GB'")
    manifest = json.loads((OUT/'manifest.json').read_text())
    stands = pd.read_parquet(ROOT/'data/external/stand_maps/stand_coords.parquet').dropna(subset=['stand','lat'])
    runways = pd.read_parquet(ROOT/'data/external/stand_maps/runway_ends.parquet')
    origin = runways.groupby('airport')[['lat','lon']].mean()
    result = {}
    for view, description in manifest['views'].items():
        path = ROOT / description['query']['path']
        assert sha(path) == description['query']['sha256']
        query = pd.read_parquet(path)
        chosen = [query.sample(n=64, random_state=20261010)]
        for _, month in query.groupby(query.day.str[:7]):
            ordered = month.sort_values('to_ms')
            chosen.extend([ordered.head(4), ordered.tail(4)])
        chosen.extend([query.loc[query.clock_ms.isna()].head(12),
                       query.loc[query.hex.isna()].head(12)])
        sample = pd.concat(chosen).drop_duplicates('mvt_id').sort_values('mvt_id').reset_index(drop=True)
        con.register('audit_ids', sample[['mvt_id']])
        raw = con.execute('''SELECT MVT_ID_mvt mvt_id, PHASE_mvt phase,
            ADEP_mvt airport, STAND_mvt stand, RUNWAY_mvt runway,
            epoch_ms(MVT_TIME_UTC_mvt)::BIGINT to_ms,
            epoch_ms(AOBT_3_flt)::BIGINT clock_ms
            FROM read_parquet(?) JOIN audit_ids ON MVT_ID_mvt=mvt_id
            ORDER BY MVT_ID_mvt''', [str(ROOT/description['raw_ranking']['path'])]).df()
        assert raw.phase.eq('DEP').all()
        pd.testing.assert_frame_equal(sample[list(raw.drop(columns='phase').columns)],
            raw.drop(columns='phase'), check_dtype=False)
        ident = con.execute('''SELECT mvt_id, phase, airport, epoch_ms(t)::BIGINT to_ms, hex
            FROM read_parquet(?) JOIN audit_ids USING(mvt_id) ORDER BY mvt_id''',
            [str(ROOT/description['identity']['path'])]).df()
        assert ident.phase.eq('DEP').all()
        pd.testing.assert_frame_equal(sample[['mvt_id','airport','to_ms','hex']],
            ident[['mvt_id','airport','to_ms','hex']], check_dtype=False)
        for kind, field, table in [('stand','stand',stands),('runway','runway',runways)]:
            lookup = {(r.airport,getattr(r,field)): (r.lat,r.lon) for r in table.itertuples()}
            for row in sample.itertuples():
                expected = lookup.get((row.airport,getattr(row,field)), (np.nan,np.nan))
                np.testing.assert_allclose([getattr(row,f'{kind}_lat'),getattr(row,f'{kind}_lon')],
                                           expected, rtol=0,atol=0,equal_nan=True)
        for row in sample.itertuples():
            np.testing.assert_allclose([row.origin_lat,row.origin_lon],
                                       origin.loc[row.airport].to_numpy(),rtol=0,atol=0)
        expected_day = pd.to_datetime(sample.to_ms,unit='ms',utc=True).dt.strftime('%Y-%m-%d')
        assert sample.day.eq(expected_day).all()
        result[view] = dict(rows=len(query), checked_rows=len(sample),
            months=sorted(sample.day.str[:7].unique().tolist()),
            missing_clock=int(sample.clock_ms.isna().sum()),
            missing_identity=int(sample.hex.isna().sum()),
            query_sha256=description['query']['sha256'],
            fields='raw phase/ID/airport/stand/runway/takeoff/AOBT, audited hex, static stand/runway/origin geometry')
    report = dict(status='passed',source_sha256=sha(__file__),
                  prepared_manifest_sha256=sha(OUT/'manifest.json'),views=result,
                  departure_labels_or_block_read=False,positions_or_model_cache_read=False)
    REPORT.parent.mkdir(parents=True,exist_ok=True)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
