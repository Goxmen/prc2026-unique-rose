"""Independent original-clock/position arithmetic audit with at most eight source days."""
import hashlib
import json
from pathlib import Path
import sys
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/derived/aobt_track_observation_20261010'
REPORT=ROOT/'tmp/research_20261010/error_mechanisms/aobt_observation_audit.json'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()

def main():
    m=json.loads((OUT/'features.json').read_text());cols=m['settings']['features']
    assert m['settings']['labels_used'] is False
    assert sha(ROOT/'VALIDATION/aobt_track_observation.py')==m['source_sha256']
    assert sha(OUT/'features.parquet')==m['feature_sha256']
    ident=ROOT/'data/derived/aircraft_rotation_20261010/identity.parquet'
    assert sha(ident)==m['identity_sha256']
    coords=ROOT/'data/external/stand_maps/stand_coords.parquet'
    assert sha(coords)==m['stand_map_sha256']
    for path,digest in m['inputs'].items(): assert sha(ROOT/path)==digest
    con=duckdb.connect(config={'threads':2,'memory_limit':'2GB'})
    con.execute("SET TimeZone='UTC'")
    con.read_parquet(str(OUT/'features.parquet')).create_view('f')
    con.read_parquet(str(ident)).create_view('identity')
    # This explicit projection never loads target or departure block.
    con.execute('''CREATE TEMP TABLE q AS SELECT r.MVT_ID_mvt mvt_id,r.ADEP_mvt airport,r.STAND_mvt stand,
        epoch_ms(r.MVT_TIME_UTC_mvt) takeoff_ms,epoch_ms(r.AOBT_3_flt) clock_ms,
        i.hex,strftime(r.AOBT_3_flt,'%Y-%m-%d') AS day
        FROM read_parquet(?) r JOIN identity i ON r.MVT_ID_mvt=i.mvt_id WHERE r.PHASE_mvt='DEP' ''',
        [str(ROOT/'data/training_2025-*.parquet')])
    n,unique=con.execute('SELECT count(*),count(DISTINCT mvt_id) FROM q').fetchone()
    assert n==unique==m['rows']
    assert con.execute('SELECT count(*) FROM q FULL OUTER JOIN f USING(mvt_id) WHERE q.mvt_id IS NULL OR f.mvt_id IS NULL').fetchone()[0]==0
    eligible='q.hex IS NOT NULL AND q.clock_ms IS NOT NULL AND (q.takeoff_ms-q.clock_ms)/1000 BETWEEN -180 AND 14400'
    allnan=' AND '.join(f'f.{c} IS NULL' for c in cols)
    invalid=con.execute(f'SELECT count(*) FROM q JOIN f USING(mvt_id) WHERE NOT coalesce(({eligible}),false) AND NOT({allnan})').fetchone()[0]
    assert invalid==0
    bounds={cols[0]:(-120,120),cols[1]:(0,1e6),cols[2]:(0,1e8),cols[3]:(0,1),cols[4]:(0,1),cols[5]:(0,1)}
    for c,(lo,hi) in bounds.items():
        assert con.execute(f'SELECT count(*) FROM f WHERE {c} IS NOT NULL AND NOT {c} BETWEEN {lo} AND {hi}').fetchone()[0]==0
        assert con.execute(f'SELECT count({c}) FROM f').fetchone()[0]==m['known'][c]
    assert con.execute('SELECT count(*) FROM f WHERE ao_on_ground NOT IN(0,1)').fetchone()[0]==0
    days=['2025-01-01','2025-01-02','2025-07-01','2025-07-02']
    con.register('days',pd.DataFrame({'day':days}))
    picked=con.execute('''WITH candidates AS (
      (SELECT mvt_id FROM q JOIN f USING(mvt_id) WHERE day IN(SELECT day FROM days) AND ao_point_offset_sec IS NOT NULL
       QUALIFY row_number() OVER(PARTITION BY day ORDER BY hash(mvt_id+809))<=8)
      UNION (SELECT mvt_id FROM q JOIN f USING(mvt_id) WHERE day IN(SELECT day FROM days)
             AND clock_ms%86400000 NOT BETWEEN 300000 AND 86100000 ORDER BY hash(mvt_id+809) LIMIT 12)
      UNION (SELECT mvt_id FROM q JOIN f USING(mvt_id) WHERE day IN(SELECT day FROM days)
             AND ao_point_offset_sec IS NULL AND ao_moving_after IS NOT NULL ORDER BY hash(mvt_id+809) LIMIT 6)
      UNION (SELECT mvt_id FROM q WHERE clock_ms IS NULL ORDER BY hash(mvt_id+809) LIMIT 6)
      UNION (SELECT mvt_id FROM q WHERE day IN(SELECT day FROM days) AND hex IS NULL ORDER BY hash(mvt_id+809) LIMIT 6)
      UNION (SELECT mvt_id FROM q WHERE day IN(SELECT day FROM days) AND (takeoff_ms-clock_ms)/1000 NOT BETWEEN -180 AND 14400
             ORDER BY hash(mvt_id+809) LIMIT 6)
    ) SELECT q.*,f.* EXCLUDE(mvt_id) FROM candidates JOIN q USING(mvt_id) JOIN f USING(mvt_id)''').df()
    position_days=sorted({(pd.Timestamp(d)+pd.Timedelta(days=k)).strftime('%Y-%m-%d') for d in days for k in [-1,0,1]})
    assert len(position_days)==8
    paths=[]
    for day in position_days:
        path=ROOT/'data/external/adsblol_heatmap/positions'/f'{day}.parquet'
        if path.exists():
            assert sha(path)==m['positions'][str(path.relative_to(ROOT))]
            paths.append(str(path))
    con.register('picked',picked[['hex','airport']].dropna().drop_duplicates())
    pos=con.execute('''SELECT p.hex,p.airport,epoch_ms(p.ts) ms,p.lat,p.lon,p.alt,p.gs FROM read_parquet(?) p
        WHERE (hex,airport) IN(SELECT hex,airport FROM picked)''',[paths]).df()
    stands=pd.read_parquet(coords).dropna(subset=['stand','lat']).set_index(['airport','stand'])
    values=0;near_ties=0;midnight=0;missing_clock=0;missing_id=0;outside=0;all_missing=0
    samples=[]
    for r in picked.itertuples(index=False):
        expected={c:np.nan for c in cols}
        actual_clock=pd.notna(r.clock_ms)
        known=pd.notna(r.hex)
        usable=actual_clock and known and -180<=(r.takeoff_ms-r.clock_ms)/1000<=14400
        missing_clock+=int(not actual_clock);missing_id+=int(not known)
        outside+=int(actual_clock and not (-180<=(r.takeoff_ms-r.clock_ms)/1000<=14400))
        boundary=actual_clock and (r.clock_ms%86400000<300000 or r.clock_ms%86400000>86100000)
        midnight+=int(boundary)
        found=pos[(pos.hex==r.hex)&(pos.airport==r.airport)&(pos.ms>=r.clock_ms-300000)&(pos.ms<=r.clock_ms+300000)&(pos.ms<=r.takeoff_ms+60000)].copy() if usable else pos.iloc[:0].copy()
        if len(found):
            found['delta']=found.ms-r.clock_ms
            found['distance']=found.delta.abs()
            order=found.sort_values(['distance','ms','lat','lon'])
            closest=order.iloc[0]
            near_ties+=int((found.distance==closest.distance).sum()>1)
            if closest.distance<=120000:
                expected['ao_point_offset_sec']=closest.delta/1000
                expected['ao_speed_kt']=closest.gs/10 if closest.gs>=0 else np.nan
                expected['ao_on_ground']=float(closest.alt==-123) if pd.notna(closest.alt) else np.nan
                if (r.airport,r.stand) in stands.index:
                    slat,slon=stands.loc[(r.airport,r.stand),['lat','lon']]
                    plat,plon=closest.lat/1e6,closest.lon/1e6
                    expected['ao_stand_distance_m']=np.hypot((slat-plat)*111320,(slon-plon)*111320*np.cos(np.deg2rad((slat+plat)/2)))
            ground=found[(found.alt==-123)&(found.gs>=0)]
            for name,mask in [('ao_moving_before',ground.delta<0),('ao_moving_after',ground.delta>=0)]:
                window=ground.loc[mask]
                if len(window):expected[name]=float((window.gs>=50).mean())
        all_missing+=int(all(pd.isna(v) for v in expected.values()))
        for c,wanted in expected.items():
            got=getattr(r,c);wanted=np.float32(wanted)
            assert (pd.isna(got) and pd.isna(wanted)) or np.isclose(got,wanted,atol=.002,rtol=0),(r.mvt_id,c,got,wanted)
            values+=1
        samples.append(dict(mvt_id=int(r.mvt_id),day=r.day,points=len(found),midnight_boundary=bool(boundary)))
    # Explicit independent tie/cutoff checks complement naturally occurring samples.
    sys.path.insert(0,str(ROOT/'VALIDATION'))
    import aobt_track_observation as ao
    t=int(pd.Timestamp('2025-07-01T00:00:00Z').timestamp()*1000)
    query=pd.DataFrame(dict(mvt_id=[1],airport=['X'],stand=['A'],to_ms=[t],clock_ms=[t],hex=[123]))
    synthetic=pd.DataFrame(dict(hex=123,airport='X',ts=pd.to_datetime(t+np.array([-300,-30,30,60,61,300])*1000,unit='ms',utc=True),
        lat=50000000,lon=4000000,alt=[-123,-123,-123,-123,-123,-123],gs=[0,0,50,100,100,100]))
    stand=pd.DataFrame(dict(airport=['X'],stand=['A'],lat=[50.],lon=[4.]))
    got=ao.observe(con,query,synthetic,stand).iloc[0]
    np.testing.assert_allclose(got[cols].to_numpy(float),[-30,0,0,1,0,1],rtol=0,atol=0)
    report=dict(status='passed',manifest_sha256=sha(OUT/'features.json'),source_sha256=m['source_sha256'],all_rows=n,
        raw_sample_rows=len(picked),raw_sample_values=values,raw_position_days=len(paths),raw_position_rows=len(pos),
        midnight_rows=midnight,missing_clock_rows=missing_clock,missing_identity_rows=missing_id,outside_clock_range_rows=outside,
        all_missing_samples=all_missing,natural_nearest_ties=near_ties,synthetic_midnight_tie_and_takeoff_cutoff=True,samples=samples)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
