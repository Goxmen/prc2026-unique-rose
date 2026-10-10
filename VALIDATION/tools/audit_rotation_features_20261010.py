"""Independent bounded arithmetic audit of fold-scoped aircraft rotation tables."""
import hashlib, json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/derived/aircraft_rotation_20261010'
REPORT=ROOT/'tmp/research_20261010/error_mechanisms/rotation_feature_audit.json'
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def diff(a,b):return (a-b).total_seconds()
manifest=json.loads((OUT/'folds.json').read_text())
assert sha(ROOT/'VALIDATION/aircraft_rotation.py')==manifest['source_sha256']
assert sha(OUT/'identity.parquet')==manifest['identity_sha256']
con=duckdb.connect();con.execute("SET threads=2; SET TimeZone='UTC'; SET memory_limit='2GB'")
con.read_parquet(str(OUT/'identity.parquet')).create_view('events')
cols=manifest['settings']['feature_names']; summaries=[]; raw_checks=[]
expected_folds={'2025-01':[f'2025-{m:02d}' for m in range(3,12)],'2025-07':['2025-01','2025-02','2025-03','2025-09','2025-10','2025-11','2025-12']}
for fold,parts in manifest['folds'].items():
 for part,m in parts.items():
    p=Path(m['path']);assert sha(p)==m['sha256']
    allowed=expected_folds[fold]+([fold] if part=='held' else [])
    assert m['reference_months']==allowed
    con.read_parquet(str(p)).create_view('features', replace=True)
    # Stratified fixed sample: dispersed IDs, beginning-of-month boundaries,
    # missing identities, and changed-stand arrivals. No target enters selection.
    q=con.execute('''WITH chosen AS (
      (SELECT mvt_id FROM features ORDER BY hash(mvt_id+713) LIMIT 24)
      UNION (SELECT f.mvt_id FROM features f JOIN events e USING(mvt_id) WHERE day(e.t)=1 ORDER BY hash(f.mvt_id) LIMIT 12)
      UNION (SELECT mvt_id FROM features WHERE ar_aircraft='<MISSING>' ORDER BY mvt_id LIMIT 4)
      UNION (SELECT mvt_id FROM features WHERE ar_prev_same_stand=0 ORDER BY hash(mvt_id) LIMIT 6)
    ) SELECT e.*,f.* EXCLUDE(mvt_id) FROM chosen c JOIN events e USING(mvt_id) JOIN features f USING(mvt_id)''').df()
    con.register('picked',q[['hex','airport','stand']]);con.register('allowed',pd.DataFrame({'month':allowed}))
    refs=con.execute('''SELECT e.* FROM events e WHERE month IN(SELECT month FROM allowed) AND (
       hex IN(SELECT hex FROM picked WHERE hex IS NOT NULL) OR
       (phase='ARR' AND (airport,stand) IN(SELECT airport,stand FROM picked WHERE stand IS NOT NULL)))''').df()
    assert set(refs.month)<=set(allowed)
    nvalues=0; arrival_ids=[]
    for r in q.itertuples():
      ident=not pd.isna(r.hex)
      exp={'ar_aircraft':format(int(r.hex),'06x') if ident else '<MISSING>'}
      for c in cols[1:]:exp[c]=np.nan
      own=refs[(refs.phase=='ARR') & refs.hex.eq(r.hex) & refs.inblock.notna() & refs.t.lt(r.t)] if ident else refs.iloc[:0]
      a=own.sort_values('t').iloc[-1] if len(own) else None
      valid=a is not None and a.airport==r.airport and 0<diff(r.t,a.inblock)<=259200
      st=refs[(refs.phase=='ARR') & refs.airport.eq(r.airport) & refs.stand.eq(r.stand) & refs.inblock.lt(r.t)] if pd.notna(r.stand) else refs.iloc[:0]
      s=st.sort_values('inblock').iloc[-1] if len(st) else None
      if ident and (r.t-pd.Timedelta(hours=24)).strftime('%Y-%m') in allowed:
        exp['ar_departures_24h']=float(((refs.phase=='DEP') & refs.hex.eq(r.hex) & refs.t.ge(r.t-pd.Timedelta(hours=24)) & refs.t.lt(r.t)).sum())
      if ident and s is not None and pd.notna(s.hex):exp['ar_stand_same_aircraft']=float(r.hex==s.hex)
      if valid:
        arrival_ids.append(int(a.mvt_id))
        exp['ar_prev_inblock_age_sec']=diff(r.t,a.inblock)
        exp['ar_prev_taxiin_sec']=a.taxiin
        exp['ar_prev_arrival_delay_sec']=diff(a.inblock,a.arrival_schedule) if pd.notna(a.arrival_schedule) else np.nan
        if pd.notna(r.stand) and pd.notna(a.stand):exp['ar_prev_same_stand']=float(r.stand==a.stand)
        if s is not None:exp['ar_prev_minus_stand_age_sec']=diff(s.inblock,a.inblock)
      for c in cols:
        actual=getattr(r,c);wanted=exp[c]
        if c=='ar_aircraft':assert actual==wanted,(fold,part,r.mvt_id,c,actual,wanted)
        else:assert (pd.isna(actual) and pd.isna(wanted)) or np.isclose(actual,np.float32(wanted),atol=.02,rtol=0),(fold,part,r.mvt_id,c,actual,wanted)
        nvalues+=1
    # Independently check arrival in-block against supplied raw taxi-in labels.
    ids=pd.DataFrame({'mvt_id':list(set(arrival_ids))});con.register('arrival_sample',ids)
    raw=con.execute('''SELECT r.MVT_ID_mvt mvt_id,r.PHASE_mvt phase,r.TAXITIME_SEC_mvt taxiin,
       r.MVT_TIME_UTC_mvt landing,r.SCHED_TIME_UTC_mvt schedule,e.inblock,e.arrival_schedule,e.taxiin cached_taxiin
       FROM read_parquet(?) r JOIN arrival_sample a ON r.MVT_ID_mvt=a.mvt_id JOIN events e USING(mvt_id)''',[[str(next((ROOT/'data').glob(f'training_{x}-01_*.parquet'))) for x in allowed]]).df()
    assert len(raw)==len(ids)
    assert raw.phase.eq('ARR').all() and raw.taxiin.ge(0).all()
    assert (raw.landing+pd.to_timedelta(raw.taxiin,unit='s')).eq(raw.inblock).all()
    assert raw.schedule.eq(raw.arrival_schedule).all() and raw.taxiin.eq(raw.cached_taxiin).all()
    summaries.append(dict(fold=fold,part=part,rows=len(q),values=nvalues,boundary_rows=int(q.t.dt.day.eq(1).sum()),raw_arrivals=len(raw),reference_months=allowed,sha256=m['sha256']))
report=dict(status='passed',folds_sha256=sha(OUT/'folds.json'),source_sha256=manifest['source_sha256'],identity_sha256=manifest['identity_sha256'],sampled_values=sum(x['values'] for x in summaries),folds=summaries)
REPORT.parent.mkdir(parents=True,exist_ok=True);REPORT.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
