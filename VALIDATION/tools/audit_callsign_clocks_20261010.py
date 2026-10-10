"""Independent bounded source-interval and all-row validity audit; no targets read."""
import hashlib
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/derived/callsign_clocks_20261010'
ROT=ROOT/'data/derived/aircraft_rotation_20261010'
REPORT=ROOT/'tmp/research_20261010/error_mechanisms/callsign_feature_audit.json'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1048576),b''): h.update(block)
    return h.hexdigest()

def equal(a,b):
    return (pd.isna(a) and pd.isna(b)) or np.isclose(a,b,atol=.002,rtol=0)

def main():
    m=json.loads((OUT/'manifest.json').read_text())
    cols=m['settings']['feature_names']
    assert sha(ROOT/'VALIDATION/callsign_clocks.py')==m['source_sha256']
    assert sha(ROT/'identity.parquet')==m['identity_sha256']
    assert sha(ROT/'folds.json')==m['rotation_manifest_sha256']
    assert sha(OUT/'shared.parquet')==m['shared_sha256']
    con=duckdb.connect(config={'threads':2,'memory_limit':'2GB'})
    con.execute("SET TimeZone='UTC'")
    con.read_parquet(str(OUT/'shared.parquet')).create_view('s')
    con.read_parquet(str(ROT/'identity.parquet')).create_view('i')
    selected=con.execute('''WITH chosen AS (
      (SELECT mvt_id FROM s ORDER BY hash(mvt_id+317) LIMIT 10)
      UNION (SELECT mvt_id FROM s WHERE cc_ambiguous=1 ORDER BY hash(mvt_id+317) LIMIT 8)
      UNION (SELECT mvt_id FROM s WHERE cc_censored=1 ORDER BY hash(mvt_id+317) LIMIT 6)
      UNION (SELECT mvt_id FROM s WHERE floor(call_start_ms/86400000)<floor(to_ms/86400000)
             AND cc_call_start_sec IS NOT NULL ORDER BY hash(mvt_id+317) LIMIT 8)
      UNION (SELECT s.mvt_id FROM s JOIN i USING(mvt_id) WHERE i.hex IS NULL ORDER BY hash(s.mvt_id+317) LIMIT 6)
      UNION (SELECT s.mvt_id FROM s JOIN i USING(mvt_id) WHERE i.hex IS NOT NULL AND s.call_start_ms IS NULL
             ORDER BY hash(s.mvt_id+317) LIMIT 6)
    ) SELECT s.*,i.hex,i.cs,i.t FROM chosen JOIN s USING(mvt_id) JOIN i USING(mvt_id) ORDER BY t''').df()
    selected['day']=selected.t.dt.strftime('%Y-%m-%d')
    picked_hex=selected[['hex']].dropna().drop_duplicates()
    con.register('picked_hex',picked_hex)
    history=con.execute("SELECT hex,cs,t FROM i WHERE phase='DEP' AND hex IN(SELECT hex FROM picked_hex)").df()
    cache={};checked_sources={}; rows=[]
    def source(day):
        if day not in cache:
            p=ROOT/'data/external/adsblol_heatmap/callsigns'/f'{day}.parquet'
            if p.exists():
                expected=next(x['sha256'] for x in m['call_sources'] if x['path']==str(p))
                assert sha(p)==expected
                checked_sources[day]=expected
                frame=pd.read_parquet(p)
                frame=frame[frame.records>=3].copy()
                frame['cs']=frame.callsign.fillna('').str.replace(' ','',regex=False).str.upper().str.strip()
                frame=frame[frame.cs.ne('')]
                frame['day']=day
            else:
                frame=pd.DataFrame(columns=['hex','cs','first_ms','last_ms','squawk','day'])
            cache[day]=frame
        return cache[day]
    bridge_examples=0
    for r in selected.itertuples(index=False):
        raw=pd.concat([source((pd.Timestamp(r.day)+pd.Timedelta(days=k)).strftime('%Y-%m-%d')) for k in [-2,-1,0,1]],ignore_index=True)
        aircraft=raw[raw.hex.eq(r.hex)] if pd.notna(r.hex) else raw.iloc[:0]
        cur=aircraft[aircraft.cs.eq(r.cs)]
        others=aircraft[~aircraft.cs.eq(r.cs)]
        active=cur[(cur.first_ms-180000<=r.to_ms)&(cur.last_ms+180000>=r.to_ms)]
        expected={c:np.nan for c in ['call_start_ms','squawk_start_ms','previous_call_end_ms','active_interval_count',*cols]}
        bridged=False
        if len(active):
            envelopes=cur.groupby('day',sort=False).agg(first=('first_ms','min'),last=('last_ms','max'))
            eligible=envelopes[(envelopes['first']-180000<=r.to_ms)&(envelopes['last']+180000>=r.to_ms)]
            chosen=sorted(eligible.index,key=lambda d:(d!=r.day,abs((pd.Timestamp(d)-pd.Timestamp(r.day)).days),d))[0]
            def earliest(first,intervals):
                initial=first
                while True:
                    previous=[a for a,b in intervals if a<first and 0<=first-b<=600000]
                    if not previous: return first,first!=initial
                    first=min(previous)
            call_start,bridged=earliest(envelopes.loc[chosen,'first'],list(envelopes[['first','last']].itertuples(index=False,name=None)))
            current_squawk=active.loc[active.first_ms.idxmax()]
            sq=cur[cur.squawk.eq(current_squawk.squawk)]
            squawk_start,_=earliest(current_squawk.first_ms,list(sq[['first_ms','last_ms']].itertuples(index=False,name=None)))
            previous=others[(others.last_ms>=r.to_ms-14400000)&(others.last_ms<r.to_ms)]
            end=previous.last_ms.max() if len(previous) else np.nan
            earlier=history[history.hex.eq(r.hex)&history.cs.eq(r.cs)&history.t.dt.strftime('%Y-%m-%d').eq(r.day)&history.t.lt(r.t)]
            previous_departure=earlier.t.max().value//1000000 if len(earlier) else np.nan
            ambiguous=bool(((others.first_ms<r.to_ms)&(others.last_ms>call_start)).any() or active.squawk.nunique()>1 or (pd.notna(previous_departure) and call_start<=previous_departure<r.to_ms))
            expected.update(call_start_ms=call_start,squawk_start_ms=squawk_start,previous_call_end_ms=end,active_interval_count=len(active),cc_ambiguous=float(ambiguous),cc_censored=float(call_start%1800000<=20000 or call_start>r.to_ms))
            for name,key,lower in [('cc_call_start_sec','call_start_ms',-180),('cc_squawk_start_sec','squawk_start_ms',-180),('cc_previous_call_end_sec','previous_call_end_ms',0)]:
                age=(r.to_ms-expected[key])/1000
                expected[name]=age if lower<=age<=14400 else np.nan
        for name,value in expected.items():
            if name=='cc_before_own_arrival': continue
            assert equal(getattr(r,name),value),(r.mvt_id,name,getattr(r,name),value)
        bridge_examples+=int(bridged)
        rows.append(dict(mvt_id=int(r.mvt_id),day=r.day,missing_identity=bool(pd.isna(r.hex)),active_intervals=int(len(active)),bridged=bool(bridged)))
    folds=[]
    for month,parts in m['folds'].items():
        for part,meta in parts.items():
            path=OUT/f'{month}_{part}.parquet';rot=ROT/f'{month}_{part}.parquet'
            assert sha(path)==meta['sha256'] and sha(rot)==meta['rotation_sha256']
            con.read_parquet(str(path)).create_view('f',replace=True)
            con.read_parquet(str(rot)).create_view('a',replace=True)
            schema=con.execute('DESCRIBE f').df()
            assert schema.column_name.tolist()==['mvt_id',*cols]
            assert schema.column_type.tolist()[1:]==['FLOAT']*6
            total,distinct,missing=con.execute('SELECT count(*),count(DISTINCT mvt_id),count(*) FILTER(WHERE i.hex IS NULL) FROM f JOIN i USING(mvt_id)').fetchone()
            assert total==distinct==meta['rows']
            allnan=' AND '.join(f'f.{c} IS NULL' for c in cols)
            assert con.execute(f'SELECT count(*) FROM f JOIN i USING(mvt_id) WHERE i.hex IS NULL AND NOT({allnan})').fetchone()[0]==0
            for name in cols[:3]:
                low=0 if name==cols[2] else -180
                assert con.execute(f'SELECT count(*) FROM f WHERE {name} IS NOT NULL AND NOT {name} BETWEEN {low} AND 14400').fetchone()[0]==0
            for name in cols[3:]:
                assert con.execute(f'SELECT count(*) FROM f WHERE {name} IS NOT NULL AND {name} NOT IN (0,1)').fetchone()[0]==0
            for name in [*cols[:3],*cols[4:]]:
                assert con.execute(f'SELECT count(*) FROM f JOIN s USING(mvt_id) WHERE f.{name} IS DISTINCT FROM s.{name}::FLOAT').fetchone()[0]==0
            assert con.execute('''SELECT count(*) FROM f JOIN s USING(mvt_id) JOIN a USING(mvt_id)
                WHERE f.cc_before_own_arrival IS DISTINCT FROM CASE WHEN s.call_start_ms IS NOT NULL AND a.ar_prev_inblock_age_sec IS NOT NULL
                THEN (((s.to_ms-s.call_start_ms)/1000)>a.ar_prev_inblock_age_sec)::FLOAT ELSE NULL END''').fetchone()[0]==0
            for name in cols:
                assert con.execute(f'SELECT count({name}) FROM f').fetchone()[0]==meta['nonmissing'][name]
            folds.append(dict(month=month,part=part,rows=total,missing_identity_rows=missing,reference_months=meta['reference_months']))
    report=dict(status='passed',manifest_sha256=sha(OUT/'manifest.json'),source_sha256=m['source_sha256'],raw_sample_rows=len(rows),raw_sample_values=len(rows)*9,bridge_examples=bridge_examples,raw_source_files=len(checked_sources),folds=folds,samples=rows)
    REPORT.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))

if __name__=='__main__':main()
