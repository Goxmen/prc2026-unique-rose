"""Cache the publicly embedded EUROCONTROL daily/runway taxi-out aggregates.

Uses the report's anonymous public API, exactly as its public browser client.
The resource key is published in the iframe URL and is not an account credential.
No private endpoint, browser login or per-flight target is requested.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request
import uuid

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/external/eurocontrol_daily_taxi_20261011'
RESEARCH=ROOT/'tmp/research_20261011/daily_taxi_source'
KEY='9ce24e97-d21b-4de7-84af-0314128a1f6f'
API='https://wabi-west-europe-d-primary-api.analysis.windows.net/public/reports/'
PAGE='https://ansperformance.eu/efficiency/taxiout/'
AIRPORTS=['EDDF','EDDM','EGLL','EHAM','LEBL','LEMD','LFPG','LIRF','LSZH','LTFM']
COLUMNS=['day','airport','phase','runway','n','total_min','ref_min','add_min']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def record(path):return dict(path=str(Path(path).relative_to(ROOT)),sha256=digest(path))


def decode(response):
    assert len(response['results'])==1
    data=response['results'][0]['result']['data']
    sets=data['dsr']['DS'];assert len(sets)==1
    dataset=sets[0];assert len(dataset['PH'])==1
    rows=dataset['PH'][0]['DM0'];assert len(rows)<30000,'Truncated window; request smaller scope'
    schema=rows[0]['S'];assert len(schema)==8
    expected=['TAXI_OUT.'+n for n in ['MVT_DD','ADEP_ICAO','SRC_PHASE','AP_C_RWY']]
    expected+=['Sum(TAXI_OUT.'+n+')' for n in ['N','TAXI_OUT_TIME','REF_TAXI_OUT_TIME','ADD_TAXI_OUT_TIME']]
    assert [s['Name'] for s in data['descriptor']['Select']]==expected
    previous=[None]*len(schema);output=[]
    for row in rows:
        assert set(row)<=set(['S','C','R','Ø']),set(row)
        if 'S' in row:assert row['S']==schema
        assert not row.get('R',0)&row.get('Ø',0)
        supplied=iter(row.get('C',[]));current=[]
        for i,s in enumerate(schema):
            if row.get('R',0)&(1<<i):value=previous[i]
            elif row.get('Ø',0)&(1<<i):value=None
            else:value=next(supplied)
            current.append(value)
        sentinel=object();assert next(supplied,sentinel) is sentinel
        decoded=[]
        for value,s in zip(current,schema):
            if value is not None and 'DN' in s:value=dataset['ValueDicts'][s['DN']][int(value)]
            elif value is not None and s['T']==3:value=float(value)
            decoded.append(value)
        output.append(decoded);previous=current
    for event in data.get('metrics',{}).get('Events',[]):
        if 'RowCount' in event.get('Metrics',{}):assert event['Metrics']['RowCount']==len(rows)
    frame=pd.DataFrame(output,columns=COLUMNS)
    frame['day']=pd.to_datetime(frame.day,unit='ms',utc=True).dt.strftime('%Y-%m-%d')
    assert frame[COLUMNS[:4]].notna().all().all()
    assert np.isfinite(frame[COLUMNS[4:]]).all().all()
    assert (frame.n>0).all() and np.equal(frame.n,np.rint(frame.n)).all()
    assert not frame.duplicated(COLUMNS[:4]).any()
    np.testing.assert_allclose(frame.total_min,frame.ref_min+frame.add_min,atol=1e-7,rtol=0)
    return frame


def column(name):return {'Column':{'Expression':{'SourceRef':{'Source':'t'}},'Property':name}}


def request(airport,models):
    select=[dict(column(n),Name='TAXI_OUT.'+n) for n in ['MVT_DD','ADEP_ICAO','SRC_PHASE','AP_C_RWY']]
    select += [{'Aggregation':{'Expression':column(n),'Function':0},'Name':'Sum(TAXI_OUT.'+n+')'}
        for n in ['N','TAXI_OUT_TIME','REF_TAXI_OUT_TIME','ADD_TAXI_OUT_TIME']]
    where=[{'Condition':{'In':{'Expressions':[column('ADEP_ICAO')],'Values':[[{'Literal':{'Value':repr(airport)}}]]}}}]
    for kind,date in [(2,'2025-01-01'),(3,'2026-08-01')]:
        where.append({'Condition':{'Comparison':{'ComparisonKind':kind,'Left':column('MVT_DD'),
            'Right':{'Literal':{'Value':f"datetime'{date}T00:00:00'"}}}}})
    query=dict(Version=2,From=[dict(Name='t',Entity='TAXI_OUT',Type=0)],Select=select,Where=where,
        OrderBy=[dict(Direction=1,Expression=column(n)) for n in ['MVT_DD','AP_C_RWY']])
    binding=dict(Primary=dict(Groupings=[dict(Projections=list(range(8)))]),
        DataReduction=dict(DataVolume=4,Primary=dict(Window=dict(Count=30000))),Version=1)
    command=dict(Commands=[dict(SemanticQueryDataShapeCommand=dict(Query=query,Binding=binding,ExecutionMetricsKind=1))])
    return dict(version='1.0.0',queries=[dict(Query=command,ApplicationContext=dict(
        DatasetId=models['models'][0]['dbName'],Sources=[dict(ReportId=models['exploration']['reportId'])]))],
        cancelQueries=[],modelId=models['models'][0]['id'])


def fetch():
    OUT.mkdir(parents=True,exist_ok=True)
    manifest_path=OUT/'manifest.json'
    assert not manifest_path.exists(),'Complete cached source exists; verify and reuse it'
    for name in ['taxiout.html','powerbi.html','models.json','schema.json']:
        src=RESEARCH/name;dest=OUT/name
        if dest.exists():assert digest(src)==digest(dest)
        else:shutil.copy2(src,dest)
    models=json.loads((OUT/'models.json').read_text())
    assert models['models'][0]['id']==3679962
    frames=[];receipts=[]
    for airport in AIRPORTS:
        rp=OUT/f'{airport}_request.json';sp=OUT/f'{airport}_response.json'
        if airport=='EHAM' and (RESEARCH/'eham_response.json').exists() and not sp.exists():
            shutil.copy2(RESEARCH/'eham_request.json',rp);shutil.copy2(RESEARCH/'eham_response.json',sp)
        if not sp.exists():
            payload=request(airport,models)
            if rp.exists():assert json.loads(rp.read_text())==payload
            else:rp.write_text(json.dumps(payload,indent=2)+'\n')
            headers={'Accept':'application/json','Content-Type':'application/json','User-Agent':'Mozilla/5.0',
                'ActivityId':str(uuid.uuid4()),'RequestId':str(uuid.uuid4()),'X-PowerBI-ResourceKey':KEY}
            req=urllib.request.Request(API+'querydata?synchronous=true',data=json.dumps(payload).encode(),headers=headers)
            with urllib.request.urlopen(req,timeout=30) as response:body=response.read()
            json.loads(body)
            temporary=sp.with_suffix('.json.tmp');temporary.write_bytes(body);temporary.replace(sp)
        frame=decode(json.loads(sp.read_text()))
        assert set(frame.airport)=={airport} and set(frame.phase)=={'DEP'}
        original_rows=len(frame)
        frame=frame[frame.day.ge('2025-01-01')&frame.day.lt('2026-08-01')].copy()
        assert len(frame) and frame.day.min()=='2025-01-01' and frame.day.max()=='2026-07-31'
        frames.append(frame)
        receipts.append(dict(airport=airport,raw_rows=original_rows,selected_rows=len(frame),
            days=frame.day.nunique(),request=record(rp),response=record(sp)))
        print(receipts[-1],flush=True)
    data=pd.concat(frames,ignore_index=True)
    assert not data.duplicated(COLUMNS[:4]).any()
    output=OUT/'daily_runway.parquet'
    if output.exists():pd.testing.assert_frame_equal(pd.read_parquet(output),data,check_exact=True)
    else:data.to_parquet(output,index=False)
    manifest=dict(fetched_at=datetime.now(timezone.utc).isoformat(),source_page=PAGE,
        public_model_id=3679962,public_model_refresh=models['models'][0]['LastRefreshTime'],
        source=record(__file__),metadata=[record(OUT/n) for n in ['taxiout.html','powerbi.html','models.json','schema.json']],
        receipts=receipts,output=record(output),rows=len(data),columns=COLUMNS,
        units='N flight count; taxi/reference/additional totals in minutes; source date decoded from UTC epoch',
        validation='compression and arithmetic checks only; population identity, local date and model benefit still unverified',
        reuse='Official publicly embedded EUROCONTROL dashboard; retain attribution and unmodified responses; source terms follow-up required')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(dict(rows=len(data),airports=10,first=data.day.min(),last=data.day.max(),manifest=str(manifest_path)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['fetch'])
    args=parser.parse_args();globals()[args.stage]()
