from pathlib import Path
import hashlib,json
import duckdb
import pandas as pd

root=Path(__file__).resolve().parents[2];out=root/'VALIDATION/reports/research_network_20261010/monthly_population_probe_20261011';out.mkdir(exist_ok=True)
paths=sorted((root/'data').glob('training_2025-*.parquet'))
monthly=root/'VALIDATION/runs/v3_20261011_monthly_taxi/rows2025.parquet'
source=root/'data/external/eurocontrol_performance_20261011/taxi_out_additional_time_2025.csv'
with duckdb.connect(config={'threads':1}) as con:
 con.read_parquet(str(monthly)).create_view('rows')
 con.read_parquet([str(p) for p in paths]).create_view('raw')
 f=con.execute('''with eligible as (
 select r.*, p.STAND_mvt stand,p.RUNWAY_mvt runway
 from rows r join raw p on r.mvt_id=p.MVT_ID_mvt
 where r.eligible and r.y between 0 and 7200 and r.local_month between '2025-01' and '2025-12'
 ), p10 as (
 select airport,stand,runway,quantile_cont(y,0.1) q10
 from eligible group by airport,stand,runway
 ), cells as (
 select e.airport,e.stand,e.runway,count(*) n,sum(case when e.y<=p.q10 then 1 else 0 end) lower_n
 from eligible e join p10 p using(airport,stand,runway) group by e.airport,e.stand,e.runway
 ) select e.airport,e.local_month,count(*) basic_n,avg(e.y) basic_mean,
 sum(case when c.lower_n>=10 then 1 else 0 end) reference_n,
 avg(case when c.lower_n>=10 then e.y else null end) reference_mean
 from eligible e join cells c using(airport,stand,runway)
 where e.local_month='2025-12' group by e.airport,e.local_month''').fetchdf()
s=pd.read_csv(source);s=s[s.MONTH_NUM.eq(12)].rename(columns={'APT_ICAO':'airport'})
s['source_mean']=60*(s.TOTAL_REF_TIME_MIN+s.TOTAL_ADD_TIME_MIN)/s.TOTAL_REF_NB_FL
f=f.merge(s[['airport','TF','VALID_FL','NO_REF','TOTAL_REF_NB_FL','source_mean']],on='airport',validate='one_to_one')
f['basic_minus_valid']=f.basic_n-f.VALID_FL
f['reference_count_gap']=f.reference_n-f.TOTAL_REF_NB_FL
f['reference_mean_gap']=f.reference_mean-f.source_mean
f['basic_mean_gap']=f.basic_mean-f.source_mean
f.to_csv(out/'december.csv',index=False)
record={'purpose':'Descriptive source-population audit only; no CV scoring, model training, candidate or upload',
 'reference':'https://ansperformance.eu/library/ATXOT_indicator_documentation_mar23.pdf',
 'scope':'December2025: previous12local months approximately covered by supplied2025 data',
 'rule':'stand/runway present,0<=taxi<=7200; reference cell requires at least10 values <= its linear P10',
 'limitations':['No helicopter or post-off-block deicing exclusion','First local hours of January missing because source starts UTC January1','Same development data already examined; not independent validation','No new prediction or hidden-label inference'],
 'inputs':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [monthly,source]],
 'result':f.to_dict('records')}
(out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
print(f.round(3).to_string(index=False))
