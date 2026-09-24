"""Independent raw-clock, weather-time, traffic and aggregation checks for step 4."""
import json
from pathlib import Path
import sys

import duckdb
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parent))
import investigate_arrival_shift as base


def main():
    out=base.OUT
    audit=json.loads((out/'audit.json').read_text())
    assert audit['source_sha256']==base.sha(Path(base.__file__))
    assert audit['design_sha256']==base.sha(out/'design.json')
    assert audit['features']['sha256']==base.sha(base.CACHE/'arrival_features.parquet')
    for file,h in audit['outputs'].items():assert base.sha(out/file)==h
    follow=json.loads((out/'snow_duration_audit.json').read_text())
    for file,h in follow['outputs'].items():assert base.sha(out/file)==h
    for file,h in follow['cache'].items():assert base.sha(base.CACHE/file)==h
    a=pd.read_parquet(base.CACHE/'arrival_features.parquet')
    assert a.mvt_id.is_unique
    old=json.loads((base.ROOT/'VALIDATION/reports/recording_regime_2026.json').read_text())
    count=0
    for row in old['by_airport']:
        if row['airport'] not in base.AIRPORTS or row['phase']!='ARR':continue
        g=a[a.airport.eq(row['airport'])&a.month.eq(row['ym'])]
        assert len(g)==row['n']
        np.testing.assert_allclose(g.taxi_sec.quantile(.999),row['label_p999'],rtol=1e-12)
        np.testing.assert_allclose(g.taxi_sec.gt(7200).mean()*100,row['pct_label_over_2h'],rtol=1e-12,atol=1e-12)
        count+=1
    con=duckdb.connect();con.execute("SET TimeZone='UTC'");con.execute('SET threads=6')
    paths=[str(p) for p in sorted((base.ROOT/'data').glob('training_*.parquet'))]+[str(base.ROOT/'data/ranking.parquet')]
    con.read_parquet(paths).create_view('raw')
    raw=con.sql("""SELECT MVT_ID_mvt AS mvt_id, TAXITIME_SEC_mvt AS y,
        epoch(BLOCK_TIME_UTC_mvt-MVT_TIME_UTC_mvt) AS difference
        FROM raw WHERE PHASE_mvt='ARR' AND ADES_mvt IN ('EHAM','EDDF','LSZH')""").df().set_index('mvt_id')
    np.testing.assert_array_equal(raw.loc[a.mvt_id,'y'],a.taxi_sec)
    np.testing.assert_array_equal(raw.y,raw.difference)
    events=con.sql("""SELECT CASE WHEN PHASE_mvt='ARR' THEN ADES_mvt ELSE ADEP_mvt END AS airport,
        PHASE_mvt AS phase,CAST(MVT_TIME_UTC_mvt AS TIMESTAMP) AS stamp
        FROM raw WHERE (PHASE_mvt='ARR' AND ADES_mvt IN ('EHAM','EDDF','LSZH'))
        OR (PHASE_mvt='DEP' AND ADEP_mvt IN ('EHAM','EDDF','LSZH'))""").df()
    events['month']=events.stamp.dt.strftime('%Y-%m')
    samples=[]
    for _,g in a[a.month.isin(['2025-01','2026-01','2025-07','2026-07'])].groupby(['airport','month']):
        g=g.sort_values('mvt_id');samples.extend(g.index[np.linspace(0,len(g)-1,5,dtype=int)])
    metar=pd.read_parquet(base.weather.METAR);metar['utc']=base.weather._utc_naive(metar.valid)
    traffic_checks=0
    for _,r in a.loc[samples].iterrows():
        wx=metar[metar.station.eq(r.airport)&metar.utc.le(r.landing)]
        assert wx.utc.max()==r.valid
        e=events[events.airport.eq(r.airport)&events.month.eq(r.month)]
        for phase,col in [('ARR','arrivals_prev30'),('DEP','departures_prev30')]:
            if pd.isna(r[col]):continue
            n=(e.phase.eq(phase)&e.stamp.ge(r.landing-pd.Timedelta(minutes=30))&e.stamp.lt(r.landing)).sum()
            assert n==r[col]
            traffic_checks+=1
    # Reconstruct the snow-duration statistic directly from raw report lookups,
    # without the rolling implementation used to build the saved grid.
    s=pd.read_parquet(base.CACHE/'EHAM_snow_duration.parquet')
    reports=metar[metar.station.eq('EHAM')].copy()
    tokens=reports.wxcodes.fillna('').str.replace(r'[+-]','',regex=True).str.split()
    reports['snow']=tokens.map(lambda ts:any('SN' in t or 'SG' in t for t in ts))
    reports=reports.groupby('utc').snow.max().sort_index()
    chosen=s[s.month.isin(['2025-01','2026-01'])].sort_values('mvt_id')
    chosen=chosen.iloc[np.linspace(0,len(chosen)-1,40,dtype=int)]
    for _,r in chosen.iterrows():
        ticks=pd.date_range(end=r['sample'],periods=48,freq='30min')
        loc=reports.index.get_indexer(ticks,method='pad')
        known=(loc>=0)&((ticks-reports.index[np.maximum(loc,0)])<=pd.Timedelta(minutes=90))
        expected=float(reports.iloc[loc[known]].mean()) if known.sum()>=40 else np.nan
        np.testing.assert_allclose(r.snow_fraction24h,expected,rtol=1e-12,equal_nan=True)
    c=pd.read_csv(out/'conditional_tail_rates.csv');cells=pd.read_csv(out/'supported_cells.csv')
    checks=0
    for _,r in c[c.threshold.eq(1800)].iterrows():
        q=cells[cells.case.eq(r.case)&cells.stage.eq(r.stage)]
        assert int(q.n_target.sum())==r.target_supported_n
        assert int(q.n_source.sum())==r.source_supported_n
        if len(q):
            expected=100*((q.source_tail_n/q.n_source)*q.n_target).sum()/q.n_target.sum()
            np.testing.assert_allclose(expected,r.expected_target_tail_pct,rtol=1e-12,atol=1e-12)
            np.testing.assert_allclose(q.target_tail_n.sum(),r.target_supported_n*r.target_supported_tail_pct/100,atol=1e-8)
        checks+=1
    # Main factual totals, including overlap and all-year severity support.
    jan=s[s.month.eq('2026-01')];oldjan=s[s.month.eq('2025-01')];allyear=s[s.month.str.startswith('2025')]
    tail=jan.taxi_sec.gt(1800);longsnow=jan.snow_fraction24h.gt(.25)
    summary=dict(target_n=len(jan),tail30_n=int(tail.sum()),
        tail30_with_recent_snow=int((tail&jan.snow_prev24h.eq(1)).sum()),
        prolonged_snow_n=int(longsnow.sum()),tail30_in_prolonged_snow=int((tail&longsnow).sum()),
        old_january_prolonged_snow_n=int(oldjan.snow_fraction24h.gt(.25).sum()),
        all_2025_prolonged_snow_n=int(allyear.snow_fraction24h.gt(.25).sum()),
        above_full2025_snow_duration_max_n=int(jan.snow_fraction24h.gt(allyear.snow_fraction24h.max()).sum()),
        above_full2025_snow_duration_max_tail_n=int((tail&jan.snow_fraction24h.gt(allyear.snow_fraction24h.max())).sum()))
    result=dict(passed=True,source_sha256=base.sha(Path(__file__)),monthly_reference_checks=count,
        raw_label_arithmetic_rows=len(a),weather_time_checks=len(samples),traffic_checks=traffic_checks,
        independent_snow_duration_checks=len(chosen),standardization_arithmetic_checks=checks,
        main_counts=summary)
    base.write(out/'independent_audit.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
