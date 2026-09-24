"""Assemble audited step-2 scores/neighbours for later saved-error analysis.

Uses original raw kNN and corrected weather kNN. No departure labels or model
losses are loaded. Both probability views are retained, with calibration limits.
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

import check_train_ranking_similarity as base
import check_similarity_weather_scale as corrected


def main():
    source_manifest=json.loads((base.SOURCE/'manifest.json').read_text())
    stand_keys=set()
    for month,digest in source_manifest['files'].items():
        if not month.startswith('2025-'):
            continue
        path=base.SOURCE/f'{month}.parquet';assert base.sha(path)==digest
        f=pd.read_parquet(path,columns=['airport','stand']).dropna().drop_duplicates()
        stand_keys.update(f.itertuples(index=False,name=None))
    joined=[];slices=[]
    key=['mvt_id','date','airport','label','fold']
    for case in ['january_year','july_year']:
        original=base.frame_for(base.CASES[case])
        raw_parts=[];rich_parts=[]
        for airport in sorted(original.airport.unique()):
            for fold in range(4):
                raw=pd.read_parquet(base.CACHE/f'knn_{case}_{airport}_fold{fold}.parquet')
                raw_parts.append(raw[key+[c for c in raw if c.startswith('raw_')]])
                rich=pd.read_parquet(corrected.CACHE/f'{case}_{airport}_fold{fold}.parquet')
                rich_parts.append(rich.rename(columns={c:f'rich_{c}' for c in rich if c not in key}))
        rows=pd.concat(raw_parts,ignore_index=True).merge(pd.concat(rich_parts,ignore_index=True),
                on=key,validate='one_to_one',how='inner')
        assert rows.mvt_id.is_unique and set(rows.mvt_id)==set(original.mvt_id)
        for view in ['raw','rich']:
            p=pd.concat([pd.read_parquet(base.CACHE/f'{case}_fold{fold}_{view}.parquet') for fold in range(4)],ignore_index=True)
            rows=rows.merge(p[key+['propensity']].rename(columns={'propensity':f'{view}_propensity'}),on=key,validate='one_to_one')
        columns=['mvt_id','no_flight_id','takeoff_minus_sched_min','wx_snow','stand','aircraft','airline']
        rows=rows.merge(original[columns],on='mvt_id',validate='one_to_one')
        assert len(rows)==len(original)
        rows['case']=case
        rows['new_stand_full_2025']=[pd.notna(s) and (a,s) not in stand_keys for a,s in zip(rows.airport,rows.stand)]
        for airport,f in rows.groupby('airport'):
            masks={'all':np.ones(len(f),bool),'no_flight_id':f.no_flight_id.eq(1),
                   'schedule_gap_over4h':f.takeoff_minus_sched_min.gt(240),
                   'snow':f.wx_snow.eq(1),'new_stand_full_2025':f.new_stand_full_2025}
            for name,mask in masks.items():
                for label in [0,1]:
                    g=f.loc[mask & f.label.eq(label)]
                    slices.append(dict(case=case,airport=airport,slice=name,label=label,n=len(g),
                        raw_far_pct=float(100*g.raw_far5.mean()) if len(g) else None,
                        rich_far_pct=float(100*g.rich_far5.mean()) if len(g) else None,
                        raw_propensity_median=float(g.raw_propensity.median()) if len(g) else None,
                        rich_propensity_median=float(g.rich_propensity.median()) if len(g) else None))
        joined.append(rows)
    all_rows=pd.concat(joined,ignore_index=True)
    assert all_rows.mvt_id.is_unique
    assert all_rows.label.eq(1).sum()==344841 and all_rows.label.eq(0).sum()==344419
    output=base.CACHE/'similarity_rows.parquet';all_rows.to_parquet(output,index=False)
    slice_path=base.OUT/'slices_corrected.csv';pd.DataFrame(slices).to_csv(slice_path,index=False)
    primary=pd.read_csv(base.OUT/'knn.csv');primary=primary[primary.view.eq('raw')]
    rich=pd.read_csv(base.OUT/'knn_weather_scale_corrected.csv').assign(view='rich')
    counts=primary[['case','airport','k','n_source','n_target']]
    rich=rich.merge(counts,on=['case','airport','k'],validate='one_to_one')
    combined=pd.concat([primary,rich],ignore_index=True).sort_values(['case','airport','view','k'])
    combined.to_csv(base.OUT/'knn_current.csv',index=False)
    base.write_json(base.OUT/'delivery.json',dict(
        total_rows=len(all_rows),source_rows=int(all_rows.label.eq(0).sum()),ranking_rows=int(all_rows.label.eq(1).sum()),
        row_cache=str(output.relative_to(base.ROOT)),row_cache_sha256=base.sha(output),
        neighbour_columns='raw_neighbour1..10 and rich_neighbour1..10, source-2025 donor movement IDs',
        distance_columns='raw_d1/d5/d10 and corrected rich_d1/d5/d10; far flags use source-calibration q95 per airport/fold',
        propensity_columns='raw_propensity and rich_propensity: balanced diagnostic priors, calibration varies by held-out week',
        source='January and July 2025; each query excludes its entire date block from the donor bank',
        current_distance_summary='knn_current.csv; rich rows use the documented weather-scale correction',
        original_distance_summary='knn.csv is retained for provenance; its rich rows have the scale problem and are superseded',
        current_slice_summary='slices_corrected.csv; old slices.csv retains initial rich distances',
        limitation='Distances measure support under this chosen feature geometry and source-bank size; they do not prove an absence of similar rows in all 2025 training, or measure prediction cost.',
        source_cache_sha256=source_manifest['files'],script_sha256=base.sha(Path(__file__)),
        outputs={p.name:base.sha(p) for p in [base.OUT/'knn_current.csv',slice_path]}))
    print('Delivered',len(all_rows),'audited source/ranking rows with both propensity views and nearest donor IDs.')


if __name__=='__main__':
    main()
