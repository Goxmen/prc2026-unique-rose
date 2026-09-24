"""Independent arithmetic and row-identity checks for the completed step-2 run."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'tmp/train_ranking_step1'
CACHE = ROOT/'tmp/train_ranking_step2'
OUT = ROOT/'VALIDATION/reports/train_ranking_step2'


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_distances(reference, query, design, corrected_weather=False):
    """Direct column-by-column calculation; no encoding or nearest helper reuse."""
    groups = {}
    for group, columns in design['feature_groups'].items():
        categorical = [c for c in columns if c in design['categoricals']]
        numeric = [c for c in columns if c not in categorical]
        components = []
        if categorical:
            terms=[]
            for c in categorical:
                source=reference[c].astype('string').fillna('<NULL>')
                value='<NULL>' if pd.isna(query[c]) else str(query[c])
                terms.append(source.ne(value).to_numpy(float))
            components.append(np.mean(terms,axis=0))
        if numeric:
            terms=[]
            for c in numeric:
                a=reference[c].to_numpy(dtype=float,na_value=np.nan,copy=True)
                b=float(query[c]) if pd.notna(query[c]) else np.nan
                if c.startswith('takeoff_minus_'):
                    a=np.sign(a)*np.log1p(np.abs(a)); b=np.sign(b)*np.log1p(abs(b))
                valid=np.isfinite(a); present=np.isfinite(b)
                if valid.any():
                    lo,q25,median,q75,hi=np.quantile(a[valid],[.1,.25,.5,.75,.9])
                    scale=q75-q25 if q75-q25>1e-8 else hi-lo
                    if corrected_weather and group=='weather' and scale<=1e-8:
                        scale=float(a[valid].max()-a[valid].min())
                    constant=scale<=1e-8
                    scale=scale if not constant else 1.
                else:
                    median,scale,constant=0.,1.,True
                az=(np.where(valid,a,median)-median)/scale
                bz=(b-median)/scale if present else 0.
                if corrected_weather and group=='weather':
                    if not valid.any():
                        az=np.zeros(len(a));bz=0.
                    elif constant:
                        az=np.zeros(len(a));bz=float(present and not np.isclose(b,median,rtol=1e-10,atol=1e-10))
                terms.append(((az-bz)**2+(valid != present).astype(float))/2)
            components.append(np.mean(terms,axis=0))
        groups[group]=np.mean(components,axis=0)
    return {view:np.sqrt(np.mean([groups[g] for g in names],axis=0))
            for view,names in design['views'].items()}


def main():
    design=json.loads((OUT/'design.json').read_text())
    signature=hashlib.sha256(json.dumps(design,sort_keys=True).encode()).hexdigest()
    months={m:pd.read_parquet(SOURCE/f'{m}.parquet') for pair in design['cases'].values() for m in pair}
    pair_rows={case:pd.concat([months[m][['mvt_id','date','airport']].assign(label=i)
                              for i,m in enumerate(pair)]).set_index('mvt_id')
               for case,pair in design['cases'].items()}
    neighbour_rows={case:pd.concat([months[m] for m in design['cases'][case]]).set_index('mvt_id')
                    for case in ['january_year','july_year']}
    checked_models=0;checked_knn=0;checked_queries=0
    for checkpoint in sorted(CACHE.glob('*fold*.json')):
        record=json.loads(checkpoint.read_text())
        assert record['design_sha256']==signature
        for name,digest in record['files'].items():
            assert hash_file(CACHE/name)==digest,name
        if checkpoint.name.startswith('knn_'):
            case=record['case'];month=design['cases'][case][0]
            source=months[month];reference=source[source.airport.eq(record['airport']) & source.date.isin(pd.to_datetime(record['reference_dates']))].sort_values('mvt_id').reset_index(drop=True)
            assert len(reference)==record['n_reference']
            output=pd.read_parquet(CACHE/next(iter(record['files'])))
            all_raw=neighbour_rows[case]
            assert output.mvt_id.is_unique and not output.mvt_id.isin(reference.mvt_id).any()
            for view in design['views']:
                neighbour_columns=[f'{view}_neighbour{k}' for k in range(1,11)]
                ids=output[neighbour_columns].to_numpy().ravel()
                assert pd.Index(ids).isin(reference.mvt_id).all()
                assert (np.diff(output[[f'{view}_d{k}' for k in [1,5,10]]].to_numpy(),axis=1)>=-1e-12).all()
                for k in [1,5,10]:
                    np.testing.assert_array_equal(output[f'{view}_far{k}'],output[f'{view}_d{k}']>record['thresholds'][f'{view}_k{k}'])
            # One source and one target query per bank, preferring a missing-ID
            # case to exercise the missing-value arithmetic where available.
            for label in [0,1]:
                candidates=output[output.label.eq(label)].sort_values('mvt_id')
                missing=all_raw.loc[candidates.mvt_id,'no_flight_id'].eq(1).to_numpy()
                row=candidates.iloc[np.flatnonzero(missing)[0] if missing.any() else 0]
                raw=all_raw.loc[row.mvt_id]
                distances=independent_distances(reference,raw,design)
                for view,d in distances.items():
                    expected=np.sort(d,kind='stable')
                    for k in [1,5,10]:
                        np.testing.assert_allclose(row[f'{view}_d{k}'],expected[k-1],rtol=1e-10,atol=1e-10)
                    ids=[row[f'{view}_neighbour{k}'] for k in range(1,11)]
                    positions=pd.Index(reference.mvt_id).get_indexer(ids)
                    assert (d[positions]<=expected[9]+1e-10).all()
                checked_queries+=1
            checked_knn+=1
        else:
            output=pd.read_parquet(CACHE/f'{checkpoint.stem}.parquet')
            assert output.mvt_id.is_unique
            if record['case']!='january_null':
                expected=pair_rows[record['case']].loc[output.mvt_id]
                for column in ['date','airport','label']:
                    np.testing.assert_array_equal(output[column].to_numpy(),expected[column].to_numpy())
            roles={k:pd.DatetimeIndex(v) for k,v in record['date_roles'].items()}
            for a,b in [('fit','cal'),('fit','test'),('cal','test')]:
                assert all(abs(x-y)>pd.Timedelta(days=1) for x in roles[a] for y in roles[b]),checkpoint.name
            fit=record['calibration']
            np.testing.assert_allclose(output.propensity,expit(fit['slope']*output.logit+fit['intercept']))
            for score in record['scores']:
                f=output[output.airport.eq(score['airport'])]
                assert abs(roc_auc_score(f.label,f.logit)-score['raw_auc'])<1e-12
            checked_models+=1
    assert checked_models==33 and checked_knn==80 and checked_queries==160
    correction=ROOT/'tmp/train_ranking_step2_weather_scale'
    corrected_banks=0;corrected_queries=0
    for checkpoint in sorted(correction.glob('*fold*.json')):
        d=json.loads(checkpoint.read_text())
        assert hash_file(correction/d['file'])==d['sha256']
        old=json.loads((CACHE/f'knn_{checkpoint.name}').read_text())
        source=months[design['cases'][d['case']][0]]
        reference=source[source.airport.eq(d['airport']) & source.date.isin(pd.to_datetime(old['reference_dates']))].sort_values('mvt_id').reset_index(drop=True)
        assert len(reference)==d['n_reference']==old['n_reference']
        output=pd.read_parquet(correction/d['file'])
        previous=pd.read_parquet(CACHE/next(iter(old['files'])),columns=['mvt_id'])
        assert set(output.mvt_id)==set(previous.mvt_id) and output.mvt_id.is_unique
        ids=output[[f'neighbour{k}' for k in range(1,11)]].to_numpy().ravel()
        assert pd.Index(ids).isin(reference.mvt_id).all()
        for k in [1,5,10]:
            np.testing.assert_array_equal(output[f'far{k}'],output[f'd{k}']>d['thresholds'][str(k)])
        for label in [0,1]:
            row=output[output.label.eq(label)].sort_values('mvt_id').iloc[0]
            query=neighbour_rows[d['case']].loc[row.mvt_id]
            distance=independent_distances(reference,query,design,corrected_weather=True)['rich']
            expected=np.sort(distance,kind='stable')
            for k in [1,5,10]:
                np.testing.assert_allclose(row[f'd{k}'],expected[k-1],rtol=1e-10,atol=1e-10)
            positions=pd.Index(reference.mvt_id).get_indexer([row[f'neighbour{k}'] for k in range(1,11)])
            assert (distance[positions]<=expected[9]+1e-10).all()
            corrected_queries+=1
        corrected_banks+=1
    assert corrected_banks==80 and corrected_queries==160
    result=dict(status='passed',classifier_checkpoints=checked_models,knn_banks=checked_knn,
                independently_recomputed_queries=checked_queries,query_views=2,
                corrected_weather_banks=corrected_banks,corrected_weather_queries_recomputed=corrected_queries,
                date_embargo='verified for all classifier fit/cal/test date sets',
                neighbours='all saved IDs belong to the correct source-airport donor bank; all distance ordering/threshold flags verified',
                distance_arithmetic='first/fifth/tenth neighbour distances independently recomputed over all donor rows for 160 queries',
                design_sha256=signature,script_sha256=hash_file(Path(__file__)))
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
