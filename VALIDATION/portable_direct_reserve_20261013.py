"""Serialize the retained latest direct mixture as a distinct portable baseline.

R(tabm_direct) only: nearest-even nonnegative integers, no source correction,
new model inference, fitting or upload. Freeze, build and independent SQL audit
each need the one declared design; build additionally needs root allocation.
The optional comparison reads audited saved arrays once and changes nothing.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / 'VALIDATION/runs'
OUT = RUNS / 'v3_20261013_portable_direct_reserve'
DIRECT = RUNS / 'v3_20261012_final_public_direct_components'
CV = RUNS / 'v3_20261013_portable_direct_cv'
CHAIN = RUNS / 'v3_20261013_portable_public_source'
AUDITOR = ROOT / 'VALIDATION/tools/audit_portable_direct_reserve_20261013.py'
COUNTS = dict(cv=344419, old=344841, final=670790)
RAW_SHA = dict(old='9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
               final='4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f')


def read(path):
    return json.loads(Path(path).read_text())


def art(path):
    path = Path(path); digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return dict(path=str(path.relative_to(ROOT)), sha256=digest.hexdigest())


def verify(record):
    path = ROOT / record['path']; assert art(path) == record, path
    return path


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def runtime():
    return dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, pyarrow=pa.__version__)


def freeze():
    assert not (OUT / 'identity.json').exists()
    sources = {Path(__file__), AUDITOR}; specs = {}
    cv_result, cv_audit = read(CV / 'result.json'), read(CV / 'audit.json')
    assert cv_audit['status'] == 'pass' and cv_audit['result'] == art(CV / 'result.json')
    assert cv_result['identity'] == cv_audit['identity'] == art(CV / 'identity.json')
    production = read(DIRECT / 'independent_audit.json')
    assert production['status'] == 'pass' and production['identity'] == art(DIRECT / 'identity.json')
    sources.update([CV / name for name in ['identity.json', 'result.json', 'audit.json']])
    sources.update([DIRECT / name for name in ['identity.json', 'independent_audit.json']])
    for view in COUNTS:
        if view == 'cv':
            record = cv_result
            raw = []
            for month in ['2025-01', '2025-07']:
                found = list((ROOT / 'data').glob(f'training_{month}-01_*.parquet'))
                assert len(found) == 1; raw.extend(found)
            spec = dict(prediction=record['prediction'], raw=[art(p) for p in raw])
        else:
            record = read(DIRECT / f'{view}.json')
            assert record['identity'] == production['identity'] and record['rows'] == COUNTS[view]
            assert record['prediction'] == production['views'][view]['prediction']
            path = ROOT / 'data' / ('ranking.parquet' if view == 'old' else 'final_ranking.parquet')
            assert art(path)['sha256'] == RAW_SHA[view]
            raw = [path]
            template = ROOT / 'data' / ('submitting.parquet' if view == 'old' else 'final_submitting.parquet')
            assert record['template'] == art(template)
            spec = dict(prediction=record['prediction'], raw=[art(path)], template=art(template))
            sources.update([DIRECT / f'{view}.json', template])
        assert record['rows'] == COUNTS[view]
        sources.update(raw); sources.add(verify(spec['prediction'])); specs[view] = spec
    OUT.mkdir(parents=True, exist_ok=True)
    identity = dict(name='portable_direct_reserve_v1', transformation='R(tabm_direct), nonnegative nearest-even integer',
                    sources=[art(path) for path in sorted(sources)], specs=specs, runtime=runtime(),
                    source=art(__file__), auditor=art(AUDITOR), no_fit=True, no_model_inference=True,
                    no_source_correction=True, no_upload=True, selected=False)
    write(OUT / 'identity.json', identity)
    print(json.dumps(dict(status='frozen', identity=art(OUT / 'identity.json'))))


def guard():
    identity = read(OUT / 'identity.json')
    assert identity['source'] == art(__file__) and identity['runtime'] == runtime()
    for record in identity['sources']:
        verify(record)
    return identity


def build():
    identity = guard(); allocation = read(OUT / 'allocation.json')
    assert allocation['authority'] == '/root' and allocation['identity'] == art(OUT / 'identity.json')
    assert allocation['state'] == 'released_for_direct_reserve_serialization'
    assert allocation['views'] == list(COUNTS) and not (OUT / 'result.json').exists()
    outputs = {}; months = {}
    for view, spec in identity['specs'].items():
        assert not (OUT / view).exists(); folder = OUT / view; folder.mkdir()
        packet = pd.read_parquet(verify(spec['prediction']))
        ids = packet.mvt_id.to_numpy(); assert len(ids) == len(set(ids)) == COUNTS[view]
        raw = pd.concat([pd.read_parquet(verify(a), columns=['MVT_ID_mvt', 'ADEP_mvt', 'MVT_TIME_UTC_mvt']) for a in spec['raw']])
        assert raw.MVT_ID_mvt.is_unique
        raw = raw.set_index('MVT_ID_mvt').loc[ids]
        values = packet.tabm_direct.to_numpy(float); assert np.isfinite(values).all()
        submitted = np.rint(np.maximum(values, 0)).astype(np.int64)
        frame = pd.DataFrame(dict(mvt_id=ids, month=raw.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m').to_numpy(),
            day=raw.MVT_TIME_UTC_mvt.dt.strftime('%Y-%m-%d').to_numpy(), airport=raw.ADEP_mvt.to_numpy(),
            raw_direct=values, submitted=submitted))
        if view == 'cv':
            for key in ['month', 'day']:
                np.testing.assert_array_equal(frame[key], packet[key])
            frame['y'] = packet.y.to_numpy()
            months = {m:float(np.sqrt(np.mean((f.submitted-f.y)**2))) for m,f in frame.groupby('month')}
        else:
            template = pq.read_table(verify(spec['template']))
            np.testing.assert_array_equal(template['MVT_ID_mvt'].to_numpy(), ids)
            column = template.schema.get_field_index('TAXITIME_SEC_mvt'); field = template.schema.field(column)
            candidate = template.set_column(column, field, pa.array(submitted, type=field.type))
            assert candidate.schema == template.schema
            for name in template.column_names:
                if name != 'TAXITIME_SEC_mvt': assert candidate[name].equals(template[name])
            pq.write_table(candidate, folder / 'candidate.parquet')
            assert pq.read_table(folder / 'candidate.parquet').equals(candidate)
        frame.to_parquet(folder / 'predictions.parquet', index=False)
        pd.testing.assert_frame_equal(frame, pd.read_parquet(folder / 'predictions.parquet'), check_exact=True)
        outputs[view] = dict(rows=len(frame), prediction=art(folder / 'predictions.parquet'))
        if view != 'cv': outputs[view]['candidate'] = art(folder / 'candidate.parquet')
    write(OUT / 'result.json', dict(identity=art(OUT / 'identity.json'), allocation=art(OUT / 'allocation.json'),
        views=outputs, months=months, cv=float(np.mean(list(months.values()))), no_fit=True, no_model_inference=True,
        no_upload=True, selected=False, independent_audit_pending=True))
    print(json.dumps(dict(status='built', result=art(OUT / 'result.json'), cv=np.mean(list(months.values())))))


def compare():
    guard(); result=read(OUT/'result.json'); audit=read(OUT/'audit.json')
    assert audit['status']=='pass' and audit['result']==art(OUT/'result.json')
    assert not (OUT/'chain_comparison.json').exists(), 'One saved-array comparison, no tuning'
    rows=[]; inputs=[art(OUT/'result.json'),art(OUT/'audit.json')]; missing=[]
    for view in COUNTS:
        path=CHAIN/view/'audit.json'
        if not path.exists():missing.append(view);continue
        checked=read(path); parent=read(CHAIN/view/'result.json')
        assert checked['status']=='pass' and checked['result']==art(CHAIN/view/'result.json')
        assert checked['identity']==parent['identity']
        direct=pd.read_parquet(verify(result['views'][view]['prediction']))
        chain=pd.read_parquet(verify(parent['prediction']))
        assert len(chain)==len(direct) and chain.mvt_id.is_unique
        chain=chain.set_index('mvt_id').loc[direct.mvt_id].reset_index()
        for key in ['month','day','airport']:np.testing.assert_array_equal(chain[key],direct[key])
        if view=='cv':np.testing.assert_array_equal(chain.y,direct.y)
        np.testing.assert_array_equal(chain.direct_reserve,direct.submitted)
        groups=[('all','all',np.arange(len(direct)))]
        for month,ix in direct.groupby('month',sort=True).indices.items():groups.append((month,'all',ix))
        for airport,ix in direct.groupby('airport',sort=True).indices.items():groups.append(('all',airport,ix))
        for keys,ix in direct.groupby(['month','airport'],sort=True).indices.items():groups.append((*keys,ix))
        for month,airport,at in groups:
            a=direct.iloc[at];b=chain.iloc[at];delta=b.submitted.to_numpy(float)-a.submitted.to_numpy(float)
            row=dict(view=view,month=month,airport=airport,rows=len(at),changed=int(np.count_nonzero(delta)),
                     rms=float(np.sqrt(np.mean(delta**2))),mean_delta=float(delta.mean()),max_abs=float(abs(delta).max()))
            if view=='cv':
                row.update(direct_rmse=float(np.sqrt(np.mean((a.submitted-a.y)**2))),
                           chain_rmse=float(np.sqrt(np.mean((b.submitted-b.y)**2))))
                row['rmse_change']=row['chain_rmse']-row['direct_rmse']
            rows.append(row)
        inputs.extend([art(path),art(CHAIN/view/'result.json'),parent['prediction'],result['views'][view]['prediction']])
    assert 'cv' not in missing
    score=read(CHAIN/'score.json');assert score['result']==art(CHAIN/'cv/result.json')
    assert read(CHAIN/'cv/audit.json')['metrics']['score']==art(CHAIN/'score.json')
    inputs.append(art(CHAIN/'score.json'))
    value=dict(source=art(__file__),inputs=inputs,rows=rows,chain_views_unavailable=missing,
               equal_month_cv=dict(direct=result['cv'],chain=score['cv']['submitted'],
               paired=score['paired']['direct_reserve']),no_tuning=True,no_upload=True)
    write(OUT/'chain_comparison.json',value)
    print(json.dumps(dict(status='compared',views=[v for v in COUNTS if v not in missing],result=art(OUT/'chain_comparison.json'))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','build','compare'])
    args=parser.parse_args();globals()[args.stage]()
