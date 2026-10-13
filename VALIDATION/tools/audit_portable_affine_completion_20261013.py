"""Additive exact-ID alignment audit around the immutable numerical auditor.

Only final component loading changes: require unique, equal ID sets and align
to the unchanged template. Old/CV arithmetic and the one saved solve are reused.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_portable_affine'
COMPLETION=OUT/'completion_final_order_20261013'
BASE_SOURCE=ROOT/'VALIDATION/tools/audit_portable_affine_20261013.py'
spec=importlib.util.spec_from_file_location('immutable_portable_affine_auditor',BASE_SOURCE)
base=importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def main():
    assert not (OUT/'independent_audit.json').exists()
    assert not (OUT/'base_numerical_audit.json').exists()
    original=base.read(OUT/'identity.json')
    frozen=base.read(COMPLETION/'identity.json')
    finished=base.read(COMPLETION/'result.json')
    result=base.read(OUT/'result.json')
    assert frozen['original_identity']==finished['original_identity']==base.art(OUT/'identity.json')
    assert finished['identity']==base.art(COMPLETION/'identity.json')
    assert result['identity']==base.art(OUT/'identity.json')
    assert result['completion_identity']==base.art(COMPLETION/'identity.json')
    assert result['completion_result']==base.art(COMPLETION/'result.json')
    assert base.art(Path(__file__)) in frozen['sources']
    assert base.art(BASE_SOURCE) in frozen['sources']
    for item in frozen['sources']:
        base.verify(item)
    for item in original['sources']:
        base.verify(item)
    for key,path in [('geometry',OUT/'geometry.json'),('old',OUT/'old.parquet'),('cv',OUT/'cv.parquet')]:
        actual=base.art(path)
        assert finished[key]==actual and actual in frozen['sources']
    assert result['geometry']==finished['geometry']
    assert result['predictions']['old']==finished['old'] and result['predictions']['cv']==finished['cv']
    assert result['predictions']['final']==finished['final']
    assert result['final_candidate']==finished['final_candidate']
    base.verify(finished['final']);base.verify(finished['final_candidate'])
    ids=pq.read_table(ROOT/'data/final_submitting.parquet')['MVT_ID_mvt'].to_numpy()
    assert len(ids)==len(np.unique(ids))==670790
    assert len(finished['alignments'])==len(original['versions'])
    supplied={int(a['version']):a for a in finished['alignments']}
    assert set(supplied)==set(original['versions'])
    alignments=[]
    for version in original['versions']:
        record=original['components'][str(version)]
        columns=[record['id_column'],record['value_column']]
        frame=pd.read_parquet(base.verify(record['final']),columns=columns)
        movement=frame[columns[0]].to_numpy()
        assert len(movement)==len(ids) and pd.Index(movement).is_unique and not pd.isna(movement).any()
        np.testing.assert_array_equal(np.sort(movement),np.sort(ids))
        item=dict(version=version,input=record['final'],id_column=columns[0],value_column=columns[1],
                  rows=len(ids),reordered_rows=int(np.count_nonzero(movement!=ids)),unique_ids=True,exact_id_set=True)
        assert supplied[version]==item,(version,supplied[version],item)
        alignments.append(item)
    original_loader=base.component

    def aligned_final(record,view,ids):
        if view!='final':
            return original_loader(record,view,ids)
        frame=pd.read_parquet(base.verify(record[view]),columns=[record['id_column'],record['value_column']])
        index=pd.Index(frame[record['id_column']])
        assert index.is_unique and not index.hasnans and len(index)==len(ids)
        np.testing.assert_array_equal(np.sort(index.to_numpy()),np.sort(ids))
        aligned=frame.set_index(record['id_column']).loc[ids]
        np.testing.assert_array_equal(aligned.index,ids)
        values=aligned[record['value_column']].to_numpy(float)
        assert np.isfinite(values).all() and (values>=0).all()
        np.testing.assert_array_equal(values,np.rint(values))
        return values

    # Explicit adapter changes alignment only. Every original SQL/geometry/CV/
    # serialization/template check executes from the unchanged pinned source.
    base.component=aligned_final
    base.main()
    numerical=OUT/'base_numerical_audit.json'
    (OUT/'independent_audit.json').replace(numerical)
    report=base.read(numerical)
    assert report['status']=='pass' and report['result']==base.art(OUT/'result.json')
    report.update(source=base.art(Path(__file__)),original_numerical_auditor=base.art(BASE_SOURCE),
        base_audit=base.art(numerical),completion_identity=base.art(COMPLETION/'identity.json'),
        completion_result=base.art(COMPLETION/'result.json'),alignments=alignments,
        final_alignment='Exact unique-ID set reindexing to original template; no values or numerical recipe changed.',
        solve_repeated=False,old_or_cv_outputs_recomputed=False)
    (OUT/'independent_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(audit=base.art(OUT/'independent_audit.json'),components=len(alignments),
        reordered_components=sum(a['reordered_rows']>0 for a in alignments))))


if __name__=='__main__':main()
