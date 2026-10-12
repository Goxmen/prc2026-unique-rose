"""Combine two audited integer v78 deltas once on v79; no version or upload."""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import tabm_other_runway_combo_cv as cv

ROOT,RUNS=cv.ROOT,cv.RUNS
OUT=RUNS/'v3_20261012_tabm_other_runway_combo_submit'
TABM=RUNS/'v3_20261012_nn_public_tabm_submit_stack'
RUNWAY=RUNS/'v3_20261012_joint_public_other_runway_submit'
V79=cv.P79
V78=RUNS/'v3_20261011_joint_public_candidate/submission_v78'
read,art,verify,write=cv.read,cv.art,cv.verify,cv.write
DESIGN=dict(formula=cv.DESIGN['formula'],coefficients=[1,1],rows=344841,
    inputs='audited fixed TabM one-of-three replacement and robust other-runway serialized deltas versus exactv78',
    clipping='sum original integer v78 deltas onto exactv79, then one nonnegative clip and rounding',
    standalone_transports='independently verify each; do not substitute the separately clipped v79 deltas',
    selection=cv.DESIGN['selection'],new_bounds=False,no_fit=True,no_search=True,no_version=True,no_upload=True)


def prepare():
    assert not (OUT/'identity.json').exists() and not (OUT/'result.json').exists()
    cv.guard();score=read(cv.OUT/'result.json')
    assert score['identity']==art(cv.OUT/'identity.json')
    assert score['paired']['control']['decision']=='improvement'
    assert score['all_row_sql_arithmetic_truth_calendar_scores']=='pass'
    verify(score['scored'])
    sources=[Path(__file__),Path(cv.__file__),cv.OUT/'identity.json',cv.OUT/'result.json',
        cv.OUT/'scored.parquet',ROOT/'data/submitting.parquet',V78/'manifest.json',V79/'manifest.json']
    parts={}
    for name,directory in [('tabm',TABM),('runway',RUNWAY)]:
        result=read(directory/'result.json');audit=read(directory/'audit.json')
        assert audit['status']=='pass' and audit['result']==art(directory/'result.json')
        identity_path=verify(result['identity']);identity=read(identity_path)
        verify(identity['source'])
        for record in identity['sources']:verify(record)
        assert art(directory/'predictions.parquet') in result['artifacts']
        for record in result['artifacts']:verify(record)
        parts[name]=dict(identity=art(identity_path),result=art(directory/'result.json'),audit=art(directory/'audit.json'),
            predictions=art(directory/'predictions.parquet'))
        sources += [identity_path,directory/'result.json',directory/'audit.json',directory/'predictions.parquet',
            *[ROOT/a['path'] for a in identity['sources']],*[ROOT/a['path'] for a in result['artifacts']]]
    controls={}
    for version,directory in [(78,V78),(79,V79)]:
        manifest=read(directory/'manifest.json')
        assert manifest['state']=='scored' and manifest['version']==version
        record=dict(path=manifest['file'],sha256=manifest['sha256']);verify(record)
        controls[str(version)]=record;sources.append(ROOT/record['path'])
    write(OUT/'identity.json',dict(design=DESIGN,source=art(__file__),cv=art(cv.OUT/'result.json'),
        parts=parts,controls=controls,sources=[art(p) for p in sorted(set(sources))]))


def guard():
    identity=read(OUT/'identity.json')
    assert identity['design']==DESIGN and identity['source']==art(__file__)
    for record in identity['sources']:verify(record)
    return identity


def build():
    identity=guard();assert not (OUT/'result.json').exists(),'Preserve built combination'
    template=pq.read_table(ROOT/'data/submitting.parquet');ids=template['MVT_ID_mvt'].to_numpy()
    assert len(ids)==len(np.unique(ids))==344841
    controls={}
    for version,record in identity['controls'].items():
        table=pq.read_table(verify(record));assert table.schema==template.schema
        for col in template.column_names:
            if col!='TAXITIME_SEC_mvt':assert table[col].equals(template[col])
        controls[version]=table['TAXITIME_SEC_mvt'].to_numpy().astype(float)
    tabm,runway=[pd.read_parquet(verify(identity['parts'][name]['predictions'])) for name in ['tabm','runway']]
    for f in [tabm,runway]:
        assert len(f)==len(ids) and f.mvt_id.is_unique
        np.testing.assert_array_equal(f.mvt_id,ids)
        np.testing.assert_array_equal(f.v78,controls['78']);np.testing.assert_array_equal(f.v79,controls['79'])
    for f,column,transported in [(tabm,'tabm_v78','tabm_v79'),(runway,'robust_v78','robust_v79')]:
        np.testing.assert_array_equal(f.serialized_delta,f[column]-f.v78)
        np.testing.assert_array_equal(f[transported],np.rint(np.maximum(0,f.v79+f.serialized_delta)))
    pred=cv.combine(controls['79'],controls['78'],tabm.tabm_v78,runway.robust_v78)
    zero=tabm.serialized_delta.eq(0)&runway.serialized_delta.eq(0)
    np.testing.assert_array_equal(pred[zero],controls['79'][zero])
    alternative=np.rint(np.maximum(0,tabm.tabm_v79.to_numpy()+runway.robust_v79.to_numpy()-controls['79']))
    field=template.schema.field('TAXITIME_SEC_mvt')
    table=template.set_column(template.schema.get_field_index(field.name),field,pa.array(pred,type=field.type,safe=True))
    assert table.schema==template.schema
    for col in template.column_names:
        if col!=field.name:assert table[col].equals(template[col])
    np.testing.assert_array_equal(table[field.name].to_numpy(),pred)
    pq.write_table(table,OUT/'candidate.parquet');assert pq.read_table(OUT/'candidate.parquet').equals(table)
    f=pd.DataFrame(dict(mvt_id=ids,v79=controls['79'],v78=controls['78'],
        tabm_delta=tabm.serialized_delta,runway_delta=runway.serialized_delta,
        tabm_v79=tabm.tabm_v79,runway_v79=runway.robust_v79,pred=pred))
    f.to_parquet(OUT/'predictions.parquet',index=False)
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        for name,path in [('f',OUT/'predictions.parquet'),('t',TABM/'predictions.parquet'),
            ('r',RUNWAY/'predictions.parquet'),('o',OUT/'candidate.parquet')]:con.read_parquet(str(path)).create_view(name)
        n,u,bad=con.execute('''SELECT count(*),count(distinct f.mvt_id),sum(CASE WHEN
            f.v79 IS DISTINCT FROM t.v79 OR f.v79 IS DISTINCT FROM r.v79 OR f.v78 IS DISTINCT FROM t.v78 OR
            f.tabm_delta IS DISTINCT FROM t.tabm_v78-t.v78 OR f.runway_delta IS DISTINCT FROM r.robust_v78-r.v78 OR
            f.pred IS DISTINCT FROM round_even(greatest(0,f.v79+f.tabm_delta+f.runway_delta),0) OR
            f.pred IS DISTINCT FROM o.TAXITIME_SEC_mvt OR
            (f.tabm_delta=0 AND f.runway_delta=0 AND f.pred IS DISTINCT FROM f.v79)
            THEN 1 ELSE 0 END) FROM f JOIN t USING(mvt_id) JOIN r USING(mvt_id) JOIN o ON f.mvt_id=o.MVT_ID_mvt''').fetchone()
        assert n==u==344841 and bad==0
    score=read(cv.OUT/'result.json')
    result=dict(design=DESIGN,identity=art(OUT/'identity.json'),rows=n,
        cv=score['cv'],paired=score['paired'],cv_source=art(cv.OUT/'result.json'),
        changed=int(np.sum(pred!=controls['79'])),standalone_transports_exact=True,
        separately_clipped_deltas_would_differ_rows=int(np.sum(pred!=alternative)),
        clipping_note='Diagnostic only; no alternative candidate produced or selected.',
        artifacts=[art(OUT/p) for p in ['candidate.parquet','predictions.parquet']],no_version=True,no_upload=True)
    write(OUT/'result.json',result)
    write(OUT/'audit.json',dict(status='pass',result=art(OUT/'result.json'),identity=art(OUT/'identity.json'),
        rows=n,all_row_fixed_formula_ids_control_transport_and_serialization=True,
        scope='Independent SQL arithmetic within builder plus prior separate component audits; no claim of independent whole-pipeline reimplementation.'))
    print(dict(rows=n,changed=result['changed'],cv=score['cv']['combined'],
        separately_clipped_difference=result['separately_clipped_deltas_would_differ_rows']),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','build'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
