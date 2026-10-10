"""Build only the predeclared half sparse3 / half callsign15 combination.

The callsign main tree, other two trees, half CatBoost/R20 probability and all
corrections remain the independently audited combined candidate. Require its
exact raw/serialized replay at sparse weight zero before producing either view.
No fitting, weight selection, availability gate or upload entry point exists.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

import callsign_lgb_submit as combined
import callsign_submit_fast as calls_worker
import nn_sparse_track_submit as sparse
import sparse_track_combo_score as cv

parent=combined.parent
ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261010_sparse_callsign_submit'
WEIGHT=.5
DESIGN=dict(parent='audited callsign15/main-callsign-LGB/half-Cat3-R20 combined candidate',
    neural='0.5 sparse3 + 0.5 callsign15 on every row',sparse_weight=WEIGHT,
    network_share=.71,other_components='unchanged parent tree/probability/corrections',
    views=['old','final'],availability_gate=False,fit=False,upload=False,
    control='sparse weight0 exactly reproduces all parent parts and serialized columns')


def verified(record):
    return parent.features.verify_artifact(record)


def ordered(path,ids,columns=None):
    frame=pd.read_parquet(path,columns=columns)
    if not frame.mvt_id.is_unique:
        raise ValueError('Duplicate prediction/query IDs: '+str(path))
    np.testing.assert_array_equal(frame.mvt_id.to_numpy(),np.asarray(ids))
    return frame


def mix_neural(ids,calls,sparse_frame,weight=WEIGHT):
    """Only weight0 control and the frozen half mixture are accepted."""
    if weight not in (0.,WEIGHT): raise ValueError('No weight search is permitted')
    np.testing.assert_array_equal(calls.mvt_id,ids)
    np.testing.assert_array_equal(sparse_frame.mvt_id,ids)
    if pd.Index(ids).has_duplicates: raise ValueError('Duplicate IDs within ranking view')
    a,b=calls.base.to_numpy(float),sparse_frame.base.to_numpy(float)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or (a<0).any() or (b<0).any():
        raise ValueError('Invalid ordinary neural predictions')
    return a.copy() if weight==0 else .5*(a+b)


def serialized(template,ids,values):
    """Original fill operation plus independent ID-based rounded-value audit."""
    table=parent.hs.fill_template(template,ids,values)
    if table.schema!=template.schema or table.column_names!=template.column_names:
        raise ValueError('Serialized schema changed')
    for name in template.column_names:
        if name!='TAXITIME_SEC_mvt' and not table[name].equals(template[name]):
            raise ValueError('Protected template column changed: '+name)
    expected=np.rint(pd.Series(np.asarray(values),index=np.asarray(ids)).loc[
        template['MVT_ID_mvt'].to_numpy()].to_numpy())
    np.testing.assert_array_equal(table['TAXITIME_SEC_mvt'].to_numpy(),expected)
    if not np.isfinite(expected).all() or (expected<0).any():
        raise ValueError('Invalid serialized predictions')
    return table


def check_zero(replayed,reference,table,expected):
    pd.testing.assert_frame_equal(replayed,reference,check_dtype=False,
                                  check_categorical=False,check_exact=True)
    if table.schema!=expected.schema or table.column_names!=expected.column_names:
        raise ValueError('Zero-weight parent schema differs')
    for name in table.column_names:
        if not table[name].equals(expected[name]):
            raise ValueError('Zero-weight parent column differs: '+name)


def guard():
    # Full source verification is deliberately confined to an allocated build slot.
    _,_,identity=sparse.guard()
    calls_worker.guard()
    combined.guard()
    evidence=parent.read(cv.OUT/'result.json')
    if evidence['design']!=cv.DESIGN or evidence['source_sha256']!=parent.sha256(Path(cv.__file__)):
        raise ValueError('Combined CV design/source changed')
    for name,digest in evidence['inputs'].items():
        if parent.sha256(ROOT/name)!=digest: raise ValueError('Combined CV input changed: '+name)
    for control in ['callsign_combination','half_direct3_combination','matched_v58']:
        key='half_sparse3_combination_vs_'+control
        if evidence['comparisons'][key]['decision']!='improvement':
            raise ValueError('Fixed sparse combination lacks passing evidence: '+key)
    audit_path=combined.OUT/'candidate_audit.json'
    audit=parent.read(audit_path)
    if not audit['passed']: raise ValueError('Parent combined candidate has not passed independent audit')
    verified(audit['source'])
    current=dict(design=DESIGN,source=parent.artifact(Path(__file__)),
        helpers={str(Path(m.__file__).relative_to(ROOT)):parent.sha256(Path(m.__file__))
                 for m in [combined,parent,calls_worker,sparse,cv]},
        sparse_identity=parent.artifact(sparse.IDENTITY),chosen_recipe=parent.artifact(sparse.DECISION),
        combined_cv=parent.artifact(cv.OUT/'result.json'),
        combined_cv_predictions=parent.artifact(cv.OUT/'scored_errors.parquet'),
        parent_candidate_audit=parent.artifact(audit_path))
    OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'design.json'
    if path.exists():
        if parent.read(path)!=current: raise ValueError('Frozen candidate design/evidence changed')
    elif list(OUT.glob('**/*.parquet')):
        raise ValueError('Unrecorded candidate artifacts require inspection')
    else: parent.write_json(path,current)
    return identity,audit


def sparse_mean(view,ids,identity):
    pool_path=sparse.OUT/'pooled.json'
    record=parent.read(pool_path)
    if (record['identity']!=parent.artifact(sparse.IDENTITY) or record['members']!=sparse.SEEDS
            or record['rule']!='equal mean of three float32 saved member predictions'):
        raise ValueError('Sparse pool recipe changed')
    members=[sparse.completed_member(k,identity) for k in range(3)]
    if any(row is None for row in members): raise ValueError('Three completed sparse members required')
    if record['models']!=[parent.artifact(sparse.member_paths(k)[1]) for k in range(3)]:
        raise ValueError('Sparse pool member list changed')
    frames=[ordered(verified(row['predictions'][view]),ids) for row in members]
    pooled=ordered(verified(record['outputs'][view]),ids)
    np.testing.assert_array_equal(pooled.base,np.mean([frame.base.to_numpy(float) for frame in frames],axis=0))
    for frame in frames: np.testing.assert_array_equal(frame.sequence_count,pooled.sequence_count)
    return pooled,dict(record=parent.artifact(pool_path),prediction=record['outputs'][view])


def build(view):
    directory=OUT/'candidates'/view
    if directory.exists(): raise FileExistsError('Preserve already built sparse candidate')
    identity,audit=guard()
    base_dir=combined.OUT/'candidates'/view
    manifest_path=base_dir/'manifest.json'
    manifest=parent.read(manifest_path)
    checked=audit['views'][view]
    if (manifest['view']!=view or manifest['design']!=combined.DESIGN
            or manifest['source_sha256']!=parent.sha256(Path(combined.__file__))
            or manifest['design_sha256']!=parent.sha256(combined.OUT/'design.json')
            or checked['manifest']!=parent.artifact(manifest_path)
            or checked['output']!=manifest['output']):
        raise ValueError('Audited combined parent/view changed')
    reference=pd.read_parquet(verified(checked['parts']))
    data=parent.ranking_data(view);ids=data['rank_ids']
    np.testing.assert_array_equal(reference.mvt_id,ids)
    control=parent.parent_control(view,data)
    _,original_tree,original_p,parent_sources=parent.parent_parts(view,data)
    if parent_sources!=manifest['parent_parts']: raise ValueError('Parent component sources changed')
    components,tree_sources=combined.parent_components(view,ids)
    if tree_sources!=manifest['parent_tree_components']: raise ValueError('Parent tree sources changed')
    model=combined.completed_model()
    if model is None or manifest['model_record']!=parent.artifact(combined.OUT/'model.json'):
        raise ValueError('Parent main tree model changed')
    main_record=parent.read(verified(manifest['main_prediction']))
    if main_record['model_sha256']!=model['model']['sha256'] or main_record['view']!=view:
        raise ValueError('Main tree prediction namespace changed')
    main=ordered(verified(main_record['prediction']),ids)
    reproduced,tree=combined.replace_main(components,ids,main)
    np.testing.assert_array_equal(reproduced,original_tree)
    values,network_records=[],[]
    for group in parent.GROUPS:
        for k in parent.NETS:
            member=calls_worker.completed_member(group,k)
            if member is None: raise ValueError('All15 callsign members required')
            values.append(ordered(verified(member['predictions'][view]),ids).base.to_numpy(float))
            network_records.append(parent.artifact(parent.member_paths(group,k)[1]))
    if network_records!=manifest['network_records']: raise ValueError('Calls15 parent members changed')
    calls=pd.DataFrame(dict(mvt_id=ids,base=np.mean(values,axis=0)))
    sparse_frame,sparse_record=sparse_mean(view,ids,identity)
    raw=pd.concat([data['months'][m]['raw'] for m in parent.features.RANK_MONTHS[view]],ignore_index=True)
    query_record=sparse.parent.read(sparse.ranking_cache.OUT/'manifest.json')['views'][view]['query']
    query=pd.read_parquet(verified(query_record)).set_index('mvt_id').loc[ids]
    np.testing.assert_array_equal(query.airport,raw.ADEP_mvt)
    milliseconds=lambda s:pd.to_datetime(s,utc=True).astype('datetime64[ms, UTC]').astype('int64').where(pd.notna(s))
    np.testing.assert_array_equal(query.to_ms,milliseconds(raw.MVT_TIME_UTC_mvt))
    np.testing.assert_array_equal(query.clock_ms.to_numpy(float,na_value=np.nan),
                                  milliseconds(raw.AOBT_3_flt).to_numpy(float,na_value=np.nan))
    index_path=sparse.ranking_cache.OUT/view/'full_index.parquet'
    counts=pd.read_parquet(index_path,columns=['mvt_id','count']).set_index('mvt_id').loc[ids,'count']
    np.testing.assert_array_equal(counts,sparse_frame.sequence_count)
    cat_record=manifest['rome']['record']
    probability,cat_provenance=parent.catboost_probability(verified(cat_record),view,ids,raw.ADEP_mvt)
    if cat_provenance!=manifest['rome']: raise ValueError('Fixed Rome part provenance changed')
    probability=.5*original_p+.5*probability
    zero,zero_masks=parent.final_frame(data,mix_neural(ids,calls,sparse_frame,0),tree,probability)
    zero['main_lgb_original']=components.lgb.to_numpy(float)
    zero['main_lgb_callsign']=main.pred.to_numpy(float)
    template=pq.read_table(verified(manifest['template']))
    check_zero(zero,reference,serialized(template,ids,zero.pred.to_numpy()),pq.read_table(verified(manifest['output'])))
    if zero_masks!=manifest['correction_masks']: raise ValueError('Zero-weight correction masks changed')
    frame,masks=parent.final_frame(data,mix_neural(ids,calls,sparse_frame),tree,probability)
    frame['calls_nn']=calls.base.to_numpy()
    frame['sparse_nn']=sparse_frame.base.to_numpy()
    frame['sequence_count']=sparse_frame.sequence_count.to_numpy()
    frame['main_lgb_original']=components.lgb.to_numpy(float)
    frame['main_lgb_callsign']=main.pred.to_numpy(float)
    table=serialized(template,ids,frame.pred.to_numpy())
    directory.mkdir(parents=True)
    output=directory/'candidate.parquet';pq.write_table(table,output)
    reread=pq.read_table(output)
    if not reread.equals(table,check_metadata=True): raise ValueError('Serialized round trip changed')
    parent.save_frame(directory/'parts.parquet',frame)
    parent.write_json(directory/'manifest.json',dict(state='prepared_not_uploaded',view=view,design=DESIGN,
        design_record=parent.artifact(OUT/'design.json'),source=parent.artifact(Path(__file__)),
        output=parent.artifact(output),parts=parent.artifact(directory/'parts.parquet'),
        template=manifest['template'],parent_candidate=parent.artifact(manifest_path),
        parent_audit=parent.artifact(combined.OUT/'candidate_audit.json'),
        exact_original_parent=control,exact_weight_zero_raw_and_serialized=True,
        sparse_pool=sparse_record,view_query=query_record,rows=len(frame),
        correction_masks=masks,rule_triggered=int(frame.rule_triggered.sum()),
        cv_evidence=parent.artifact(cv.OUT/'result.json'),no_upload=True))
    parent.log(f'{view}: fixed half sparse3/calls15 candidate built and serialized; no upload')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--view',choices=parent.VIEWS,required=True)
    with threadpool_limits(2): build(parser.parse_args().view)
