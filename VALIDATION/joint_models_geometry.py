"""Append v78 to the frozen basis; evaluate only after actual official scores."""
import argparse
from pathlib import Path
import types

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import joint_public_candidate as candidate
import daily_models_geometry as parent
import public_models_geometry as base
from records import sha256,write_json

ROOT=parent.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261011_joint_models_geometry'
ADDITIONS=[*parent.ADDITIONS,78]
DESIGN=dict(parent.DESIGN,parent='audited51-vector basis plus selected joint-source increment',
    additions=ADDITIONS,new_directions='v78 fixed joint public taxi and departure moments',
    preparation='reuse all51 frozen ranking/CV vectors; append one exact serialized vector')
art,read,verify=parent.art,parent.read,parent.verify
transport=base.transport


def prepare():
    assert not (OUT/'prepared.json').exists(),'Preserve prepared geometry'
    previous=read(parent.OUT/'prepared.json');old_audit=read(parent.OUT/'audit.json')
    assert old_audit['status']=='pass' and old_audit['prepared']==art(parent.OUT/'prepared.json')
    assert len(previous['versions'])==51 and 78 not in previous['versions']
    assert previous['design']==parent.DESIGN
    for path,digest in previous['sources'].items():assert sha256(ROOT/path)==digest,path
    ranking=pd.read_parquet(verify(previous['ranking']));held=pd.read_parquet(verify(previous['cv']))
    path=candidate.OUT/'submission_v78/manifest.json';manifest=read(path);audit=read(path.parent/'audit.json')
    assert manifest['version']==78 and manifest['rows']==344841
    assert manifest['state'] in ['prepared','uploaded','scored']
    assert not manifest['audit']['independent_package_audit_pending']
    assert audit['status']=='pass' and audit['file']==art(ROOT/manifest['file'])
    for source,digest in manifest['sources'].items():assert sha256(ROOT/source)==digest,source
    assert sha256(ROOT/manifest['file'])==manifest['sha256']
    frame=pd.read_parquet(ROOT/manifest['file']);np.testing.assert_array_equal(ranking.mvt_id,frame.MVT_ID_mvt)
    validation=pd.read_parquet(path.parent/'cv.parquet').set_index('mvt_id').loc[held.mvt_id]
    assert len(validation)==len(held)==344419 and validation.index.is_unique
    for field in ['month','day','y']:np.testing.assert_array_equal(validation[field],held[field])
    values=validation.pred.to_numpy(float);np.testing.assert_array_equal(values,np.rint(np.maximum(values,0)))
    assert np.isfinite(values).all()
    ranking['v78']=frame.TAXITIME_SEC_mvt.to_numpy(float);held['v78']=values
    OUT.mkdir(parents=True,exist_ok=True);snapshot=OUT/'manifest_v78_snapshot.json'
    assert not snapshot.exists(),'Inspect partial preparation before retry'
    write_json(snapshot,manifest)
    additions=dict(previous['additions'])
    additions['78']=dict(snapshot=art(snapshot),live_manifest=str(path.relative_to(ROOT)),
        ranking=art(ROOT/manifest['file']),cv=art(path.parent/'cv.parquet'),
        cv_record=manifest['cv_record'],cv_column='pred',cv_serialization='already nonnegative nearest integer')
    sources=dict(previous['sources'])
    for source in [Path(__file__),Path(parent.__file__),Path(base.__file__),Path(candidate.__file__),
        parent.OUT/'prepared.json',parent.OUT/'audit.json',verify(previous['ranking']),
        verify(previous['cv']),snapshot,path.parent/'audit.json',ROOT/manifest['file'],
        path.parent/'cv.parquet',verify(manifest['cv_record'])]:
        sources[str(source.relative_to(ROOT))]=sha256(source)
    ranking.to_parquet(OUT/'ranking_basis.parquet',index=False);held.to_parquet(OUT/'cv_basis.parquet',index=False)
    write_json(OUT/'prepared.json',dict(design=DESIGN,versions=[*previous['versions'],78],
        known_scores=previous['known_scores'],additions=additions,sources=sources,
        ranking=art(OUT/'ranking_basis.parquet'),cv=art(OUT/'cv_basis.parquet'),
        checks=base.check_geometry(),historical_cv_caveat=previous['historical_cv_caveat'],
        reused_basis=art(parent.OUT/'prepared.json')))
    print(dict(versions=52,ranking_rows=len(ranking),cv_rows=len(held),required_new_scores=[75,76,77,78]),flush=True)


def inherit(function):
    scope=dict(function.__globals__,OUT=OUT,ADDITIONS=ADDITIONS,DESIGN=DESIGN,__file__=__file__)
    result=types.FunctionType(function.__code__,scope,function.__name__,function.__defaults__,function.__closure__)
    assert result.__code__ is function.__code__
    return result


def audit():
    for name in ['ranking_basis.parquet','cv_basis.parquet']:
        old=pd.read_parquet(parent.OUT/name);new=pd.read_parquet(OUT/name)
        pd.testing.assert_frame_equal(new[old.columns],old,check_exact=True)
    inherit(base.audit)()


def evaluate():
    inherit(base.evaluate)()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['prepare','audit','evaluate'])
    with threadpool_limits(1):globals()[parser.parse_args().stage]()
