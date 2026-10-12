"""Separate membership/recipe/SQL audit of the fixed Rome TabM experiment.

Does not import the candidate runner or repeat training, inference or selection.
Original track/preprocessing and reload checks remain in their retained receipts.
"""
import json
from pathlib import Path
import time

import duckdb
import numpy as np
import pandas as pd

import audit_public_cohort_scores_20261012 as common

ROOT,RUNS=common.ROOT,common.RUNS
read,art,verify,metrics=common.read,common.artifact,common.verify,common.validate_metrics
TRAIN=RUNS/'v3_20261012_rome_tabm_tracks'
STACK=RUNS/'v3_20261012_rome_tabm_tracks_stack'
TRANSPORT=RUNS/'v3_20261012_rome_tabm_tracks_delta_transport'
RAW=RUNS/'v3_20261011_daily_offblock_alignment/rows2025.parquet'
OUT=ROOT/'VALIDATION/reports/rome_tabm_tracks_20261012/audit.json'
ARMS=['tabular','physical']


def main():
    started=time.monotonic();assert not OUT.exists()
    identity=read(TRAIN/'identity.json');pre=read(TRAIN/'preflight.json')
    result=read(STACK/'result.json');audit=read(STACK/'audit.json')
    transport=read(TRANSPORT/'rome_tabm_tracks/result.json');frozen=read(TRANSPORT/'design.json')
    assert pre['status']==audit['status']=='pass'
    assert result['identity']==audit['identity']==pre['identity']==art(TRAIN/'identity.json')
    assert audit['result']==art(STACK/'result.json')
    assert result['design']['probability_share']==1/3 and result['design']['no_search']
    assert identity['design']['model']['seed']==42 and identity['design']['model']['k']==32
    assert identity['design']['model']['max_epochs']==48
    assert identity['design']['no_blend_search'] and identity['design']['no_architecture_search']
    assert pre['feasibility_counts_independently_reproduced']
    assert result['exact_v77_and_v78_controls'] and result['unchanged_outside_rome']
    assert transport['identity']==art(TRANSPORT/'design.json') and transport['design']==frozen['design']
    assert frozen['design']['coefficient']==1 and frozen['design']['new_bounds'] is False
    assert frozen['design']['experiments']==dict(rome_tabm_tracks=dict(directory=STACK.name,control='control',arms=ARMS))
    for record in [*frozen['sources'],*result['members'],*result['artifacts'],*transport['sources'],transport['scored']]:verify(record)
    # Large immutable source bytes are checked by the fit and preflight receipts.
    for record in identity['sources']:
        if record['path'].endswith(('.py','.json')):verify(record)
    raw=pd.read_parquet(RAW).set_index('mvt_id');members=[]
    packets=[ROOT/r['path'] for r in identity['sources'] if '/inputs/' in r['path'] and r['path'].endswith('.pkl')]
    assert len(packets)==2
    for packet in packets:
        verify(next(r for r in identity['sources'] if ROOT/r['path']==packet))
        month=packet.stem;d=pd.read_pickle(packet);r=raw.loc[d['train_ids']]
        assert r.airport.eq('LIRF').all() and month not in set(r.month)
        np.testing.assert_array_equal(d['target'],(r.gap-r.y).abs().le(60).astype(int))
        assert not np.intersect1d(d['train_ids'],d['ids']).size
        assert raw.loc[d['ids'][d['use']],'airport'].eq('LIRF').all()
        last=max(r.month);inner_ids=d['train_ids'][r.month.ne(last).to_numpy()]
        for arm in ARMS:
            inner=read(TRAIN/f'{month}_{arm}_inner.json');full=read(TRAIN/f'{month}_{arm}_full.json')
            selected=min(inner['curve'],key=lambda row:row['inner_logloss'])['epoch']
            assert inner['epochs']==48 and full['epochs']==selected
            assert [v['epoch'] for v in inner['curve']]==list(range(1,49))
            assert [v['epoch'] for v in full['curve']]==list(range(1,selected+1))
            for phase,rec,ids in [('inner',inner,inner_ids),('full',full,d['train_ids'])]:
                assert rec['identity']==art(TRAIN/'identity.json') and rec['month']==month
                assert rec['phase']==phase and rec['arm']==arm and rec['seed']==42
                assert rec['device']=='NVIDIA GeForce RTX 3080' and rec['saved_before_held_inference']
                assert len(rec['columns'])==len(set(rec['columns']))==75
                for key in ['model','preprocessing','training_ids']:verify(rec[key])
                np.testing.assert_array_equal(pd.read_parquet(ROOT/rec['training_ids']['path']).mvt_id,ids)
                assert rec['rows']==len(ids) and rec['months']==sorted(set(raw.loc[ids,'month']))
                members.append(art(TRAIN/f'{month}_{arm}_{phase}.json'))
            rec=read(TRAIN/f'{month}_{arm}_predictions.json')
            assert rec['full_model']==art(TRAIN/f'{month}_{arm}_full.json')
            assert rec['inner_model']==art(TRAIN/f'{month}_{arm}_inner.json')
            assert rec['all_rome_reload_exact'] and rec['selected_epoch']==selected and rec['selection_month']==last
            verify(rec['prediction'])
            p=pd.read_parquet(ROOT/rec['prediction']['path']);np.testing.assert_array_equal(p.mvt_id,d['ids'])
            assert len(p)==rec['rows'] and int(np.sum(d['use']))==rec['rome_rows']
            assert p.p.between(0,1).all() and (p.loc[~d['use'],'p']==0).all()
            members.append(art(TRAIN/f'{month}_{arm}_predictions.json'))
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        for view,path in [('f',STACK/'scored.parquet'),('old',common.PARENT78/'scored.parquet'),
            ('p',common.PARENT79/'cv.parquet'),('r',RAW),('t',TRANSPORT/'rome_tabm_tracks/scored.parquet')]:
            con.read_parquet(str(path)).create_view(view)
        terms=' OR '.join(f't.{arm} IS DISTINCT FROM round_even(greatest(0,p.pred+f.{arm}-f.control),0) OR '
            f'(f.airport<>\'LIRF\' AND f.{arm} IS DISTINCT FROM f.control)' for arm in ARMS)
        n,u,bad=con.execute(f'''SELECT count(*),count(DISTINCT f.mvt_id),sum(CASE WHEN
            f.y IS DISTINCT FROM r.y OR f.month IS DISTINCT FROM r.month OR f.day IS DISTINCT FROM r.day OR
            f.airport IS DISTINCT FROM r.airport OR f.control IS DISTINCT FROM old.candidate OR
            f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day OR
            t.y IS DISTINCT FROM p.y OR t.month IS DISTINCT FROM p.month OR t.day IS DISTINCT FROM p.day OR
            t.control IS DISTINCT FROM p.pred OR {terms} THEN 1 ELSE 0 END)
            FROM f JOIN r USING(mvt_id) JOIN old USING(mvt_id) JOIN p USING(mvt_id) JOIN t USING(mvt_id)''').fetchone()
        assert n==u==344419 and bad==0
        for view in ['f','t']:
            terms=' OR '.join(f'{c} IS NULL OR NOT isfinite({c}) OR {c}<0 OR {c}<>round_even({c},0)' for c in ['control',*ARMS])
            assert con.execute(f'SELECT count(*) FROM {view} WHERE {terms}').fetchone()[0]==0
        primary=metrics(con,'f',result,['control',*ARMS])
        secondary=metrics(con,'t',transport,['control',*ARMS])
    effects={name:{arm:r['cv'][arm]-r['cv']['control'] for arm in ARMS}
        for name,r in [('v78',result),('v79',transport)]}
    value=dict(status='pass',rows=n,seconds=time.monotonic()-started,source=art(__file__),members=members,
        effects=effects,sql_scores=primary,sql_transport_scores=secondary,
        inputs=[art(p) for p in [TRAIN/'identity.json',TRAIN/'preflight.json',STACK/'result.json',STACK/'audit.json',
            STACK/'scored.parquet',TRANSPORT/'design.json',TRANSPORT/'rome_tabm_tracks/result.json',
            TRANSPORT/'rome_tabm_tracks/scored.parquet',RAW]],
        original_binary_targets_and_inner_full_memberships=True,inner_only_stopping_and_frozen_recipes=True,
        all_row_truth_calendar_integer_transport_and_nonrome_invariance=True,
        scope='Separate saved-artifact, membership, selection and SQL arithmetic checks. Raw track transforms, '
            'encoder fitting, GPU checks and model reloads remain in their retained preflight/fit receipts.',
        limitation='Fixed-seed reused development experiment, not independent confirmation.')
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass',rows=n,effects=effects),indent=2))


if __name__=='__main__':main()
