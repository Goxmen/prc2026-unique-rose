"""Training-only feasibility of end-to-end Rome TabM track classification.

The frozen-encoder tree and plain PLE classifiers already failed. A classifier
which learns physical tracks jointly with a TabM backbone is untested. This
check reads only four shared training months, with no Jan/Jul departure labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261012_rome_tabm_track_pilot'
RAW=ROOT/'VALIDATION/runs/v3_20261011_daily_offblock_alignment/rows2025.parquet'
INDEX=ROOT/'data/derived/sparse_track_sequence_20261010/full_index.parquet'
MONTHS=['2025-03','2025-09','2025-10','2025-11']
DESIGN=dict(months=MONTHS,airport='LIRF',target='abs(gap-y)<=60; original classifier label',
    observations='count>=2 in retained full physical track index; no learned encoder',
    primary='at least100 observed copy and100 observed non-copy training rows across fixed months',
    reports='monthly classes, usable-track and ground-observation counts; no model or prediction costs',
    distinctness='jointly trained raw-track TabM binary classifier; earlier frozen-encoder and PLE classifiers remain closed',
    proposed_comparator='same TabM and9 quality inputs with physical representation disabled',
    no_held_departure_labels=True,no_fit=True,no_cv=True,no_parameter_search=True)


def art(p):
    return dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def read(p):
    return json.loads(p.read_text())


def freeze():
    assert not (OUT/'identity.json').exists()
    manifest=INDEX.with_name('full.json')
    assert read(manifest)['outputs'][INDEX.name]==art(INDEX)['sha256']
    paths=[Path(__file__),RAW,INDEX,manifest,
           ROOT/'VALIDATION/rome_neural_jan_jul.py',ROOT/'VALIDATION/rome_neural_model.py',
           ROOT/'VALIDATION/reports/research_network_20261010/rome_frozen_track_input_coverage.json']
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'identity.json').write_text(json.dumps(dict(design=DESIGN,sources=[art(p) for p in paths]),indent=2)+'\n')


def pilot():
    identity=read(OUT/'identity.json');assert identity['design']==DESIGN
    assert not (OUT/'result.json').exists()
    for r in identity['sources']:
        assert art(ROOT/r['path'])==r
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        con.read_parquet(str(INDEX)).create_view('idx')
        con.read_parquet(str(RAW)).create_view('raw')
        query='''SELECT r.mvt_id,r.month,abs(r.gap-r.y)<=60 AS copy,i.count>=2 AS usable,
            i.raw_ground_count>0 AS ground,r.has_id,r.gap
            FROM raw r JOIN idx i USING(mvt_id) WHERE r.airport='LIRF' AND r.month IN (SELECT unnest(?))'''
        frame=con.execute(query,[MONTHS]).df()
        assert frame.mvt_id.is_unique and set(frame.month)==set(MONTHS)
        expected=con.execute("SELECT count(*) FROM raw WHERE airport='LIRF' AND month IN (SELECT unnest(?))",[MONTHS]).fetchone()[0]
        assert len(frame)==expected
    records=[]
    for month,q in [('ALL',frame),*frame.groupby('month')]:
        for label in [False,True]:
            f=q.loc[q['copy'].eq(label)]
            records.append(dict(month=month,copy=label,rows=len(f),usable=int(f.usable.sum()),
                ground=int(f.ground.sum()),no_id=int((~f.has_id).sum()),
                no_id_long=int((~f.has_id&f.gap.gt(3600)).sum())))
    allowed=all(r['usable']>=100 for r in records if r['month']=='ALL')
    result=dict(identity=art(OUT/'identity.json'),rows=len(frame),records=records,
                enough_observed_examples=allowed,no_fit=True,no_cv=True,
                interpretation='Feasibility only; input coverage does not establish label predictability or gain.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','pilot'])
    globals()[parser.parse_args().stage]()
