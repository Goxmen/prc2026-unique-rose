"""Independent final shared-ID transport audit; no transport-builder imports.

Verifies construction and source provenance only. Hidden departure block-clock
coupling and final accuracy remain unverified; no new CV score is produced.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import time
import duckdb
import numpy as np
import pyarrow.parquet as pq
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'VALIDATION/runs/v3_20261013_final_shared_clock_transport'
V80=ROOT/'VALIDATION/runs/v3_20261012_tabm_pool3_candidate/submission_v80'
EXPECTED_V80='50b986030af2efaab89b69bb69522ade6b060cc7f58fa324954e3fb06e0ef6f4'
EXPECTED_TEMPLATE='b3445d80e29bf599a5583ca9675f63ddf57b51fdf1ca3f498facb38ea84f3924'
RAW_SHA={'ranking':'9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86',
         'final_ranking':'4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f'}


def read(p):return json.loads(Path(p).read_text())
def art(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):h.update(block)
    return dict(path=str(p.relative_to(ROOT)),sha256=h.hexdigest())
def verify(a):
    path=ROOT/a['path'];assert art(path)=={k:a[k] for k in ['path','sha256']};return path


def provenance(identity,result):
    assert identity['coefficient']==result['coefficient']==2 and identity['auditor']==art(Path(__file__))
    assert result['identity']==art(OUT/'identity.json')
    for a in identity['sources']:verify(a)
    for key in ['auditor','v80','portable','template','ranking','final_ranking','source_review','portable_result','portable_audit']:
        assert identity[key] in identity['sources'],key
    assert identity['template']==dict(path='data/final_submitting.parquet',sha256=EXPECTED_TEMPLATE)
    assert result['qualifications']==identity['qualifications']
    q=identity['qualifications']
    assert q['not_exact_v80_final_replay'] and not q['old_leaderboard_improvement_claim'] and not q['final_score_known']
    assert 'unverified' in q['departure_assumption'] and 'No genuine paired-departure' in q['validation_limit']
    review=read(verify(identity['source_review']))
    assert review['status']=='pass' and review['auditor']==identity['auditor']
    assert review['builder'] in identity['sources'] and review['assessment'] in identity['sources']
    allocation=read(verify(result['allocation']))
    assert allocation['authority']=='/root' and allocation['identity']==art(OUT/'identity.json')
    assert allocation['action']=='build_fixed_shared_clock_transport'
    parent=read(verify(identity['portable_result']));audit=read(verify(identity['portable_audit']))
    assert audit['status']=='pass' and audit['result']==identity['portable_result']
    assert audit['identity']==parent['identity'] and parent['identity'] in identity['sources'];verify(parent['identity'])
    spec=identity['portable_spec'];assert spec['prediction']==identity['portable']
    if spec['kind']=='audited_final_candidate':
        assert parent['view']==audit['view']=='final' and parent['rows']==audit['rows']==670790
        assert parent['candidate']==audit['candidate']==identity['portable']
        assert (spec['id_column'],spec['value_column'])==('MVT_ID_mvt','TAXITIME_SEC_mvt')
    elif spec['kind']=='audited_nested_final_candidate':
        assert set(parent['views'])==set(audit['views'])=={'old','cv','final'}
        final,checked=parent['views']['final'],audit['views']['final']
        assert final['rows']==checked['rows']==670790
        assert final['candidate']==checked['candidate']==identity['portable']
        assert final['prediction']==checked['prediction']
        assert (spec['id_column'],spec['value_column'])==('MVT_ID_mvt','TAXITIME_SEC_mvt')
    else:
        assert spec['kind']=='audited_three_view_submitted'
        assert set(parent['predictions'])=={'old','cv','final'} and parent['predictions']==audit['predictions']
        assert parent['predictions']['final']==identity['portable']
        assert parent['rows']['final']==670790
        assert (spec['id_column'],spec['value_column'])==('mvt_id','submitted')
    manifest=read(V80/'manifest.json');official=read(V80/'official_result.json')
    assert art(V80/'manifest.json') in identity['sources'] and art(V80/'official_result.json') in identity['sources']
    assert manifest['version']==80 and manifest['state']=='scored' and manifest['rows']==344841
    assert manifest['sha256']==manifest['remote_sha256']==identity['v80']['sha256']==EXPECTED_V80
    assert identity['v80']['path']==manifest['file'] and official==manifest['official_result']
    assert official['status']=='Succeeded' and official['file']=='unique-rose_v80.parquet' and official['used_pairs']==344841
    production=read(verify(manifest['component']));check=read(verify(manifest['audit']))
    assert check['status']=='pass' and check['result']==manifest['component'] and check['identity']==production['identity']
    assert check['file'] in production['artifacts'] and check['file']['sha256']==EXPECTED_V80
    for key,sha in RAW_SHA.items():assert identity[key]['sha256']==sha
    return spec


def audit():
    started=time.monotonic();assert not (OUT/'audit.json').exists()
    identity,result=read(OUT/'identity.json'),read(OUT/'result.json');spec=provenance(identity,result)
    template=pq.read_table(verify(identity['template']));candidate=pq.read_table(verify(result['candidate']))
    trace=pq.read_table(verify(result['trace']))
    assert template.schema==candidate.schema and template.num_rows==candidate.num_rows==trace.num_rows==670790
    for col in template.column_names:
        if col!='TAXITIME_SEC_mvt':assert candidate[col].equals(template[col])
    np.testing.assert_array_equal(trace['MVT_ID_mvt'].to_numpy(),template['MVT_ID_mvt'].to_numpy())
    np.testing.assert_array_equal(trace['submitted'].to_numpy(),candidate['TAXITIME_SEC_mvt'].to_numpy())
    with duckdb.connect(config={'threads':1,'memory_limit':'1GB'}) as con:
        con.execute("SET TimeZone='UTC'")
        for name,record in [('old_raw',identity['ranking']),('final_raw',identity['final_ranking']),
            ('old',identity['v80']),('fallback',identity['portable']),('candidate',result['candidate']),('trace',result['trace'])]:
            con.read_parquet(str(verify(record))).create_view(name)
        for name,n in [('old',344841),('candidate',670790),('trace',670790)]:
            assert con.execute(f'SELECT count(*),count(distinct MVT_ID_mvt) FROM {name}').fetchone()==(n,n)
        for name in ['old_raw','final_raw']:
            assert con.execute(f'SELECT count(*)=count(distinct MVT_ID_mvt) FROM {name}').fetchone()[0]
        idcol,valuecol=spec['id_column'],spec['value_column']
        assert con.execute(f'SELECT count(*),count(distinct "{idcol}") FROM fallback').fetchone()==(670790,670790)
        con.execute(f'''CREATE TEMP VIEW b AS SELECT "{idcol}" AS id,CAST("{valuecol}" AS DOUBLE) AS pred FROM fallback''')
        assert con.execute('SELECT count(*) FROM b WHERE pred IS NULL OR NOT isfinite(pred) OR pred<0 OR pred!=round_even(pred,0)').fetchone()[0]==0
        assert con.execute('''SELECT count(*) FROM old o LEFT JOIN candidate c USING(MVT_ID_mvt) WHERE c.MVT_ID_mvt IS NULL''').fetchone()[0]==0
        con.execute('''CREATE TEMP VIEW checked AS SELECT c.MVT_ID_mvt AS id,c.TAXITIME_SEC_mvt AS pred,t.shared,
            t.delta_takeoff_seconds AS dt_saved,t.submitted AS traced,b.pred AS fallback,o.TAXITIME_SEC_mvt AS prior,
            o.MVT_ID_mvt IS NOT NULL AS is_shared,
            epoch(f.MVT_TIME_UTC_mvt)-epoch(r.MVT_TIME_UTC_mvt) AS actual_dt,
            f.PHASE_mvt AS final_phase,r.PHASE_mvt AS old_phase,f.ADEP_mvt AS airport,
            strftime(f.MVT_TIME_UTC_mvt,'%Y-%m') AS utc_month,
            f.BLOCK_TIME_UTC_mvt AS final_block,f.TAXITIME_SEC_mvt AS final_label,
            r.BLOCK_TIME_UTC_mvt AS old_block,r.TAXITIME_SEC_mvt AS old_label
            FROM candidate c JOIN trace t USING(MVT_ID_mvt) JOIN b ON b.id=c.MVT_ID_mvt
            JOIN final_raw f USING(MVT_ID_mvt) LEFT JOIN old o USING(MVT_ID_mvt)
            LEFT JOIN old_raw r ON o.MVT_ID_mvt=r.MVT_ID_mvt''')
        n,u,shared,portable,bad=con.execute('''SELECT count(*),count(distinct id),count(*) FILTER(WHERE is_shared),count(*) FILTER(WHERE NOT is_shared),
            count(*) FILTER(WHERE pred IS NULL OR NOT isfinite(pred) OR pred<0 OR pred>2147483647 OR pred!=round_even(pred,0)
                OR pred IS DISTINCT FROM traced OR shared IS DISTINCT FROM is_shared OR final_phase IS DISTINCT FROM 'DEP'
                OR final_block IS NOT NULL OR final_label IS NOT NULL
                OR (is_shared AND (old_phase IS DISTINCT FROM 'DEP' OR old_block IS NOT NULL OR old_label IS NOT NULL
                    OR actual_dt IS NULL OR NOT isfinite(actual_dt) OR actual_dt!=round_even(actual_dt,0) OR abs(actual_dt)>12
                    OR dt_saved IS DISTINCT FROM actual_dt OR pred IS DISTINCT FROM round_even(greatest(CAST(prior AS DOUBLE)+2*actual_dt,0.),0)))
                OR (NOT is_shared AND (dt_saved IS DISTINCT FROM 0. OR pred IS DISTINCT FROM fallback))) FROM checked''').fetchone()
        assert n==u==result['rows']==670790 and shared==result['shared_rows']==344841 and portable==result['portable_rows']==325949 and bad==0,(n,u,shared,portable,bad)
        groups=con.execute('''SELECT utc_month,airport,count(*) AS rows,count(*) FILTER(WHERE is_shared) AS shared_rows,
            count(*) FILTER(WHERE NOT is_shared) AS portable_rows,count(*) FILTER(WHERE is_shared AND prior+2*actual_dt<0) AS zero_clipped,
            max(abs(actual_dt)) FILTER(WHERE is_shared) AS max_dt,min(pred) AS minimum,max(pred) AS maximum
            FROM checked GROUP BY utc_month,airport ORDER BY utc_month,airport''').fetchdf().to_dict('records')
    # Null sample maxima are legitimate on the two nonshared months.
    for row in groups:
        for key,val in row.items():
            if isinstance(val,float) and np.isnan(val):row[key]=None
    value=dict(status='pass',view='final',source=art(Path(__file__)),identity=art(OUT/'identity.json'),result=art(OUT/'result.json'),
        candidate=result['candidate'],trace=result['trace'],rows=n,shared_rows=shared,portable_rows=portable,groups=groups,
        all_SQL_raw_ID_source_clock_integer_clamp_fallback_and_trace_exact=True,all_template_columns_schema_order_exact=True,
        allocation=result['allocation'],qualifications=identity['qualifications'],
        departure_block_phase_assumption_verified=False,final_accuracy_verified=False,no_new_CV_score=True,no_fit=True,no_upload=True,
        seconds=time.monotonic()-started)
    (OUT/'audit.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass',rows=n,shared_rows=shared,portable_rows=portable,audit=art(OUT/'audit.json'),seconds=value['seconds'])))

if __name__=='__main__':audit()
