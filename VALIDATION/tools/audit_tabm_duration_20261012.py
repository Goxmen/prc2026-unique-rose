"""Independent saved-model contracts and all-row SQL audit; no new fitting."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import duckdb
import torch
import tabm_other_runway_combo_cv as common

ROOT,RUNS=common.ROOT,common.RUNS
RUN=RUNS/'v3_20261012_nn_public_tabm_duration'
STACK=RUNS/'v3_20261012_nn_public_tabm_duration_stack'
read,art,verify,write=common.read,common.art,common.verify,common.write


def main():
    assert not (STACK/'independent_audit.json').exists()
    identity=read(RUN/'identity.json');result=read(STACK/'combined_result.json')
    raw=read(STACK/'result.json');parity=read(RUN/'update_parity.json')
    assert parity['status']=='pass' and parity['identity']==art(RUN/'identity.json')
    assert result['training_identity']==art(RUN/'identity.json') and result['exact_old_controls_and_combined_replay']
    assert read(RUN/'audit.json')['status']=='pass'
    for record in [*identity['files'],*result['sources'],*raw['sources'],result['prediction'],raw['scored']]:verify(record)
    members=[]
    for month,early_stop in [('2025-01',21),('2025-07',17)]:
        final_curve=None
        for arm,epochs in [('tabm',24),('early',early_stop)]:
            path=RUN/'networks'/f'{month}_{arm}.json';r=read(path)
            assert r['identity']==art(RUN/'identity.json') and r['design']==identity['design']
            assert r['seed']==42 and r['month']==month and r['epochs']==epochs and r['all_held_reloaded_exact']
            for key in ['model','prediction','preprocessing']:verify(r[key])
            assert r['preprocessing']==identity['folds'][month]['preprocessing']
            model=torch.load(verify(r['model']),map_location='cpu',weights_only=False)
            assert model['identity']==art(RUN/'identity.json') and model['design']==identity['design']
            assert [v['epoch'] for v in model['curve']]==list(range(1,epochs+1))
            assert all(torch.isfinite(v).all() for v in model['state_dict'].values())
            if arm=='tabm':
                final_curve=model['curve'];assert r['curve']==final_curve
                assert r['inputs']==identity['folds'][month]
                assert {k:r[k] for k in ['host','gpu','torch']}==identity['execution']
            else:
                assert model['curve']==final_curve[:epochs] and r['same_trajectory_as_final']
            members.append(art(path));del model
    with duckdb.connect(config={'threads':1,'memory_limit':'256MB'}) as con:
        for name,path in [('f',STACK/'scored.parquet'),('s',STACK/'combined.parquet'),
            ('old',RUNS/'v3_20261012_nn_public_tabm_pool3_stack/scored.parquet'),
            ('selected',RUNS/'v3_20261012_tabm_pool3_other_runway_combo/scored.parquet'),
            ('p',common.P79/'cv.parquet'),('r',common.RUNWAY/'scored.parquet')]:
            con.read_parquet(str(path)).create_view(name)
            assert con.execute(f'SELECT count(*),count(DISTINCT mvt_id) FROM {name}').fetchone()==(344419,344419)
        cols=['three_tabm','matched_duration','complete_schedule']
        formula=' OR '.join(f's.{c} IS DISTINCT FROM round_even(greatest(0,p.pred+f.{c}-old.selected+r.candidate-old.selected),0)' for c in cols)
        bad=con.execute(f'''SELECT count(*) FROM s JOIN f USING(mvt_id) JOIN old USING(mvt_id)
            JOIN selected USING(mvt_id) JOIN p USING(mvt_id) JOIN r USING(mvt_id)
            WHERE s.y IS DISTINCT FROM p.y OR s.month IS DISTINCT FROM p.month OR s.day IS DISTINCT FROM p.day
            OR f.y IS DISTINCT FROM p.y OR f.month IS DISTINCT FROM p.month OR f.day IS DISTINCT FROM p.day
            OR f.selected IS DISTINCT FROM old.selected OR f.one_tabm IS DISTINCT FROM old.one_tabm
            OR f.three_tabm IS DISTINCT FROM old.three_tabm OR s.three_tabm IS DISTINCT FROM selected.three_tabm_runway
            OR {formula}''').fetchone()[0]
        assert bad==0
        for view,rec in [('f',raw),('s',result)]:
            for name in rec['cv']:
                scores=dict(con.execute(f'SELECT month,sqrt(avg(pow({name}-y,2))) FROM {view} GROUP BY month').fetchall())
                for m,value in scores.items():assert abs(value-rec['scores'][m][name])<1e-8
                assert abs(sum(scores.values())/2-rec['cv'][name])<1e-8
    value=dict(status='pass',source=art(__file__),identity=art(RUN/'identity.json'),result=art(STACK/'combined_result.json'),
        members=members,rows=344419,all_saved_model_contracts_and_exact_epoch_prefix_curves=True,
        all_row_ids_truth_calendar_original_controls_and_integer_transport=True,independent_SQL_scores=True,
        update_parity=art(RUN/'update_parity.json'),training_guard_and_all_row_model_reloads='Inherited audited fit receipts',
        scope='Independent provenance and arithmetic audit; training input/model reload already checked by the frozen runner. Reused development data.')
    write(STACK/'independent_audit.json',value)
    print(json.dumps(dict(status='pass',rows=344419,gate=result['gate_pass'],cv=result['cv'])))


if __name__=='__main__':
    torch.set_num_threads(1);main()
