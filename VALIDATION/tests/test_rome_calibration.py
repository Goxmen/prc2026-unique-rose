"""Synthetic checks only: LightGBM training is forbidden throughout this module."""
import contextlib
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import overnight_models
import protocol
import rome_calibration as c
import rome_calibration_run as run
import rome_calibration_report as report
from VALIDATION.tests.test_v3 import departures


def setUpModule():
    guard=patch('lightgbm.train',side_effect=AssertionError('Unit checks must not train boosters'))
    guard.start()
    unittest.addModuleCleanup(guard.stop)


def raw_frame(gaps,airports=None,noid=None):
    n=len(gaps)
    t=pd.Timestamp('2025-07-01 22:30',tz='UTC')
    f=pd.DataFrame(dict(ADEP_mvt=airports or ['LIRF']*n,
        FLIGHT_ID_mvt=[np.nan if b else 1. for b in (noid or [True]*n)],
        MVT_TIME_UTC_mvt=[t]*n,
        SCHED_TIME_UTC_mvt=[t-pd.Timedelta(seconds=g) if np.isfinite(g) else pd.NaT for g in gaps],
        LOBT_flt=pd.Series([pd.NaT]*n,dtype='datetime64[ns, UTC]')))
    return f


class CalibrationTests(unittest.TestCase):
    def test_label_precedence_and_boundaries(self):
        y=np.array([1200,1140,1139,86400,86401,93600,93601,87000])
        gap=np.array([1200,1200,1200,np.nan,1000,1000,1000,87000])
        self.assertEqual(c.outcomes(y,gap,3).tolist(),[1,1,0,0,2,2,0,1])
        self.assertEqual(c.outcomes(y,gap,2).tolist(),[1,1,0,0,0,0,0,1])
        with self.assertRaises(ValueError):c.outcomes([np.nan],[1])

    def test_objective_gradient_matches_finite_difference(self):
        logp=np.log([[.7,.2,.1],[.1,.8,.1],[.05,.05,.9],[.3,.3,.4]])
        y=np.array([0,1,2,0]);t=np.array([.8,-.4,.2]);eps=1e-6
        _,gradient=c.calibration_objective(t,logp,y)
        numerical=[]
        for i in range(3):
            d=np.zeros(3);d[i]=eps
            numerical.append((c.calibration_objective(t+d,logp,y)[0]-c.calibration_objective(t-d,logp,y)[0])/(2*eps))
        np.testing.assert_allclose(gradient,numerical,atol=1e-7)

    def test_calibration_learns_separate_id_strata(self):
        p=np.tile([.2,.8],(400,1));y=np.r_[np.ones(80),np.zeros(120),np.ones(20),np.zeros(180)].astype(int)
        noid=np.r_[np.zeros(200,bool),np.ones(200,bool)]
        fitted=c.fit_calibrator(p,y,noid);q=c.calibrate(p,noid,fitted)
        self.assertLess(q[:200,1].mean(),.8)
        self.assertLess(q[200:,1].mean(),q[:200,1].mean())
        self.assertLess(-np.log(q[np.arange(400),y]).mean(),-np.log(p[np.arange(400),y]).mean())
        for g in fitted['groups'].values():
            self.assertEqual(g['rows'],200)
            self.assertLess(g['objective_after'],g['objective_before'])

    def test_small_stratum_identity_and_absent_day_are_finite(self):
        p=np.tile([.4,.6,0.],(25,1));y=np.arange(25)%2;noid=np.arange(25)<6
        fitted=c.fit_calibrator(p,y,noid);q=c.calibrate(p,noid,fitted)
        self.assertEqual(fitted['groups']['id_missing']['theta'],[1.,0.,0.])
        self.assertEqual(fitted['groups']['id_present']['theta'],[1.,0.,0.])
        self.assertTrue(np.isfinite(q).all());np.testing.assert_allclose(q.sum(axis=1),1)
        np.testing.assert_allclose(q[:,:2],p[:,:2],atol=1e-8)

    def test_invalid_probabilities_are_rejected(self):
        for p in ([[.8,.8]],[[-.2,1.2]],[[np.nan,1.]],[[1.]]):
            with self.subTest(p=p),self.assertRaises(ValueError):c.fit_calibrator(p,[0],[False])
        with self.assertRaises(ValueError):c.constrain_day([[.3,.4,.3]],[True,False],1)

    def test_day_gate_uses_rome_local_dates_and_support_shrinkage(self):
        gaps=[50400,50399,50400,50400,172800]
        raw=raw_frame(gaps,['LIRF','LIRF','EGLL','LIRF','LIRF'],[True,True,True,False,True])
        eligible=c.day_eligible(raw,gaps)
        self.assertEqual(eligible.tolist(),[True,False,False,False,False])
        p=np.tile([.2,.3,.5],(5,1));q=c.constrain_day(p,eligible,20)
        np.testing.assert_allclose(q[0],[.3,.45,.25])
        np.testing.assert_allclose(q[1:],[ [.4,.6,0.] ]*4)
        np.testing.assert_allclose(c.constrain_day(p,eligible,0)[:,2],0)

    def test_mixture_preserves_parent_clock_and_late_fallback(self):
        gap=np.array([6000,4000,-1000,np.nan]);raw=raw_frame(gap)
        p=np.tile([.8,.2],(4,1));ordinary=np.array([7000,7000,100,900])
        np.testing.assert_allclose(c.mixture(ordinary,gap,p,raw,1000),[2000,1120,0,900])
        np.testing.assert_allclose(c.mixture(ordinary,gap,p,raw,None),[6800,6400,0,900])
        bounded=c.binary_schedule_scope(p,gap)
        np.testing.assert_array_equal(bounded[-1],[1,0]);np.testing.assert_array_equal(p[-1],[.8,.2])
        day=c.mixture([1,100000],[60000,60000],[[0,0,1],[0,0,1]],raw_frame([60000,60000]),1000)
        np.testing.assert_allclose(day,[86460,93600])

    def test_postprocessing_ignores_targets_and_preserves_other_airports(self):
        raw=raw_frame([60000,6000,1000],['LIRF','EGLL','LIRF'],[True,True,False])
        raw.loc[2,'LOBT_flt']=raw.loc[2,'MVT_TIME_UTC_mvt']-pd.Timedelta(seconds=9000)
        control=np.array([60000.,1000.,5340.]);rome=np.array([True,False,True])
        before=c.postprocess(control,[70000,1000],raw,rome)
        raw['taxi_sec']=[1,1e9,1];raw['y']=1e9;raw['MVT_TIME_OFFBLOCK']=pd.Timestamp('1800-01-01')
        after=c.postprocess(control,[70000,1000],raw,rome)
        np.testing.assert_array_equal(before,[70000,1000,5340]);np.testing.assert_array_equal(after,before)

    def test_classifier_adapter_sets_only_intended_multiclass_changes(self):
        X=pd.DataFrame({'gap':[1,2,3]})
        with patch.object(c.signals,'_train',return_value=object()) as train:
            _,info=c.fit_classifier(X,[1000,3000,87000],[2000,3000,60000],42,3)
        params,args_X,labels,rounds,_=train.call_args.args
        self.assertEqual(params['num_class'],3);self.assertEqual(params['num_leaves'],31)
        self.assertEqual(params['num_threads'],6);self.assertEqual(rounds,300)
        self.assertIs(args_X,X);np.testing.assert_array_equal(labels,[0,1,2])
        self.assertEqual(info['counts'],[1,1,1])


class ScopeAndResumeTests(unittest.TestCase):
    def test_all_inner_jobs_are_chronological_inside_outer_training(self):
        jobs=[j for j in run.shared.schedule() if j['scheme']=='v3']
        self.assertEqual(len(jobs),6)
        for outer in jobs:
            inner=run.inner_jobs(outer)
            self.assertEqual([len(j['train']) for j in inner],[3,4,5])
            for j in inner:
                self.assertEqual(j['train'],outer['train'][:len(j['train'])])
                train=departures([j['start'],j['end']]);valid=departures([j['evals'][0]['month']+'-01'])
                valid['mvt_id']=10
                run.check_inner_scope(outer,j,train,valid)
                with self.assertRaises(ValueError):run.check_inner_scope(outer,j,train,valid.assign(mvt_id=0))
                with self.assertRaises(ValueError):
                    run.check_inner_scope(outer,j,train,valid.assign(MVT_TIME_UTC_mvt=pd.Timestamp(outer['end'],tz='UTC')+pd.Timedelta(days=1)))

    def test_feature_path_filters_training_to_rome_and_never_reads_validation_labels(self):
        td=departures(['2025-01-01','2025-02-01','2025-03-01'],[1000,1200,800])
        td.loc[1,'ADEP_mvt']='EGLL';td['FLIGHT_mvt']='ITY100'
        vd=departures(['2025-04-01','2025-04-02'],[1000,100000]);vd['mvt_id']=[10.,11.]
        vd['FLIGHT_mvt']='NEW999';vd.loc[1,'ADEP_mvt']='EGLL'
        for f in (td,vd):
            for name in ('raw_flight_id_missing','raw_plan_block_missing','raw_aobt3_missing'):f[name]=False
        # Arrival inputs include the observable landing/taxi-in fields used by the stand witness.
        ta=td.copy();ta['ADES_mvt']='LIRF';va=vd.copy();va['ADES_mvt']='LIRF'
        cfg=copy.deepcopy(overnight_models.BASE_CONFIG)
        X,y,cats,raw=run.training_inputs(td,ta,cfg)
        self.assertEqual(len(X.columns),54);self.assertTrue(raw.ADEP_mvt.eq('LIRF').all())
        np.testing.assert_array_equal(y,[1000,800])
        V,vr=run.validation_inputs(td,ta,vd,va,'2025-03-31',cats,cfg)
        changed=vd.assign(taxi_sec=[199999,1],OFFBLOCK_TIME_mvt=pd.Timestamp('1800-01-01'))
        W,_=run.validation_inputs(td,ta,changed,va,'2025-03-31',cats,cfg)
        pd.testing.assert_frame_equal(V,W)
        self.assertEqual(len(V),1);self.assertEqual(V.columns.tolist(),X.columns.tolist())

    def test_saved_classifier_is_loaded_without_refitting(self):
        progress={'models':{'done':{'path':'models/done.txt'}}}
        with patch.object(run.lgb,'Booster',return_value='saved') as load,patch.object(c,'fit_classifier') as fit:
            value=run.fitted_model(Path('/tmp'),progress,'done',None,None,None,42,3,{'prediction_threads':20})
        self.assertEqual(value,'saved');fit.assert_not_called();load.assert_called_once()

    def test_outer_completed_window_skips_all_training(self):
        outer={'evals':[{'fold':'A1'}]};progress={'predictions':{'A1__42':{}}}
        with patch.object(run.shared,'split_job') as split:
            run.outer_predictions(None,progress,None,None,outer,None,None,{'seeds':[42]})
        split.assert_not_called()

    def test_outer_prediction_serialization_replays_missing_clock_and_all_rows(self):
        class Stub:
            def __init__(self,model_file,**kwargs):self.binary='binary' in model_file
            def predict(self,X,**kwargs):
                return np.array([.2,.9]) if self.binary else np.array([[.7,.2,.1],[.9,.05,.05]])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir();out=root/'candidate';out.mkdir()
            raw=raw_frame([6000,np.nan,6000],['LIRF','LIRF','EGLL']);raw['mvt_id']=[1.,2.,3.]
            raw['taxi_sec']=[1000.,900.,1000.]
            f=raw.copy();f['airport']=f.ADEP_mvt;f['mvt_time']=f.MVT_TIME_UTC_mvt;f['y']=f.taxi_sec
            f['has_flight_plan']=False;f['arrival_history_available']=True;f['arrival_history_days_available']=7
            f['raw_flight_id_missing']=True;f['fold']='A1';f['month']='2025-07';f['seed']=42
            f['pred_noid_cap']=[2000.,900.,1000.];f['ordinary']=[1000.,900.,1000.]
            f['p']=[.2,0,0];f['blended']=[2000.,900.,1000.];f['gap_sec']=[6000,np.nan,6000];f['rule_triggered']=False
            f.to_parquet(parent/'parent.parquet',index=False)
            source={'parameters':{'window':{'late_mean':1000}},'folds':{'A1':{'predictions':{'42':'parent.parquet'}}}}
            parent_progress={'models':{'window__classifier_stand__42':{'path':'binary.txt'}}}
            progress={'models':{'window__c3__42':{'path':'three.txt'}},'predictions':{},'windows':{},'folds':{},
                      'calibrators':{'window__42':{'path':'cal.json'}}}
            fitted={str(k):c.fit_calibrator(np.full((2,k),1/k),[0,0],[True,True]) for k in (2,3)}
            (out/'cal.json').write_text(json.dumps({'fitted':fitted}))
            V=pd.DataFrame({'mvt_minus_sched_sec':[6000.,np.nan]});vr=raw.iloc[:2].copy()
            outer={'id':'window','freeze':'2025-06-30','evals':[{'fold':'A1','month':'2025-07'}]}
            with patch.object(run,'PARENT',parent),patch.object(run.shared,'split_job',return_value=(raw,raw,{'2025-07':(raw,raw)})),\
                 patch.object(run,'training_inputs',return_value=(V,np.array([1000.,900.]),{},vr)),\
                 patch.object(run,'validation_inputs',return_value=(V,vr)),patch.object(run,'fitted_model',return_value=object()),\
                 patch.object(run.lgb,'Booster',Stub),contextlib.redirect_stdout(io.StringIO()):
                run.outer_predictions(out,progress,source,parent_progress,outer,None,None,
                    {'seeds':[42],'smoke':False,'prediction_threads':1})
            saved=pd.read_parquet(out/'oof_A1_seed42.parquet')
            self.assertEqual(saved.mvt_id.tolist(),[1.,2.,3.]);self.assertEqual(len(progress['folds']),1)
            self.assertEqual(progress['predictions']['A1__42']['binary_replay_max_difference'],0)
            for col in c.COLUMNS.values():
                self.assertEqual(saved.loc[2,col],1000.);self.assertTrue(np.isfinite(saved[col]).all())
            np.testing.assert_allclose(saved.pred_binary_uncal,saved.blended,atol=1e-8)
            self.assertFalse(saved.day_eligible.any())

    def test_dry_run_creates_nothing_and_loads_no_dataset(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);parent=root/'parent';parent.mkdir()
            (parent/'result.json').write_text(json.dumps({'config':{}}));(parent/'progress.json').write_text('{}')
            out=root/'v3_preview'
            with patch.object(run,'PARENT',parent),patch.object(run,'RUNS',root),patch.object(run.shared,'load_inputs') as load,contextlib.redirect_stdout(io.StringIO()) as stream:
                run.main(['--output-dir',str(out),'--dry-run'])
            data=json.loads(stream.getvalue());self.assertEqual(data['new_boosters'],126)
            self.assertEqual(data['ordinary_fits'],0);self.assertEqual(data['uploads'],0)
            self.assertEqual(len(data['config']['jobs']),6);self.assertFalse(out.exists());load.assert_not_called()


class ResultTests(unittest.TestCase):
    def test_ledger_updates_are_idempotent_and_preserve_official_scores(self):
        original='# Submissions\n\n| # | Name | Recipe | A | B | Combined | LB | Status |\n|---|---|---|---|---|---|---|---|\n| 37 | Existing | original | 1.1 | 2.2 | 3.3 | **280.17** | v37 |\n\nKeep this unrelated note.\n'
        result={name:{'scores':dict(zip(report.SCORES,[200.12346,190.23456,195.34567]))} for name in ('binary','three')}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'SUBMISSIONS.md';path.write_text(original)
            report.update_submissions(path,'test_run',result,'run/report.md');first=path.read_text()
            report.update_submissions(path,'test_run',result,'run/report.md');self.assertEqual(first,path.read_text())
            self.assertIn('| 38 | **Rome-binary-cal**',first);self.assertIn('| 39 | **Rome-three-cal**',first)
            self.assertIn('200.1235 | 190.2346 | 195.3457 | — | Not submitted',first)
            self.assertIn('**280.17** | v37',first);self.assertIn('Keep this unrelated note.',first)
            result['three']['scores']['R_A']=float('nan')
            with self.assertRaises(ValueError):report.update_submissions(path,'test_run',result,'run/report.md')
            self.assertEqual(first,path.read_text())

    def test_ledger_preserves_weighted_column_and_marks_unscored_additions(self):
        original=('| Order | Name | Recipe | A | B | Combined | Weighted CV | LB | Status |\n'
                  '|---|---|---|---|---|---|---|---|---|\n'
                  '| 37 | Existing | original | 1.1 | 2.2 | 3.3 | 267.5814 | **279.3805** | v38 |\n')
        results={name:{'scores':dict(zip(report.SCORES,[200.,190.,195.]))} for name in ('binary','three')}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'SUBMISSIONS.md';path.write_text(original)
            report.update_submissions(path,'test_run',results,'run/report.md')
            first=path.read_text()
            report.update_submissions(path,'test_run',results,'run/report.md')
            self.assertEqual(first,path.read_text())
        self.assertIn('| 267.5814 | **279.3805** | v38 |',first)
        rows=[[c.strip() for c in line.split('|')[1:-1]] for line in first.splitlines()
              if line.startswith('| 38 |') or line.startswith('| 39 |')]
        self.assertEqual(len(rows),2)
        for row in rows:
            self.assertEqual(len(row),9)
            self.assertEqual(row[-3:],['-','—','Not submitted'])

    def test_partial_report_cannot_touch_submission_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);path=root/'SUBMISSIONS.md';path.write_text('unchanged')
            record={'folds':{},'config':{'seeds':[42,142,242],'sample':1}}
            with self.assertRaisesRegex(ValueError,'Only complete'):report.finish(record,root,root)
            self.assertEqual(path.read_text(),'unchanged')

    def test_diagnostics_average_squared_losses_not_ensemble_predictions(self):
        frame=pd.DataFrame(dict(mvt_id=[1,1],weight=[.5,.5],y=[1000,1000],gap_sec=[60000,60000],
            raw_flight_id_missing=[True,True],day_eligible=[False,False],pred_noid_cap=[8000,1000]))
        for col in c.COLUMNS.values():frame[col]=frame.pred_noid_cap
        for stage,k in report.PROBABILITIES.items():
            p=np.array([.7,.3] if k==2 else [.6,.3,.1])
            for i in range(k):frame[f'{stage}_p{i}']=p[i]
        d=report.diagnostic_summary(frame);stats=d['cohorts']['target_at_most_2h']
        self.assertEqual(stats['unique_flights'],1);self.assertEqual(stats['weighted_rows'],1)
        self.assertAlmostEqual(stats['predictions']['binary']['rmse'],np.sqrt(49e6/2))
        self.assertEqual(stats['predictions']['binary']['above_2h_fraction'],.5)
        self.assertEqual(d['cohorts']['day_class']['unique_flights'],0)
        self.assertAlmostEqual(d['probabilities']['binary_raw']['all']['logloss'],-np.log(.7))
        self.assertAlmostEqual(d['probabilities']['binary_raw']['all']['multiclass_brier'],.18)

    def test_synthetic_scoring_report_and_ledger_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp);directory=root/'VALIDATION/runs/v3_synthetic';directory.mkdir(parents=True)
            record=dict(run_id='v3_synthetic',baseline_id='parent',folds={},windows={},
                fits={'new_classifiers':126},protocol=protocol.VERSION,score_version=protocol.SCORE_VERSION,
                ranking_weights={'jan_fraction':.5},data_identity=[],prediction_column='pred_noid_cap',
                config=dict(seeds=[42,142,242],sample=1,smoke=False,train_months=6,early_stopping_rounds=0,
                            bootstrap_reps=40,bootstrap_seed=0,block_days=1))
            for fold in protocol.A_FOLDS+protocol.B_FOLDS:
                info=asdict(fold);info['train_start']=str(protocol.train_start(fold,6).date());info['predictions']={}
                for seed in record['config']['seeds']:
                    f=pd.DataFrame(dict(mvt_id=np.arange(3)+int(fold.val[-2:])*10,
                        mvt_time=pd.to_datetime([fold.val+'-10']*3,utc=True),airport=['LIRF','LIRF','EGLL'],
                        y=[1000.,87000.,1500.],has_flight_plan=[True,False,True],
                        arrival_history_available=True,arrival_history_days_available=7,
                        raw_flight_id_missing=[False,True,False],fold=fold.name,seed=seed,
                        day_eligible=[False,True,False],gap_sec=[1000.,60000.,1000.],
                        pred_noid_cap=[1200.,60000.,1400.]))
                    for name,column in c.COLUMNS.items():
                        f[column]=[1150.,70000. if name.startswith('three') else 65000.,1400.]
                    for stage,k in report.PROBABILITIES.items():
                        for i in range(k):f[f'{stage}_p{i}']=1/k
                    path=f'{fold.name}_{seed}.parquet';f.to_parquet(directory/path,index=False)
                    info['predictions'][str(seed)]=path
                record['folds'][fold.name]=info
            parent=root/'VALIDATION/runs/parent';parent.mkdir()
            control_report=run.reporting.report(record,directory)
            run.write_json(parent/'result.json',{'report':control_report})
            ledger=root/'SUBMISSIONS.md';ledger.write_text('| 37 | Existing | recipe | 1 | 2 | 3 | 4 | v37 |\n')
            progress={'comparisons':{}}
            for candidate,baseline,block in [('binary','control',1),('three','control',1),
                ('binary','binary_uncal',1),('three','binary',1),('three','three_uncal',1),
                ('binary_cal_hard','control',1),('binary_uncal','control',1),
                ('binary','control',3),('three','control',3)]:
                run.compare(record,directory,progress,candidate,baseline,block)
            record['comparisons']=progress['comparisons'];outputs=report.finish(record,directory,root)
            self.assertEqual(outputs['ledger'],'SUBMISSIONS.md');self.assertIn('Rome-three-cal',ledger.read_text())
            scorecard=json.loads((directory/'scorecard.json').read_text())
            self.assertLess(scorecard['three']['scores']['R_A'],scorecard['binary']['scores']['R_A'])
            diagnostics=json.loads((directory/'diagnostics.json').read_text())
            self.assertEqual(diagnostics['R_A']['cohorts']['day_class']['unique_flights'],6)
            self.assertEqual(diagnostics['R_B_long']['cohorts']['day_class']['unique_flights'],2)
            self.assertAlmostEqual(diagnostics['R_B_long']['cohorts']['day_class']['weighted_rows'],2)
            self.assertIn('Probability and failure diagnostics',(directory/'report.md').read_text())
            self.assertEqual(len(pd.read_csv(directory/'day_cases.csv')),27)


if __name__=='__main__':unittest.main()
