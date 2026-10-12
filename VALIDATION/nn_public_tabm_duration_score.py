"""Frozen matched-duration comparison through the exact selected correction stack."""
import ast,inspect,textwrap
import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import nn_public_tabm_duration as run
import nn_public_tabm_pool3_stack as parent
import tabm_pool3_other_runway_combo_cv as combo

ROOT=run.ROOT
OUT=ROOT/'VALIDATION/runs/v3_20261012_nn_public_tabm_duration_stack'
ARMS=['selected','one_tabm','three_tabm','matched_duration','complete_schedule']
DESIGN=dict(training=run.DESIGN,formula=combo.DESIGN['formula'],
    primary='complete_schedule versus same-trajectory matched_duration',
    actual='complete_schedule versus retained three_tabm; exact existing pool/runway replay',
    scope='Replace seed42 only; retain1042/2042 and every correction/weight',
    no_weight_search=True,no_upload=True)


def pools(month,ids,new,members,sources):
    extra={}
    for name,stem,epochs in [('matched_duration','early',run.original.STOP[month]),('complete_schedule','tabm',24)]:
        path=run.OUT/'networks'/f'{month}_{stem}.json';rec=run.read(path)
        assert rec['identity']==run.art(run.MANIFEST) and rec['design']==run.DESIGN
        assert rec['month']==month and rec['epochs']==epochs and rec['seed']==42 and rec['all_held_reloaded_exact']
        for key in ['model','prediction','preprocessing']:run.verify(rec[key])
        f=pd.read_parquet(run.verify(rec['prediction']));np.testing.assert_array_equal(f.mvt_id,ids)
        sources.extend([run.art(path),rec['prediction'],rec['model']])
        extra[name]=np.mean([f.base.to_numpy(float),new[1],new[2]],axis=0)
    return dict(selected=np.mean(members,axis=0),one_tabm=np.mean([new[0],*members[1:]],axis=0),
        three_tabm=np.mean([new[k] for k in range(3)],axis=0),**extra)


def scoring_function():
    tree=ast.parse(textwrap.dedent(inspect.getsource(parent.score)))
    class Edit(ast.NodeTransformer):
        arms=0;comparisons=0
        def visit_Assign(self,node):
            if len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
                if node.targets[0].id=='arms':
                    self.arms+=1;return ast.parse('arms=pools(month,ids,new,members,sources)').body[0]
                if node.targets[0].id=='comparisons':
                    self.comparisons+=1
                    return ast.parse("comparisons=[('matched_duration','complete_schedule'),('three_tabm','complete_schedule'),('three_tabm','matched_duration')]").body[0]
            return self.generic_visit(node)
    edit=Edit();tree=edit.visit(tree);assert edit.arms==edit.comparisons==1
    scope=dict(parent.score.__globals__,OUT=OUT,ARMS=ARMS,DESIGN=DESIGN,pools=pools,__file__=__file__)
    exec(compile(ast.fix_missing_locations(tree),'<tabm-duration-scoring-pools-only>','exec'),scope)
    return scope['score']


def main():
    run.guard();assert run.read(run.OUT/'audit.json')['status']=='pass'
    scoring_function()()
    f=pd.read_parquet(OUT/'scored.parquet');ids=f.mvt_id
    old=pd.read_parquet(parent.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    for name in ['selected','one_tabm','three_tabm','y','month','day']:np.testing.assert_array_equal(f[name],old[name])
    p=pd.read_parquet(combo.first.P79/'cv.parquet').set_index('mvt_id').loc[ids]
    runway=pd.read_parquet(combo.first.RUNWAY/'scored.parquet').set_index('mvt_id').loc[ids]
    selected=pd.read_parquet(combo.OUT/'scored.parquet').set_index('mvt_id').loc[ids]
    out=f[['mvt_id','month','day','y']].copy()
    for name in ARMS[2:]:out[name]=combo.first.combine(p.pred,old.selected,f[name],runway.candidate)
    np.testing.assert_array_equal(out.three_tabm,selected.three_tabm_runway)
    scores={m:{c:float(np.sqrt(np.mean((g[c]-g.y)**2))) for c in ARMS[2:]} for m,g in out.groupby('month')}
    paired={name:combo.first.bootstrap()(out.day,out.month,(out[name]-out.y)**2,(out.complete_schedule-out.y)**2)
        for name in ['matched_duration','three_tabm']}
    raw=run.read(OUT/'result.json')
    gate=all(p['decision']=='improvement' for p in paired.values()) and all(
        g['complete_schedule']<g[k] for g in scores.values() for k in ['matched_duration','three_tabm']) and all(
        raw['paired']['complete_schedule_vs_'+k]['decision']=='improvement' for k in ['matched_duration','three_tabm'])
    out.to_parquet(OUT/'combined.parquet',index=False)
    with duckdb.connect(config={'threads':1}) as con:
        con.register('f',out)
        for name in ARMS[2:]:
            for m,value in con.execute(f'SELECT month,sqrt(avg(pow({name}-y,2))) FROM f GROUP BY month').fetchall():
                assert abs(value-scores[m][name])<1e-8
    value=dict(status='scored',training_identity=run.art(run.MANIFEST),stack=run.art(OUT/'result.json'),
        prediction=run.art(OUT/'combined.parquet'),scores=scores,
        cv={c:float(np.mean([g[c] for g in scores.values()])) for c in ARMS[2:]},paired=paired,gate_pass=gate,
        exact_old_controls_and_combined_replay=True,rows=len(out),
        sources=[run.art(p) for p in [__file__,run.__file__,parent.__file__,combo.__file__,combo.OUT/'scored.parquet',
            parent.OUT/'scored.parquet',combo.first.P79/'cv.parquet',combo.first.RUNWAY/'scored.parquet']],
        independent_model_and_score_audit_required=True,no_upload=True)
    run.write(OUT/'combined_result.json',value)
    print(dict(cv=value['cv'],paired=paired,gate_pass=gate),flush=True)


if __name__=='__main__':
    with threadpool_limits(1):main()
