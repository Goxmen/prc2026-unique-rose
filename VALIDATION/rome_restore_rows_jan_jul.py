"""Classifier-only restoration of rule-covered training rows; fixed R20/v58.

Separate from the ADSB.lol feature screen: no new inputs. The saved before_drop
matrix restores no-ID Rome departures with schedule gap >=4 hours to the Rome
classifier. Common inputs and labels must match the prepared R20 feature screen
exactly after its ADSB columns are removed. Ordinary model parts, final rules and
correction parameters stay v58. Fixed 1,700 rounds/seed42, no checkpoint tuning.
"""
import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import rome_adsblol_jan_jul as screen

ROOT=screen.ROOT
SOURCE=screen.OUT
OUT=screen.ts.RUNS/'v3_20261010_rome_restore_rows_jan_jul'
DESIGN=dict(parent='v58',part='Rome recording classifier',new_features=[],
    retained_classifier_extras=list(screen.ts.CLASSIFIER),seed=42,rounds=1700,trial='R20',
    network_share=.71,heldout_months=screen.ts.SPLITS,
    training_membership='all original before_drop rows for classifier only; ordinary remains unchanged',
    checkpoint_selection=False,exploratory_reused_development_months=True)


def prepare():
    (OUT/'inputs').mkdir(parents=True,exist_ok=True)
    for month in screen.ts.SPLITS:
        path=OUT/'inputs'/f'{month}.pkl'
        if path.exists():
            continue
        original=pd.read_pickle(SOURCE/'inputs'/f'{month}.pkl')
        names=[c for c in original['A'] if c not in screen.NEW]
        src=screen.ts.split_dir(screen.hpo.SPLITS_FROM,month)/'before_drop.pkl'
        before=pd.read_pickle(src)
        lirf=np.asarray(before['airport'])=='LIRF'
        X,y=before['X'].loc[lirf],np.asarray(before['y'],float)[lirf]
        raw=screen.ts.minimal_raw(np.full(len(X),'LIRF'),X.index)
        A=screen.signals.classifier_matrix(X,raw,screen.ts.KIND,screen.ts.CLASSIFIER)
        target=screen.rome_hybrid.substituted(X,y,raw).astype(float)
        restored=np.asarray(before['rule'],bool)[lirf]
        common=A.loc[~restored]
        assert len(A)==len(original['A'])+int(restored.sum())
        assert list(A)==names
        pd.testing.assert_frame_equal(common,original['A'][names],check_categorical=False,check_dtype=False)
        np.testing.assert_array_equal(target[~restored],original['target'])
        assert (X.loc[restored,screen.rome_hybrid.GAP]>=14400).all()
        assert (X.loc[restored,'clock_raw_flight_id_missing']==1).all()
        B=original['B'][names]
        temp=path.with_suffix('.tmp')
        pd.to_pickle(dict(A=A,B=B,target=target,use=original['use'],ids=original['ids'],
                          train_index=A.index.to_numpy(),restored=restored),temp)
        temp.replace(path)
        screen.write_json(path.with_suffix('.json'),dict(design=DESIGN,columns=list(A),rows=len(A),
            restored_rows=int(restored.sum()),restored_positive=int(target[restored].sum()),
            common_rows_verified=len(common),source_before_sha256=screen.sha256(src),
            original_matrix_sha256=screen.sha256(SOURCE/'inputs'/f'{month}.pkl'),
            prepared_sha256=screen.sha256(path),wrapper_sha256=screen.sha256(Path(__file__))))
        screen.log(f'{month}: restored {int(restored.sum())} classifier rows; {len(common):,} common rows identical')
        del before,X,A,B,original
        gc.collect()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','fit','score'])
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args()
    if args.threads<1:
        parser.error('--threads must be positive')
    # Reuse only the fixed fitting/scoring implementation, with independent
    # frozen design, input cache, output files and candidate name.
    screen.OUT,screen.DESIGN,screen.ARM=OUT,DESIGN,'restore_rule_rows'
    screen.ensure_design()
    with threadpool_limits(2):
        if args.stage=='prepare': prepare()
        elif args.stage=='fit': screen.fit(args.threads)
        else: screen.score()


if __name__=='__main__':
    main()
