"""Mask known plans and their true twins to audit fold-scoped recovery precision."""

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness import load
from modeling import month_rows, period_rows
from recovery import PLAN_FIELDS, recover_frame, summarize_audit
from records import write_json
from common import connect


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--probes-per-month', type=int, default=5000)
    ap.add_argument('--mode', choices=['exact', 'relaxed'], default='exact')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    con = connect()
    con.execute('SET threads=2')
    dep, arr = load(con, 1.)
    con.close()
    train_dep = period_rows(dep, '2025-01-01', '2025-06-30')
    train_arr = period_rows(arr, '2025-01-01', '2025-06-30')
    results = {'mode': args.mode, 'reference_tables': 'Fitted on January–June only; July–December probes excluded.',
               'context': 'January–June plus each evaluation month separately', 'months': {}}
    if args.resume:
        results = json.loads(args.output.read_text())
        if results['mode'] != args.mode or any(v['probes'] != args.probes_per_month for v in results['months'].values()):
            raise ValueError('Resume requires the same recovery mode and probe count')
    for month in [f'2025-{m:02}' for m in range(7, 13)]:
        if month in results['months']:
            continue
        vd = month_rows(dep, month)
        pool = pd.concat([train_arr, month_rows(arr, month)], ignore_index=True)
        known = vd.loc[vd.FLIGHT_ID_mvt.notna() & vd.ADEP_flt.notna()
                       & vd.FLIGHT_ID_mvt.isin(pool.FLIGHT_ID_mvt)]
        # A deterministic random sample; truth used only to select/score masked probes.
        probes = known.sample(n=min(args.probes_per_month, len(known)), random_state=42).sort_values('mvt_id')
        masked = vd.copy()
        mask = masked.mvt_id.isin(probes.mvt_id)
        for field in PLAN_FIELDS:
            masked.loc[mask, field] = pd.NaT if pd.api.types.is_datetime64_any_dtype(masked[field]) else None
        masked.loc[mask, 'FLIGHT_ID_mvt'] = np.nan
        claims = pd.concat([train_dep, masked], ignore_index=True)
        truth = probes.set_index('mvt_id').FLIGHT_ID_mvt
        item = {'probes': len(probes)}
        for hidden in [False, True]:
            donors = pool.loc[~pool.FLIGHT_ID_mvt.isin(truth)] if hidden else pool
            _, audit = recover_frame(masked, donors, claims, mode=args.mode,
                                     reference_dep=train_dep, reference_arr=train_arr)
            audit = audit.loc[audit.mvt_id.isin(truth.index)].copy()
            audit['true_flight_id'] = audit.mvt_id.map(truth)
            audit['correct'] = audit.donor_flight_id.eq(audit.true_flight_id)
            found = audit.stage.ne('unrecovered')
            count = int(found.sum())
            result = dict(summarize_audit(audit), correct=int((found & audit.correct).sum()),
                          incorrect=int((found & ~audit.correct).sum()),
                          precision=float(audit.loc[found, 'correct'].mean()) if count else None,
                          returned_fraction=count / len(probes),
                          by_stage=audit.groupby('stage').apply(
                              lambda g: pd.Series({'probes': len(g),
                                  'returned': len(g) if g.name != 'unrecovered' else 0,
                                  'incorrect': int((~g.correct).sum()) if g.name != 'unrecovered' else 0}),
                              include_groups=False).reset_index().to_dict('records'),
                          by_airport=audit.groupby('ADEP_mvt').apply(
                              lambda g: pd.Series({'probes': len(g),
                                  'returned': int(g.stage.ne('unrecovered').sum()),
                                  'incorrect': int((g.stage.ne('unrecovered') & ~g.correct).sum())}),
                              include_groups=False).reset_index().to_dict('records'))
            label = 'true_twin_hidden' if hidden else 'plan_hidden'
            item[label] = result
            audit.to_parquet(args.output.with_name(f'{args.output.stem}_{month}_{label}.parquet'), index=False)
        results['months'][month] = item
        write_json(args.output, results)
        print(month, {key: {k:v for k,v in value.items() if k in ['recovered','correct','incorrect','precision','returned_fraction']}
                      for key,value in item.items() if isinstance(value, dict)}, flush=True)
        del pool, masked, claims
        gc.collect()


if __name__ == '__main__':
    main()
