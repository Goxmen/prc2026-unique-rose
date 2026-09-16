"""Held-out static-field accuracy with causal/frozen training references."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'pipeline'))
from common import connect, TRAIN_GLOB
from movement_data import load_phase
from modeling import month_rows, period_rows
from recovery import public, recover_exact
from recovery_history import POLICY, impute_static
from records import write_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    con = connect()
    con.execute('SET threads=2')
    dep = load_phase(con, TRAIN_GLOB, 'DEP')
    con.close()
    train = period_rows(dep, '2025-01-01', '2025-06-30')
    result = {'reference': 'Raw matched departures in January–June only', 'groups': {}}
    cases = (['training_causal'] + [f'2025-{m:02}' for m in range(7, 13)]
             + [f'2025-{m:02}_delayed_ge_3h' for m in range(7, 13)])
    for name in cases:
        delayed = name.endswith('delayed_ge_3h')
        pool = train if name == 'training_causal' else month_rows(dep, name[:7])
        known = pool.loc[pool.FLIGHT_ID_mvt.notna() & pool.ADEP_flt.notna()]
        if delayed:
            known = known.loc[(known.MVT_TIME_UTC_mvt-known.SCHED_TIME_UTC_mvt).dt.total_seconds().ge(10800)]
        probes = known.sample(min(2000 if delayed else 5000, len(known)), random_state=42).sort_values('mvt_id')
        ref = train.loc[~train.mvt_id.isin(probes.mvt_id)]
        masked = public(probes).copy()
        for field in POLICY:
            masked.loc[:, field] = None
        _, audit = recover_exact(masked, masked.iloc[:0])
        filled, audit = impute_static(masked, public(ref), audit, causal=name=='training_causal')
        fields = {}
        for field in POLICY:
            comparable = probes[field].notna()
            recovered = filled[field].notna() & comparable
            correct = filled[field].eq(probes[field]) & recovered
            fields[field] = {'known': int(comparable.sum()), 'filled': int(recovered.sum()),
                             'correct': int(correct.sum()), 'incorrect': int((recovered & ~correct).sum()),
                             'accuracy': float(correct.sum()/recovered.sum()) if recovered.any() else None}
        result['groups'][name] = {'probes': len(probes), 'fields': fields}
        audit.to_parquet(args.output.with_name(f'{args.output.stem}_{name}.parquet'), index=False)
        write_json(args.output, result)
        print(name, fields, flush=True)


if __name__ == '__main__':
    main()
