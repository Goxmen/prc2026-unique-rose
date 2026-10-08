"""Leaderboard files for the nearest-neighbour training sets: v58 with every part refitted on 2025's training
flights nearest to each 2026 test month.

The user's idea (2026-10-07, practice runner `similar_rows_jan_jul.py`): every 2026 flight to be predicted picks
its K nearest 2025 training flights at the same airport; each part is refitted on the picked flights only,
once per test month, and predicts that month. The distance is the practice runner's (every regressor input,
numbers as percentiles, categories as match/mismatch, each weighted by its gain share in a fast LightGBM
fitted on all 2025 training flights). The user chose to pick K on the leaderboard, since the practice test
cannot copy the real setting (2025's own January and July are in the real training data).

Parts and settings are v54's (`hpo_submit.py`: networks N09 x 15, main LightGBM LB1, per-airport P04, CatBoost
K07), fitted by v54's own functions; blend 0.71 x networks + 0.29 x mean(trees) as v58, with v54's Rome
probabilities, corrections and four-hour rule.

Run: TORCH_DISABLE_NATIVE_JIT=1 .venv/bin/python VALIDATION/similar_rows_submit.py knn
     .venv/bin/python VALIDATION/similar_rows_submit.py fit --part {nn,catboost,lgb,airport} --k 10
     .venv/bin/python VALIDATION/similar_rows_submit.py prepare --k 10 --version 59
"""
import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time

os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT', '1')

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline'), str(ROOT / 'VALIDATION/tools')]
import hpo_submit as hps
import ideas_jan_jul as ideas
import nn_improve_submit as nis
import overnight_run
from records import RUNS, runtime_identity, sha256, source_snapshot, write_json
from rome_hybrid_submit import MONTHS, now, submission
import similar_rows_jan_jul as sr
from submit import fill_template

OUT = RUNS / 'v3_20261007_similar_rows_submit'
W_NN = 0.71                     # v58
V54 = hps.OUT / 'submission_v54'


def log(message):
    print(f'[{time.strftime("%H:%M:%S")}] {message}', flush=True)


def month_slices(data):
    out, offset = {}, 0
    for m in MONTHS:
        n = len(data['months'][m]['raw'])
        out[m] = slice(offset, offset + n)
        offset += n
    return out


def run_knn():
    data = pd.read_pickle(hps.INPUTS)
    cols = data['names'] + ideas.COMBO
    weight = sr.input_weights(ideas, dict(X=data['Xn'], y=data['yn']), cols)
    log(f'weights: {list(weight.sort_values(ascending=False).head(8).round(3).items())}')
    tap = data['Xn'].airport.astype(str).to_numpy()
    long = data['yn'] > 3600
    record = dict(weights={c: round(float(w), 5) for c, w in weight.sort_values(ascending=False).items()})
    for m, sl in month_slices(data).items():
        started = time.monotonic()
        V = data['Vn'].iloc[sl]
        vap = V.airport.astype(str).to_numpy()
        first = np.zeros(len(tap), np.int32)
        for ap in np.unique(vap):
            t = tap == ap
            tr, te = sr.distance_space(data['Xn'].loc[t, cols], V.loc[vap == ap, cols], weight)
            first[t], _, _, _ = sr.neighbours(tr, te)
        record[m] = dict(minutes=(time.monotonic() - started) / 60, test_rows=len(V),
                         kept={f'k{k}': dict(all=round(float(((first > 0) & (first <= k)).mean()), 4),
                                             over_1h=round(float(((first > 0) & (first <= k))[long].mean()), 4))
                               for k in sr.KNN_K})
        log(f"knn {m}: {record[m]['minutes']:.1f} min, kept {record[m]['kept']}")
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({'first_pick': first}).to_parquet(OUT / f'knn_{m}.parquet')
    write_json(OUT / 'knn.json', record)


def month_data(data, m, sl, k):
    """v54's full-year inputs restricted to month m's K-nearest training flights and m's ranking rows."""
    first = pd.read_parquet(OUT / f'knn_{m}.parquet').first_pick.to_numpy()
    if len(first) != len(data['Xn']):
        raise ValueError('neighbour rows differ from training rows')
    keep = (first > 0) & (first <= k)
    return dict(Xn=data['Xn'].loc[keep], yn=data['yn'][keep], tmonth=data['tmonth'][keep], Vn=data['Vn'].iloc[sl],
                rank_ids=data['rank_ids'][sl], names=data['names']), int(keep.sum())


def fit(part, k):
    started = time.monotonic()
    path = OUT / 'parts' / f'{part}_k{k}.parquet'
    if path.exists():
        raise ValueError(f'{path.name} exists; inspect it before refitting')
    spec = hps.searched()[part]
    data = pd.read_pickle(hps.INPUTS)
    hps.WORK = OUT / 'work' / f'k{k}' / part          # v54's functions save their models under WORK/models
    frames, record = [], {}
    for m, sl in month_slices(data).items():
        sub, rows = month_data(data, m, sl, k)
        hps.WORK = OUT / 'work' / f'k{k}' / part / m
        if part == 'nn':
            frame, rec = hps.network_part(sub, spec)
        elif part == 'catboost':
            frame, rec = hps.catboost_part(sub, spec)
        else:
            with threadpool_limits(limits=1, user_api='blas'):
                frame, rec = hps.gbm_part(sub, part, spec)
        record[m] = dict(rec, training_rows=rows)
        frames.append(frame)
        log(f'{part} k{k} {m}: {rows:,} training rows')
    frame = pd.concat(frames, ignore_index=True)
    if not np.array_equal(frame.mvt_id.to_numpy(), data['rank_ids']):
        raise ValueError(f'{part}: rows differ from the ranking rows')
    path.parent.mkdir(parents=True, exist_ok=True)
    overnight_run.save_frame(path, frame)
    import torch
    write_json(path.with_suffix('.json'), dict(trial=spec['trial'], k=k, months=record, finished_at=now(), host=socket.gethostname(),
                                               minutes=(time.monotonic() - started) / 60,
                                               device=torch.cuda.get_device_name() if part in ('nn', 'catboost') else 'cpu'))
    log(f'{part} k{k} done in {(time.monotonic() - started) / 60:.1f} min')


def part_values(part, k, rank_ids):
    f = pd.read_parquet(OUT / 'parts' / f'{part}_k{k}.parquet')
    if not np.array_equal(f.mvt_id.to_numpy(), rank_ids):
        raise ValueError(f'{part} k{k}: rows differ from the ranking rows')
    cols = [c for c in f.columns if c != 'mvt_id']
    if part == 'nn' and len(cols) != len(hps.ids.GROUPS) * hps.hpo.NN_FINAL_NETS:
        raise ValueError('nn: expected 15 networks')
    return f[cols].to_numpy(float).mean(axis=1)


def prepare(k, version):
    started = time.monotonic()
    directory = OUT / f'submission_v{version}'
    if directory.exists() or submission(version).exists():
        raise ValueError(f'v{version} is already prepared; inspect it before continuing')
    v58 = ts_read(RUNS / 'v3_20261006_blend_weight_submit' / 'submission_v58' / 'submission_v58.json')
    if sha256(submission(58)) != v58['sha256']:
        raise ValueError('v58 file differs from its manifest')
    data = pd.read_pickle(hps.INPUTS)
    rank_ids = data['rank_ids']
    parts = {p: part_values(p, k, rank_ids) for p in hps.PARTS}
    nn_all, gbm_all = parts['nn'], np.mean([parts[p] for p in hps.GBM], axis=0)
    directory.mkdir(parents=True)
    pieces, summary = [], {}
    for m, sl in month_slices(data).items():
        d = data['months'][m]
        saved = pd.read_parquet(V54 / f'ranking_{m}.parquet')
        if not np.array_equal(saved.mvt_id.to_numpy(), d['raw'].mvt_id.to_numpy()):
            raise ValueError(f'{m}: v54 rows differ')
        base = W_NN * nn_all[sl] + (1 - W_NN) * gbm_all[sl]
        frame, masks = nis.final(base, saved.p.to_numpy(float), d['V'], d['raw'], data['parameters'], data['fitted'])
        frame['nn_base'], frame['gbm_base'], frame['blend_base'], frame['month'] = nn_all[sl], gbm_all[sl], base, m
        for q in hps.PARTS:
            frame[f'{q}_part'] = parts[q][sl]
        overnight_run.save_frame(directory / f'ranking_{m}.parquet', frame)
        summary[m] = dict(rows=len(frame), correction_masks=masks,
                          part_vs_v54={q: dict(corr=float(np.corrcoef(parts[q][sl], saved[f'{q}_part'])[0, 1]),
                                               mean_change=float((parts[q][sl] - saved[f'{q}_part']).mean()))
                                       for q in hps.PARTS})
        pieces.append(frame)
    frame = pd.concat(pieces, ignore_index=True).set_index('mvt_id').reindex(rank_ids).reset_index()
    if not np.isfinite(frame.pred).all() or (frame.pred < 0).any() or frame.rule_triggered.sum() != 43:
        raise ValueError('Invalid predictions or changed rule eligibility')
    template = pq.read_table(nis.TEMPLATE)
    table = fill_template(template, frame.mvt_id, frame.pred.to_numpy())
    nis.transfer.validate_submission(table, template)
    path = submission(version)
    pq.write_table(table, path)
    mine = table.to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt
    other = pq.read_table(submission(58)).to_pandas().set_index('MVT_ID_mvt').TAXITIME_SEC_mvt.reindex(mine.index)
    a, b = mine.to_numpy(float), other.to_numpy(float)
    against = dict(v58=dict(mean_change=float((a - b).mean()), mean_abs_change=float(np.abs(a - b).mean()),
                            rms_change=float(np.sqrt(((a - b) ** 2).mean())), corr=float(np.corrcoef(a, b)[0, 1])))
    manifest = dict(version=version, state='prepared', file=str(path.relative_to(ROOT)), sha256=sha256(path),
                    parent='v58', change=f'every part refitted per test month on the {k} nearest 2025 training '
                                         f'flights of each test flight', k=k, w_nn=W_NN,
                    parts={p: ts_read(OUT / 'parts' / f'{p}_k{k}.json') for p in hps.PARTS},
                    neighbours=ts_read(OUT / 'knn.json'), summary=summary, against=against,
                    runtime=runtime_identity(), source=source_snapshot(directory),
                    minutes=(time.monotonic() - started) / 60, prepared_at=now())
    write_json(directory / f'submission_v{version}.json', manifest)
    log(json.dumps(dict(summary=summary, against=against), indent=1))
    log(f'prepared {path}; not uploaded')


def ts_read(path):
    return json.loads(Path(path).read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('action', choices=['knn', 'fit', 'prepare'])
    ap.add_argument('--part', choices=hps.PARTS)
    ap.add_argument('--k', type=int)
    ap.add_argument('--version', type=int)
    args = ap.parse_args()
    if args.action == 'knn':
        run_knn()
    elif args.action == 'fit':
        fit(args.part, args.k)
    else:
        prepare(args.k, args.version)


if __name__ == '__main__':
    main()
