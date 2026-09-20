"""Adversarial validation: what separates 2025 from the 2026 ranking set, and what
an experimental reweighting of validation does to the CV score.

Four discriminators, each answering a different question, plus a null guard:

  D0  null       Jan+Jul 2025 split at random          -- lands at AUC 0.5
  D1  naive      all 2025 departures vs ranking        -- the uncontrolled run
  D2  season     Jan+Jul 2025 vs Jan+Jul 2026          -- year effect, season held
  D3  within     Jan+Jul 2025 vs Nov+Dec 2025          -- season effect, year held

Read these with care. AUC measures how easily two sets are told apart, not how
much their differences cost taxi-out predictions, so D3 landing above D2 says
nothing about whether 2026 is harmless. D0 shows chance performance on random
labels inside 2025; it cannot rule out a cross-year calendar signal, because a
year column would pass it while making D1 and D2 trivial -- the guard for that is
EXCLUDED below, asserted at load time. Training gain is not evidence either: the
high-cardinality categoricals take most of the gain even in D0, so held-out
permutation importance is the measure reported here.

WEIGHTING. Each scored group gets its own density ratio, estimated from its own
source months against the ranking set: R_A from July-December, R_B_long from
November-December. An earlier version of this file estimated one ratio on
July-December and applied it to both, which weighted a November-December group
with the wrong source; that error is why the combined figure moved from 252.30 to
269.40. Estimation is cross-fitted over both classes, so no row is weighted by a
model that saw it and held-out probabilities exist for both classes, and the
source/target class-count correction is applied per half.

269.40 IS AN EXPERIMENTAL DIAGNOSTIC. It has not been validated as a leaderboard
forecast on any period where the answer is known, its calibration against ranking
proportions is imperfect, the feature set omits weather, traffic and history, and
weights cannot represent feature combinations absent from the source at all. The
remainder against the official score is unexplained; nothing here measures a
label-side effect.

No taxi-out fits, no uploads, no change to any recorded score. The September 20
audit in `tmp/adversarial_review_20260920/` was taken against the previous
revision of this file and is preserved there.
"""

import json
from pathlib import Path
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'VALIDATION'), str(ROOT / 'pipeline')]
from common import connect  # noqa: E402
from records import write_json  # noqa: E402
from reporting import group_loss, load_losses  # noqa: E402

DATA = ROOT / 'data'
REPORTS = ROOT / 'VALIDATION/reports'
RUN = ROOT / 'VALIDATION/runs/v3_20260919_artifact_experiments'
OUT = REPORTS / 'adversarial_shift.json'

CATEGORICAL = ['airport', 'ades', 'ac_type', 'operator', 'market', 'wake',
               'rule_mvt', 'rule_flt', 'ftype', 'stand', 'runway']
NUMERIC = ['hour', 'dow', 'no_flight_id', 'no_aobt3', 'no_lobt', 'no_iobt',
           'no_eobt1', 'no_stand', 'no_runway', 'no_type', 'no_operator',
           'mvt_minus_sched', 'mvt_minus_lobt', 'mvt_minus_iobt', 'mvt_minus_eobt1',
           'mvt_minus_aobt3', 'aobt3_minus_sched', 'eobt1_minus_lobt', 'iobt_minus_lobt',
           'type_mismatch', 'adep_mismatch', 'diverted']
FEATURES = CATEGORICAL + NUMERIC
EXCLUDED = ['ym', 'month', 'date', 'year']      # calendar fields, asserted absent
BALANCE = ['no_flight_id', 'no_aobt3', 'no_lobt', 'no_stand', 'no_operator',
           'no_type', 'type_mismatch', 'diverted']
SUPPORT_KEYS = [['airport', 'stand'], ['airport', 'stand', 'runway'], ['airport', 'ades']]
PARAMS = dict(objective='binary', learning_rate=0.05, num_leaves=63, verbose=-1,
              min_data_per_group=100, cat_smooth=10, num_threads=6,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1)
ROUNDS = 300
SEED = 42
EPS = 1e-6
PERMUTATION_ROWS = 200_000
VAL_MONTHS = ['2025-07', '2025-08', '2025-09', '2025-10', '2025-11', '2025-12']
LONG_MONTHS = ['2025-11', '2025-12']
RANK_MONTHS = ['2026-01', '2026-07']
OFFICIAL = 279.3805
GROUP_SOURCE = {'R_A': VAL_MONTHS, 'R_B_long': LONG_MONTHS}
GROUP_FOLDS = {'R_A': ['A1', 'A2', 'A3', 'A4', 'A5', 'A6'], 'R_B_long': ['B3', 'B4', 'B7']}

QUERY = """
WITH src AS (
    SELECT * FROM read_parquet($training)
    UNION ALL BY NAME
    SELECT * FROM read_parquet($ranking)
)
SELECT
    MVT_ID_mvt AS mvt_id,
    strftime(MVT_TIME_UTC_mvt, '%Y-%m') AS ym,
    ADEP_mvt AS airport, ADES_mvt AS ades,
    AIRCRAFT_TYPE_mvt AS ac_type, AIRCRAFT_OPERATOR_flt AS operator,
    MARKET_SEGMENT_flt AS market, WK_TBL_CAT_flt AS wake,
    FLIGHT_RULE_mvt AS rule_mvt, FLIGHT_RULE_flt AS rule_flt,
    FLIGHT_TYPE_flt AS ftype, STAND_mvt AS stand, RUNWAY_mvt AS runway,
    hour(MVT_TIME_UTC_mvt) AS hour, dayofweek(MVT_TIME_UTC_mvt) AS dow,
    (FLIGHT_ID_mvt IS NULL)::INT AS no_flight_id,
    (AOBT_3_flt IS NULL)::INT AS no_aobt3,
    (LOBT_flt IS NULL)::INT AS no_lobt,
    (IOBT_flt IS NULL)::INT AS no_iobt,
    (EOBT_1_flt IS NULL)::INT AS no_eobt1,
    (STAND_mvt IS NULL)::INT AS no_stand,
    (RUNWAY_mvt IS NULL)::INT AS no_runway,
    (AIRCRAFT_TYPE_mvt IS NULL)::INT AS no_type,
    (AIRCRAFT_OPERATOR_flt IS NULL)::INT AS no_operator,
    epoch(MVT_TIME_UTC_mvt - SCHED_TIME_UTC_mvt) AS mvt_minus_sched,
    epoch(MVT_TIME_UTC_mvt - LOBT_flt) AS mvt_minus_lobt,
    epoch(MVT_TIME_UTC_mvt - IOBT_flt) AS mvt_minus_iobt,
    epoch(MVT_TIME_UTC_mvt - EOBT_1_flt) AS mvt_minus_eobt1,
    epoch(MVT_TIME_UTC_mvt - AOBT_3_flt) AS mvt_minus_aobt3,
    epoch(AOBT_3_flt - SCHED_TIME_UTC_mvt) AS aobt3_minus_sched,
    epoch(EOBT_1_flt - LOBT_flt) AS eobt1_minus_lobt,
    epoch(IOBT_flt - LOBT_flt) AS iobt_minus_lobt,
    (AIRCRAFT_TYPE_mvt IS DISTINCT FROM AIRCRAFT_TYPE_flt)::INT AS type_mismatch,
    (ADEP_mvt IS DISTINCT FROM ADEP_flt)::INT AS adep_mismatch,
    (ADES_FILED_flt IS DISTINCT FROM ADES_flt)::INT AS diverted
FROM src
WHERE PHASE_mvt = 'DEP'
"""


def load():
    con = connect()
    con.execute('SET threads=6')
    df = con.execute(QUERY, {
        'training': [str(p) for p in sorted(DATA.glob('training_*.parquet'))],
        'ranking': [str(DATA / 'ranking.parquet')]}).df()
    for c in CATEGORICAL:                      # one shared code space for both years
        df[c] = df[c].astype('category')
    for c in NUMERIC:
        df[c] = df[c].astype('float32')
    df['year'] = df.ym.str.slice(0, 4)
    assert not set(FEATURES) & set(EXCLUDED), 'a calendar field reached the feature set'
    assert df.mvt_id.notna().all() and not df.mvt_id.duplicated().any()
    return df


def fit(train, label, seed=SEED):
    ds = lgb.Dataset(train[FEATURES], label=label, categorical_feature=CATEGORICAL,
                     free_raw_data=False)
    return lgb.train({**PARAMS, 'seed': seed, 'data_random_seed': seed}, ds, ROUNDS)


def permutation_importance(model, frame, y, rng, top=12):
    """Held-out importance: AUC lost when one column is shuffled. Training gain is not used."""
    if len(frame) > PERMUTATION_ROWS:
        pick = rng.choice(len(frame), PERMUTATION_ROWS, replace=False)
        frame, y = frame.iloc[pick], y[pick]
    x = frame[FEATURES].copy()                 # one copy; columns are restored in place
    base = float(roc_auc_score(y, model.predict(x)))
    drops = []
    for column in FEATURES:
        original = x[column].copy()
        x[column] = original.array[rng.permutation(len(x))]   # keeps categorical dtype
        drops.append((column, base - float(roc_auc_score(y, model.predict(x)))))
        x[column] = original
    drops.sort(key=lambda kv: -kv[1])
    return {'holdout_auc': base, 'n_rows': int(len(x)),
            'top_features': [{'feature': f, 'auc_drop': float(d)} for f, d in drops[:top]]}


def discriminate(df, name, question, neg, pos, permute=True):
    """Fit on a random 70%, score AUC and held-out permutation importance on the rest."""
    part = pd.concat([df.loc[neg], df.loc[pos]])
    y = np.r_[np.zeros(int(neg.sum())), np.ones(int(pos.sum()))]
    rng = np.random.default_rng(SEED)
    holdout = rng.random(len(part)) < 0.30
    t0 = time.time()
    model = fit(part[~holdout], y[~holdout])
    auc = float(roc_auc_score(y[holdout], model.predict(part[holdout][FEATURES])))
    out = {'name': name, 'question': question, 'auc': auc,
           'n_negative': int(neg.sum()), 'n_positive': int(pos.sum())}
    if permute:
        out['permutation'] = permutation_importance(
            model, part[holdout].reset_index(drop=True), y[holdout], rng)
    print(f'  {name}: AUC {auc:.4f}  ({int(neg.sum()):,} vs {int(pos.sum()):,} rows, '
          f'{time.time() - t0:.0f}s)')
    return out


def calibration(p, y, bins=10):
    """Reliability of the held-out probabilities -- a different question from balance."""
    order = np.argsort(p)
    edges = np.array_split(order, bins)
    table, ece = [], 0.0
    for part in edges:
        mean_p, observed = float(p[part].mean()), float(y[part].mean())
        table.append({'n': len(part), 'mean_predicted': mean_p, 'observed': observed})
        ece += len(part) / len(p) * abs(mean_p - observed)
    return {'expected_calibration_error': float(ece),
            'brier': float(brier_score_loss(y, p)), 'bins': table}


def cross_fitted(df, source, target, name):
    """Density ratio for `source`, cross-fitted over both classes.

    Both source and target are halved. Each model trains on the complementary
    halves and predicts the held-out half of both, so every source row gets an
    out-of-sample propensity and held-out probabilities exist for both classes.
    The class-count correction uses the counts of the half that trained the model.
    """
    src, tgt = df.index[source], df.index[target]
    rng = np.random.default_rng(SEED)
    src_half = rng.random(len(src)) < 0.5
    tgt_half = rng.random(len(tgt)) < 0.5
    out = pd.DataFrame(index=src, columns=['p', 'odds', 'ratio', 'half'], dtype=float)
    held_p, held_y, counts = [], [], []
    for k, (sk, tk) in enumerate([(src_half, tgt_half), (~src_half, ~tgt_half)]):
        t0 = time.time()
        model = fit(pd.concat([df.loc[src[~sk]], df.loc[tgt[~tk]]]),
                    np.r_[np.zeros(int((~sk).sum())), np.ones(int((~tk).sum()))])
        ps = np.clip(model.predict(df.loc[src[sk], FEATURES]), EPS, 1 - EPS)
        pt = np.clip(model.predict(df.loc[tgt[tk], FEATURES]), EPS, 1 - EPS)
        correction = float((~sk).sum()) / float((~tk).sum())   # per-half prior correction
        out.loc[src[sk], 'p'] = ps
        out.loc[src[sk], 'odds'] = ps / (1 - ps)
        out.loc[src[sk], 'ratio'] = ps / (1 - ps) * correction
        out.loc[src[sk], 'half'] = k
        held_p.append(np.r_[ps, pt])
        held_y.append(np.r_[np.zeros(len(ps)), np.ones(len(pt))])
        counts.append({'half': k, 'source_train': int((~sk).sum()), 'target_train': int((~tk).sum()),
                       'source_holdout': int(sk.sum()), 'target_holdout': int(tk.sum()),
                       'class_count_correction': correction})
        print(f'  {name} half {k}: correction {correction:.4f}, {time.time() - t0:.0f}s')
    assert out[['p', 'odds', 'ratio']].notna().all().all(), f'{name}: a source row has no propensity'
    out['mvt_id'] = df.loc[src, 'mvt_id'].to_numpy()
    return (out.set_index('mvt_id'), counts,
            calibration(np.concatenate(held_p), np.concatenate(held_y)))


def concentration(sq, w):
    """How much of the weighted error, and of the increase, sits in very few flights."""
    share = w / w.sum()
    weighted = share * sq
    delta = sq * (share - 1 / len(sq))          # contribution to the change in mean squared error
    order = np.argsort(-delta)
    positive = delta.clip(min=0)
    top1pct = max(1, len(sq) // 100)
    return {
        'ess_pct': float(100 * w.sum() ** 2 / np.sum(w ** 2) / len(w)),
        'weight_p50': float(np.quantile(w, 0.5)), 'weight_p90': float(np.quantile(w, 0.9)),
        'weight_p99': float(np.quantile(w, 0.99)), 'weight_max': float(w.max()),
        'weight_share_top_1pct': float(100 * np.sort(w)[-top1pct:].sum() / w.sum()),
        'top10_weighted_sse_pct': float(100 * np.sort(weighted)[-10:].sum() / weighted.sum()),
        'top1pct_weighted_sse_pct': float(100 * np.sort(weighted)[-top1pct:].sum() / weighted.sum()),
        'mse_delta': float(delta.sum()),
        'top10_mse_delta': float(delta[order[:10]].sum()),
        'top10_share_of_positive_delta_pct': float(100 * positive[order[:10]].sum() / positive.sum()),
    }


def balance_table(df, frame, w, rank_mask):
    """Do the weighted proportions match ranking's? Separate question from calibration."""
    rows = []
    for column in BALANCE:
        values = frame[column].to_numpy()
        rows.append({'feature': column,
                     'source_pct': float(100 * values.mean()),
                     'weighted_pct': float(100 * np.average(values, weights=w)),
                     'ranking_pct': float(100 * df.loc[rank_mask, column].mean())})
        rows[-1]['residual_pct_points'] = rows[-1]['weighted_pct'] - rows[-1]['ranking_pct']
    for ap in sorted(df.airport.dropna().unique()):
        mask = frame.airport.eq(ap).to_numpy()
        rows.append({'feature': f'airport={ap}',
                     'source_pct': float(100 * mask.mean()),
                     'weighted_pct': float(100 * np.average(mask, weights=w)),
                     'ranking_pct': float(100 * df.loc[rank_mask, 'airport'].eq(ap).mean())})
        rows[-1]['residual_pct_points'] = rows[-1]['weighted_pct'] - rows[-1]['ranking_pct']
    return rows


def unseen_support(df, source, rank_mask):
    """Ranking combinations absent from the source. Weights cannot reach these."""
    out = []
    for cols in SUPPORT_KEYS:
        src = pd.MultiIndex.from_frame(df.loc[source, cols].astype('string').fillna('<NA>'))
        tgt = pd.MultiIndex.from_frame(df.loc[rank_mask, cols].astype('string').fillna('<NA>'))
        unseen = ~tgt.isin(src)
        out.append({'keys': cols, 'ranking_unseen_n': int(unseen.sum()),
                    'ranking_unseen_pct': float(100 * unseen.mean()),
                    'note': 'no source rows to upweight; reweighting cannot represent these'})
    return out


def main():
    print('loading departures...')
    df = load()
    print(f'  {len(df):,} departure rows, {len(FEATURES)} features; '
          f'calendar fields excluded: {", ".join(EXCLUDED)}')

    jan_jul_25 = df.ym.isin(['2025-01', '2025-07'])
    jan_jul_26 = df.ym.isin(RANK_MONTHS)
    nov_dec_25 = df.ym.isin(LONG_MONTHS)

    print('\ndiscriminators (AUC measures separability, not cost):')
    rng = np.random.default_rng(7)
    coin = pd.Series(rng.random(len(df)) < 0.5, index=df.index) & jan_jul_25
    discriminators = [
        discriminate(df, 'D0 null', 'Jan+Jul 2025 split at random', jan_jul_25 & ~coin, coin),
        discriminate(df, 'D1 naive', 'all 2025 vs the ranking set', df.year.eq('2025'), jan_jul_26),
        discriminate(df, 'D2 season-matched', 'Jan+Jul 2025 vs Jan+Jul 2026', jan_jul_25, jan_jul_26),
        discriminate(df, 'D3 within-year', 'Jan+Jul 2025 vs Nov+Dec 2025', jan_jul_25, nov_dec_25),
    ]

    record = json.loads((RUN / 'result.json').read_text())
    record = {**record, 'prediction_column': 'pred_combined'}
    record['folds'] = {k: v for k, v in record['folds'].items()
                       if k in GROUP_FOLDS['R_A'] + GROUP_FOLDS['R_B_long']}
    losses = load_losses(record, RUN)
    groups = {name: group_loss(losses, folds) for name, folds in GROUP_FOLDS.items()}
    del losses
    mix = record['ranking_weights']['jan_fraction']
    combine = lambda v: float(np.sqrt(mix * v['R_A'] ** 2 + (1 - mix) * v['R_B_long'] ** 2))
    indexed = df.set_index('mvt_id')

    print('\nper-group density ratios, each from its own source months:')
    per_group, plain, weighted, clipped = {}, {}, {}, {}
    for name, months in GROUP_SOURCE.items():
        source = df.ym.isin(months)
        print(f' {name} (source {months[0]}..{months[-1]}):')
        ratios, counts, calib = cross_fitted(df, source, jan_jul_26, name)
        loss = groups[name]
        sq = loss.sq.mean(axis=0)
        w = ratios.ratio.reindex(loss.meta.index).to_numpy()
        assert np.isfinite(w).all() and (w > 0).all(), f'{name}: a scored flight has no finite weight'
        cap = np.quantile(w, 0.99)
        plain[name] = float(np.sqrt(sq.mean()))
        weighted[name] = float(np.sqrt(np.average(sq, weights=w)))
        clipped[name] = float(np.sqrt(np.average(sq, weights=np.minimum(w, cap))))
        per_group[name] = {
            'source_months': months, 'n': int(len(sq)),
            'rmse': plain[name], 'rmse_weighted': weighted[name],
            'rmse_weighted_clipped_p99': clipped[name],
            'rmse_weighted_uncorrected_odds': float(np.sqrt(np.average(
                sq, weights=ratios.odds.reindex(loss.meta.index).to_numpy()))),
            'cross_fit_counts': counts,
            'probability_calibration_heldout': calib,
            'balance_vs_ranking': balance_table(df, indexed.loc[loss.meta.index], w, jan_jul_26),
            'weight_and_error_concentration': concentration(sq, w),
            'unseen_combinations': unseen_support(df, source, jan_jul_26),
        }
        c = per_group[name]['weight_and_error_concentration']
        noid = per_group[name]['balance_vs_ranking'][0]
        print(f"   {plain[name]:.4f} → {weighted[name]:.4f} weighted "
              f"(clipped {clipped[name]:.4f}), ESS {c['ess_pct']:.1f}%")
        print(f"   calibration: ECE {calib['expected_calibration_error']:.4f}, "
              f"Brier {calib['brier']:.4f}")
        print(f"   no_flight_id: source {noid['source_pct']:.3f}% → weighted "
              f"{noid['weighted_pct']:.3f}% vs ranking {noid['ranking_pct']:.3f}% "
              f"(residual {noid['residual_pct_points']:+.3f} pp)")
        print(f"   concentration: top 10 flights are {c['top10_share_of_positive_delta_pct']:.1f}% "
              f"of the positive change in mean squared error")

    cv, wtd = combine(plain), combine(weighted)
    accounting = {
        'status': 'EXPERIMENTAL DIAGNOSTIC — not validated as a leaderboard forecast',
        'cv_combined': cv, 'cv_reweighted': wtd, 'cv_reweighted_clipped_p99': combine(clipped),
        'official': OFFICIAL, 'gap_total': OFFICIAL - cv,
        'moved_by_reweighting': wtd - cv, 'remainder': OFFICIAL - wtd,
        'remainder_note': 'unexplained; not attributed to labels, and not shown to be '
                          'recoverable error. Better weights, omitted features (weather, '
                          'traffic, history) or unseen combinations could all move it.',
    }
    print(f"\ncombined {cv:.4f} → {wtd:.4f} weighted; official {OFFICIAL}; "
          f"remainder {accounting['remainder']:.2f} s (unexplained)")

    write_json(OUT, {'discriminators': discriminators, 'groups': per_group,
                     'accounting': accounting,
                     'features': {'used': FEATURES, 'excluded': EXCLUDED},
                     'params': {**PARAMS, 'rounds': ROUNDS, 'seed': SEED}})
    print(f'wrote {OUT}')


if __name__ == '__main__':
    main()
