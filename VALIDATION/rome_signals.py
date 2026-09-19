"""Rome recording signals on top of the Rome hybrid: new inputs and two label-shape corrections.

At LIRF a departure without a flight ID is one of two things. Either its off-block
was never reported and was logged at the scheduled time (label = takeoff minus
schedule), or it really was late: a non-substituted LIRF departure loses its
flight ID only when its recorded off-block is more than an hour after schedule.
The hybrid's classifier decides between the two without the airline, because
`operator` comes from the flight plan that these rows lack. This module adds:

Classifier inputs (each can be switched on alone):

    airline                     airline code read from FLIGHT_mvt: the first three
                                characters when they are letters (ICAO, e.g. ITY),
                                otherwise the first two (IATA, e.g. LH at EDDF)
    handler_arr_sched_share     share of the airline's arrivals at this airport in
                                the same UTC month whose in-block equals the schedule
                                (within 60 s), shrunk toward the airport-month share
    handler_arr_n               number of those arrivals
    flight_number_format        0 clean, 1 mangled (ICAO prefix + last five callsign
                                characters, e.g. ITYTY630; only LIRF writes these),
                                2 missing

Arrival in-block is landing plus taxi-in; both are supplied for ranking arrivals.
Only arrivals in the frame the caller passes are used, so a validation month sees
its own arrivals and a training row sees its training window's.

Prediction corrections (fitted on the training window's labels only):

    late      For LIRF rows without a flight ID and 1 h < gap < 4 h, the
              not-substituted component of the blend becomes
              min(late mean, max(gap - 3600, 60)) instead of the ordinary
              regressor. The late mean is the mean label of the window's LIRF
              no-ID departures whose recorded off-block was at least an hour late.
    dayshift  When the four-hour rule fires, the takeoff is on a later local
              (Europe/Rome) day than the schedule and the gap is at least 14 h,
              the recorded off-block is sometimes the actual clock time on the
              schedule's date, so the label is days * 86400 + taxi. The prediction
              becomes w * (days * 86400 + ordinary) + (1 - w) * gap, with w the
              window's share of such rule rows whose label was day-shifted.

With no extra inputs and both corrections off, `predict` reproduces
`rome_hybrid.predict` exactly.
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

import modeling
import overnight_models as models
import rome_hybrid
from common import CATEGORICAL
from registered_rules import apply_rule

AIRPORT = rome_hybrid.AIRPORT
GAP = rome_hybrid.GAP
TZ = 'Europe/Rome'
AIRLINE = 'airline'
FORMAT = 'flight_number_format'
HANDLER = ['handler_arr_sched_share', 'handler_arr_n']
SIGNALS = [AIRLINE, *HANDLER, FORMAT]
HANDLER_MATCH_SEC = 60
HANDLER_PRIOR_WEIGHT = 10
LATE_DELAY_SEC = 3600
LATE_FLOOR_SEC = 60
RULE_GAP_SEC = 14400
DAYSHIFT_GAP_SEC = 14 * 3600
DAY_SEC = 86400
SETTINGS = dict(signals=SIGNALS, handler_match_sec=HANDLER_MATCH_SEC,
                handler_prior_weight=HANDLER_PRIOR_WEIGHT, handler_scope='airport x airline x UTC month',
                late_delay_sec=LATE_DELAY_SEC, late_floor_sec=LATE_FLOOR_SEC,
                dayshift_gap_sec=DAYSHIFT_GAP_SEC, timezone=TZ)


def airline_code(flight):
    """ICAO prefix when the first three characters are letters, otherwise the IATA pair."""
    f = pd.Series(flight, dtype=object)
    text = f.where(f.map(lambda x: isinstance(x, str) and len(x) >= 3))
    first3 = text.str[:3]
    alpha = first3.str.isalpha().fillna(False).astype(bool)
    return first3.where(alpha, text.str[:2])


def mangled(flight):
    """ICAO prefix followed by the last five callsign characters, e.g. ITY630 -> ITYTY630."""
    def one(x):
        if not isinstance(x, str) or len(x) != 8 or not x[:3].isalpha():
            return False
        return any(x[3:3 + k] == x[3 - k:3] for k in (1, 2, 3))
    return pd.Series(flight, dtype=object).map(one).to_numpy(bool)


def flight_number_format(flight):
    f = pd.Series(flight, dtype=object)
    return np.select([f.isna().to_numpy(), mangled(f)], [2, 1], 0).astype('float32')


def _utc(values):
    return pd.to_datetime(values, utc=True)


def _month(values):
    return _utc(values).dt.tz_localize(None).dt.to_period('M')


def handler_rates(raw, arr_events):
    """Arrival schedule-substitution share of each departure's airline, airport and UTC month."""
    arr = arr_events.loc[:, ['ADES_mvt', 'FLIGHT_mvt', 'MVT_TIME_UTC_mvt', 'SCHED_TIME_UTC_mvt', 'taxi_sec']]
    in_block = _utc(arr.MVT_TIME_UTC_mvt) + pd.to_timedelta(arr.taxi_sec, unit='s')
    offset = (in_block - _utc(arr.SCHED_TIME_UTC_mvt)).dt.total_seconds()
    ok = offset.notna().to_numpy()
    a = pd.DataFrame({'airport': arr.ADES_mvt.to_numpy(), 'airline': airline_code(arr.FLIGHT_mvt).to_numpy(),
                      'month': _month(arr.MVT_TIME_UTC_mvt).to_numpy(),
                      'match': (offset.abs() <= HANDLER_MATCH_SEC).to_numpy(float)})[ok]
    prior = a.groupby(['airport', 'month']).match.mean()
    by_airline = a.dropna(subset=['airline']).groupby(['airport', 'airline', 'month']).match.agg(['sum', 'size'])
    d = pd.DataFrame({'airport': raw.ADEP_mvt.to_numpy(), 'airline': airline_code(raw.FLIGHT_mvt).to_numpy(),
                      'month': _month(raw.MVT_TIME_UTC_mvt).to_numpy()})
    base = prior.reindex(pd.MultiIndex.from_frame(d[['airport', 'month']])).to_numpy(float)
    hit = by_airline.reindex(pd.MultiIndex.from_frame(d[['airport', 'airline', 'month']]))
    n = hit['size'].fillna(0).to_numpy(float)
    share = (hit['sum'].fillna(0).to_numpy(float) + HANDLER_PRIOR_WEIGHT * base) / (n + HANDLER_PRIOR_WEIGHT)
    unknown = d.airline.isna().to_numpy()
    return pd.DataFrame({'handler_arr_sched_share': np.where(unknown, np.nan, share),
                         'handler_arr_n': np.where(unknown, np.nan, n)}, index=raw.index).astype('float32')


def add_signals(X, raw, arr_events):
    """Append SIGNALS to a matrix aligned with `raw`; `airline` becomes a pandas categorical."""
    if not X.index.equals(raw.index):
        raise ValueError('Matrix and raw movement rows must align before adding Rome signals')
    overlap = set(X.columns) & set(SIGNALS)
    if overlap:
        raise ValueError(f'Rome signals already present: {sorted(overlap)}')
    signals = handler_rates(raw, arr_events)
    signals.insert(0, AIRLINE, pd.Categorical(airline_code(raw.FLIGHT_mvt).to_numpy()))
    signals[FORMAT] = flight_number_format(raw.FLIGHT_mvt)
    return pd.concat([X, signals.loc[:, SIGNALS]], axis=1)


def _categorical(columns):
    return [c for c in [*CATEGORICAL, AIRLINE] if c in columns]


def _train(params, X, y, rounds, desc):
    dataset = lgb.Dataset(X, y, categorical_feature=_categorical(X.columns))
    with tqdm(total=rounds, desc=desc, unit='rnd', leave=False) as bar:
        return lgb.train(params, dataset, num_boost_round=rounds,
                         callbacks=[lambda env: bar.update(env.iteration + 1 - bar.n)])


def classifier_matrix(X, raw, kind, extra=()):
    base = rome_hybrid.classifier_matrix(X, raw, kind)
    return pd.concat([base, X.loc[:, list(extra)]], axis=1) if extra else base


def fit_classifier(X, y, raw, kind, seed, threads=8, extra=()):
    """The hybrid's LIRF substitution classifier, with optional extra inputs."""
    s = rome_hybrid.substituted(X, y, raw)
    lirf = raw['ADEP_mvt'].to_numpy() == AIRPORT
    if not s.any() or s[lirf].all():
        raise ValueError('Need both schedule-recorded and ordinary LIRF training rows')
    params = dict(rome_hybrid.CLASSIFIER_PARAMS, seed=int(seed), bagging_seed=int(seed) + 1,
                  feature_fraction_seed=int(seed) + 2, num_threads=int(threads))
    return _train(params, classifier_matrix(X.loc[lirf], raw.loc[lirf], kind, extra),
                  s[lirf].astype(float), rome_hybrid.CLASSIFIER_ROUNDS, f'classifier+{len(extra)}/{seed}')


def fit_ordinary(X, y, raw, kind, seed, rounds=800, threads=8, extra=()):
    """The hybrid's ordinary regressor (schedule-recorded rows excluded), with optional extra inputs."""
    s = rome_hybrid.substituted(X, y, raw)
    params = models.model_params('control', seed, threads)
    names = rome_hybrid.regressor_features(kind, extra)
    return _train(params, X.loc[~s, names], np.asarray(y, float)[~s], rounds, f'ordinary+{len(extra)}/{seed}')


def local_days(raw):
    """Whole local (Europe/Rome) calendar days from schedule to takeoff."""
    day = lambda s: _utc(s).dt.tz_convert(TZ).dt.tz_localize(None).dt.normalize()
    return (day(raw.MVT_TIME_UTC_mvt) - day(raw.SCHED_TIME_UTC_mvt)).dt.days.to_numpy(float)


def _no_id_lirf(raw):
    return (raw['ADEP_mvt'].to_numpy() == AIRPORT) & raw['FLIGHT_ID_mvt'].isna().to_numpy()


def late_rows(gap, raw):
    return _no_id_lirf(raw) & (gap > LATE_DELAY_SEC) & (gap < RULE_GAP_SEC)


def dayshift_rows(gap, raw, days):
    return _no_id_lirf(raw) & (gap >= DAYSHIFT_GAP_SEC) & (days >= 1)


def fit_parameters(X, y, raw):
    """Late mean and day-shift weight from training-window labels only."""
    y = np.asarray(y, float)
    gap = X[GAP].to_numpy(float)
    late = late_rows(gap, raw) & (gap - y >= LATE_DELAY_SEC)
    days = local_days(raw)
    rule = dayshift_rows(gap, raw, days)
    shifted = rule & (y - DAY_SEC * days >= 0) & (y - DAY_SEC * days <= 3600)
    return dict(late_mean=float(y[late].mean()) if late.any() else None, late_rows=int(late.sum()),
                dayshift_weight=float(shifted.sum() / rule.sum()) if rule.any() else 0.0,
                dayshift_rows=int(rule.sum()), dayshift_shifted=int(shifted.sum()))


def ordinary_prediction(ordinary, V, kind, threads=8, extra=()):
    return modeling.predict(ordinary, V.loc[:, rome_hybrid.regressor_features(kind, extra)], threads=threads)


def substitution_probability(classifier, V, raw, kind, threads=8, extra=()):
    """P(schedule-recorded) for LIRF rows with a finite gap; zero elsewhere."""
    lirf = (raw['ADEP_mvt'].to_numpy() == AIRPORT) & np.isfinite(V[GAP].to_numpy(float))
    p = np.zeros(len(V))
    if lirf.any():
        p[lirf] = classifier.predict(classifier_matrix(V.loc[lirf], raw.loc[lirf], kind, extra), num_threads=threads)
    return p


def predict(ordinary, classifier, V, raw, kind, threads=8, extra=(), classifier_extra=(),
            late_mean=None, dayshift_weight=None):
    """Hybrid prediction with optional classifier inputs and the two corrections."""
    base = ordinary_prediction(ordinary, V, kind, threads, extra)
    p = substitution_probability(classifier, V, raw, kind, threads, classifier_extra)
    return combine(base, p, V, raw, late_mean, dayshift_weight)


def combine(base, p, V, raw, late_mean=None, dayshift_weight=None):
    """Blend, four-hour rule and corrections from ordinary predictions and probabilities."""
    gap = V[GAP].to_numpy(float)
    lirf = (raw['ADEP_mvt'].to_numpy() == AIRPORT) & np.isfinite(gap)
    other = base.copy()
    late = np.zeros(len(V), bool)
    if late_mean is not None and np.isfinite(late_mean):
        late = late_rows(gap, raw)
        other[late] = np.minimum(late_mean, np.maximum(gap[late] - LATE_DELAY_SEC, LATE_FLOOR_SEC))
    blended = np.where(lirf, np.clip(p * np.where(lirf, gap, 0) + (1 - p) * other, 0, None), base)
    final, audit = apply_rule(blended, raw, rome_hybrid.RULE)
    fired = audit.rule_triggered.to_numpy()
    shifted = np.zeros(len(V), bool)
    if dayshift_weight:
        days = local_days(raw)
        shifted = fired & dayshift_rows(gap, raw, days)
        w = float(dayshift_weight)
        final[shifted] = w * (DAY_SEC * days[shifted] + base[shifted]) + (1 - w) * gap[shifted]
    return dict(ordinary=base, p=p, blended=blended, final=final, rule_triggered=fired,
                late_rows=late, dayshift_rows=shifted)
