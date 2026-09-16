"""Observable clock disagreements and a fixed baseline for residual regression."""

import numpy as np
import pandas as pd

CLOCKS = {'schedule': 'SCHED_TIME_UTC_mvt', 'lobt': 'LOBT_flt', 'iobt': 'IOBT_flt',
          'eobt1': 'EOBT_1_flt', 'aobt3': 'AOBT_3_flt'}
BASELINE_ORDER = ('aobt3', 'eobt1', 'iobt', 'lobt', 'schedule')
PROVENANCE = ('raw_flight_id_missing', 'raw_plan_block_missing', 'raw_aobt3_missing')


def clock_features(frame):
    # Explicit allowlist: labels, movement block stamps and donor targets cannot enter.
    raw = frame.loc[:, ['MVT_TIME_UTC_mvt', *CLOCKS.values(), *PROVENANCE]]
    movement = pd.to_datetime(raw.MVT_TIME_UTC_mvt, utc=True)
    gaps = pd.DataFrame({name: (movement-pd.to_datetime(raw[col], utc=True)).dt.total_seconds()
                         for name, col in CLOCKS.items()}, index=frame.index)
    gaps = gaps.where(np.isfinite(gaps))
    features = {}
    for left, right in [('aobt3', 'schedule'), ('aobt3', 'eobt1'),
                        ('eobt1', 'iobt'), ('iobt', 'lobt')]:
        # timestamp(left) - timestamp(right) = gap(right) - gap(left)
        features[f'clock_{left}_minus_{right}_sec'] = gaps[right]-gaps[left]
    for name in CLOCKS:
        features[f'clock_{name}_available'] = gaps[name].notna().astype(np.int8)
    for name in PROVENANCE:
        if raw[name].isna().any():
            raise ValueError(f'missing original provenance flag: {name}')
        features['clock_'+name] = raw[name].astype(np.int8)
    features['clock_available_count'] = gaps.notna().sum(axis=1)
    features['clock_span_sec'] = gaps.max(axis=1)-gaps.min(axis=1)
    features['clock_std_sec'] = gaps.std(axis=1, ddof=0)
    features['clock_median_sec'] = gaps.median(axis=1)
    baseline = pd.Series(np.nan, index=frame.index)
    source = pd.Series(len(BASELINE_ORDER), index=frame.index, dtype=np.int8)
    for code, name in enumerate(BASELINE_ORDER):
        selected = baseline.isna() & gaps[name].notna()
        baseline.loc[selected] = gaps.loc[selected, name]
        source.loc[selected] = code
    features['clock_baseline_sec'] = baseline.fillna(0).clip(lower=0)
    features['clock_baseline_source'] = source
    return pd.DataFrame(features, index=frame.index)


def target_offset(X, mode='direct'):
    if mode == 'direct':
        return np.zeros(len(X))
    if mode != 'clock_residual':
        raise ValueError(f'unknown target mode: {mode}')
    offset = X['clock_baseline_sec'].to_numpy(float)
    if not np.isfinite(offset).all():
        raise ValueError('residual baseline must be finite for every row')
    return offset
