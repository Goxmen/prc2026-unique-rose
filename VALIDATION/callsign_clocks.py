"""Retained callsign/squawk clocks, with explicit ambiguity and censor flags.

The cache stores DAILY (hex, callsign, squawk) min/max/count aggregates, not
contiguous callsign episodes. Calls were retained only for aircraft occurring
near a challenge airport in the same half-hour source chunk. These timestamps
are observation clocks, never asserted to be actual switches or off-blocks.

No departure labels or block timestamps are read. Aircraft identity reuses the
audited input-only cache. The own-arrival comparison uses the already audited,
fold-scoped rotation feature table. Other ambiguity checks use callsign records
or earlier departures on the query's same UTC day, hence in its allowed month.
"""
import argparse
from collections import defaultdict
from datetime import date, timedelta
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

import aircraft_rotation as rotation

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / 'data/derived/callsign_clocks_20261010'
FEATURE_NAMES = ['cc_call_start_sec', 'cc_squawk_start_sec', 'cc_previous_call_end_sec',
                 'cc_before_own_arrival', 'cc_ambiguous', 'cc_censored']
SETTINGS = dict(min_records=3, near_event_slack_sec=180, lookback_sec=14400,
                day_boundary_bridge_sec=600, source_boundary_period_sec=1800,
                source_boundary_tolerance_sec=20, feature_names=FEATURE_NAMES,
                source_semantics='daily hex/callsign/squawk aggregates; no contiguous episodes inferred',
                ambiguity='overlapping other callsign, multiple active squawks, or earlier same-call departure that UTC day',
                censor='call start in first 20 seconds after half-hour boundary or after supplied takeoff')


def bridge_start(start, intervals):
    """Only extend across adjacent source dates with a small observed gap."""
    start = int(start)
    while True:
        candidates = [first for first,last in intervals
                      if first < start and 0 <= start-last <= SETTINGS['day_boundary_bridge_sec']*1000]
        if not candidates:
            return start
        new = min(candidates)
        if new == start:
            return start
        start = new


def row_clocks(t, day, current, others, previous_departure=np.nan):
    """One aircraft's retained intervals; current rows are (first,last,squawk,day)."""
    slack = SETTINGS['near_event_slack_sec']*1000
    active = [r for r in current if r[0]-slack <= t <= r[1]+slack]
    if not active:
        return [np.nan]*6
    daily = {}
    for first,last,squawk,source_day in current:
        if source_day not in daily:
            daily[source_day] = [first,last]
        else:
            daily[source_day][0] = min(daily[source_day][0],first)
            daily[source_day][1] = max(daily[source_day][1],last)
    eligible = [s for s,(first,last) in daily.items() if first-slack <= t <= last+slack]
    chosen = min(eligible,key=lambda s:(s != day,abs((date.fromisoformat(s)-date.fromisoformat(day)).days),s))
    start = bridge_start(daily[chosen][0],list(daily.values()))
    # A change of squawk can create a later retained start under the same callsign.
    selected = max(active,key=lambda r:r[0])
    squawk_intervals = [(first,last) for first,last,squawk,_ in current if squawk == selected[2]]
    squawk_start = bridge_start(selected[0],squawk_intervals)
    ends = [last for first,last in others
            if t-SETTINGS['lookback_sec']*1000 <= last < t]
    previous_end = max(ends) if ends else np.nan
    other_overlap = any(first < t and last > start for first,last in others)
    ambiguity = (other_overlap
                 or len({r[2] for r in active}) > 1
                 or (np.isfinite(previous_departure) and start <= previous_departure < t))
    censor = (start % (SETTINGS['source_boundary_period_sec']*1000)
              <= SETTINGS['source_boundary_tolerance_sec']*1000) or start > t
    return [start,squawk_start,previous_end,float(ambiguity),float(censor),len(active)]


def day_clocks(queries, calls, day):
    """Shared input-only clocks, without any fold-dependent arrival reference."""
    c = calls[calls.records.ge(SETTINGS['min_records'])].copy()
    c['cs'] = c.callsign.fillna('').str.strip().str.upper().str.replace(' ','',regex=False)
    c = c[c.cs.ne('')]
    groups,by_hex = defaultdict(list),defaultdict(list)
    for hex_id,cs,first,last,squawk,source_day in c[
            ['hex','cs','first_ms','last_ms','squawk','source_day']].itertuples(index=False,name=None):
        groups[(hex_id,cs)].append((first,last,squawk,source_day))
        by_hex[hex_id].append((cs,first,last))
    rows = []
    for q in queries.itertuples(index=False):
        current = groups.get((q.hex,q.cs),[])
        others = [(a,b) for cs,a,b in by_hex.get(q.hex,[]) if cs != q.cs]
        values = row_clocks(q.to_ms,day,current,others,q.previous_departure_ms)
        rows.append([q.mvt_id,q.to_ms,*values])
    frame = pd.DataFrame(rows,columns=['mvt_id','to_ms','call_start_ms','squawk_start_ms',
                         'previous_call_end_ms','cc_ambiguous','cc_censored','active_interval_count'])
    for name,source in [('cc_call_start_sec','call_start_ms'),
                        ('cc_squawk_start_sec','squawk_start_ms'),
                        ('cc_previous_call_end_sec','previous_call_end_ms')]:
        age = (frame.to_ms-frame[source])/1000
        lower = 0 if name == 'cc_previous_call_end_sec' else -SETTINGS['near_event_slack_sec']
        frame[name] = age.where(age.between(lower,SETTINGS['lookback_sec']))
    return frame


def fold_features(shared, arrival_table):
    """Only this arrival comparison depends on explicitly scoped movement history."""
    raw = shared.reindex(arrival_table.index)
    if raw.to_ms.isna().any():
        raise ValueError('Shared callsign table misses departure IDs')
    frame = raw[['cc_call_start_sec','cc_squawk_start_sec','cc_previous_call_end_sec',
                 'cc_ambiguous','cc_censored']].copy()
    call_age = (raw.to_ms-raw.call_start_ms)/1000
    own_age = arrival_table.ar_prev_inblock_age_sec
    frame['cc_before_own_arrival'] = np.where(call_age.notna() & own_age.notna(),
                                             (call_age > own_age).astype(float), np.nan)
    return frame[FEATURE_NAMES].astype(np.float32)


def build(out=DEFAULT):
    import pyarrow as pa
    pa.set_cpu_count(2)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'shared.parquet').exists():
        raise FileExistsError('Existing callsign clocks must be audited before reuse')
    manifest = json.loads((rotation.DEFAULT/'identity.json').read_text())
    folds = json.loads((rotation.DEFAULT/'folds.json').read_text())
    identity_path = rotation.DEFAULT/'identity.parquet'
    if rotation.sha256(identity_path) != manifest['identity_sha256'] or manifest['identity_sha256'] != folds['identity_sha256']:
        raise ValueError('Audited aircraft identity changed')
    columns = ['mvt_id','phase','airport','t','cs','hex']
    dep = pd.read_parquet(identity_path,columns=columns)
    dep = dep[dep.phase.eq('DEP')].copy()
    dep['day'] = dep.t.dt.strftime('%Y-%m-%d')
    dep['to_ms'] = dep.t.astype('datetime64[ms, UTC]').astype('int64')
    known = dep.hex.notna()
    history = dep.loc[known].sort_values(['hex','cs','day','to_ms'])
    previous = history.groupby(['hex','cs','day'],sort=False).to_ms.shift()
    dep['previous_departure_ms'] = previous.reindex(dep.index)
    query_days = dep.groupby('day',sort=True)
    count_days = query_days.ngroups
    started = time.monotonic(); frames=[];sources=set()
    for i,(day,query_day) in enumerate(query_days):
        stamp = date.fromisoformat(day)
        call_frames=[]
        for delta in [-2,-1,0,1]:
            s = str(stamp+timedelta(days=delta))
            path = rotation.HEATMAP/'callsigns'/f'{s}.parquet'
            if path.exists():
                call_frames.append(pd.read_parquet(path).assign(source_day=s));sources.add(path)
        q = query_day[['mvt_id','hex','cs','to_ms','previous_departure_ms']]
        calls = pd.concat(call_frames,ignore_index=True) if call_frames else pd.DataFrame(
            columns=['hex','callsign','records','first_ms','last_ms','squawk','source_day'])
        frames.append(day_clocks(q,calls,day))
        if (i+1)%30 == 0:
            print(f'{i+1}/{count_days} days in {time.monotonic()-started:.1f}s',flush=True)
    shared = pd.concat(frames,ignore_index=True).set_index('mvt_id')
    if shared.index.duplicated().any():
        raise ValueError('Duplicate shared feature movement IDs')
    shared.reset_index().to_parquet(out/'shared.parquet',index=False)
    reports={}
    for month in rotation.FOLDS:
        reports[month]={}
        for part in ['train','held']:
            arrival_path = rotation.DEFAULT/f'{month}_{part}.parquet'
            if rotation.sha256(arrival_path) != folds['folds'][month][part]['sha256']:
                raise ValueError('Fold-scoped arrival features changed')
            arrival = rotation.load_features(arrival_path)
            features = fold_features(shared,arrival)
            path = out/f'{month}_{part}.parquet'
            features.reset_index().to_parquet(path,index=False)
            reports[month][part]=dict(rows=len(features),sha256=rotation.sha256(path),
                rotation_sha256=folds['folds'][month][part]['sha256'],
                reference_months=folds['folds'][month][part]['reference_months'],
                nonmissing={c:int(features[c].notna().sum()) for c in FEATURE_NAMES})
    rotation.write_json(out/'manifest.json',dict(settings=SETTINGS,
        source_sha256=rotation.sha256(Path(__file__)),
        identity_sha256=manifest['identity_sha256'],
        rotation_manifest_sha256=rotation.sha256(rotation.DEFAULT/'folds.json'),
        shared_sha256=rotation.sha256(out/'shared.parquet'),folds=reports,
        call_sources=[dict(path=str(p),sha256=rotation.sha256(p)) for p in sorted(sources)],
        elapsed_sec=time.monotonic()-started))


def load_features(path):
    table = pd.read_parquet(path).set_index('mvt_id')
    if table.index.duplicated().any() or list(table.columns) != FEATURE_NAMES:
        raise ValueError('Invalid callsign feature table')
    return table


def add_features(X, ids, table):
    if len(X) != len(ids) or set(X.columns)&set(FEATURE_NAMES):
        raise ValueError('Input shape or feature overlap')
    if not pd.Index(ids).isin(table.index).all():
        raise ValueError('Missing movement IDs in callsign feature table')
    values=table.loc[np.asarray(ids)].copy();values.index=X.index
    return pd.concat([X,values],axis=1)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--out',type=Path,default=DEFAULT)
    build(parser.parse_args().out)
