"""Fresh, embargoed inputs for final-month model selection.

This module performs no work when imported. ``prepare_fold`` is called only by
the user-launched validation runner. It constructs a new split, never borrows
historical Jan/Jul predictions, training matrices, classifiers, or calibrations.
The only target-dependent fitting here is the established training-only late
mean/correction calibration; no predictive model is fitted by this module.

The returned dictionary has ordinary-regression X/y/V, complete aligned raw
frames train_raw/raw, train_ids/ids/tmonth, core_X/core_y/core_raw/core_V for
Rome classifiers and prediction corrections, core_public/held_public, and
feature_groups. ``core_*`` training rows have the existing rule exclusion but
retain Rome substituted targets; ordinary X/y exclude those substituted rows.
All model matrices are label-free; raw.taxi_sec is retained solely for scoring.

prepare_fold(out, month, train_months, threads, include_era=False) saves
out/inputs.pkl and out/inputs.json, verifies their identities on resume, and
returns the dictionary. It does not alter any historical experiment artifact.
"""
from __future__ import annotations

import gc
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "final_month_inputs_v1"
FOLDS = {
    "2025-01": [f"2025-{m:02d}" for m in range(3, 12)],
    "2025-02": [f"2025-{m:02d}" for m in range(4, 13)],
    "2025-06": [f"2025-{m:02d}" for m in [1, 2, 3, 4, 8, 9, 10, 11, 12]],
    "2025-07": [f"2025-{m:02d}" for m in [1, 2, 3, 4, 5, 9, 10, 11, 12]],
}


def _sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "sha256": _sha256(path)}


def _write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _source_closure(paths):
    """Hash a stable local import graph, independent of lazy imports in the runner."""
    pending, seen = list(paths), set()
    roots = [ROOT / "VALIDATION", ROOT / "pipeline", ROOT / "VALIDATION/tools"]
    while pending:
        path = Path(pending.pop()).resolve()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import)
                     else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                relative = Path(*name.split(".")).with_suffix(".py")
                for root in roots:
                    candidate = root / relative
                    if candidate.is_file():
                        pending.append(candidate)
                        break
    return seen


def _checked_record(path, key, expected):
    record = json.loads(Path(path).read_text())
    if record[key] != expected:
        raise ValueError(f"Changed historical input-only cache: {path} [{key}]")
    return record


def observable_held(frame):
    """Remove every known departure-truth alias from a copy before feature APIs."""
    import numpy as np
    import pandas as pd
    out = frame.copy()
    for column in out.columns:
        if column in {"taxi_sec", "TAXITIME_SEC_mvt", "y"}:
            out[column] = np.nan
        elif "BLOCK_TIME_UTC" in column.upper():
            out[column] = pd.NaT
    return out


def _scope_adsb(table, context):
    """Recompute movement-neighbour summaries from only permitted input rows."""
    import adsblol
    import numpy as np
    import pandas as pd

    ids = context.mvt_id.to_numpy()
    if not pd.Index(ids).isin(table.index).all():
        raise ValueError("ADSB own-observation cache is missing requested IDs")
    scoped = table.loc[ids].reset_index().copy()
    # Identity clocks are supplied movement inputs, even on source-unmatched rows.
    scoped["airport"] = context.ADEP_mvt.to_numpy()
    scoped["to_ms"] = pd.to_datetime(context.MVT_TIME_UTC_mvt, utc=True).astype(
        "datetime64[ms, UTC]").astype("int64").to_numpy()
    scoped = adsblol.neighbours(scoped)
    scoped = adsblol.prior_mean(scoped, "ab_stand_exit_sec",
                               lambda f: f.ab_exit_gap_sec <= adsblol.OWN_EXIT_GAP_SEC,
                               "ab_nb_exit")
    # Coverage is not selected in the original134 but keep its scope honest.
    day = scoped.to_ms // 86_400_000
    seen = scoped.ab_seg_sec.notna().astype(float).where(scoped.ab_matched.notna())
    scoped["ab_cov_day"] = seen.groupby([scoped.airport, day]).transform("mean")
    if scoped.mvt_id.duplicated().any():
        raise ValueError("Duplicate scoped ADSB identity")
    return scoped.set_index("mvt_id")


def _stand_xy(tables):
    import numpy as np
    import pandas as pd
    import extra_features as ef

    coords, _, origin = tables
    o = origin.reindex(coords.airport)
    cosine = np.cos(np.radians(o.lat.to_numpy()))
    return pd.DataFrame({"x": (coords.lon.to_numpy() - o.lon.to_numpy()) * 111_320 * cosine,
                         "y": (coords.lat.to_numpy() - o.lat.to_numpy()) * 110_540},
                        index=pd.Index(ef.key(coords.airport, coords.stand)))


def _scope_combo(dep, arr, tables):
    """No departure labels or block times enter peer/arrival/runway features."""
    import pandas as pd
    import extra_features as ef

    dcols = ["mvt_id", "ADEP_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "AOBT_3_flt"]
    acols = ["ADES_mvt", "STAND_mvt", "RUNWAY_mvt", "MVT_TIME_UTC_mvt", "taxi_sec"]
    d, a = dep[dcols].reset_index(drop=True), arr[acols].reset_index(drop=True)
    frame = pd.concat([ef.arrival_features(d, a), ef.peer_features(d, _stand_xy(tables)),
                       ef.runway_features(d, a)], axis=1)
    frame.index = pd.Index(d.mvt_id.to_numpy(), name="mvt_id")
    for col in ef.CATEGORICAL:
        frame[col] = frame[col].astype("category")
    return frame


def _append(X, ids, table, columns):
    import pandas as pd
    if set(columns) & set(X.columns):
        raise ValueError("Attempted to append duplicate feature columns")
    if not pd.Index(ids).isin(table.index).all():
        raise ValueError("Feature table is missing requested movement IDs")
    extra = table.loc[ids, columns].copy()
    extra.index = X.index
    return pd.concat([X, extra], axis=1)


def _build(out, month, train_months, threads, include_era):
    import numpy as np
    import pandas as pd
    import adsblol
    import aircraft_rotation as rotation
    import callsign_clocks as clocks
    import extra_features as ef
    import modeling
    import opdi
    import overnight_run
    import public_delay_features as public
    import public_daily_taxi_features as daily
    import rome_hybrid
    import seasonal
    import tuning_seasonal as ts
    import stand_features as sf

    raw_paths = sorted((ROOT / "data").glob("training_2025-*.parquet"))
    if len(raw_paths) != 12:
        raise ValueError("Expected all 12 original 2025 training files")
    record_path = ts.V43 / "result.json"
    config = dict(json.loads(record_path.read_text())["config"], threads=threads,
                  prediction_threads=threads, fit_workers=1)
    # Recovery would change the training raw/label mapping. Existing selected
    # preparation was recovery-off, so fail rather than silently approximate it.
    if config.get("recovery", "off") != "off":
        raise ValueError("This fresh builder requires the selected recovery-off recipe")
    identity_path = rotation.DEFAULT / "identity.parquet"
    shared_path = clocks.DEFAULT / "shared.parquet"
    rotation_record = _checked_record(rotation.DEFAULT / "identity.json", "identity_sha256", _sha256(identity_path))
    calls_record = _checked_record(clocks.DEFAULT / "manifest.json", "shared_sha256", _sha256(shared_path))
    if calls_record["identity_sha256"] != rotation_record["identity_sha256"]:
        raise ValueError("Callsign observations and rotation identity disagree")
    if rotation_record["settings"] != rotation.SETTINGS or calls_record["settings"] != clocks.SETTINGS:
        raise ValueError("Input-only identity/callsign cache settings changed")
    source_paths = set(raw_paths + [Path(__file__), record_path, ts.weather.METAR,
        sf.COORDS, sf.ENDS, opdi.FEATURES_PATH, adsblol.FEATURES_PATH, adsblol.STAND_PATH,
        identity_path, rotation.DEFAULT / "identity.json", shared_path, clocks.DEFAULT / "manifest.json",
        public.base.SOURCE / "all_pre_departure_delays_2025.csv",
        daily.source.SOURCE / "daily_runway.parquet", daily.source.SOURCE / "manifest.json",
        daily.source.REPORT / "public_daily_taxi_source_audit.json"])
    if include_era:
        import external_conditions as era
        source_paths.update([era.CACHE / "weather_manifest.json", *sorted((era.CACHE / "weather").glob("*.json"))])
    # Hash a fixed local source graph rather than sys.modules, which changes as
    # models import optional GPU packages between folds and on resume.
    source_paths.update(_source_closure([Path(__file__)]))
    identity = dict(schema=SCHEMA, month=month, train_months=train_months,
                    include_era=include_era, config=config,
                    sources=[_artifact(p) for p in sorted(source_paths)])
    cache, receipt = out / "inputs.pkl", out / "inputs.json"
    if cache.exists() or receipt.exists():
        if not (cache.exists() and receipt.exists()):
            raise ValueError(f"Incomplete fold cache; inspect before resuming: {out}")
        saved = json.loads(receipt.read_text())
        if saved["identity"] != identity or saved["artifact"] != _artifact(cache):
            raise ValueError(f"Fold cache provenance changed: {out}")
        data = pd.read_pickle(cache)
        data["_input_artifact"] = dict(path=str(cache.resolve()), sha256=_sha256(cache))
        return data
    if json.loads((daily.source.REPORT / "public_daily_taxi_source_audit.json").read_text())["status"] != "pass":
        raise ValueError("Public daily taxi source audit has not passed")

    dep, arr = overnight_run.load_inputs(1.)
    td, ta, held = seasonal.split_frames(dep, arr, dict(train=train_months, val=[month]))
    vd, va = held[month]
    del dep, arr, held
    vd = vd.reset_index(drop=True)
    # Labels survive only in vd, for later scoring. Every feature API below gets
    # this physically label-masked held frame; training reference frames remain td.
    vd_inputs = observable_held(vd)
    events = pd.concat([td, vd_inputs], ignore_index=True)
    arrivals = pd.concat([ta, va], ignore_index=True)
    if set(pd.to_datetime(td.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m")) != set(train_months):
        raise ValueError("Training row months do not match the frozen fold")
    if set(pd.to_datetime(vd.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m")) != {month}:
        raise ValueError("Held row months do not match the frozen fold")
    if not np.isfinite(vd.taxi_sec.to_numpy(float)).all():
        raise ValueError("All held departures need finite raw targets")
    freeze = pd.Period(max(train_months), "M").end_time.strftime("%Y-%m-%d")
    tables = sf.load()
    reports = ts.weather.load_reports()
    grid = ts.features.snow_grid(pd.read_parquet(ts.weather.METAR))
    X45, y_all, cats, raw_all = overnight_run.prepare_training(td, ta, config)
    parameters = ts.signals.fit_parameters(X45, y_all, raw_all)
    corrections, _ = ts.v42_submit.fit_corrections(td, ta, config, parameters["late_mean"])
    drop = ts.recipe.rule_rows(raw_all, config["prediction_rule"])
    core_raw = raw_all.loc[~drop].copy()
    core_y = np.asarray(y_all, float)[~drop]
    core_X = ts.recipe.add_inputs(X45.loc[~drop].copy(), core_raw, reports, td, ta)
    core_X = ts.features.add_features(core_X, core_raw, ta, grid)
    core_X = sf.add_features(core_X, core_raw, tables)
    core_V = modeling.validation_matrix(td, ta, vd_inputs, va, freeze, cats, config)
    core_V = ts.recipe.add_inputs(core_V, vd_inputs, reports, events, arrivals)
    core_V = ts.features.add_features(core_V, vd_inputs, arrivals, grid)
    core_V = sf.add_features(core_V, vd_inputs, tables)
    core_V = ts.taxi_occupancy.add_occupancy(core_V, vd_inputs, modeling.observable(events))
    if not core_X.index.equals(core_raw.index) or not core_V.index.equals(vd.index):
        raise ValueError("Core matrices and raw IDs are not aligned")
    substituted = rome_hybrid.substituted(core_X, core_y, core_raw)
    train_raw = core_raw.loc[~substituted].copy()
    y = core_y[~substituted]
    np.testing.assert_array_equal(y, train_raw.taxi_sec.to_numpy(float))
    names = rome_hybrid.regressor_features(ts.KIND, ts.ORDINARY)
    X, V = core_X.loc[~substituted, names].copy(), core_V.loc[:, names].copy()
    train_ids, ids = train_raw.mvt_id.to_numpy(), vd.mvt_id.to_numpy()
    if np.intersect1d(train_ids, ids).size:
        raise ValueError("Training/held ID overlap")
    del X45, y_all, raw_all
    gc.collect()

    ab = adsblol.load_all()
    combo = ef.ARRIVAL_FEATURES + ef.PEER_FEATURES + ef.RUNWAY_FEATURES
    original = adsblol.ADSB_FEATURES + adsblol.STAND_FEATURES
    for part, context_dep, context_arr, raw in [("train", td, ta, train_raw),
                                               ("held", events, arrivals, vd_inputs)]:
        frame = X if part == "train" else V
        scoped_ab = _scope_adsb(ab, context_dep)
        frame = adsblol.add_adsblol(frame, raw, scoped_ab, original)
        del scoped_ab
        extra = _scope_combo(context_dep, context_arr, tables)
        frame = _append(frame, raw.mvt_id.to_numpy(), extra, combo)
        del extra
        if part == "train":
            X = frame
        else:
            V = frame
        gc.collect()
    del ab
    base134 = names + original
    combo156 = base134 + combo
    if len(base134) != 134 or len(combo156) != 156 or list(X) != combo156 or list(V) != combo156:
        raise ValueError("Established ordinary feature counts/order changed")
    # Make the freshly scoped categorical dictionary training-only and common to
    # both matrices; unseen held runway sets become missing, never new fit levels.
    for column in ef.CATEGORICAL:
        categories = X[column].cat.categories
        V[column] = pd.Categorical(V[column], categories=categories)

    identity_events = pd.read_parquet(identity_path)
    shared = pd.read_parquet(shared_path).set_index("mvt_id")
    if not shared.index.is_unique:
        raise ValueError("Duplicate callsign shared-cache IDs")
    con = rotation.connection()
    try:
        for part, qids, allowed in [("train", train_ids, train_months),
                                    ("held", ids, train_months + [month])]:
            arrival = rotation.feature_table(identity_events, qids, allowed, con)
            clock_features = clocks.fold_features(shared, arrival)
            if part == "train":
                X = clocks.add_features(X, train_ids, clock_features)
            else:
                V = clocks.add_features(V, ids, clock_features)
    finally:
        con.close()
    del identity_events, shared

    public_inputs = public.load(raw_paths)  # SQL selects inputs only, no DEP target.
    pmonths = public_inputs.day.str[:7]
    public_source = public.base.public(2025)
    public_tables = {}
    for part, allowed in [("train", train_months), ("held", train_months + [month])]:
        public_tables[part] = public.build(public_inputs.loc[pmonths.isin(allowed)], public_source).set_index("mvt_id")
    X = _append(X, train_ids, public_tables["train"], public.FEATURES)
    V = _append(V, ids, public_tables["held"], public.FEATURES)
    core_public = public_tables["train"].loc[core_raw.mvt_id, public.FEATURES].copy()
    core_public.index = core_X.index
    held_public = public_tables["held"].loc[ids, public.FEATURES].copy()
    held_public.index = core_V.index
    del public_inputs, public_tables

    taxi_source = pd.read_parquet(daily.source.SOURCE / "daily_runway.parquet")
    for part, raw in [("train", train_raw), ("held", vd_inputs)]:
        rows = raw[["mvt_id", "ADEP_mvt", "MVT_TIME_UTC_mvt", "RUNWAY_mvt"]].rename(
            columns={"ADEP_mvt": "airport", "MVT_TIME_UTC_mvt": "takeoff", "RUNWAY_mvt": "runway"}).reset_index(drop=True)
        rows["local_day"] = ""
        for airport, zone in daily.source.ZONES.items():
            mask = rows.airport.eq(airport)
            rows.loc[mask, "local_day"] = rows.loc[mask, "takeoff"].dt.tz_convert(zone).dt.strftime("%Y-%m-%d")
        if not rows.local_day.str.len().eq(10).all():
            raise ValueError("Unknown airport timezone for public daily taxi input")
        table = daily.build(rows, taxi_source).set_index("mvt_id")
        if part == "train":
            X = _append(X, train_ids, table, daily.FEATURES)
        else:
            V = _append(V, ids, table, daily.FEATURES)

    calls162 = combo156 + clocks.FEATURE_NAMES
    public174 = calls162 + public.FEATURES
    daily184 = public174 + daily.FEATURES
    groups = dict(base134=base134, combo156=combo156, calls162=calls162,
                  public174=public174, daily184=daily184,
                  airport168=combo156 + public.FEATURES,
                  public12=list(public.FEATURES), daily10=list(daily.FEATURES),
                  classifier=list(ts.CLASSIFIER))
    if len(X.columns) != 184 or list(X) != daily184 or list(V) != daily184:
        raise ValueError("Full ordinary184 feature set changed")
    if include_era:
        table = era.weather_table()
        A, B = era.weather_join(train_raw, table), era.weather_join(vd_inputs, table)
        if A.isna().any().any() or B.isna().any().any():
            raise ValueError("ERA weather coverage incomplete; no silent fallback")
        X, V = pd.concat([X, A], axis=1), pd.concat([V, B], axis=1)
        groups["era195"] = daily184 + era.WEATHER
    categorical = [c for c in X if isinstance(X[c].dtype, pd.CategoricalDtype)]
    for column in categorical:
        V[column] = pd.Categorical(V[column], categories=X[column].cat.categories)
    forbidden = {"taxi_sec", "TAXITIME_SEC_mvt", "BLOCK_TIME_UTC_mvt", "y"}
    if forbidden.intersection(X.columns) or forbidden.intersection(V.columns):
        raise ValueError("Departure label leaked into model columns")
    meta = dict(month=month, train_months=train_months, freeze=freeze, config=config,
                training_rows=len(td), ordinary_rows=len(X), core_rows=len(core_X),
                heldout_rows=len(V), excluded_rule_rows=int(drop.sum()),
                substituted_rows=int(substituted.sum()), parameters=parameters,
                corrections=corrections, regressor_features=names,
                feature_context={"train": train_months, "held": train_months + [month]},
                feature_note="Fresh fold-scoped histories and input-only peers; held labels masked before feature APIs")
    data = dict(X=X, y=y, V=V, raw=vd, train_raw=train_raw,
                train_ids=train_ids, ids=ids,
                tmonth=pd.to_datetime(train_raw.MVT_TIME_UTC_mvt, utc=True).dt.strftime("%Y-%m").to_numpy(),
                core_X=core_X, core_y=core_y, core_raw=core_raw, core_V=core_V,
                core_ids=core_raw.mvt_id.to_numpy(), core_public=core_public, held_public=held_public,
                feature_groups=groups, categorical=categorical, meta=meta)
    out.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".pkl.tmp")
    pd.to_pickle(data, tmp)
    tmp.replace(cache)
    _write_json(receipt, dict(identity=identity, artifact=_artifact(cache),
                             rows={"ordinary": len(X), "core": len(core_X), "held": len(V)},
                             feature_groups=groups))
    data["_input_artifact"] = dict(path=str(cache.resolve()), sha256=_sha256(cache))
    return data


def prepare_fold(out, month, train_months, threads=4, include_era=False):
    """Build or verify one fresh fold; called only after the user starts the run."""
    train_months = list(train_months)
    if month not in FOLDS or train_months != FOLDS[month]:
        raise ValueError("Fold must match the frozen Jan/Feb/Jun/Jul one-month embargo")
    if not isinstance(threads, int) or not 1 <= threads <= 16:
        raise ValueError("threads must be an integer between 1 and 16")
    sys.path[:0] = [str(ROOT / "VALIDATION"), str(ROOT / "pipeline"), str(ROOT / "VALIDATION/tools")]
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=threads):
        return _build(Path(out).resolve(), month, train_months, threads, bool(include_era))
