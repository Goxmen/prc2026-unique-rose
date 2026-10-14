"""Complete fixed v80 production assembly after missing full-year model refits.

The historical source files are imported unchanged. Their old-weight replay
guards remain intact; this separate run explicitly admits fresh model weights
and reports differences on the old inputs instead of calling them exact replay.
Only missing base predictors are fitted elsewhere. This module recalculates all
prediction-dependent downstream stages, preserving their saved calibration and
the selected ordered affine coefficients. It neither fits nor uploads.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "VALIDATION/runs"
DEFAULT_OUT = RUNS / "v3_20261013_v80_refit_final"
CATALOG = RUNS / "v3_20261012_final_basis_catalog/partial_30.json"
SELECTED = RUNS / "v3_20261012_final_selected_stack/old/identity.json"
GEOMETRY79 = RUNS / "v3_20261012_target205_geometry"
MANIFEST79 = RUNS / "v3_20261012_target205_candidate/submission_v79/manifest.json"
TREE_VERSIONS = [2, 3, 5, 6, 7, 9, 26, 27, 28, 29, 43, 44]
NEURAL_VERSIONS = [47, 48, 49, 50, 51, 53]
VIEWS = ("old", "final")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def art(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def verify(record):
    path = ROOT / record["path"]
    if sha(path) != record["sha256"]:
        raise ValueError(f"Artifact changed: {record['path']}")
    return path


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parquet(frame, path):
    temporary = Path(path).with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def serial(values):
    import numpy as np
    x = np.asarray(values, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite prediction")
    x = np.rint(np.maximum(x, 0))
    if x.max(initial=0) > np.iinfo(np.int32).max:
        raise ValueError("Prediction exceeds template integer range")
    return x.astype(np.int32)


def align(frame, ids, id_column="mvt_id"):
    import numpy as np
    if len(frame) != len(ids) or not frame[id_column].is_unique:
        raise ValueError("Missing or duplicate prediction IDs")
    if set(frame[id_column]) != set(ids):
        raise ValueError("Prediction ID population differs from template")
    result = frame.set_index(id_column).loc[ids].reset_index()
    np.testing.assert_array_equal(result[id_column], ids)
    return result


def affine(matrix, weights):
    import numpy as np
    from threadpoolctl import threadpool_limits
    p, w = np.asarray(matrix, float), np.asarray(weights, float)
    if p.ndim != 2 or p.shape[1] != len(w) or not np.isfinite(w).all():
        raise ValueError("Invalid ordered affine basis")
    np.testing.assert_array_equal(p, serial(p))
    # Historical selected-stack assembly also used one BLAS thread. Keep this
    # fixed independently of the user's training thread budget and replay host.
    with threadpool_limits(1):
        return p @ w


def independent_bound(ids, rows, values, matches):
    """Independent NumPy formulation of the two fixed support constraints."""
    import numpy as np
    x = np.asarray(values, float).copy()
    matched = rows.has_id.to_numpy(bool) & np.isfinite(rows.nm_gap.to_numpy(float))
    gap = rows.nm_gap.to_numpy(float)
    x[matched] = np.maximum(0, np.minimum(np.maximum(x[matched], gap[matched] - 3606), gap[matched] + 3606))
    x = np.maximum(0, x)
    witnesses = matches.loc[matches.accepted, ["mvt_id", "recovered_gap"]]
    if not witnesses.mvt_id.is_unique:
        raise ValueError("Ambiguous accepted recovered-plan witnesses")
    recovered = witnesses.set_index("mvt_id").reindex(ids).recovered_gap.to_numpy(float)
    low, high = recovered - 3606, recovered + 3606
    inside = np.isfinite(recovered) & (x > low) & (x < high)
    lower = (low >= 0) & (x - low <= high - x)
    x[inside & lower] = low[inside & lower]
    x[inside & ~lower] = high[inside & ~lower]
    return x


def independent_monthly(rows, prediction, source, coefficients, airports):
    """Recompute monthly cell means and fixed half-strength shifts directly."""
    import numpy as np
    f = rows.assign(pred=np.asarray(prediction, float))
    keep = f.eligible & f.pred.le(7200)
    keys = ["airport", "local_month"]
    cells = f.loc[keep].groupby(keys, as_index=False).agg(mean=("pred", "mean"), count=("pred", "size"))
    cells = cells.merge(source, on=keys, how="left", validate="one_to_one")
    intercept = dict(zip(airports, coefficients[:len(airports)]))
    composition = cells.airport.map(intercept).to_numpy(float, copy=True)
    composition += cells.invalid.to_numpy(float) * coefficients[-2]
    composition += cells.noref.to_numpy(float) * coefficients[-1]
    raw = cells.mean_sec.to_numpy(float) + composition - cells["mean"].to_numpy(float)
    cells["shift"] = np.where(cells.invalid.lt(.05) & cells["count"].ge(100), .5 * np.clip(raw, -120, 120), 0.)
    shift = f[keys].merge(cells[keys + ["shift"]], on=keys, how="left", validate="many_to_one")["shift"].fillna(0).to_numpy(float)
    return np.maximum(0, f.pred.to_numpy(float) + np.where(keep, shift, 0.))


def independent_daily_shift(rows, prediction, source, params, airports):
    """Recompute the saved fixed daily composition/gain without producer code."""
    import numpy as np
    f = rows.assign(pred=np.asarray(prediction, float))
    keep = f.eligible & f.pred.between(0, 7200)
    keys = ["airport", "local_day", "runway"]
    cells = f.loc[keep].groupby(keys, as_index=False).agg(mean=("pred", "mean"), count=("pred", "size"))
    cells = cells.merge(source, on=keys, how="left", validate="one_to_one")
    n = len(airports)
    intercept = dict(zip(airports, params["coefficients"][:n]))
    slopes = dict(zip(airports, params["coefficients"][n:]))
    ratio = cells.source_n.to_numpy(float) / cells["count"].to_numpy(float)
    composition = cells.airport.map(intercept).to_numpy(float, copy=True)
    composition += cells.airport.map(slopes).to_numpy(float) * (1 - ratio) / .1
    variance = 200.**2 / cells["count"].to_numpy(float) + 15.**2
    noise = cells.airport.map(params["noise"]).to_numpy(float)
    gain = variance / (variance + noise**2)
    supported = cells["count"].ge(20) & cells.source_n.ge(20) & (ratio >= .5) & (ratio <= 1.2) & np.isfinite(cells.source_mean)
    raw = cells.source_mean.to_numpy(float) + composition - cells["mean"].to_numpy(float)
    cells["shift"] = np.where(supported, gain * np.clip(raw, -120, 120), 0.)
    shift = f[keys].merge(cells[keys + ["shift"]], on=keys, how="left", validate="many_to_one")["shift"].fillna(0).to_numpy(float)
    return np.where(keep, shift, 0.)


def modules():
    sys.path.insert(0, str(ROOT / "VALIDATION"))
    import final_public_corrections as corrections
    import final_public_direct_components as direct
    import final_public_quality_components as quality
    import final_joint_public_inference as joint
    return corrections, direct, quality, joint


def check_audit(path):
    value = read(path)
    if value.get("status") != "pass":
        raise ValueError(f"Upstream audit did not pass: {path}")
    return value


def collect_inputs(out, mods):
    """Authenticate every reused vector/packet and the relevant original code."""
    c, d, q, j = mods
    files = {Path(__file__), CATALOG, SELECTED, GEOMETRY79 / "prepared.json", GEOMETRY79 / "result.json", MANIFEST79, c.MANIFEST, d.MANIFEST,
             c.V71 / "geometry.json", c.V73 / "geometry.json",
             c.V72 / "parameters.json", c.V74 / "parameters.json",
             q.original.OUT / "parameters.json", q.original.OUT / "result.json", q.original.MANIFEST,
             c.inputs.MANIFEST, c.inputs.OUT / "audit.json",
             d.OUT / "independent_audit.json", j.MANIFEST,
             j.OUT / "prepared.json", j.OUT / "input_audit.json",
             c.base.SOURCE / "all_pre_departure_delays_2026.csv",
             c.base.SOURCE / "taxi_out_additional_time_2026.csv",
             q.original.daily.SOURCE / "daily_runway.parquet"}
    catalog = read(CATALOG)
    if len(catalog["components"]) != 30:
        raise ValueError("Expected the reviewed 30-component retained catalog")
    for version, component in catalog["components"].items():
        audit = check_audit(verify(component["audit"]))
        result = read(verify(component["result"])) if "result" in component else None
        if not c.audit_binds(audit, component["final"], component.get("result"), result):
            raise ValueError(f"Unbound retained final component v{version}")
        files.update(verify(component[k]) for k in ("old", "final", "audit"))
        if "result" in component:
            files.add(verify(component["result"]))
    ca = check_audit(c.inputs.OUT / "audit.json")
    if ca["identity"] != art(c.inputs.MANIFEST):
        raise ValueError("Correction input audit identity changed")
    calibration = read(q.original.OUT / "result.json")
    if calibration["identity"] != art(q.original.MANIFEST) or art(q.original.OUT / "parameters.json") not in calibration["artifacts"]:
        raise ValueError("Saved daily calibration is not bound to the original fit")
    if read(q.original.OUT / "parameters.json")["allowed"] != [f"2025-{month:02d}" for month in range(1, 13)]:
        raise ValueError("Daily calibration was not fitted on the complete 2025 scope")
    da = check_audit(d.OUT / "independent_audit.json")
    if da["identity"] != art(d.MANIFEST):
        raise ValueError("Direct model audit identity changed")
    ja = check_audit(j.OUT / "input_audit.json")
    if ja["identity"] != art(j.MANIFEST):
        raise ValueError("Joint posterior input audit identity changed")
    for view in VIEWS:
        receipt_path = c.inputs.OUT / view / "prepared.json"
        receipt = read(receipt_path)
        if receipt["identity"] != art(c.inputs.MANIFEST):
            raise ValueError("Prepared correction inputs changed")
        files.add(receipt_path)
        files.update(verify(r) for r in receipt["outputs"])
        dr = read(d.OUT / f"{view}.json")
        if dr["identity"] != art(d.MANIFEST) or da["views"][view]["prediction"] != dr["prediction"]:
            raise ValueError("Direct model values are not audited")
        files.update([d.OUT / f"{view}.json", verify(dr["prediction"])])
        jr, audit = read(j.OUT / view / "result.json"), check_audit(j.OUT / view / "audit.json")
        if (jr["identity"] != art(j.MANIFEST) or audit["result"] != art(j.OUT / view / "result.json")
                or audit["prediction"] != jr["prediction"]):
            raise ValueError("Joint posterior values are not audited")
        files.update([j.OUT / view / "result.json", j.OUT / view / "audit.json", verify(jr["prediction"])])
        files.add(ROOT / "data" / ("submitting.parquet" if view == "old" else "final_submitting.parquet"))
        files.add(ROOT / "data" / ("ranking.parquet" if view == "old" else "final_ranking.parquet"))
        raw = (RUNS / "v3_20261011_public_daily_taxi_features/inputsranking.parquet" if view == "old"
               else RUNS / "v3_20261012_final_daily_main_inference/features/inputsfinal.parquet")
        raw_manifest = read(raw.parent / "manifest.json")
        raw_audit = check_audit(raw.parent / "audit.json")
        if raw_audit["manifest"] != art(raw.parent / "manifest.json"):
            raise ValueError("Raw daily feature audit changed")
        if raw_manifest["records"]["ranking" if view == "old" else "final"]["inputs"] != art(raw):
            raise ValueError("Raw daily features changed")
        files.update([raw, raw.parent / "manifest.json", raw.parent / "audit.json"])
    for family in ("trees", "neural"):
        receipt_path = out / family / "result.json"
        receipt = check_audit(receipt_path)
        files.add(receipt_path)
        for view in VIEWS:
            prediction = out / family / "components" / f"{view}.parquet"
            recorded = receipt["predictions"][view]
            if {key: recorded.get(key) for key in ("path", "sha256")} != art(prediction):
                raise ValueError(f"Refit output is not authenticated: {family}/{view}")
            files.add(prediction)
    # The correction calibration is reused from its full-2025 fit, never refitted
    # against new predictions. Authenticate the bytes against the historical port.
    expected = {r["path"]: r["sha256"] for r in read(c.MANIFEST)["sources"]}
    expected.update(read(q.original.MANIFEST)["sources"])
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name:
            p = Path(name).resolve()
            if p.is_file() and p.is_relative_to(ROOT / "VALIDATION"):
                files.add(p)
    for p in files:
        relative = str(p.relative_to(ROOT))
        if relative in expected and sha(p) != expected[relative]:
            raise ValueError(f"Historical source/input changed: {relative}")
    files.update(ROOT / f"submissions/unique-rose_v{v}.parquet" for v in [71, 72, 73, 74, 76, 77, 78, 79, 80])
    return sorted(files)


def freeze(out, mods):
    c, _, _, _ = mods
    target = out / "assembly_identity.json"
    if target.exists():
        identity = read(target)
        for record in identity["sources"]:
            verify(record)
        return identity
    files = collect_inputs(out, mods)
    selected = read(SELECTED)
    prepared79, geometry79, manifest79 = read(GEOMETRY79 / "prepared.json"), read(GEOMETRY79 / "result.json"), read(MANIFEST79)
    if (manifest79["prepared"] != art(GEOMETRY79 / "prepared.json")
            or manifest79["geometry"] != art(GEOMETRY79 / "result.json")
            or geometry79["prepared"] != manifest79["prepared"]
            or manifest79["weights"] != geometry79["solution"]["weights"]
            or selected["versions"] != prepared79["versions"]
            or selected["weights"] != [geometry79["solution"]["weights"][str(v)] for v in selected["versions"]]):
        raise ValueError("Selected v79 weights differ from the actual submitted recipe")
    g71, g73 = read(c.V71 / "geometry.json"), read(c.V73 / "geometry.json")
    if len(g71["versions"]) != 48 or g73["versions"] != g71["versions"] + [72]:
        raise ValueError("Unexpected historical nested basis")
    if len(selected["versions"]) != 51 or 75 in selected["versions"]:
        raise ValueError("Unexpected selected v79 basis")
    import platform, numpy, pandas, scipy
    identity = dict(schema="v80_full_recipe_refit_assembly_v1", sources=[art(p) for p in files],
        geometry71=g71, geometry73=g73, versions=selected["versions"], weights=selected["weights"],
        fresh_versions=TREE_VERSIONS + NEURAL_VERSIONS,
        retained_provenance={v: r["provenance"] for v, r in read(CATALOG)["components"].items()},
        runtime=dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__, scipy=scipy.__version__),
        recipe="v80 fixed 48/49/51 affine and full public correction chain; missing full-2025 models refitted",
        fresh_weight_old_predictions="Diagnostic only; no original-weight replay claim or ranking score claim",
        unchanged_posteriors="Joint/runway increments are independent of the refitted components: their only prediction prior is retained v70",
        no_fit=True, no_upload=True)
    write(target, identity)
    return identity


def basis(view, out, identity, c):
    import numpy as np
    import pandas as pd
    ids = c.template(view)
    catalog = read(CATALOG)["components"]
    historical_basis = read(c.MANIFEST)["old_basis"]
    result = pd.DataFrame({"mvt_id": ids})
    frames = {name: align(pd.read_parquet(out / name / "components" / f"{view}.parquet"), ids)
              for name in ("trees", "neural")}
    for version in identity["geometry71"]["versions"]:
        if str(version) in catalog:
            record = catalog[str(version)]
            values = align(pd.read_parquet(verify(record[view])), ids,
                           record.get("old_id_column", record.get("id_column", "mvt_id")) if view == "old"
                           else record.get("id_column", "mvt_id"))
            column = record.get("old_value_column", record.get("value_column", "submitted")) if view == "old" else record.get("value_column", "submitted")
            value = values[column].to_numpy(float)
            if view == "old":
                expected = c.vector(historical_basis[str(version)], ids)
                np.testing.assert_array_equal(value, expected, err_msg=f"Retained v{version}")
        else:
            family = "trees" if version in TREE_VERSIONS else "neural"
            value = frames[family][f"v{version}"].to_numpy(float)
        np.testing.assert_array_equal(value, serial(value))
        result[f"v{version}"] = value.astype(np.int32)
    return result


def bracketed_tilt(original, values, weights, target, events):
    """Extend only a failed root bracket; preserve every successful old result.

    Same recovery as portable_public_tilt_recovery_20261013: no new prior,
    tolerance, calibration, or iteration count.
    """
    import numpy as np
    from scipy.special import logsumexp
    result = original(values, weights, target)
    if np.isclose(values @ result, target, atol=1e-5, rtol=1e-8):
        return result
    if not (np.isfinite(values).all() and np.isfinite(weights).all()
            and np.isfinite(target) and (weights >= 0).all() and weights.sum() > 0
            and values.min() < target < values.max()):
        raise ValueError("Invalid or infeasible tilt support")
    z = (values - target) / max(np.ptp(values), 1.)
    lw = np.log(np.maximum(weights, 1e-300))
    def at(parameter):
        mass = np.exp(lw + parameter * z - logsumexp(lw + parameter * z))
        return mass, float(mass @ z)
    lo, hi = -128., 128.
    low, high = at(lo)[1], at(hi)[1]
    if low <= 0 <= high:
        raise ValueError("Tilt failed despite a valid original bracket; requires separate diagnosis")
    for _ in range(32):
        if low <= 0 <= high:
            break
        if low > 0: lo *= 2
        if high < 0: hi *= 2
        low, high = at(lo)[1], at(hi)[1]
        if not np.isfinite([lo, hi, low, high]).all():
            raise ValueError("Nonfinite tilt bracket")
    else:
        raise ValueError("Unable to bracket tilt root")
    bracket = [lo, hi]
    for _ in range(55):
        mid = (lo + hi) / 2
        mass, residual = at(mid)
        if residual > 0: hi = mid
        else: lo = mid
    np.testing.assert_allclose(values @ mass, target, atol=1e-5, rtol=1e-8)
    event = dict(values=values.tolist(), base_weights=weights.tolist(), target=float(target),
                 original_mean=float(values @ result), weights=mass.tolist(), bracket=bracket,
                 root=float(mid), iterations=55, atol=1e-5, rtol=1e-8)
    audit_tilt_event(event)
    events.append(event)
    return mass


def audit_tilt_event(event):
    """Independent Brent root and max-shift normalization, on repaired cases."""
    import numpy as np
    from scipy.optimize import brentq
    values, base = np.asarray(event["values"]), np.asarray(event["base_weights"])
    target = event["target"]
    z = (values - target) / max(float(np.ptp(values)), 1.)
    def at(parameter):
        logits = np.log(np.maximum(base, 1e-300)) + parameter * z
        mass = np.exp(logits - logits.max())
        return mass / mass.sum()
    root = brentq(lambda x: float(at(x) @ z), *event["bracket"], xtol=1e-12, rtol=1e-14)
    reference = at(root)
    np.testing.assert_allclose(reference, event["weights"], atol=1e-10, rtol=0)
    np.testing.assert_allclose(reference @ values, target, atol=1e-5, rtol=1e-8)


@contextmanager
def repaired_tilt(posterior, events):
    original = posterior.tilt
    posterior.tilt = lambda values, weights, target: bracketed_tilt(original, values, weights, target, events)
    try:
        yield
    finally:
        posterior.tilt = original


def daily_posterior(folder, identity_record, ids, rows, p71, c):
    """Checkpoint each independent airport-day; never repeat completed solves."""
    import numpy as np
    import pandas as pd
    params = read(c.V72 / "parameters.json")["parameters"]
    source = c.base.public(2026).set_index(["airport", "day"])
    gaussian, student = [np.full(len(ids), np.nan) for _ in range(2)]
    cells = []
    checkpoints = folder / "posterior_days"
    checkpoints.mkdir(parents=True, exist_ok=True)
    for (airport, day), at in rows.groupby(["airport", "day"]).indices.items():
        p = checkpoints / f"{airport}_{day}.parquet"
        r = p.with_suffix(".json")
        if r.exists():
            previous = read(r)
            if previous["identity"] != identity_record:
                raise ValueError("Posterior checkpoint identity changed")
            frame = pd.read_parquet(verify(previous["prediction"]))
            np.testing.assert_array_equal(frame.mvt_id, ids[at])
            np.testing.assert_array_equal(frame.v71, p71[at])
            gaussian[at], student[at] = frame.gaussian, frame.student
            cells.append(previous["diagnostic"])
            continue
        group, public = rows.iloc[at], source.loc[(airport, day)]
        if not np.isfinite([public.total, public.n_public]).all():
            raise ValueError(f"Missing daily public source: {airport}/{day}")
        args = (group.gap.to_numpy(float), p71[at].astype(float), group.has_id.to_numpy(bool),
                group.nm_gap.to_numpy(float), airport, params[airport], float(public.total),
                float(public.n_public - len(group)))
        recovery = []
        with repaired_tilt(c.heavy.old, recovery):
            gaussian[at], ginfo = c.heavy.condition(*args, logprior=c.heavy.GAUSSIAN)
            student[at], sinfo = c.heavy.condition(*args)
        frame = pd.DataFrame(dict(mvt_id=ids[at], v71=p71[at], gaussian=gaussian[at], student=student[at]))
        parquet(frame, p)
        diagnostic = dict(airport=airport, day=day, rows=len(group), public_total=float(public.total),
                          public_count=float(public.n_public), gaussian_offset=ginfo["offset"], **sinfo)
        write(r, dict(identity=identity_record, prediction=art(p), diagnostic=diagnostic,
                      tilt_recovery=recovery))
        cells.append(diagnostic)
    if not np.isfinite(gaussian).all() or not np.isfinite(student).all():
        raise ValueError("Incomplete daily posterior")
    return gaussian, student, pd.DataFrame(cells)


def public_stages(view, folder, base, rows, matches, identity, c):
    import numpy as np
    import pandas as pd
    ids = base.mvt_id.to_numpy()
    P = base[[f"v{v}" for v in identity["geometry71"]["versions"]]].to_numpy(float)
    w71 = [identity["geometry71"]["weights"][str(v)] for v in identity["geometry71"]["versions"]]
    raw71 = affine(P, w71)
    p71 = serial(raw71)
    gaussian, student, days = daily_posterior(folder, art(folder.parent / "assembly_identity.json"), ids, rows, p71, c)
    p72, refined72 = serial(gaussian), serial(student)
    P73 = np.column_stack([P, p72.astype(float)])
    del P
    raw73 = affine(P73, [identity["geometry73"]["weights"][str(v)] for v in identity["geometry73"]["versions"]])
    del P73
    p73 = serial(raw73)
    refined73 = serial(raw73 + identity["geometry73"]["weights"]["72"] * (refined72.astype(float) - p72))
    monthly = align(pd.read_parquet(c.inputs.OUT / view / "monthly.parquet"), ids)
    params = read(c.V74 / "parameters.json")["full_year"]
    source = c.constraints.monthly.public(2026)
    monthly74, old_cells = c.constraints.monthly.apply(monthly, p73.astype(float), source, params)
    half74, new_cells = c.constraints.monthly.apply(monthly, refined73.astype(float), source, params)
    full74 = np.maximum(0, 2 * half74 - refined73.astype(float))
    raw74, raw_refined74 = [c.constrain(ids, rows, x, matches) for x in [monthly74, full74]]
    frame = pd.DataFrame(dict(mvt_id=ids, raw71=raw71, v71=p71, raw72=gaussian,
        raw_refined72=student, v72=p72, refined72=refined72, raw73=raw73, v73=p73,
        refined73=refined73, monthly74=monthly74, half_monthly74=half74, full_monthly74=full74,
        raw74=raw74, raw_refined74=raw_refined74, v74=serial(raw74), refined74=serial(raw_refined74)))
    parquet(frame, folder / "public_stages.parquet")
    parquet(days, folder / "public_days.parquet")
    parquet(pd.concat([old_cells.assign(branch="original"), new_cells.assign(branch="refined")], ignore_index=True), folder / "monthly_cells.parquet")
    return frame


def quality_stages(view, folder, public, rows, matches, mods):
    import numpy as np
    import pandas as pd
    c, direct, quality, _ = mods
    original = quality.original
    ids = public.mvt_id.to_numpy()
    frame = align(pd.read_parquet(direct.OUT / f"{view}.parquet"), ids)
    monthly = align(pd.read_parquet(c.inputs.OUT / view / "monthly.parquet"), ids)
    raw_path = (RUNS / "v3_20261011_public_daily_taxi_features/inputsranking.parquet" if view == "old"
                else RUNS / "v3_20261012_final_daily_main_inference/features/inputsfinal.parquet")
    inputs = align(pd.read_parquet(raw_path), ids)
    inputs["eligible"] = monthly.eligible.to_numpy()
    inputs["month"] = inputs.takeoff.dt.strftime("%Y-%m")
    inputs = inputs.drop(columns="takeoff")
    if "y" in inputs:
        raise ValueError("Departure targets in inference inputs")
    source, params = original.daily.public(), read(original.OUT / "parameters.json")
    q = inputs[["airport", "local_day"]].assign(local_month=inputs.local_day.str[:7]).merge(
        original.constraints.monthly.public(2026)[["airport", "local_month", "invalid"]],
        on=["airport", "local_month"], how="left", validate="many_to_one")
    eligible = q.invalid.lt(.05).to_numpy()
    result = inputs.copy()
    result["quality_eligible"], result["invalid_share"] = eligible, q.invalid.to_numpy()
    result["v76_float"] = c.transport(ids, rows, public.v74, frame.v76_direct, frame.control, matches)
    result["v76"] = serial(result.v76_float)
    cells = []
    # Airport/cat intermediary comparisons do not enter v79/v80. Main and TabM
    # are both recomputed because daily eligibility and shifts depend on parent.
    for arm in ["main", "tabm"]:
        value = c.transport(ids, rows, public.refined74, frame[arm + "_direct"], frame.control, matches)
        refined = np.rint(np.maximum(value, 0))
        _, cell, shift = original.daily.apply(inputs, refined, source, params)
        applied = np.where(eligible, shift, 0.)
        bounded = c.constrain(ids, rows, refined + applied, matches)
        for key, vector in dict(refined_float=value, refined=refined, shift=shift, applied=applied,
                                bounded=bounded, pred=serial(bounded)).items():
            result[arm + "_" + key] = vector
        cells.append(cell.assign(arm=arm))
    parquet(result, folder / "quality_stages.parquet")
    parquet(pd.concat(cells, ignore_index=True), folder / "quality_cells.parquet")
    return result


def build(view, out, identity, mods):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    c, _, _, joint = mods
    folder = out / view
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "result.json"
    if receipt_path.exists():
        result = read(receipt_path)
        if result["identity"] != art(out / "assembly_identity.json"):
            raise ValueError("Assembly resume identity changed")
        for record in result["outputs"]:
            verify(record)
        return
    base = basis(view, out, identity, c)
    parquet(base, folder / "basis48.parquet")
    ids = base.mvt_id.to_numpy()
    rows = align(pd.read_parquet(c.inputs.OUT / view / "daily.parquet"), ids)
    matches = pd.read_parquet(c.inputs.OUT / view / "matches.parquet")
    public = public_stages(view, folder, base, rows, matches, identity, c)
    quality = quality_stages(view, folder, public, rows, matches, mods)
    increments = align(pd.read_parquet(joint.OUT / view / "increments.parquet"), ids)
    constrain = lambda x: serial(c.constrain(ids, rows, x, matches))
    v78 = constrain(quality.main_pred.to_numpy(float) + .5 * increments.increment.to_numpy())
    tabm78 = constrain(quality.tabm_pred.to_numpy(float) + .5 * increments.increment.to_numpy())
    robust78 = constrain(v78.astype(float) + .5 * increments.robust_delta.to_numpy())
    np.testing.assert_array_equal(robust78[~increments.robust_active], v78[~increments.robust_active])
    recent = {72: public.v72, 74: public.v74, 76: quality.v76, 77: quality.main_pred, 78: v78}
    matrix = pd.DataFrame({"mvt_id": ids})
    for version in identity["versions"]:
        matrix[f"v{version}"] = recent[version] if version in recent else base[f"v{version}"]
    raw79 = affine(matrix.drop(columns="mvt_id").to_numpy(float), identity["weights"])
    v79 = serial(raw79)
    tabm_delta, robust_delta = tabm78.astype(float) - v78, robust78.astype(float) - v78
    v80 = serial(v79.astype(float) + tabm_delta + robust_delta)
    stages = pd.DataFrame(dict(mvt_id=ids, v78=v78, raw79=raw79, v79=v79, tabm_v78=tabm78,
                              robust_v78=robust78, tabm_delta=tabm_delta, robust_delta=robust_delta, v80=v80))
    parquet(matrix, folder / "basis51.parquet")
    parquet(stages, folder / "stages.parquet")
    template_path = ROOT / "data" / ("submitting.parquet" if view == "old" else "final_submitting.parquet")
    template = pq.read_table(template_path)
    from submit import fill_template
    candidate = fill_template(template, ids, v80)
    if candidate.schema != template.schema:
        raise ValueError("Final schema changed")
    for name in template.column_names:
        if name != "TAXITIME_SEC_mvt" and not candidate[name].equals(template[name]):
            raise ValueError(f"Template column changed: {name}")
    pq.write_table(candidate, folder / "candidate.parquet")
    if not pq.read_table(folder / "candidate.parquet").equals(candidate):
        raise ValueError("Candidate readback differs")
    files = ["basis48.parquet", "basis51.parquet", "public_stages.parquet", "public_days.parquet",
             "monthly_cells.parquet", "quality_stages.parquet", "quality_cells.parquet", "stages.parquet", "candidate.parquet"]
    write(receipt_path, dict(identity=art(out / "assembly_identity.json"), view=view, rows=len(ids),
        outputs=[art(folder / name) for name in files], candidate=art(folder / "candidate.parquet"),
        no_fit=True, no_upload=True, old_weight_equality_required=False, independent_audit_pending=True))
    print(json.dumps(dict(stage="v80_assembly", view=view, rows=len(ids), candidate=str(folder / "candidate.parquet"))), flush=True)


def audit(view, out, identity, mods):
    """Independent ordered sums, support bounds, stage linkage and final schema."""
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    c, direct, quality_module, joint = mods
    folder = out / view
    result = read(folder / "result.json")
    if result["identity"] != art(out / "assembly_identity.json"):
        raise ValueError("Result identity mismatch")
    for record in result["outputs"]:
        verify(record)
    for receipt in sorted((folder / "posterior_days").glob("*.json")):
        for event in read(receipt).get("tilt_recovery", []):
            audit_tilt_event(event)
    template_path = ROOT / "data" / ("submitting.parquet" if view == "old" else "final_submitting.parquet")
    template = pq.read_table(template_path)
    ids = template["MVT_ID_mvt"].to_numpy()
    expected_rows = 344841 if view == "old" else 670790
    if len(ids) != expected_rows or len(np.unique(ids)) != len(ids):
        raise ValueError("Template population changed")
    load = lambda name: align(pd.read_parquet(folder / name), ids)
    base, matrix, public, quality, stages = [load(n + ".parquet") for n in
        ["basis48", "basis51", "public_stages", "quality_stages", "stages"]]
    rows = align(pd.read_parquet(c.inputs.OUT / view / "daily.parquet"), ids)
    matches = pd.read_parquet(c.inputs.OUT / view / "matches.parquet")
    direct_values = align(pd.read_parquet(direct.OUT / f"{view}.parquet"), ids)
    increments = align(pd.read_parquet(joint.OUT / view / "increments.parquet"), ids)
    bound = lambda value: independent_bound(ids, rows, value, matches)
    sums = {}
    for label, frame, versions, weights, actual in [
        ("71", base, identity["geometry71"]["versions"], identity["geometry71"]["weights"], public.raw71),
        ("73", base.assign(v72=public.v72), identity["geometry73"]["versions"], identity["geometry73"]["weights"], public.raw73),
        ("79", matrix, identity["versions"], dict(zip(map(str, identity["versions"]), identity["weights"])), stages.raw79)]:
        # Explicit sequential accumulation is independent of NumPy's matrix BLAS.
        total = np.zeros(len(ids), dtype=np.float64)
        for version in versions:
            value = frame[f"v{version}"].to_numpy(float)
            np.testing.assert_array_equal(value, serial(value))
            total += value * float(weights[str(version)])
        error = float(np.max(np.abs(total - actual)))
        np.testing.assert_allclose(total, actual, atol=1e-8, rtol=0)
        sums[label] = error
    for source, output in [("raw71", "v71"), ("raw72", "v72"), ("raw_refined72", "refined72"), ("raw73", "v73")]:
        np.testing.assert_array_equal(serial(public[source]), public[output])
    monthly_rows = align(pd.read_parquet(c.inputs.OUT / view / "monthly.parquet"), ids)
    monthly_source = c.constraints.monthly.public(2026)
    monthly_params = read(c.V74 / "parameters.json")["full_year"]
    for prediction, actual in [(public.v73, public.monthly74), (public.refined73, public.half_monthly74)]:
        independently = independent_monthly(monthly_rows, prediction, monthly_source,
                                             monthly_params["coefficients"], list(c.constraints.monthly.ZONES))
        np.testing.assert_allclose(independently, actual, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(public.refined73, serial(public.raw73.to_numpy() + identity["geometry73"]["weights"]["72"] * (public.refined72.to_numpy(float) - public.v72.to_numpy(float))))
    np.testing.assert_array_equal(public.full_monthly74, np.maximum(0, 2 * public.half_monthly74 - public.refined73))
    for incoming, raw, integer in [("monthly74", "raw74", "v74"), ("full_monthly74", "raw_refined74", "refined74")]:
        np.testing.assert_array_equal(bound(public[incoming]), public[raw])
        np.testing.assert_array_equal(serial(public[raw]), public[integer])
    np.testing.assert_array_equal(quality.v76_float, bound(public.v74.to_numpy(float) + direct_values.v76_direct.to_numpy() - direct_values.control.to_numpy()))
    np.testing.assert_array_equal(quality.v76, serial(quality.v76_float))
    for arm in ["main", "tabm"]:
        expected = bound(public.refined74.to_numpy(float) + direct_values[arm + "_direct"].to_numpy() - direct_values.control.to_numpy())
        np.testing.assert_array_equal(quality[arm + "_refined_float"], expected)
        np.testing.assert_array_equal(quality[arm + "_refined"], serial(expected))
        expected_shift = independent_daily_shift(quality, quality[arm + "_refined"],
            quality_module.original.daily.public(), read(quality_module.original.OUT / "parameters.json"),
            list(quality_module.original.daily.ZONES))
        np.testing.assert_allclose(quality[arm + "_shift"], expected_shift, atol=1e-8, rtol=0)
        np.testing.assert_array_equal(quality[arm + "_applied"], np.where(quality.quality_eligible, quality[arm + "_shift"], 0.))
        np.testing.assert_array_equal(quality[arm + "_bounded"], bound(quality[arm + "_refined"].to_numpy(float) + quality[arm + "_applied"].to_numpy()))
        np.testing.assert_array_equal(quality[arm + "_pred"], serial(quality[arm + "_bounded"]))
    expected_v78 = serial(bound(quality.main_pred.to_numpy(float) + .5 * increments.increment.to_numpy()))
    expected_tabm = serial(bound(quality.tabm_pred.to_numpy(float) + .5 * increments.increment.to_numpy()))
    expected_robust = serial(bound(expected_v78.astype(float) + .5 * increments.robust_delta.to_numpy()))
    for name, expected in [("v78", expected_v78), ("tabm_v78", expected_tabm), ("robust_v78", expected_robust), ("v79", serial(stages.raw79))]:
        np.testing.assert_array_equal(stages[name], expected)
    np.testing.assert_array_equal(stages.tabm_delta, expected_tabm.astype(float) - expected_v78)
    np.testing.assert_array_equal(stages.robust_delta, expected_robust.astype(float) - expected_v78)
    expected = serial(stages.v79.to_numpy(float) + stages.tabm_delta.to_numpy() + stages.robust_delta.to_numpy())
    np.testing.assert_array_equal(stages.v80, expected)
    for version, values in {72: public.v72, 74: public.v74, 76: quality.v76, 77: quality.main_pred, 78: expected_v78}.items():
        np.testing.assert_array_equal(matrix[f"v{version}"], values)
    for version in set(identity["versions"]) - {72, 74, 76, 77, 78}:
        np.testing.assert_array_equal(matrix[f"v{version}"], base[f"v{version}"])
    candidate = pq.read_table(folder / "candidate.parquet")
    if candidate.schema != template.schema or candidate.num_rows != expected_rows:
        raise ValueError("Candidate schema/population mismatch")
    for name in template.column_names:
        if name != "TAXITIME_SEC_mvt" and not candidate[name].equals(template[name]):
            raise ValueError(f"Candidate modifies {name}")
    np.testing.assert_array_equal(candidate["TAXITIME_SEC_mvt"].to_numpy(), expected)
    diagnostic = {}
    if view == "old":
        values = {71: public.v71, 72: public.v72, 73: public.v73, 74: public.v74,
                  76: quality.v76, 77: quality.main_pred, 78: stages.v78, 79: stages.v79, 80: stages.v80}
        for version, value in values.items():
            original = pd.read_parquet(ROOT / f"submissions/unique-rose_v{version}.parquet")
            np.testing.assert_array_equal(original.MVT_ID_mvt, ids)
            delta = np.asarray(value, float) - original.TAXITIME_SEC_mvt.to_numpy(float)
            diagnostic[str(version)] = dict(rms_seconds=float(np.sqrt(np.mean(delta ** 2))),
                changed_rows=int(np.count_nonzero(delta)), max_abs_seconds=float(np.max(np.abs(delta))))
    receipt = dict(status="pass", identity=art(out / "assembly_identity.json"), result=art(folder / "result.json"),
        view=view, rows=len(ids), candidate=art(folder / "candidate.parquet"), ordered_sum_max_error=sums,
        independent_support_and_transport_checks=True, all_template_columns_exact=True,
        finite_nonnegative_integer=True, diagnostic_vs_original_old_predictions=diagnostic,
        diagnostic_is_not_validation_or_leaderboard_score=True, no_original_weight_replay_claim=True)
    write(folder / "audit.json", receipt)
    return receipt


def required_files():
    """Read JSON metadata only; no parquet reads, hashes, imports or inference."""
    paths = {Path(__file__), CATALOG, SELECTED, GEOMETRY79 / "prepared.json", GEOMETRY79 / "result.json", MANIFEST79,
        RUNS / "v3_20261012_final_public_corrections/identity.json",
        RUNS / "v3_20261011_physical_affine/submission_v71/geometry.json",
        RUNS / "v3_20261011_expanded_affine/submission_v73/geometry.json",
        RUNS / "v3_20261011_daily_delay_submit/submission_v72/parameters.json",
        RUNS / "v3_20261011_public_constraints/submission_v74/parameters.json",
        RUNS / "v3_20261011_public_daily_taxi_submit/parameters.json",
        RUNS / "v3_20261011_public_daily_taxi_submit/result.json",
        RUNS / "v3_20261011_public_daily_taxi_submit/identity.json",
        RUNS / "v3_20261012_final_public_direct_components/identity.json",
        RUNS / "v3_20261012_final_public_direct_components/independent_audit.json",
        RUNS / "v3_20261012_final_correction_inputs/identity.json",
        RUNS / "v3_20261012_final_correction_inputs/audit.json",
        RUNS / "v3_20261012_final_joint_public_inference/identity.json",
        RUNS / "v3_20261012_final_joint_public_inference/prepared.json",
        RUNS / "v3_20261012_final_joint_public_inference/input_audit.json"}
    for record in read(CATALOG)["components"].values():
        for key in ["old", "final", "audit", "result"]:
            if key in record:
                paths.add(ROOT / record[key]["path"])
    for view in VIEWS:
        prefix = "" if view == "old" else "final_"
        paths.update([ROOT / f"data/{prefix}submitting.parquet", ROOT / f"data/{prefix}ranking.parquet"])
        for name in ["prepared.json", "daily.parquet", "monthly.parquet", "matches.parquet", "clocks.parquet"]:
            paths.add(RUNS / f"v3_20261012_final_correction_inputs/{view}/{name}")
        for name in ["result.json", "audit.json", "increments.parquet"]:
            paths.add(RUNS / f"v3_20261012_final_joint_public_inference/{view}/{name}")
        paths.update(RUNS / f"v3_20261012_final_public_direct_components/{view}.{ext}" for ext in ["json", "parquet"])
        raw = (RUNS / "v3_20261011_public_daily_taxi_features/inputsranking.parquet" if view == "old"
               else RUNS / "v3_20261012_final_daily_main_inference/features/inputsfinal.parquet")
        paths.update([raw, raw.parent / "manifest.json", raw.parent / "audit.json"])
    # Source and public tables are recorded in the original correction identity.
    identity = read(RUNS / "v3_20261012_final_public_corrections/identity.json")
    for record in identity["sources"]:
        if record["path"].endswith(".py") or record["path"].endswith("all_pre_departure_delays_2026.csv") or record["path"].endswith("taxi_out_additional_time_2026.csv"):
            paths.add(ROOT / record["path"])
    paths.add(ROOT / "data/external/eurocontrol_daily_taxi_20261011/daily_runway.parquet")
    paths.update(ROOT / f"submissions/unique-rose_v{v}.parquet" for v in [71, 72, 73, 74, 76, 77, 78, 79, 80])
    return sorted(paths)


def plan(out):
    selected = read(SELECTED)
    return dict(stage="assembly_plan", out=str(out), versions=selected["versions"],
        required_fresh_versions=TREE_VERSIONS + NEURAL_VERSIONS, retained_versions=sorted(map(int, read(CATALOG)["components"])),
        regenerates=["v71", "Gaussian and t5 daily posterior", "v73", "monthly and plan corrections", "v76", "v77", "v78", "v79", "three-TabM and runway v80"],
        reuses="Audited direct component models and joint/runway inference with unchanged v70 prior",
        inputs_ready={name: (out / name / "result.json").is_file() for name in ["trees", "neural"]},
        required_files=[str(p.relative_to(ROOT)) for p in required_files()],
        candidate=str(out / "final/candidate.parquet"), no_fit=True, no_upload=True)


def run(out, threads=4):
    from threadpoolctl import threadpool_limits
    mods = modules()
    # One numerical thread matches the original selected correction/affine
    # pipeline. --threads remains a shared runner interface, not a recipe change.
    with threadpool_limits(1):
        identity = freeze(out, mods)
        for view in VIEWS:
            build(view, out, identity, mods)
            audit(view, out, identity, mods)
            gc.collect()
        final = read(out / "final/audit.json")
        write(out / "audit.json", dict(status="pass", recipe="v80 fixed recipe with full-2025 refitted missing models",
            identity=art(out / "assembly_identity.json"), candidate=final["candidate"],
            views={view: art(out / view / "audit.json") for view in VIEWS},
            rows=final["rows"], fresh_weights_not_exact_historical_replay=True,
            no_fit=True, no_upload=True))
    print(json.dumps(dict(stage="assembly_complete", status="pass", candidate=str(out / "final/candidate.parquet"))), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["plan", "run"])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.stage == "plan":
        print(json.dumps(plan(args.out), indent=2))
    else:
        run(args.out.resolve(), args.threads)
