"""Fixed, explicitly conditional old/final shared-movement transport.

Arrival evidence does not prove the hidden departure block-clock transform.
There is no authentic paired-departure Jan/Jul CV for this operator, and this is
not an exact replay of the v80 models on final inputs. No fit or upload occurs.
Freeze only after source review and the complete portable-chain comparison.
Build is a separate root-allocated action; importing/testing the pure interface
does not read any data or create a candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "VALIDATION/runs/v3_20261013_final_shared_clock_transport"
MANIFEST = OUT / "identity.json"
DESIGN = ROOT / "VALIDATION/reports/cohort_learning_20261013/final_shared_clock_transport_assessment.json"
CLOCK = ROOT / "VALIDATION/reports/final_shared_clocks_20261013/independent_numpy.json"
V80 = ROOT / "VALIDATION/runs/v3_20261012_tabm_pool3_candidate/submission_v80"
V80_SHA = "50b986030af2efaab89b69bb69522ade6b060cc7f58fa324954e3fb06e0ef6f4"
CLOCK_SHA = "0f81f1a9d0081819e48cfa59b0c46749f26ec5d7155e0f108f9cd2780952ebe2"
RAW_SHA = {"ranking": "9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86",
           "final_ranking": "4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f"}
QUALIFICATION = {
    "departure_assumption": "Hidden departure block change is the opposite of observed takeoff change up to the same rounding residual seen in arrivals; unverified for departures.",
    "arrival_evidence": "All344693 shared arrivals satisfy delta_block+delta_movement in {-1,0,1}; independently reproduced from supplied data.",
    "validation_limit": "No genuine paired-departure Jan/Jul CV; synthetic shifted-target checks are not predictive evidence.",
    "not_exact_v80_final_replay": True,
    "old_leaderboard_improvement_claim": False,
    "final_score_known": False,
}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def art(path):
    path = Path(path).resolve()
    return {"path": str(path.relative_to(ROOT)), "sha256": sha(path)}


def read(path):
    return json.loads(Path(path).read_text())


def verify(record):
    path = (ROOT / record["path"]).resolve()
    assert path.is_relative_to(ROOT) and sha(path) == record["sha256"], record
    return path


def write(path, value):
    path = Path(path)
    assert not path.exists(), "Preserve existing records; no implicit retry/overwrite."
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def vector(value, length, name):
    value = np.asarray(value)
    assert value.ndim == 1 and len(value) == length, name
    assert np.issubdtype(value.dtype, np.number) and np.isfinite(value).all(), name
    return value.astype(np.float64)


def ids(value, name):
    value = np.asarray(value)
    result = vector(value, len(value), name)
    assert np.array_equal(result, np.rint(result)) and np.max(np.abs(result), initial=0) <= 2**53, name
    index = pd.Index(result)
    assert index.is_unique, name
    return index


def transform_all_ids(final_ids, old_ids, old_prediction, old_takeoff_seconds,
                      final_takeoff_seconds, portable_prediction):
    """Pure fixed operator; all final arrays follow final_ids, old arrays old_ids.

    Returns serialized int32 predictions, exact shared-ID mask and a delta array
    (zero for new IDs). No labels, calendar selector, coefficient argument or IO.
    Fallback predictions must already be nonnegative integers, preserving their
    established serialization exactly. IDs may arrive in any unique order.
    """
    wanted, old = ids(final_ids, "final IDs"), ids(old_ids, "old IDs")
    n, m = len(wanted), len(old)
    assert n > 0 and m > 0 and m <= n
    base = vector(portable_prediction, n, "portable prediction")
    prior = vector(old_prediction, m, "v80 prediction")
    for name, values in [("portable", base), ("v80", prior)]:
        assert (values >= 0).all() and np.array_equal(values, np.rint(values)), name
        assert (values <= np.iinfo(np.int32).max).all(), name
    ot = vector(old_takeoff_seconds, m, "old takeoff")
    ft = vector(final_takeoff_seconds, n, "final takeoff")
    at = wanted.get_indexer(old)
    assert (at >= 0).all(), "Every old ID must occur exactly once in final."
    shared = np.zeros(n, dtype=bool)
    shared[at] = True
    dt = np.zeros(n, dtype=np.float64)
    dt[at] = ft[at] - ot
    assert np.array_equal(dt, np.rint(dt)) and np.max(np.abs(dt)) <= 12
    output = base.copy()
    output[at] = prior + 2 * dt[at]
    output = np.rint(np.maximum(output, 0))
    assert np.isfinite(output).all() and (output <= np.iinfo(np.int32).max).all()
    result = output.astype(np.int32)
    np.testing.assert_array_equal(result[~shared], base[~shared])
    return result, shared, dt


def freeze(portable_result, portable_audit, source_review):
    assert not MANIFEST.exists(), "One fixed recipe; do not overwrite or tune."
    portable_result, portable_audit, source_review = map(Path, [portable_result, portable_audit, source_review])
    pr, audit, review = read(portable_result), read(portable_audit), read(source_review)
    assert review["status"] == "pass" and review["builder"] == art(__file__)
    assert review["assessment"] == art(DESIGN)
    assert pr["view"] == audit["view"] == "final"
    assert audit["status"] == "pass" and audit["result"] == art(portable_result)
    assert audit["identity"] == pr["identity"] and audit["candidate"] == pr["candidate"]
    assert pr["rows"] == audit["rows"] == 670790
    portable = verify(pr["candidate"])
    portable_identity = verify(pr["identity"])
    manifest, official = read(V80 / "manifest.json"), read(V80 / "official_result.json")
    assert manifest["version"] == 80 and manifest["state"] == "scored"
    assert manifest["rows"] == 344841 and manifest["sha256"] == manifest["remote_sha256"] == V80_SHA
    assert official == manifest["official_result"] and official["status"] == "Succeeded"
    assert official["file"] == "unique-rose_v80.parquet" and official["used_pairs"] == 344841
    old = verify({"path": manifest["file"], "sha256": V80_SHA})
    component_path, checked_path = verify(manifest["component"]), verify(manifest["audit"])
    component, checked = read(component_path), read(checked_path)
    assert checked["status"] == "pass" and checked["result"] == manifest["component"]
    assert component["identity"] == checked["identity"]
    assert checked["file"] in component["artifacts"] and checked["file"]["sha256"] == V80_SHA
    assert sha(CLOCK) == CLOCK_SHA
    clock = read(CLOCK)
    assert clock["status"] == "pass" and clock["departure_hidden_values_all_null"]
    assert clock["shared_rows"] == 689534 and clock["overall"]["n"] == 344693
    assert clock["overall"]["label_identity_failures"] == clock["overall"]["outside_one_second"] == 0
    assert clock["overall"]["residual_histogram"] == [
        {"residual": -1.0, "n": 86014}, {"residual": 0.0, "n": 172383}, {"residual": 1.0, "n": 86296}]
    raw = {}
    for name, expected in RAW_SHA.items():
        record = clock["inputs"][name]
        assert record == {"path": f"data/{name}.parquet", "sha256": expected}
        raw[name] = verify(record)
    template = ROOT / "data/final_submitting.parquet"
    assert pq.ParquetFile(template).metadata.num_rows == 670790
    sources = [Path(__file__), DESIGN, CLOCK, verify(clock["source"]), source_review,
               portable_result, portable_audit, portable_identity, portable,
               V80 / "manifest.json", V80 / "official_result.json", component_path,
               checked_path, verify(checked["file"]), verify(component["identity"]), old,
               template, *raw.values()]
    write(MANIFEST, {"name": "fixed_shared_clock_transport_v1", "coefficient": 2,
        "sources": [art(p) for p in sorted(set(sources))],
        "v80": art(old), "portable": art(portable), "template": art(template),
        "ranking": art(raw["ranking"]), "final_ranking": art(raw["final_ranking"]),
        "source_review": art(source_review), "portable_result": art(portable_result),
        "portable_audit": art(portable_audit), "qualifications": QUALIFICATION,
        "selector": "Exact old submission ID membership; new IDs use unchanged portable candidate",
        "serialization": "max(0), nearest-even, int32 once after shared2dt replacement",
        "no_fit": True, "no_coefficient_search": True, "no_cv_gain_claim": True})
    print(json.dumps({"frozen": art(MANIFEST), "candidate_written": False}))


def aligned_clock(path, wanted):
    frame = pd.read_parquet(path, columns=["MVT_ID_mvt", "PHASE_mvt", "MVT_TIME_UTC_mvt"])
    assert frame.MVT_ID_mvt.notna().all() and frame.MVT_ID_mvt.is_unique
    dep = frame.loc[frame.PHASE_mvt == "DEP"].set_index("MVT_ID_mvt")
    assert len(dep) == len(wanted) and pd.Index(wanted).difference(dep.index).empty
    times = dep.loc[wanted, "MVT_TIME_UTC_mvt"]
    assert times.notna().all()
    return (times - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds().to_numpy()


def build():
    identity = read(MANIFEST)
    assert identity["coefficient"] == 2 and identity["qualifications"] == QUALIFICATION
    assert art(__file__) in identity["sources"]
    for record in identity["sources"]:
        verify(record)
    for name in ["candidate.parquet", "trace.parquet", "result.json"]:
        assert not (OUT / name).exists(), "Partial/completed build exists; audit before any retry."
    template = pq.read_table(verify(identity["template"]))
    original = pq.read_table(verify(identity["v80"]))
    portable = pq.read_table(verify(identity["portable"]))
    assert template.schema == portable.schema
    final_ids = template["MVT_ID_mvt"].to_numpy()
    old_ids = original["MVT_ID_mvt"].to_numpy()
    assert len(final_ids) == 670790 and len(old_ids) == 344841
    assert portable["MVT_ID_mvt"].equals(template["MVT_ID_mvt"])
    for name in template.column_names:
        if name != "TAXITIME_SEC_mvt":
            assert portable[name].equals(template[name])
    old_time = aligned_clock(verify(identity["ranking"]), old_ids)
    final_time = aligned_clock(verify(identity["final_ranking"]), final_ids)
    values, shared, delta = transform_all_ids(final_ids, old_ids,
        original["TAXITIME_SEC_mvt"].to_numpy(), old_time, final_time,
        portable["TAXITIME_SEC_mvt"].to_numpy())
    assert int(shared.sum()) == 344841 and int((~shared).sum()) == 325949
    field = template.schema.field("TAXITIME_SEC_mvt")
    candidate = template.set_column(template.schema.get_field_index(field.name), field,
                                    pa.array(values, type=field.type))
    assert candidate.schema == template.schema
    for name in template.column_names:
        if name != field.name:
            assert candidate[name].equals(template[name])
    trace = pd.DataFrame({"MVT_ID_mvt": final_ids, "shared": shared,
                          "delta_takeoff_seconds": delta, "submitted": values})
    trace.to_parquet(OUT / "trace.parquet", index=False)
    pq.write_table(candidate, OUT / "candidate.parquet")
    assert pq.read_table(OUT / "candidate.parquet").equals(candidate)
    pd.testing.assert_frame_equal(pd.read_parquet(OUT / "trace.parquet"), trace, check_exact=True)
    write(OUT / "result.json", {"identity": art(MANIFEST), "rows": len(values),
        "shared_rows": int(shared.sum()), "portable_rows": int((~shared).sum()),
        "candidate": art(OUT / "candidate.parquet"), "trace": art(OUT / "trace.parquet"),
        "qualifications": QUALIFICATION, "coefficient": 2,
        "independent_audit_pending": True, "selected_for_final": False,
        "no_new_cv_score": True, "no_fit": True, "no_upload": True})
    print(json.dumps({"result": art(OUT / "result.json"), "rows": len(values)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freezer = sub.add_parser("freeze")
    for name in ["portable-result", "portable-audit", "source-review"]:
        freezer.add_argument("--" + name, required=True, type=Path)
    sub.add_parser("build")
    args = parser.parse_args()
    if args.command == "freeze":
        freeze(args.portable_result, args.portable_audit, args.source_review)
    else:
        build()
