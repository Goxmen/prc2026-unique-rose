#!/usr/bin/env python3
"""Export, verify and replay accepted v83 from retained component predictions.

No training, network, credentials, upload, or imports from the research pipeline.
The export contains component predictions, not the accepted final prediction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
RUN = Path("VALIDATION/runs/v3_20261013_v80_teacher_distillation")
IDENTITY_SHA = "94b63f64ca5021036422f70304664d7128beecdd418aa051b3257227cf638d41"
MODEL_SHA = "21031645415924c157af701784ea4ea4c5a7944cad601679df373cccdcc2065e"
FINAL_SHA = "f4f6a15fe5bb0865727cb906b77e17f182d34dc4b516f657c26937ad097af85f"
VALUES_SHA = "1c0a15708b34b95eb98f8d188270e9af3483ecae36de4c117f2fb8bfe31f48a1"
VERSIONS = [1, 4, 8, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 30, 33, 34,
            37, 38, 45, 46, 54, 56, 57, 58, 60, 61, 64, 70]
ROWS = 670790


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def safe_path(root, relative):
    relative = Path(relative)
    require(not relative.is_absolute() and ".." not in relative.parts,
            f"Unsafe relative path: {relative}")
    path = (Path(root) / relative).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Path escapes bundle/root")
    return path


def checked(root, record):
    path = safe_path(root, record["path"])
    require(path.is_file(), f"Missing artifact: {record['path']}")
    require(sha256(path) == record["sha256"], f"Changed artifact: {record['path']}")
    return path


def record(path, root):
    return dict(path=str(Path(path).relative_to(root)), sha256=sha256(path),
                bytes=Path(path).stat().st_size)


def integer_vector(values):
    values = np.asarray(values, dtype=np.float64)
    require(np.isfinite(values).all(), "Nonfinite prediction")
    rounded = np.rint(np.maximum(values, 0))
    require(rounded.max(initial=0) <= np.iinfo(np.int32).max, "Prediction overflows int32")
    return rounded.astype(np.int32)


def align(frame, ids, id_column="mvt_id"):
    index = pd.Index(frame[id_column])
    require(not index.hasnans and index.is_unique, "Missing/duplicate component IDs")
    require(len(index) == len(ids) and pd.Index(ids).isin(index).all(), "Component IDs differ")
    return frame.set_index(id_column).loc[ids].reset_index()


def constrain(ids, raw, values, matches):
    """Exact v83 order: matched bound, nonnegative, recovered exclusion, rint."""
    values = np.asarray(values, dtype=np.float64).copy()
    gap = raw.nm_gap.to_numpy(float)
    eligible = raw.has_id.to_numpy(bool) & np.isfinite(gap)
    values[eligible] = np.maximum(0, np.clip(values[eligible], gap[eligible] - 3606,
                                           gap[eligible] + 3606))
    values = np.maximum(values, 0)
    accepted = matches.loc[matches.accepted].set_index("mvt_id")
    require(accepted.index.is_unique, "Duplicate accepted recovered-plan match")
    gap = accepted.reindex(ids).recovered_gap.to_numpy(float)
    low, high = gap - 3606, gap + 3606
    inside = np.isfinite(gap) & (values > low) & (values < high)
    lower = (low[inside] >= 0) & (values[inside] - low[inside] <= high[inside] - values[inside])
    values[inside] = np.where(lower, low[inside], high[inside])
    return integer_vector(values)


def values_sha256(ids, values):
    digest = hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes())
    digest.update(np.asarray(values, dtype="<i4").tobytes())
    return digest.hexdigest()


def load_sources(root):
    identity_path = checked(root, dict(path=str(RUN / "identity.json"), sha256=IDENTITY_SHA))
    model_path = checked(root, dict(path=str(RUN / "production/model.json"), sha256=MODEL_SHA))
    identity, model = read_json(identity_path), read_json(model_path)
    require(identity["versions"] == model["columns"] == VERSIONS, "Changed component order")
    return identity, model


def source_records(identity):
    specs = identity["specs"]["final"]
    return [s["artifact"] for s in specs.values()] + [
        identity["components"][str(version)]["final"]["artifact"] for version in VERSIONS]


def inventory(root):
    identity, _ = load_sources(root)
    records = source_records(identity)
    files = [dict(**r, present=safe_path(root, r["path"]).is_file(),
                  bytes=safe_path(root, r["path"]).stat().st_size
                  if safe_path(root, r["path"]).is_file() else None) for r in records]
    return dict(scope="Retained-component replay; no source hashes read in inventory",
                rows=ROWS, component_count=len(VERSIONS), files=files,
                total_present_bytes=sum(r["bytes"] or 0 for r in files))


def export(root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    require(not destination.exists(), "Export destination already exists; preserve it")
    identity, model = load_sources(root)
    candidate = checked(root, dict(path=str(RUN / "production/final/candidate.parquet"), sha256=FINAL_SHA))
    expected = pq.read_table(candidate, use_threads=False)
    expected_ids = expected["MVT_ID_mvt"].to_numpy()
    expected_values = expected["TAXITIME_SEC_mvt"].to_numpy()
    require(len(expected_ids) == ROWS, "Accepted final row count changed")
    destination.mkdir(parents=True)
    write_json(destination / "EXPORT_INCOMPLETE.json", dict(scope="Never use an incomplete export"))
    files = []

    def project(name, spec):
        original = checked(root, spec["artifact"])
        table = pq.read_table(original, columns=spec["columns"], use_threads=False)
        require(len(table) == spec["rows"], f"Changed source row count: {name}")
        path = destination / name
        pq.write_table(table, path)
        files.append(dict(**record(path, destination), original=spec["artifact"],
                          columns=spec["columns"], rows=spec["rows"]))

    for key, spec in identity["specs"]["final"].items():
        if key == "template":
            source = checked(root, spec["artifact"])
            shutil.copyfile(source, destination / "template.parquet")
            files.append(dict(**record(destination / "template.parquet", destination), original=spec["artifact"]))
        else:
            project(f"{key}.parquet", spec)
    for version in VERSIONS:
        project(f"component_{version}.parquet", identity["components"][str(version)]["final"])
    for name, source in [("identity.json", RUN / "identity.json"), ("model.json", RUN / "production/model.json")]:
        shutil.copyfile(root / source, destination / name)
        files.append(record(destination / name, destination))
    # Preserve the exact formula sources for review; replay imports none of them.
    originals = {r["path"]: r for r in identity["sources"]}
    for name in ["portable_v80_distill_20261013.py", "portable_public_source_20261013.py",
                 "final_public_corrections.py", "public_constraints_candidate.py",
                 "recovered_plan_exclusion_fixed.py", "submit.py"]:
        original = originals[f"VALIDATION/{name}"]
        source = checked(root, original)
        folder = destination / "reference_source"
        folder.mkdir(exist_ok=True)
        shutil.copyfile(source, folder / name)
        files.append(dict(**record(folder / name, destination), original=original))
    for name in ["production/audit.json", "final_submission/acceptance.json", "final_submission/official_result.json"]:
        source = root / RUN / name
        target = destination / "evidence" / name.replace("/", "_")
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        files.append(record(target, destination))
    manifest = dict(format="prc-v83-component-replay-v1", rows=ROWS, versions=VERSIONS,
                    accepted_final_sha256=FINAL_SHA,
                    expected_values_sha256=values_sha256(expected_ids, expected_values),
                    files=files, component_columns={str(v): {
                        "id": identity["components"][str(v)]["id_column"],
                        "value": identity["components"][str(v)]["value_column"]} for v in VERSIONS},
                    environment=dict(python=platform.python_version(), numpy=np.__version__,
                                     pandas=pd.__version__, pyarrow=pa.__version__),
                    scope="Exact final assembly from retained predictions and coefficients; no raw-data retraining")
    write_json(destination / "manifest.json", manifest)
    (destination / "EXPORT_INCOMPLETE.json").unlink()
    return verify(destination)


def verify(bundle, manifest_sha256=None):
    bundle = Path(bundle).resolve()
    require(not (bundle / "EXPORT_INCOMPLETE.json").exists(), "Incomplete export")
    if manifest_sha256 is not None:
        require(sha256(bundle / "manifest.json") == manifest_sha256, "Bundle manifest hash differs")
    manifest = read_json(bundle / "manifest.json")
    require(manifest["format"] == "prc-v83-component-replay-v1", "Unknown bundle format")
    require(manifest["rows"] == ROWS and manifest["versions"] == VERSIONS, "Wrong final population/components")
    require(manifest["accepted_final_sha256"] == FINAL_SHA, "Wrong accepted final identity")
    require(manifest["expected_values_sha256"] == VALUES_SHA, "Wrong accepted prediction digest")
    required = {"identity.json", "model.json", "anchor.parquet", "raw.parquet", "matches.parquet", "template.parquet"}
    required.update(f"component_{version}.parquet" for version in VERSIONS)
    require(required.issubset({r["path"] for r in manifest["files"]}), "Required packet missing from manifest")
    require(len({r["path"] for r in manifest["files"]}) == len(manifest["files"]), "Duplicate file record")
    for item in manifest["files"]:
        checked(bundle, item)
    require(sha256(bundle / "identity.json") == IDENTITY_SHA, "Changed original identity")
    require(sha256(bundle / "model.json") == MODEL_SHA, "Changed production coefficients")
    return dict(status="pass", files=len(manifest["files"]),
                bytes=sum(r["bytes"] for r in manifest["files"]),
                manifest_sha256=sha256(bundle / "manifest.json"), rows=ROWS)


def reproduce(bundle, output, manifest_sha256=None):
    bundle, output = Path(bundle).resolve(), Path(output).resolve()
    require(not output.exists(), "Output already exists; preserve it")
    require(not output.with_suffix(output.suffix + ".replay.json").exists(), "Replay receipt already exists")
    require(not output.is_relative_to(bundle), "Write regenerated output outside the bundle")
    verification = verify(bundle, manifest_sha256)
    manifest, model = read_json(bundle / "manifest.json"), read_json(bundle / "model.json")
    template = pq.read_table(bundle / "template.parquet", use_threads=False)
    ids = template["MVT_ID_mvt"].to_numpy()
    require(len(ids) == ROWS and pd.Index(ids).is_unique and not pd.Index(ids).hasnans, "Invalid template IDs")
    anchor = align(pd.read_parquet(bundle / "anchor.parquet"), ids)
    raw = align(pd.read_parquet(bundle / "raw.parquet"), ids)
    matches = pd.read_parquet(bundle / "matches.parquet")
    z = anchor.submitted.to_numpy(float)
    require(np.array_equal(z, integer_vector(z)), "Noninteger anchor")
    require(np.array_equal(z, constrain(ids, raw, z, matches)), "Anchor constraint replay differs")
    matrix = np.empty((len(ids), len(VERSIONS)), dtype=np.float64)
    for column, version in enumerate(VERSIONS):
        spec = manifest["component_columns"][str(version)]
        frame = align(pd.read_parquet(bundle / f"component_{version}.parquet"), ids, spec["id"])
        values = frame[spec["value"]].to_numpy(float)
        require(np.array_equal(values, integer_vector(values)), "Noninteger component")
        matrix[:, column] = values
    beta = np.asarray(model["beta"], dtype=np.float64)
    require(model["columns"] == VERSIONS and np.isfinite(beta).all(), "Invalid coefficient vector")
    # Preserve original C-contiguous matrix product and float64 arithmetic order.
    values = constrain(ids, raw, z + (matrix - z[:, None]) @ beta, matches)
    require(values_sha256(ids, values) == manifest["expected_values_sha256"], "Final predictions differ from accepted values")
    field = template.schema.field("TAXITIME_SEC_mvt")
    result = template.set_column(template.schema.get_field_index(field.name), field,
                                 pa.array(values, type=field.type))
    require(result.schema == template.schema and result["MVT_ID_mvt"].equals(template["MVT_ID_mvt"]),
            "Template schema/IDs changed")
    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(result, output)
    require(pq.read_table(output, use_threads=False).equals(result), "Output roundtrip changed values")
    output_sha = sha256(output)
    report = dict(status="pass", scope="Retained-component replay only", rows=len(values),
                  manifest_sha256=verification["manifest_sha256"], output_sha256=output_sha,
                  accepted_final_sha256=FINAL_SHA, byte_exact=output_sha == FINAL_SHA,
                  values_exact=True, no_training=True, no_network=True, no_upload=True)
    write_json(output.with_suffix(output.suffix + ".replay.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["inventory", "export", "verify", "reproduce"])
    parser.add_argument("--root", type=Path, default=ROOT, help="Original checkout; inventory/export only")
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest-sha256", help="Published bundle manifest checksum; verify/reproduce only")
    args = parser.parse_args()
    if args.command != "inventory" and args.bundle is None:
        parser.error("--bundle is required")
    if args.command == "reproduce" and args.output is None:
        parser.error("--output is required")
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    with threadpool_limits(1):
        if args.command == "inventory":
            result = inventory(args.root)
        elif args.command == "export":
            result = export(args.root, args.bundle)
        elif args.command == "verify":
            result = verify(args.bundle, args.manifest_sha256)
        else:
            result = reproduce(args.bundle, args.output, args.manifest_sha256)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
