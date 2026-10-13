"""Independent column-at-a-time Pandas/NumPy reproduction of public view changes.

No model/prediction, coefficient fit, departure label, or inferred departure
block clock. The fixed arrival diagnostic cannot prove a departure mechanism.
"""
from pathlib import Path
import hashlib
import json
import platform
import resource
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "VALIDATION/reports/final_shared_clocks_20261013/independent_numpy.json"
EXPECTED = {
    "ranking": "9db75e3a56fa09b87aab72e191a53da3fe763d9887ad2b23aea9cde6a6325f86",
    "final_ranking": "4b0692a792846079516a2749e034b0abda55df72c213eabdcda52a4d1f6bc69f",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    started = time.monotonic()
    assert not OUT.exists(), "Preserve existing independent receipt; do not rerun."
    paths = {k: ROOT / "data" / (k + ".parquet") for k in EXPECTED}
    for k, path in paths.items():
        assert digest(path) == EXPECTED[k], (k, "raw source hash mismatch")
    parquet = {k: pq.ParquetFile(v) for k, v in paths.items()}
    names = parquet["ranking"].schema_arrow.names
    assert names == parquet["final_ranking"].schema_arrow.names

    def read(view, column):
        return parquet[view].read(columns=[column], use_threads=False).to_pandas()[column]

    old_ids, new_ids = read("ranking", "MVT_ID_mvt"), read("final_ranking", "MVT_ID_mvt")
    assert old_ids.notna().all() and new_ids.notna().all()
    assert old_ids.is_unique and new_ids.is_unique
    index = pd.Index(new_ids).get_indexer(old_ids)
    assert (index >= 0).all() and len(old_ids) == 689534
    phase = read("ranking", "PHASE_mvt").to_numpy()
    arrival, departure = phase == "ARR", phase == "DEP"
    assert arrival.sum() == 344693 and departure.sum() == 344841
    assert (arrival | departure).all()
    fields, saved = [], {}
    keep = {"MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt", "ADES_mvt"}
    for column in names:
        a = read("ranking", column)
        b = read("final_ranking", column).iloc[index].reset_index(drop=True)
        equal = (a.eq(b) | (a.isna() & b.isna())).to_numpy()
        for label, mask in [("ARR", arrival), ("DEP", departure)]:
            fields.append({"field": column, "phase": label,
                           "rows": int(mask.sum()), "changed": int((~equal & mask).sum())})
        if column not in {"MVT_TIME_UTC_mvt", "BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"}:
            assert equal.all(), (column, "unexpected nonclock change")
        if column in keep:
            saved[column] = (a, b)
    for column in ["BLOCK_TIME_UTC_mvt", "TAXITIME_SEC_mvt"]:
        assert all(s.iloc[np.flatnonzero(departure)].isna().all() for s in saved[column])
    dt = (saved["MVT_TIME_UTC_mvt"][1] - saved["MVT_TIME_UTC_mvt"][0]).dt.total_seconds().to_numpy()
    db = (saved["BLOCK_TIME_UTC_mvt"][1] - saved["BLOCK_TIME_UTC_mvt"][0]).dt.total_seconds().to_numpy()
    dy = (saved["TAXITIME_SEC_mvt"][1] - saved["TAXITIME_SEC_mvt"][0]).to_numpy()
    assert np.isfinite(dt).all() and np.array_equal(dt, np.rint(dt))
    assert np.max(np.abs(dt)) == 12
    assert np.array_equal(dy[arrival], (db - dt)[arrival])
    residual = db + dt
    assert np.isfinite(residual[arrival]).all() and np.max(np.abs(residual[arrival])) == 1
    airports = saved["ADES_mvt"][0].to_numpy()

    def summary(mask):
        r = residual[mask]
        values, counts = np.unique(r, return_counts=True)
        return {"n": int(mask.sum()), "correlation": float(np.corrcoef(dt[mask], db[mask])[0, 1]),
                "residual_rms": float(np.sqrt(np.mean(r ** 2))),
                "maximum_absolute": float(np.max(np.abs(r))),
                "label_identity_failures": int(np.count_nonzero(dy[mask] != (db - dt)[mask])),
                "outside_one_second": int(np.count_nonzero(np.abs(r) > 1)),
                "residual_histogram": [{"residual": float(v), "n": int(n)} for v, n in zip(values, counts)]}

    overall = summary(arrival)
    assert overall["residual_histogram"] == [
        {"residual": -1.0, "n": 86014}, {"residual": 0.0, "n": 172383}, {"residual": 1.0, "n": 86296}]
    by_airport = [{"airport": str(a), **summary(arrival & (airports == a))}
                  for a in sorted(np.unique(airports[arrival]))]
    assert len(by_airport) == 10
    report = {"status": "pass", "method": "Independent Pandas ID indexer; column-wise null-aware equality and NumPy arithmetic; no original SQL/helper imports",
              "source": {"path": str(Path(__file__).relative_to(ROOT)), "sha256": digest(__file__)},
              "inputs": {k: {"path": str(p.relative_to(ROOT)), "sha256": EXPECTED[k]} for k, p in paths.items()},
              "host": platform.node(), "libraries": {"numpy": np.__version__, "pandas": pd.__version__},
              "shared_rows": int(len(old_ids)), "final_rows": int(len(new_ids)), "fields": fields,
              "overall": overall, "airports": by_airport,
              "departure_movement_delta": {"rows": int(departure.sum()), "integer_seconds": True,
                  "min": float(dt[departure].min()), "max": float(dt[departure].max()),
                  "rms": float(np.sqrt(np.mean(dt[departure] ** 2)))},
              "departure_hidden_values_all_null": True, "no_model": True, "no_candidate": True,
              "no_departure_label_claim": True, "no_cv_claim": True,
              "elapsed_seconds": time.monotonic() - started,
              "peak_rss_MiB": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "conclusion": "Visible arrival identity independently reproduced on every shared row. Hidden departure block-clock transfer remains unobserved."}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": "pass", "path": str(OUT.relative_to(ROOT)), "sha256": digest(OUT),
                      "elapsed_seconds": report["elapsed_seconds"], "peak_rss_MiB": report["peak_rss_MiB"],
                      "overall": overall}, allow_nan=False))


if __name__ == "__main__":
    main()
