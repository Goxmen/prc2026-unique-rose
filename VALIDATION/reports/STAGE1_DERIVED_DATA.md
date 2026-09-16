# Notebook cache inventory for Stage 1

Checked the existing local files on 2026-09-16; none were rebuilt or overwritten.
Cell numbers below are zero-based. The notebook itself was not changed.

| Local file in `data/derived/` | Rows | Notebook section / writer cell | Stage 1 use |
|---|---:|---|---|
| `recovered_flt_blocks.parquet` | 1,118 (892 training, 226 ranking) | §2b / 15 | Exact-twin reference for R1 |
| `recovered_flt_relaxed.parquet` | 1,260 (1,007 training, 253 ranking) | §2c, idea 1 / 18 | Relaxed-stage reference for R2 |
| `imputed_flt_static.parquet` | 25,329 (20,525 training, 4,804 ranking) | §2c, idea 2 / 19 | Static policy/reference for R3 |
| `rule_recovered.parquet` | 773 (689 training, 84 ranking) | §4a / 35 | Separate movement-rule change; outside R1–R3 |
| `dest_recovered.parquet` | 243 (214 training, 29 ranking) | §6a / 54 (reference setup in 53) | Separate movement-destination change; outside R1–R3 |
| `operator_token.parquet` | Absent locally; map expects 25,235 | §16a / 140 | Separate operator-token feature; unnecessary for Stage 1 |

The five present files agree exactly with the notebook map's expected row
counts. Their hashes, schemas inspected during the inventory, dataset counts,
relaxed-stage counts, and source consistency checks identify the existing
artifacts without executing the notebook.

## What can be reused

Reuse the original matching rules and static-field support/majority thresholds,
and use these small artifacts as regression/audit references. Each twin cache's
donor movement ID exists in the supplied data, its recovered flight ID equals
the donor's ID, and all 16 cached plan fields equal the donor's raw values
(zero missing donors or payload discrepancies).

Do not feed these caches directly into chronological validation. The relaxed
matcher learns airline mappings and route durations from all 2025 plus both
ranking months. Static history uses that same pooled history, including future
and same-day rows relative to training examples. Merely filtering cached output
rows by month cannot undo those fitted references or global ownership/candidate
decisions. Exact matching also needs to re-evaluate candidates and plan claims
inside each available context.

## Measured policy differences

- The §17 origin guard rejects **2 exact and 1 relaxed** cached matches.
- Destination consistency rejects **1 relaxed** cached match. This local cache
  check differs from the notebook map's broad statement that every recovered
  destination agrees; raw cache/source checks are authoritative for this audit.
- **65 exact** cached plans are already attached to another departure in the
  complete supplied data. The notebook exact stage did not check this; the new
  implementation checks ownership inside the allowed fold context. The relaxed
  cache has zero such claims.
- The cached static layer contains **3,087 low-confidence flight-type fills**,
  which this improvement plan excludes. Its 829 flagged aircraft-type fills are
  a separate policy; their low-confidence provenance is retained.
- No cached twin crosses a calendar-month boundary. Boundary tests are still
  necessary for future inputs and newly eligible rows.
- Notebook recovery targets missing movement flight IDs. The pipeline also
  audits IDs sitting above an empty plan block and preserves existing values
  when filling partial plan fields.

This is the necessary new work: shared pipeline implementation, fold-scoped
availability, causal/frozen history, safeguards, and paired model validation.
It does not duplicate the notebook exploration or regenerate its caches.

[Machine-readable inventory](stage1_derived_inventory.json). Reproduce with:

```bash
.venv/bin/python VALIDATION/tools/inventory_derived.py
```
