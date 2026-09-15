# Feature-sweep notebook — section rules

How every section of `eda/feature_sweep.ipynb` is shaped once it is finished. Digging is free-form;
when gunes says "restructure", the section is brought to this shape without losing any decision
or the reasoning behind it. §1–§8 are the reference: copy their form.

## 1. Process

1. **Explain first, act second.** Walk gunes through what the section currently does (one row per
   cell: content → result), give an assessment, offer 2–3 options with a recommendation, and wait.
   Never restructure, delete or rewrite a section before gunes picks an option.
2. **Keep every decision and its "why".** Restructuring changes form, not conclusions. If a
   conclusion turns out to be wrong (e.g. an in-sample number), say so in the findings rather
   than silently replacing it.
3. **Re-execute what you touch, in a clean kernel.** Rebuild the section's code cells in a temp
   notebook (setup cell + a preamble of the names earlier sections define + the section's cells),
   check there are no error outputs, then splice the executed cells back in place. Untouched cells
   keep their outputs and execution counts. Confirm recovery counts and derived-file row counts are
   identical to before.
4. **One column per section, even when the column is dull.** A section covers its own column and
   *hands* derived forms to the column that owns them — §11's wake lookup belongs to `WK_TBL_CAT_flt`,
   §16's operator prefix to `AIRCRAFT_OPERATOR_flt`. Measure a neighbouring column only as far as
   choosing your own column's form requires, then name in one line which section owns the conclusion.
   If your column turns out to be worth nothing on its own, **that is the finding** — write it and
   stop; do not annex the neighbour that is worth something. (§16 drifted exactly this way: the
   callsign is worth 0.2 s once the operator is known, so the section grew an operator study that
   `AIRCRAFT_OPERATOR_flt` should own.)
5. **Leave the rest alone.** Do not renumber other cells, do not touch other sections, do not
   commit unless asked. New sections are inserted *before* the closing "Features so far" matrix
   section, the roadmap in cell 0 gets its ✅, and the matrix gets the column (skeleton row
   "matrix") — a section is not finished until it is in the matrix.

## 2. Section skeleton (in this order)

| cell | form |
|---|---|
| heading | `## N · \`COLUMN_NAME\` — plain-words subtitle`; sub-sections `### Na · plain-words title` (never "addendum") |
| intro | one or two short paragraphs on what the column *is*, then a **step table**: `step \| question \| what came out` |
| code cells | one comment line at the top saying what the cell does; DuckDB SQL through `con.sql(...)`, printed with `.to_string(index=False)`; `display(cat_stats(...))` or `order_stats(...)` first where the column is a plain category/number |
| sub-section intro | short; a table when it states a policy (e.g. sentinel → treatment) |
| recovery cell (if any) | writes `data/derived/<name>.parquet` with a **provenance column**, joined on `MVT_ID_mvt`; prints counts per layer |
| closing state cell (if values were missing or changed) | the last code cell before the correlation cell: raw → after per file × phase, `assert` nothing is left, the final distribution, taxi by final level |
| correlation cell | `a = target_assoc([...])` then `assoc_chart(a, main, ...)`: the raw feature, its derived forms and its final form; just before the findings |
| findings | `**Findings**` then a `fact \| evidence \| what it means for us` table; then `**Decisions**` (`issue \| decision \| why`) when something was judged; then `**State after §N**` (before → after table) when anything changed; then a `**Use in modeling.**` paragraph; a `**Pipeline recipe:**` line when a derived file exists |
| matrix | **every section adds its column** to the closing "Features so far" section, no exceptions: one line in the `assoc_base` SELECT (the column in the form the pipeline should consume — the recovered rule, the recovered destination, epoch seconds for a timestamp, otherwise the raw column, aliased to the column's own name), one line in `FEATURES` in file order, then re-execute the matrix cell and add a row to its reading. Hidden and constant columns go in too (`PHASE_mvt`, `BLOCK_TIME_UTC_mvt`): a row of zeros is the record that nothing is there |

## 3. Numbers

- **Target rows** for every statistic: training departures, `TAXITIME_SEC_mvt BETWEEN 1 AND 3600`
  (`TGT`). Say so when a cell deviates (raw labels, arrivals, both phases).
- **No in-sample signal claims.** Any "feature X buys Y seconds" is a group-mean ladder fitted on
  Jan–Oct 2025 and scored on the Nov+Dec holdout, cell support ≥ 30 (≥ 200 for high-cardinality
  levels) falling back to the coarser level. The airport-mean baseline on that holdout is
  **357.3 s**; when a ladder uses a different row subset, footnote why its baseline differs.
- Marginal associations come from the correlation cell (η, |r|); gains come from the ladder. Say
  which is which.
- Every number quoted in a findings table must be visible in an output of the same section.
- Counts are exact (`2,085,047`), percentages to two decimals when small, seconds to one decimal
  in RMSE and whole seconds elsewhere.

## 4. Text

- Tables over prose. A findings row is: the fact in bold when it matters, the evidence with the
  numbers, the consequence for modeling. One idea per row.
- Plain words first, the column name second. Explain any code, sentinel or ICAO term the first time.
- Cross-reference by section number (`§2b`, `§7a`), and update older sections' forward references
  ("the runway section will…") once that section exists.
- Keep the story of *why* a decision was taken, including the alternative that was rejected and the
  evidence that rejected it (6a's history vote vs `OFF_AIRPORT` is the model).
- No "addendum", no "brainstorm", no references to conversations or research runs that are not in
  the repo (say what was validated and how, not where).

## 5. What to remove when restructuring

- Percentile tables of identifiers; raw row dumps (`.T` of many columns); `.head()` of derived files.
- Blocks that a later cell repeats; charts that only repeat a printed table; the second panel when
  the first already carries the finding.
- In-sample signal checks → replace with the honest ladder (keep the old number only if it is
  worth saying it was flattering).
- Prose that restates a table.

Keep evidence cells that justify a decision, even when they are long.

## 6. Charts

- Palette and chrome from the setup cell: `BLUE` = training, `ORANGE` = ranking (or the second
  series), `MUTED`/`BASELINE` for reference, `INK`/`INK2` for text; `style(ax, x_grid=, y_grid=)`.
- Two panels `figsize=(11, 3.8)`, `fig.tight_layout(w_pad=2.5)`; one panel `(6.4, 3.6)`.
- Titles are short claims ("Look both ways: …"), left-aligned, bold (rcParams do this); axis labels
  name the unit and the row filter ("seconds, training 2025, 1–3600 s").
- Legends off the data; annotate the one number the panel is about.

## 7. Names later sections rely on (define once, never shadow)

`T`, `R`, `ROOT`, `con`, `APS` (the ten hubs, §2), `HELI`/`HELI_ALL` (rotorcraft, §4/4a), `FMT`
(identifier format, §3), `REC`/`RELAXED` (twin layers, §2), `FINAL` (rule precedence, §4a),
`FINALD` (destination precedence, §6a), `TZ`/`dep_clock` (local clocks, §8), `wlad` (§8's window
table), `typ_wake`/`ROTORCRAFT` (type → wake lookup and the completed rotorcraft list, §11),
`TYPE_FINAL` (the plan's code first, §11a), `PFX` (the stand's prefix, §13), `lab_d`/`lab_b`/`lab_cat` (raw-label
holdout tables and the raw-label ladder, §14), `lob_d`/`lob_ladder`/`TWINS` (the plan-clock ladder and the twin layers' clocks, §15),
`TWINCS`/`hash_pfx`/`FINALOP` (the twin layers' callsigns, the hash → prefix map and the operator
precedence, §16 — the closing matrix reads `TWINCS`), `TWINSI` (the twin layers'
initial off-block clock, §21 — the closing matrix reads it too),
and the helpers `cat_stats`, `order_stats`, `style`, `TGT`, `eta`, `rho`, `cramers_v`,
`target_assoc`, `assoc_chart`. A section may use any name defined above it; it must not
redefine one that a later section reads. Running the notebook top to bottom must work; running a
section alone in a fresh kernel is not expected to.
