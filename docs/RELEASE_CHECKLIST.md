# Competition release checklist

Last checked: 13 October 2026. This checklist concerns the accepted v83-derived
final submission and its source release. The running four-month validation may
inform a later replacement; its results do not select one automatically.

## Official requirements

The [eligibility rules](https://prc-data-challenge-2026.netlify.app/eligibility.html#eligibility-for-prize)
require publicly available GitHub source under GNU GPLv3, documentation sufficient
to understand and reproduce results, documented accessible external data with
the required reuse rights, and an original solution with authorized substantive
reuse of third-party work. Organizers intend to fork participating repositories.
Team eligibility restrictions also apply; see the official page.

The [final prediction deadline](https://prc-data-challenge-2026.netlify.app/)
is **14 October 2026 at 10:00 UTC / 12:00 Europe/Amsterdam**. The website's
`CET` label conflicts with its explicit `+02:00`/UTC timestamp; use the latter.
The final filename is
[`unique-rose_final.parquet`](https://prc-data-challenge-2026.netlify.app/ranking.html#final-submission),
using the complete four-month `final_submitting.parquet` template.

The checked website does not specify a separate code-release deadline, a form
for providing the repository URL, a report template, Docker, or mandatory model
weight hosting. Absence of an explicit requirement is not organizer confirmation
of an exemption. A [JOAS paper](https://prc-data-challenge-2026.netlify.app/#outreach)
is encouraged. The official Discord channel is the organizer's recommended
place for urgent announcements and unresolved handoff details.

## Release contents

- [GPLv3 license](../LICENSE).
- [Method and validation description](METHOD.md), including original contributions
  and the limits of reused development data.
- [Data-source inventory](DATA_SOURCES.md), provider terms, attribution and
  acquisition/build instructions.
- [Third-party attribution](THIRD_PARTY.md).
- [Executable reproduction instructions](REPRODUCING.md), environment records,
  component provenance, and final output verification.
- [Four-fold model-selection source and protocol](../VALIDATION/docs/FINAL_MONTH_SELECTION.md).
- Repository secret/history audit, excluding local IDE checkpoint refs that
  are not published by an ordinary push.

The team confirmed on 13 October that the previously raised external-source use
question is resolved for this competition. This records the team's clarification;
it does not relicense provider data or assert independent written organizer
confirmation. Provider terms and attribution remain documented, and raw external
data are not bundled into git.

## Final release procedure

1. Preserve the accepted final until a replacement is explicitly selected and
   separately trained and verified. Record the selected artifact's configuration,
   source commit, component manifest, row count, prediction hash and acceptance.
2. Run the offline replay and its integrity checks described in `REPRODUCING.md`.
   A cached-component replay is distinct from a clean raw-data retraining test.
3. Run the repository audit against all intended publication branches/tags and
   the current source files. Keep credentials, private notes, raw datasets,
   row-level caches and large model bundles out of git. Do not push local
   `refs/codex/` checkpoint references with `--mirror`.
4. Commit and push the reviewed source and documentation to the private `main`
   repository. Preserve research artifacts locally; do not move or change source
   files or input caches used by an active validation run.
5. Public GitHub availability is required for prize eligibility. The standing
   owner instruction is to keep this repository private, so publication is a
   separate owner decision after reviewing the prepared release. Obtain the
   organizer's repository-handoff instructions if not otherwise announced.
6. Make required reproduction artifacts available through an appropriate
   release/download channel with the source release. A local-only artifact path
   is not a public download. Observe provider redistribution terms.

No public release or complete raw-data retraining certification is implied merely
by this checklist being present. Exact replay evidence and artifact distribution
status are recorded in the reproduction guide and release manifests.
