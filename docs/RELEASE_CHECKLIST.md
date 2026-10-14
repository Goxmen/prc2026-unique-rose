# Competition release checklist

Official requirements last checked:13 October2026. Selected artifact updated:
14 October2026. This checklist concerns the accepted v84 final and its source
release, with exact retained-component assembly replay independently verified.

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
   separately produced and verified. Record the selected artifact's configuration,
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
5. Public GitHub availability is required for prize eligibility. The owner has
   authorized publication at **14 October, 09:30 UTC / 11:30 Amsterdam**, and
   privacy until then. See [the scheduled policy](../release/publication_schedule.json).
   Obtain the organizer's repository-handoff instructions if not otherwise announced.
6. Make required reproduction artifacts available through an appropriate
   release/download channel with the source release. A local-only artifact path
   is not a public download. Observe provider redistribution terms.

No public release or complete raw-data retraining certification is implied merely
by this checklist being present. Exact replay evidence and artifact distribution
status are recorded in the reproduction guide and release manifests.

## Scheduled publication

The desktop user timer `prc-publish-final-20261014.timer` starts the guarded
publisher at 09:30 UTC (11:30 Amsterdam). It audits current source/history, verifies local/remote
`main` agreement and the exact draft assets, changes repository visibility, then
publishes the draft release. It verifies both actions and records a receipt.
The time guard prohibits early publication, including an accidental manual start.
Failures preserve receipts and the service retries; partial success is resumed
without changing the repository back to private. There is no competition upload.

```bash
systemctl --user list-timers prc-publish-final-20261014.timer
systemctl --user status prc-publish-final-20261014.service
journalctl --user -u prc-publish-final-20261014.service
cat release/artifacts/publication.json
```

The desktop and Ubuntu WSL must stay online. A persistent timer catches a missed
start when WSL next runs but cannot wake a shut-down host; network/authentication
also must remain available. A changed final model requires repackaging and an
updated, verified schedule policy. To cancel an authorization before publication:

```bash
systemctl --user disable --now prc-publish-final-20261014.timer
systemctl --user stop prc-publish-final-20261014.service
```
