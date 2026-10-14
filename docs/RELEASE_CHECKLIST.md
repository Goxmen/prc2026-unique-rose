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
4. Commit and push the reviewed source and documentation to `main`. Preserve research artifacts locally; do not move or change source
   files or input caches used by an active validation run.
5. Public GitHub availability is required for prize eligibility. The repository
   and selected release were published at **14 October, 09:25:18 UTC /
   11:25:18 Amsterdam**, following the owner's immediate-publication request.
   See [the publication policy](../release/publication_schedule.json).
   Obtain the organizer's repository-handoff instructions if not otherwise announced.
6. Make required reproduction artifacts available through an appropriate
   release/download channel with the source release. A local-only artifact path
   is not a public download. Observe provider redistribution terms.

The public release and exact replay evidence are recorded in the reproduction
guide and release manifests. Public availability does not establish a complete
raw-data retraining certification.

## Completed publication

The guarded publisher verified source/history, local/remote `main`, exact release
assets, repository visibility and published-release status. Its receipt is
`release/artifacts/publication.json`, with status `published_verified` at
09:25:18 UTC / 11:25:18 Amsterdam on 14 October 2026. The public assets are
available from the
[v84 release](https://github.com/Goxmen/PRC/releases/tag/prc-final-v84-76938ead67f3).

The owner's immediate-publication request superseded the earlier 11:30 schedule.
`prc-publish-final-20261014.timer` is disabled; no scheduled publication remains.
This publication did not change the accepted competition predictions or the
official **10:00 UTC / 12:00 Amsterdam** final-submission deadline.
