# Snow and completed-arrival features on v39/v40

Fixed before these fits or candidate scores, 24 September 2026.

Run two paired comparisons: v39 and v40, each with and without the same 12
inputs defined in [the v38 experiment](DISRUPTION_EXPERIMENT.md). Add them only
to the ordinary regressor. Each pair shares its Rome classifier, recording
corrections, rules, data, folds and seeds. Keep v40's rule-covered training-row
exclusion; fit correction parameters and build history/categories before that
exclusion. All evaluation rows remain. Use 800 ordinary rounds, 300 classifier
rounds, seeds 42/142/242, one fit at a time, six fit and 20 prediction threads.

The original OPDI files and boosters are absent here. Rebuild the 14 original
OPDI features from the verified local public downloads using unchanged
`opdi.py`. The documented October event-file 404 remains missing. Refit a
matched control for each recipe: 36 control ordinary, 36 candidate ordinary
and 36 classifier fits across both recipes. Save every model and prediction.
The original runs have different OPDI file hashes; exact historical prediction
reproduction is not assumed. Report each local control beside its historical
score, and measure feature gains against the local matched control only.

Reuse the adopted unclipped raw-feature weights and all 16 original folds.
Report Weighted CV, original combined CV, all seed/fold changes, airport/month
and missing-ID/snow slices, and paired 1-/3-day bootstrap intervals (2,000
draws, seed 0, common calendar across overlapping score groups). Average losses
over origins and seeds, not predictions. Use the fitted June–November models
for a separate January 2025 retrospective check, outside Weighted CV.

Training arrivals come only from the training window; validation arrivals
come from that window plus the current evaluation month. Only arrivals
completed strictly before takeoff enter the new summaries. Snow uses the
fixed SN/SG definition, six-/24-hour duration and coverage. No departure
labels enter these new features. Do not tune the package after seeing scores.

Neither a small gain nor a null result establishes performance during the
much longer January 2026 snow. This test measures the package on available
historical conditions; we will discuss handling rare heavy-snow conditions
afterward. Reused folds and fixed-weight bootstrap intervals do not establish
leaderboard transfer. No automatic promotion, full-year fits, uploads or
`SUBMISSIONS.md` changes.
