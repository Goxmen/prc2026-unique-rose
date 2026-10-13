# Repository publication audit

The repository remains private. This audit prepares source for review; it does
not change visibility, publish artifacts, or rewrite history.

## Re-run before publishing

From the repository root, with the final intended changes staged:

```bash
.venv/bin/python -m unittest discover -s tools/release -p 'test_audit_repository.py' -q
.venv/bin/python tools/release/audit_repository.py --output release/security_audit.json
```

The scanner is read-only apart from its JSON report. A nonzero exit means a
credential-pattern or private-filename finding needs review. Reports contain
only paths, object IDs, line numbers and rule names, never matched values. Do
not paste suspected values into issues, commit messages or documentation.

## Coverage and results

The initial 13 October audit of commit `08595ad` found no credential-pattern
matches after identifying the dynamic URL construction in
`pipeline/dashboard_data.py` as a variable-based template, not a literal
password. The current branch tree has no tracked model arrays, pickles,
checkpoints or parquet files. `.env`, `LOCAL_PROGRESS.md`, data parquet and
submission parquet paths are ignored.

The reusable scan covers:

- Current tracked file contents and staged blobs.
- New untracked Python/shell sources, Markdown/YAML/TOML documentation, and
  release-directory metadata, before those files are staged.
- Every unique blob in all locally available branch, tag and remote-branch
  histories, plus every historical changed pathname. The initial normal
  history contained 5,037 blobs totalling 136.5 MB, with no blob over 20 MB.

The machine-readable report records the current counts, findings, skipped files
and artifact-ignore gaps. Re-run it after staging changes: generated data and
untracked research artifacts are deliberately not content-scanned before staging.

This is a bounded pattern audit, not a proof of absence of secrets. Binary image,
PDF, archive and model formats, as well as objects over 20 MB, are listed but
not inspected internally. The initial scan skipped 79 binary files and their
79 historical blob versions. Review binary provenance separately.

## Keep local artifacts private

The initial worktree also contained 184 untracked model/array/pickle artifacts
(6.78 GB) and 173 untracked logs. They are research outputs, not release source;
scoped ignore rules now exclude those artifacts without deleting them. The
19:27 UTC scan reports zero untracked model/array/pickle ignore gaps. Use
explicit source/documentation paths when staging.

Local `refs/codex/turn-diffs/checkpoints/*` are not part of a normal branch push.
They contain additional ignored artifacts and are excluded from the publication
scan. Do not publish all refs, mirror this repository, or create an all-refs
bundle for release. Publish only the reviewed branch and any intentional release
tag. Ordinary `git push origin main` does not publish those checkpoint refs.

The audit does not grant permission to change this repository's visibility.
