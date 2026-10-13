#!/usr/bin/env python3
"""Read-only publication audit. Reports locations and rule names, never values.

Scans current tracked worktree files, untracked source/documentation, the index,
and all branch/tag/remote histories. Local Codex checkpoint refs are inventoried but not traversed: they
are not part of a normal branch push and can contain private ignored files.
This conservative pattern scan is not proof that no secret exists.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess


MAX_SCAN_BYTES = 20_000_000
ARTIFACT_SUFFIXES = {".pkl", ".pickle", ".pt", ".pth", ".npy", ".npz", ".cbm", ".bin", ".parquet"}
BINARY_SUFFIXES = ARTIFACT_SUFFIXES | {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".woff", ".woff2", ".zip", ".gz"}
RULES = {
    "github_token": re.compile(rb"\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b"),
    "aws_access_key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "private_key": re.compile(rb"-----BEGIN (?:RSA |EC |DSA |OPENSSH |ENCRYPTED )?PRIVATE KEY-----"),
    "slack_token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "google_api_key": re.compile(rb"\bAIza[A-Za-z0-9_-]{35}\b"),
    "openai_api_key": re.compile(rb"\bsk-(?:(?:proj|svcacct)-)?[A-Za-z0-9_-]{32,}\b"),
    "embedded_url_password": re.compile(rb"https?://[^\s/:@]{2,}:[^\s/@]{8,}@"),
}
ASSIGNMENT = re.compile(
    rb"(?im)(?<![A-Z0-9_])[\"']?([A-Z][A-Z0-9_]{0,80})[\"']?\s*[:=]\s*[\"']([A-Za-z0-9_+/=.-]{16,})[\"']"
)
SECRET_IDENTIFIER = re.compile(rb"(?i)(?:^|_)(?:GHP|API_?KEY|SECRET_?KEY|ACCESS_?KEY(?:_ID)?|SECRET_ACCESS_KEY|ACCESS_?TOKEN|AUTH_?TOKEN|PASSWORD|TOKEN)$")
PLACEHOLDER_WORDS = {"example", "placeholder", "changeme", "replace", "your_", "redacted", "dummy", "fake", "test_", "sample", "secret_here"}


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)


def sensitive_name(path: str) -> bool:
    name = Path(path).name.lower()
    if name in {"local_progress.md", "hosts.yml", "id_rsa", "id_ed25519", "id_ecdsa", "credentials.json"}:
        return True
    if name == ".env" or (name.startswith(".env.") and not name.endswith((".example", ".sample", ".template"))):
        return True
    if path.endswith((".mc/config.json", ".aws/credentials", ".config/gh/hosts.yml")):
        return True
    return False


def scan_bytes(data: bytes) -> list[dict]:
    """No matched value or line content leaves this function."""
    hits = []
    for rule, expression in RULES.items():
        for match in expression.finditer(data):
            if rule == "embedded_url_password":
                password = match.group().split(b"://", 1)[1].rsplit(b"@", 1)[0].split(b":", 1)[1]
                # A password supplied entirely by an f-string expression is not
                # a literal credential; do not suppress mixed literal values.
                if re.fullmatch(rb"\{[^{}]+\}", password):
                    continue
            hits.append({"rule": rule, "line": data.count(b"\n", 0, match.start()) + 1, "severity": "block"})
    for match in ASSIGNMENT.finditer(data):
        if not SECRET_IDENTIFIER.search(match.group(1)):
            continue
        value = match.group(2).lower()
        if any(word.encode() in value for word in PLACEHOLDER_WORDS) or len(set(value)) < 5:
            continue
        hits.append({"rule": "literal_credential_assignment", "line": data.count(b"\n", 0, match.start()) + 1, "severity": "review"})
    # Multiple patterns can identify one line; keep each distinct rule once.
    return [dict(zip(("rule", "line", "severity"), row)) for row in sorted({(x["rule"], x["line"], x["severity"]) for x in hits})]


def inspect_content(path: str, data: bytes, *, scope: str, object_id: str | None = None) -> list[dict]:
    findings = []
    if sensitive_name(path):
        findings.append({"rule": "private_file_name", "line": None, "severity": "block"})
    findings.extend(scan_bytes(data))
    return [{"scope": scope, "path": path, "object_id": object_id, **finding} for finding in findings]


def audit(root: Path) -> dict:
    tracked = git(root, "ls-files", "-z").decode().split("\0")[:-1]
    untracked = git(root, "ls-files", "--others", "--exclude-standard", "-z").decode().split("\0")[:-1]
    untracked_reviewable = [
        path for path in untracked
        if Path(path).suffix.lower() in {".py", ".md", ".toml", ".yml", ".yaml", ".sh"}
        or (path.startswith(("release/", "docs/", "tools/release/")) and Path(path).suffix.lower() in {".json", ".txt", ".csv"})
    ]
    refs = git(root, "for-each-ref", "--format=%(refname)").decode().splitlines()
    excluded_refs = [name for name in refs if not name.startswith(("refs/heads/", "refs/tags/", "refs/remotes/"))]
    raw_objects = git(root, "rev-list", "--objects", "--branches", "--tags", "--remotes").decode().splitlines()
    objects = {line.partition(" ")[0]: line.partition(" ")[2] for line in raw_objects}
    # Include staged but not yet committed blobs, with their current names.
    for record in git(root, "ls-files", "--stage", "-z").decode().split("\0"):
        if not record:
            continue
        metadata, path = record.split("\t", 1)
        _, object_id, stage = metadata.split()
        if stage == "0":
            objects.setdefault(object_id, path)
    checks = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input=("\n".join(objects) + "\n").encode(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True,
    ).stdout.decode().splitlines()
    blobs = {parts[0]: int(parts[2]) for line in checks if len(parts := line.split()) == 3 and parts[1] == "blob"}
    findings, skipped, large, artifacts = [], [], [], []
    coverage = Counter()
    # rev-list gives one path per unique blob. Inspect every changed historical
    # path too, so an empty .env cannot hide behind a shared empty __init__.py.
    historical_paths = {
        path.strip("\n") for path in git(root, "log", "--branches", "--tags", "--remotes", "--format=", "--name-only", "-z").decode().split("\0")
        if path.strip("\n")
    }
    for path in sorted(historical_paths):
        if sensitive_name(path):
            findings.append({"scope": "historical_path", "path": path, "object_id": None, "rule": "private_file_name", "line": None, "severity": "block"})
    coverage["historical_pathnames_checked"] = len(historical_paths)
    for path in tracked + untracked_reviewable:
        scope = "worktree" if path in tracked else "untracked_source_or_documentation"
        file = root / path
        if file.is_symlink():
            skipped.append({"scope": scope, "path": path, "reason": "symlink_not_followed"})
            continue
        if not file.is_file():
            skipped.append({"scope": scope, "path": path, "reason": "missing_file"})
            continue
        size = file.stat().st_size
        coverage["tracked_worktree_bytes" if scope == "worktree" else "untracked_reviewable_bytes"] += size
        if size > 5_000_000:
            large.append({"path": path, "bytes": size})
        if Path(path).suffix.lower() in ARTIFACT_SUFFIXES or Path(path).name in {"model.txt", "ordinary.txt"}:
            artifacts.append(path)
        if size > MAX_SCAN_BYTES or Path(path).suffix.lower() in BINARY_SUFFIXES:
            skipped.append({"scope": scope, "path": path, "bytes": size, "reason": "oversize" if size > MAX_SCAN_BYTES else "binary_extension"})
            if sensitive_name(path):
                findings.extend(inspect_content(path, b"", scope=scope))
            continue
        findings.extend(inspect_content(path, file.read_bytes(), scope=scope))
        coverage["worktree_files_scanned" if scope == "worktree" else "untracked_reviewable_files_scanned"] += 1
    eligible = []
    for object_id, size in blobs.items():
        path = objects[object_id]
        if size > MAX_SCAN_BYTES or Path(path).suffix.lower() in BINARY_SUFFIXES:
            skipped.append({"scope": "history_or_index", "path": path, "object_id": object_id, "bytes": size, "reason": "oversize" if size > MAX_SCAN_BYTES else "binary_extension"})
            if sensitive_name(path):
                findings.extend(inspect_content(path, b"", scope="history_or_index", object_id=object_id))
            continue
        eligible.append(object_id)
    # A single streaming git process avoids one subprocess per historical blob.
    process = subprocess.Popen(["git", "-C", str(root), "cat-file", "--batch"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdin is not None and process.stdout is not None
    try:
        for object_id in eligible:
            process.stdin.write((object_id + "\n").encode())
            process.stdin.flush()
            header = process.stdout.readline().split()
            if len(header) != 3 or header[0].decode() != object_id or header[1] != b"blob":
                raise RuntimeError("Git object streaming failed; no object content is included in this error.")
            size = int(header[2])
            data = process.stdout.read(size)
            if len(data) != size or process.stdout.read(1) != b"\n":
                raise RuntimeError("Git object stream was truncated.")
            findings.extend(inspect_content(objects[object_id], data, scope="history_or_index", object_id=object_id))
            coverage["history_blobs_scanned"] += 1
            coverage["history_bytes_scanned"] += size
    finally:
        process.stdin.close()
        process.stdout.close()
        process.wait()
    if process.returncode:
        raise RuntimeError("Git history scan exited unsuccessfully.")
    artifact_gaps = []
    for path in untracked:
        if Path(path).suffix.lower() in ARTIFACT_SUFFIXES or Path(path).name in {"model.txt", "ordinary.txt"}:
            artifact_gaps.append({"path": path, "bytes": (root / path).stat().st_size})
    ignored_checks = {}
    for path in [".env", "LOCAL_PROGRESS.md", "data/probe.parquet", "submissions/probe.parquet"]:
        ignored_checks[path] = subprocess.run(["git", "-C", str(root), "check-ignore", "--quiet", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "head": git(root, "rev-parse", "HEAD").decode().strip(),
        "is_shallow_repository": git(root, "rev-parse", "--is-shallow-repository").decode().strip() == "true",
        "scope": "Tracked worktree files, untracked source/documentation, index, and all local/remote branch and tag histories; ignored files, untracked non-source artifacts, and nonstandard refs are excluded.",
        "limitations": ["Pattern matching cannot establish absence of secrets.", "Binary-extension files and objects over 20 MB are not inspected for embedded content.", "Local Codex checkpoint refs may contain ignored/private files and must not be pushed or bundled.", "Re-run after staging final release changes; untracked raw data/artifacts are excluded."],
        "excluded_nonstandard_refs": excluded_refs,
        "coverage": {"tracked_paths": len(tracked), "history_or_index_blobs": len(blobs), "history_or_index_blob_bytes": sum(blobs.values()), **dict(coverage)},
        "findings": findings,
        "tracked_artifact_paths": artifacts,
        "tracked_files_over_5_mb": sorted(large, key=lambda item: -item["bytes"]),
        "ignored_path_checks": ignored_checks,
        "untracked_source_paths": sorted(path for path in untracked if Path(path).suffix == ".py"),
        "untracked_artifact_ignore_gaps": sorted(artifact_gaps, key=lambda item: -item["bytes"]),
        "skipped": skipped,
    }


def main() -> int:
    if hasattr(os, "nice"):
        os.nice(10)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, default=Path("release/security_audit.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    report = audit(root)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"report": str(output.relative_to(root)) if output.is_relative_to(root) else str(output), "coverage": report["coverage"], "finding_count": len(report["findings"]), "findings_by_rule": dict(Counter(item["rule"] for item in report["findings"])), "untracked_artifact_ignore_gaps": len(report["untracked_artifact_ignore_gaps"]), "untracked_source_count": len(report["untracked_source_paths"]), "skipped_count": len(report["skipped"])}))
    return 1 if report["findings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
