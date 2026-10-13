"""The scanner must identify locations without ever returning credential values."""
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("audit_repository", Path(__file__).with_name("audit_repository.py"))
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RepositoryAuditTests(unittest.TestCase):
    def test_token_location_is_redacted(self):
        synthetic = "gh" + "p_" + "Ab12" * 10
        findings = audit.inspect_content("fixture.txt", ("header\n" + synthetic).encode(), scope="test", object_id="fixture")
        self.assertEqual(findings[0]["rule"], "github_token")
        self.assertEqual(findings[0]["line"], 2)
        self.assertNotIn(synthetic, json.dumps(findings))

    def test_literal_assignment_requires_review(self):
        synthetic = "Ar8" + "bz4Pq9Sm6Rw2Ta5V"
        findings = audit.scan_bytes(('ACCESS_TOKEN = "' + synthetic + '"').encode())
        self.assertEqual(findings, [{"rule": "literal_credential_assignment", "line": 1, "severity": "review"}])
        self.assertNotIn(synthetic, json.dumps(findings))

    def test_placeholder_assignment_is_not_reported(self):
        self.assertEqual(audit.scan_bytes(b'API_KEY = "your_example_key_here"'), [])

    def test_sensitive_filename(self):
        for path in [".env", ".env.production", "keys/id_ed25519", "LOCAL_PROGRESS.md"]:
            self.assertTrue(audit.sensitive_name(path), path)
        for path in [".env.example", ".env.sample", "README.md"]:
            self.assertFalse(audit.sensitive_name(path), path)

    def test_dynamic_url_password_is_not_literal(self):
        dynamic = b'url = f"https://{account}:{encoded_password}@{endpoint}"'
        self.assertEqual(audit.scan_bytes(dynamic), [])
        literal = b'url = "https://account:' + b'Ab9cDef21Ghijk' + b'@endpoint"'
        findings = audit.scan_bytes(literal)
        self.assertEqual(findings[0]["rule"], "embedded_url_password")

    def test_deleted_history_and_untracked_source_are_scanned(self):
        synthetic = "gh" + "p_" + "Qb39" * 10
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def git(*args):
                subprocess.run(["git", "-C", str(root), "-c", "user.name=AuditFixture", "-c", "user.email=audit@example.invalid", "-c", "commit.gpgsign=false", *args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            git("init", "-q")
            (root / "historical.py").write_text("fixture = '" + synthetic + "'\n")
            git("add", "historical.py")
            git("commit", "-qm", "Synthetic fixture")
            git("rm", "historical.py")
            git("commit", "-qm", "Remove fixture")
            (root / "pending.py").write_text("fixture = '" + synthetic + "'\n")
            report = audit.audit(root)
            self.assertTrue(any(item["scope"] == "history_or_index" and item["path"] == "historical.py" for item in report["findings"]))
            self.assertTrue(any(item["scope"] == "untracked_source_or_documentation" and item["path"] == "pending.py" for item in report["findings"]))
            self.assertNotIn(synthetic, json.dumps(report))


if __name__ == "__main__":
    unittest.main()
