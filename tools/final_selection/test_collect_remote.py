import hashlib
from pathlib import Path
import tempfile
import unittest

import collect_remote as collector


class IdentityChecks(unittest.TestCase):
    def test_ready_fold_requires_both_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'identity.json'
            with self.assertRaises(ValueError):
                collector.check_identity({'ready': ['2025-02'], 'identity_sha256': 'x'}, path)
            path.write_text('fixed')
            with self.assertRaises(ValueError):
                collector.check_identity({'ready': ['2025-02']}, path)

    def test_identity_must_match(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'identity.json'
            path.write_text('fixed')
            with self.assertRaises(ValueError):
                collector.check_identity({'ready': ['2025-02'], 'identity_sha256': 'wrong'}, path)
            collector.check_identity({'ready': ['2025-02'],
                'identity_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}, path)

    def test_waiting_for_remote_is_allowed(self):
        collector.check_identity({'ready': []}, Path('/nonexistent/identity.json'))


if __name__ == '__main__':
    unittest.main()
