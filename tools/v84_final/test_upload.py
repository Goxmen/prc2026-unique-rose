"""Offline checks for the newly selected v84 final, with no remote operations."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('selected_v84_upload', Path(__file__).with_name('upload.py'))
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class SelectedV84Requirements(unittest.TestCase):
    def setUp(self):
        candidate = dict(path='final/candidate.parquet', sha256='selected-final', bytes=10)
        self.parts = dict(
            owner=dict(selected_version=84, allowed_final_upload=True,
                       selected_old_candidate_sha256=upload.OLD_SHA),
            official=dict(status='Succeeded', score=218.9025, used_pairs=344841,
                          file='unique-rose_v84.parquet'),
            independent=dict(status='pass', candidate=dict(candidate),
                             old_leaderboard_candidate={'sha256': upload.OLD_SHA}),
            replay=dict(status='pass', sha256='selected-final', rows=670790),
            candidate=candidate)

    def test_complete_selected_recipe_is_eligible_for_transport(self):
        upload.selection_checks(**self.parts)

    def test_current_owner_revocation_blocks_before_candidate_or_network(self):
        revoked = dict(allowed_final_upload=False, selected_final_version=83)
        with patch.object(upload.transport, 'read', return_value=revoked) as read, \
             patch.object(upload.transport, 'verify') as verify, \
             patch.object(upload.transport, 'mc') as remote:
            with self.assertRaisesRegex(ValueError, 'currently preserves v83'):
                upload.validate(upload.OUT)
            self.assertEqual(read.call_count, 1)
            verify.assert_not_called()
            remote.assert_not_called()

    def test_original_v80_or_other_version_cannot_use_v84_authorization(self):
        for version in [80, 83, 85]:
            parts = copy.deepcopy(self.parts)
            parts['owner']['selected_version'] = version
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'selection'):
                upload.selection_checks(**parts)

    def test_final_candidate_cannot_be_swapped_after_audit(self):
        parts = copy.deepcopy(self.parts)
        parts['candidate']['sha256'] = 'different-final'
        with self.assertRaisesRegex(ValueError, 'exact final candidate'):
            upload.selection_checks(**parts)

    def test_old_population_replay_does_not_prove_final_replay(self):
        parts = copy.deepcopy(self.parts)
        parts['replay']['rows'] = 344841
        with self.assertRaisesRegex(ValueError, 'exact final bytes'):
            upload.selection_checks(**parts)

    def test_scored_recipe_binding_is_required(self):
        parts = copy.deepcopy(self.parts)
        parts['independent']['old_leaderboard_candidate']['sha256'] = 'overnight-refit'
        with self.assertRaisesRegex(ValueError, 'selected v84 recipe'):
            upload.selection_checks(**parts)


if __name__ == '__main__':
    unittest.main()
