"""Offline replay/publication contract tests; no network or real model execution."""
import importlib.util
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd


def module(name):
    spec = importlib.util.spec_from_file_location('v80_' + name, Path(__file__).with_name(name + '.py'))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


package, replay = module('package'), module('replay')


class ReplayTests(unittest.TestCase):
    def test_rounding_after_affine_then_two_integer_deltas(self):
        versions = list(range(51))
        basis = pd.DataFrame({f'v{v}': [1, 5, 8] for v in versions})
        weights = np.zeros(51); weights[0] = .5
        shifts = pd.DataFrame({'v78': [10, 10, 10], 'tabm_v78': [12, 8, 11], 'robust_v78': [11, 10, 12]})
        np.testing.assert_array_equal(replay.calculate(basis, shifts, versions, weights), [3, 0, 7])
        # Round-before-correction is deliberate and is not a single floating blend.
        np.testing.assert_array_equal(replay.integers(np.array([2.5, 3.5, -2.])), [2, 4, 0])

    def test_bad_recipe_and_nonfinite_rejected(self):
        frame = pd.DataFrame({'v0': [1]})
        with self.assertRaises(ValueError):
            replay.calculate(frame, pd.DataFrame(), [0], [1.])
        with self.assertRaises(ValueError):
            replay.integers(np.array([float('nan')]))
        with self.assertRaises(ValueError):
            replay.integers(np.array([2.**40]))

    def test_id_alignment_is_strict(self):
        frame = pd.DataFrame({'mvt_id': [2, 1], 'x': [20, 10]})
        self.assertEqual(replay.align(frame, [1, 2]).x.tolist(), [10, 20])
        for wanted in ([1, 3], [1], [1, 1]):
            with self.assertRaises(ValueError):
                replay.align(frame, wanted)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            for path in ['../outside', '/absolute']:
                with self.assertRaises(ValueError):
                    replay.safe(temp, path)
            self.assertEqual(replay.safe(temp, 'source/replay.py'), Path(temp) / 'source/replay.py')

    def test_archive_contains_exact_bundle_members(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); bundle = root / 'bundle'; bundle.mkdir()
            (bundle / 'x.txt').write_text('safe')
            target = root / 'good.tar'
            with tarfile.open(target, 'w') as tar:
                tar.add(bundle / 'x.txt', arcname='x.txt')
            package.verify_archive(target, bundle)
            (bundle / 'x.txt').write_text('different')
            with self.assertRaises(ValueError):
                package.verify_archive(target, bundle)


class ReleaseTests(unittest.TestCase):
    def test_acceptance_must_bind_exact_candidate(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp); candidate = out / 'candidate'; candidate.write_text('artifact')
            record = package.artifact(candidate)
            prepared = dict(candidate=record)
            accepted = dict(status='accepted', candidate=record, official_stat=dict(lastModified='2026-10-14T09:00:02Z'),
                official=dict(status='Succeeded', used_pairs=670790, file='unique-rose_final.parquet',
                              bucket='prc-2026-unique-rose', inputs=dict(test_set_file='final_submitting.parquet')))
            delivered = dict(status='remote_hash_verified', candidate=record, remote=record)
            package.save(out / 'delivery/acceptance.json', accepted)
            package.save(out / 'delivery/attempt.json', dict(candidate=record, started_at='2026-10-14T09:00:00Z'))
            package.save(out / 'delivery/delivery.json', delivered)
            package.acceptance(out, prepared)
            for changed in [dict(status='pending'), dict(official=dict(accepted['official'], used_pairs=344841)),
                            dict(candidate=dict(record, sha256='0'*64)),
                            dict(official_stat=dict(lastModified='2026-10-14T08:00:00Z'))]:
                package.save(out / 'delivery/acceptance.json', dict(accepted, **changed))
                with self.assertRaises(ValueError):
                    package.acceptance(out, prepared)

    def test_commit_preserves_unrelated_staging(self):
        with patch.object(package, 'git', return_value='user-file.py') as git:
            with self.assertRaisesRegex(ValueError, 'Unrelated staged'):
                package.commit_generated([package.ROOT / p for p in package.GENERATED], Path('/tmp/absent_v80_test'))
            self.assertEqual(git.call_count, 1)

    def test_pending_generated_push_is_narrow_and_resumable(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            marker = dict(status='committed_pending_push', base_head='parent', commit='generated',
                          paths=package.GENERATED, files=[])
            package.save(out/'release/activation_commit.json', marker)
            commands = []
            def fake_git(*args):
                commands.append(args)
                if args == ('rev-parse', 'HEAD'):
                    return 'generated'
                if args == ('push', 'origin', 'main'):
                    return ''
                raise AssertionError(args)
            with patch.object(package, 'git', side_effect=fake_git), \
                 patch.object(package, 'verify_generated_commit') as verify, \
                 patch.object(package, 'api', side_effect=[{'object':{'sha':'parent'}}, {'object':{'sha':'generated'}}]):
                result = package.resume_generated_push(out)
            verify.assert_called_once()
            self.assertEqual(verify.call_args.args[1], 'generated')
            self.assertEqual(result['status'], 'pushed')
            self.assertEqual(commands.count(('push','origin','main')), 1)
            with patch.object(package, 'git', side_effect=AssertionError('no repeated push')):
                self.assertEqual(package.resume_generated_push(out)['status'], 'pushed')

    def test_changed_remote_blocks_pending_push(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            package.save(out/'release/activation_commit.json', dict(status='committed_pending_push',
                base_head='parent', commit='generated', paths=package.GENERATED, files=[]))
            with patch.object(package, 'git', return_value='generated') as git, \
                 patch.object(package, 'verify_generated_commit'), \
                 patch.object(package, 'api', return_value={'object':{'sha':'user-new-commit'}}):
                with self.assertRaisesRegex(ValueError, 'never force-push'):
                    package.resume_generated_push(out)
                self.assertEqual(git.call_count, 1)

    def test_postdeadline_repository_recovery_is_explicit(self):
        class LateClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 14, 10, 0, 1, tzinfo=timezone.utc)
        def fake_git(*args):
            if args[:2] == ('branch','--show-current'):
                return 'main'
            if args[:2] == ('rev-parse','HEAD'):
                return 'head'
            if args[0] == 'ls-files':
                return args[-1]
            if args[0] == 'diff':
                return ''
            raise AssertionError(args)
        def fake_api(endpoint):
            if '/git/ref/' in endpoint:
                return {'object':{'sha':'head'}}
            return {'id':package.REPO_ID,'full_name':package.REPO,'permissions':{'admin':True},'private':False}
        with patch.object(package,'datetime',LateClock), patch.object(package,'git',side_effect=fake_git), \
             patch.object(package,'api',side_effect=fake_api), \
             patch.object(package,'read',return_value={'not_before_utc':'2026-10-14T10:00:00Z'}):
            with self.assertRaises(ValueError):
                package.verify_repository()
            self.assertEqual(package.verify_repository(allow_after_deadline=True),'head')

    def test_activation_receipt_is_idempotent(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp);policy=out/'policy.json'
            package.save(policy,{'test':True})
            package.save(out/'delivery/acceptance.json',{'test':True})
            prepared=dict(status='private_draft_ready',release_tag='test')
            with patch.object(package,'POLICY',policy),patch.object(package,'git',return_value='same-head'):
                first=package.finish_activation(out,prepared)
                with patch.object(package,'now',side_effect=AssertionError('no new timestamp')):
                    second=package.finish_activation(out,prepared)
            self.assertEqual(first,second)

    def test_plan_and_notes_have_no_publication_or_submission(self):
        with patch.object(package, 'api', side_effect=AssertionError('no network')), \
             patch.object(package, 'command', side_effect=AssertionError('no subprocess')):
            self.assertTrue(package.plan()['no_mutations'])
            text = package.release_notes('a'*64, 'b'*64, 'c'*40)
        self.assertIn('pending competition acceptance', text)
        self.assertIn('No fresh four-month CV', text)
        self.assertIn('retained matching full-year models', text)

    def test_successful_marker_cannot_mask_changed_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp); bundle=out/'bundle'; bundle.mkdir()
            package.save(bundle/'manifest.json', {'schema':'test'})
            candidate=out/'candidate';candidate.write_text('candidate')
            package.save(out/'release/replay_audit.json', dict(status='pass',
                manifest_sha256=package.sha(bundle/'manifest.json'), candidate_sha256='expected',
                output=package.artifact(candidate)))
            candidate.write_text('tampered')
            with self.assertRaises(ValueError):
                package.audit_replay(out,bundle,'expected')


if __name__ == '__main__':
    unittest.main()
