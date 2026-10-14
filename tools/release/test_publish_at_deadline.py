import copy
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent))
import publish_at_deadline as publish


class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.policy = dict(authorized=True, repository='Goxmen/PRC', repository_id=123,
            release_id=456, release_tag='fixture', release_source_commit='source',
            not_before_utc='2026-10-14T10:00:00Z',
            assets=[dict(name='bundle', github_asset_id=789, bytes=42, sha256='abc')])
        self.repository = dict(id=123, full_name='Goxmen/PRC', permissions=dict(admin=True),
                               private=True, html_url='https://example.com/repo')
        self.release = dict(id=456, tag_name='fixture', target_commitish='source', draft=True,
            html_url='https://example.com/release',
            assets=[dict(name='bundle', id=789, size=42, digest='sha256:abc', state='uploaded')])

    def test_cannot_publish_one_microsecond_early(self):
        instant = datetime(2026, 10, 14, 9, 59, 59, 999999, tzinfo=timezone.utc)
        with patch.object(publish, 'now', return_value=instant), patch.object(publish, 'api') as api:
            with self.assertRaises(ValueError):
                publish.publish(self.policy, Path('/unused'))
            api.assert_not_called()

    def test_exact_deadline_is_allowed(self):
        publish.guard_time(self.policy, datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc))

    def test_reject_changed_asset_digest(self):
        self.release['assets'][0]['digest'] = 'sha256:changed'
        with self.assertRaises(ValueError):
            publish.verify_remote(self.policy, self.repository, self.release)

    def test_reject_wrong_repository(self):
        self.repository['id'] = 124
        with self.assertRaises(ValueError):
            publish.verify_remote(self.policy, self.repository, self.release)

    def test_retry_finishes_partial_publication_and_is_idempotent(self):
        self.repository['private'] = False
        mutations = []
        def api(endpoint, payload=None):
            if payload is not None:
                mutations.append((endpoint, payload))
                self.release['draft'] = payload['draft']
            if endpoint.endswith('/git/ref/heads/main'):
                return {'object': {'sha': 'head'}}
            if '/git/matching-refs/tags/' in endpoint:
                return [] if self.release['draft'] else [dict(ref='refs/tags/fixture',
                    object=dict(type='commit', sha='source'))]
            return copy.deepcopy(self.release if '/releases/' in endpoint else self.repository)
        def preflight(_):
            return dict(reviewed_main='head', repository_private=self.repository['private'],
                        release_draft=self.release['draft'])
        with tempfile.TemporaryDirectory() as directory, patch.object(publish, 'api', side_effect=api), \
                patch.object(publish, 'preflight', side_effect=preflight), \
                patch.object(publish, 'now', return_value=datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)):
            receipt = Path(directory) / 'receipt.json'
            self.assertEqual(publish.publish(self.policy, receipt)['status'], 'published_verified')
            self.assertEqual(publish.publish(self.policy, receipt)['status'], 'published_verified')
        self.assertEqual(mutations, [('repos/Goxmen/PRC/releases/456',
            {'draft': False, 'tag_name': 'fixture', 'target_commitish': 'source'})])

    def test_cold_publication_order(self):
        mutations = []
        def api(endpoint, payload=None):
            if payload is not None:
                mutations.append(payload)
                (self.release if '/releases/' in endpoint else self.repository).update(payload)
                # A draft edit without an explicit tag can acquire an untagged
                # name. Preserve the reviewed tag in the publication request.
                if '/releases/' in endpoint and 'tag_name' not in payload:
                    self.release['tag_name'] = 'untagged-draft'
            if endpoint.endswith('/git/ref/heads/main'):
                return {'object': {'sha': 'head'}}
            if '/git/matching-refs/tags/' in endpoint:
                return [] if self.release['draft'] else [dict(ref='refs/tags/fixture',
                    object=dict(type='commit', sha='source'))]
            return copy.deepcopy(self.release if '/releases/' in endpoint else self.repository)
        check = dict(reviewed_main='head', repository_private=True, release_draft=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(publish, 'api', side_effect=api), \
                patch.object(publish, 'preflight', return_value=check), \
                patch.object(publish, 'now', return_value=datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)):
            result = publish.publish(self.policy, Path(directory) / 'receipt.json')
        self.assertEqual(result['status'], 'published_verified')
        self.assertEqual(mutations, [{'private': False},
            {'draft': False, 'tag_name': 'fixture', 'target_commitish': 'source'}])

    def test_head_change_prevents_all_mutations(self):
        check = dict(reviewed_main='head', repository_private=True, release_draft=True)
        with tempfile.TemporaryDirectory() as directory, patch.object(publish, 'api',
                return_value={'object': {'sha': 'changed'}}) as api, \
                patch.object(publish, 'preflight', return_value=check), \
                patch.object(publish, 'now', return_value=datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc)):
            with self.assertRaises(ValueError):
                publish.publish(self.policy, Path(directory) / 'receipt.json')
            self.assertTrue(all(len(call.args) == 1 for call in api.call_args_list))

    def test_backward_clock_prevents_visibility_patch(self):
        check = dict(reviewed_main='head', repository_private=True, release_draft=True)
        times = [datetime(2026, 10, 14, 10, 0, tzinfo=timezone.utc),
                 datetime(2026, 10, 14, 9, 59, tzinfo=timezone.utc)]
        with tempfile.TemporaryDirectory() as directory, patch.object(publish, 'api',
                return_value={'object': {'sha': 'head'}}) as api, \
                patch.object(publish, 'preflight', return_value=check), \
                patch.object(publish, 'now', side_effect=times):
            with self.assertRaises(ValueError):
                publish.publish(self.policy, Path(directory) / 'receipt.json')
            self.assertTrue(all(len(call.args) == 1 for call in api.call_args_list))

    def test_wrong_existing_tag_rejected(self):
        refs = [dict(ref='refs/tags/fixture', object=dict(type='commit', sha='wrong'))]
        with patch.object(publish, 'api', return_value=refs):
            with self.assertRaises(ValueError):
                publish.verify_tag(self.policy)

    def test_published_tag_required(self):
        with patch.object(publish, 'api', return_value=[]):
            publish.verify_tag(self.policy)
            with self.assertRaises(ValueError):
                publish.verify_tag(self.policy, required=True)


if __name__ == '__main__':
    unittest.main()
