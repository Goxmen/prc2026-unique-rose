"""Exact transport restoration and refusal to replace different committed bytes."""
from pathlib import Path
import gzip
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'tools/unpack_sparse_submit_cache_20261010.py'
spec = importlib.util.spec_from_file_location('restore_sparse_cache', SOURCE)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class TestRestore(unittest.TestCase):
    def test_exact_restore_resume_and_refuse_changed_committed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root/'cache'
            out.mkdir()
            def rec(path):
                return dict(path=str(path.relative_to(root)), sha256=b.sha(path), bytes=path.stat().st_size)
            def write(path, obj):
                path.write_text(json.dumps(obj))
            prepared = out/'manifest.json'
            write(prepared, {'input_only': True})
            auditor = root/'auditor.py'
            auditor.write_text('independent audit source')
            audit = dict(passed=True, auditor=rec(auditor), prepared_manifest=rec(prepared), views={})
            packet = dict(files=[], views={})
            originals = {}
            for i, view in enumerate(['old', 'final']):
                target = out/view
                target.mkdir()
                original = bytes(range(24))*(i+1)
                originals[view] = original
                compressed = target/'full_points.bin.gz'
                compressed.write_bytes(gzip.compress(original, mtime=0))
                index = target/'full_index.parquet'
                index.write_bytes(b'audited index '+view.encode())
                design = target/'full_design.json'
                write(design, {'view': view})
                digest = b.hashlib.sha256(original).hexdigest()
                manifest = target/'full.json'
                write(manifest, dict(outputs={'full_points.bin': digest, 'full_index.parquet': b.sha(index)},
                    raw_array_bytes=len(original), design_sha256=b.sha(design),
                    settings={'ranking_view': view}, rows=1, stored_points=i+1))
                audit['views'][view] = dict(manifest=rec(manifest))
                packet['views'][view] = dict(original_sha256=digest, original_bytes=len(original), compressed=rec(compressed))
                packet['files'].extend(rec(p) for p in [compressed, index, design, manifest])
            audit_path = out/'output_audit.json'
            write(audit_path, audit)
            packet['audit'] = rec(audit_path)
            packet['files'].append(rec(audit_path))
            write(out/'return_packet.json', packet)
            with patch.object(b, 'ROOT', root), patch.object(b, 'OUT', out):
                result = b.run()
                self.assertTrue(result['passed'])
                for view in originals:
                    self.assertEqual((out/view/'full_points.bin').read_bytes(), originals[view])
                    self.assertFalse(result['views'][view]['reused'])
                self.assertTrue(all(v['reused'] for v in b.run()['views'].values()))
                (out/'old/full_points.bin').write_bytes(b'different existing bytes')
                with self.assertRaisesRegex(ValueError, 'Refusing to replace'):
                    b.run()


if __name__ == '__main__':
    unittest.main()
