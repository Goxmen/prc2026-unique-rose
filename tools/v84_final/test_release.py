"""Offline actual-state preservation and replay-contract tests."""
import importlib.util,json
from pathlib import Path
import tempfile,unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch


def module(name):
    spec=importlib.util.spec_from_file_location('v84test_'+name,Path(__file__).with_name(name+'.py'))
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj);return obj


replay,package,backup=module('replay'),module('package'),module('backup_neural_states')


class ReleaseTests(unittest.TestCase):
    def test_exact_stage_rounding(self):
        versions=list(range(51));weights=np.zeros(51);weights[0]=.5
        basis=pd.DataFrame({f'v{v}':[1,5,8] for v in versions})
        corrections=pd.DataFrame({'v78':[10,10,10],'tabm_v78':[12,8,11],'robust_v78':[11,10,12]})
        np.testing.assert_array_equal(replay.calculate(basis,corrections,versions,weights),[3,0,7])

    def test_reviewer_provenance_excludes_owner_and_canceled_research(self):
        for p in ['owner_selection.json','current_owner_authorization.json','research_status.json',
                  'current_disposition.json','hpo_weights/n42_0/trained.json']:
            self.assertFalse(package.computational_provenance(Path(p)))
        self.assertTrue(package.computational_provenance(Path('final/audit.json')))

    def test_backup_retains_real_tensors_and_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'actual.pt';pre=root/'encoder.pkl'
            state={'model':{'weight':torch.arange(6,dtype=torch.float32).reshape(2,3),'scalar':torch.tensor(7)},
                   'optimizer':{'unneeded':torch.ones(100)},'spec':{'epoch':15}}
            torch.save(state,model);pre.write_bytes(b'original encoder state')
            with patch.object(backup,'ROOT',root):
                records=[dict(backup.art(model),kind='model'),dict(backup.art(pre),kind='preprocessing')]
                inv=root/'inventory.json';inv.write_text(json.dumps({'artifacts':records,'missing':['HPO15']}))
                backup.backup(inv,root/'backup')
                got=torch.load(root/'backup/states/actual.pt',weights_only=False)
                self.assertNotIn('optimizer',got)
                self.assertEqual(backup.tensor_manifest(got),backup.tensor_manifest({k:v for k,v in state.items() if k!='optimizer'}))
                receipt=json.loads((root/'backup_receipt.json').read_text())
                self.assertTrue(receipt['every_archive_member_hash_verified'])
                self.assertFalse(receipt['complete_selected_weight_set'])
                self.assertEqual((root/'backup/states/encoder.pkl').read_bytes(),pre.read_bytes())

    def test_backup_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);model=root/'model.pt';torch.save({'state_dict':{'weight':torch.ones(1)}},model)
            with patch.object(backup,'ROOT',root):
                rec=dict(backup.art(model),kind='model');rec['sha256']='0'*64
                inv=root/'inventory.json';inv.write_text(json.dumps({'artifacts':[rec],'missing':[]}))
                with self.assertRaisesRegex(ValueError,'state changed'):backup.backup(inv,root/'backup')


if __name__=='__main__':unittest.main()
