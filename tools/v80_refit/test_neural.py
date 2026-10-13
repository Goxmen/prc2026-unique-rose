"""Small CPU contracts for original training, resume and float serialization."""
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

SPEC=importlib.util.spec_from_file_location('v80_refit_neural',Path(__file__).with_name('neural.py'))
n=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(n)


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        n.restore_original_configuration()

    def fixture(self):
        pre=dict(cards=[4],bins=[torch.tensor([-1.,0.,1.])],plain=[1],num_width=2,mu=20.,sd=3.)
        cat=torch.tensor(np.arange(13)[:,None]%4,dtype=torch.long)
        g=torch.tensor(np.linspace(-1,1,13)[:,None],dtype=torch.float32)
        plain=torch.tensor(np.linspace(.1,.7,13)[:,None],dtype=torch.float32)
        y=torch.tensor(np.linspace(-.5,.4,13),dtype=torch.float32)
        return pre,(cat,g,plain),y

    def test_original_ple_training_and_rng_parity(self):
        pre,data,target=self.fixture()
        spec=dict(family='ple134',seed=42,stop=3,checkpoints=[1,3])
        d=SimpleNamespace(train=data,held=data,cards=pre['cards'],bins=pre['bins'],device='cpu',
                          ty=target,mu=pre['mu'],sd=pre['sd'])
        original,pred,curve=n.nni.train('ple',d,np.arange(len(target)),42,3,(data,np.zeros(len(target))))
        rng=torch.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as tmp:
            actual,other=n.train_one(spec,pre,data,target,Path(tmp),device='cpu')
            for key,val in original.state_dict().items():
                torch.testing.assert_close(val,actual.state_dict()[key],atol=0,rtol=0)
            torch.testing.assert_close(rng,torch.get_rng_state(),atol=0,rtol=0)
            np.testing.assert_array_equal(pred,n.infer(actual,data,pre))
            self.assertEqual([c['train_mse_std'] for c in curve],[c['train_mse_std'] for c in other])

    def test_resume_keeps_optimizer_sampler_and_dropout(self):
        pre,data,target=self.fixture()
        spec=dict(family='ple134',seed=142,stop=3,checkpoints=[1,3])
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            whole,_=n.train_one(spec,pre,data,target,Path(a),device='cpu')
            n.train_one(spec,pre,data,target,Path(b),device='cpu',stop_override=1)
            torch.manual_seed(999);np.random.seed(123)
            resumed,_=n.train_one(spec,pre,data,target,Path(b),device='cpu')
            for key,val in whole.state_dict().items():
                torch.testing.assert_close(val,resumed.state_dict()[key],atol=0,rtol=0)
            left=torch.load(Path(a)/'latest.pt',weights_only=False)
            right=torch.load(Path(b)/'latest.pt',weights_only=False)
            self.assertEqual(left['scheduler'],right['scheduler'])
            torch.testing.assert_close(left['generator'],right['generator'],atol=0,rtol=0)
            torch.testing.assert_close(left['torch_rng'],right['torch_rng'],atol=0,rtol=0)

    def test_original_plain_training_parity(self):
        frame=pd.DataFrame({'cat':pd.Categorical(['a']*22+['b']*22), 'num':np.linspace(-3,4,44)})
        item=dict(X=frame,Vn=frame.iloc[:7],y=np.linspace(50,110,44),raw=pd.DataFrame({'taxi_sec':np.zeros(7)}))
        expected,_,_=n.nnj.fit_nn(item,42,'cpu')
        encoder=n.nnj.Encoder(frame);cat,num=encoder.transform(frame);vc,vn=encoder.transform(item['Vn'])
        y=item['y'];mu,sd=float(y.mean()),float(y.std())
        pre=dict(cards=encoder.cards,num_width=num.shape[1],mu=mu,sd=sd)
        spec=dict(family='plain134',seed=42,stop=8,checkpoints=[8])
        with tempfile.TemporaryDirectory() as tmp:
            model,_=n.train_one(spec,pre,(torch.tensor(cat),torch.tensor(num)),torch.tensor(((y-mu)/sd).astype(np.float32)),Path(tmp),'cpu')
            np.testing.assert_array_equal(expected,n.infer(model,(torch.tensor(vc),torch.tensor(vn)),pre))

    def test_chunked_original_encoder(self):
        frame=pd.DataFrame({'cat':pd.Categorical(['a']*22+['b']*22),'num':np.linspace(-3,4,44),'missing':[float('nan')]+list(range(43))})
        enc=n.nnj.Encoder(frame);expected=enc.transform(frame)
        with tempfile.TemporaryDirectory() as tmp:
            actual=n.encode(enc,frame,Path(tmp),'train',chunk=7)
            for a,b in zip(expected,actual):
                np.testing.assert_array_equal(a,b)
                b._mmap.close()

    def test_frozen_trajectory_counts(self):
        t=n.trajectories();self.assertEqual(len(t),31)
        self.assertEqual(sum(x['family']=='plain134' for x in t),1)
        self.assertEqual(sum(x['family']=='ple134' for x in t),15)
        self.assertEqual(sum(x['family']=='combo156' for x in t),15)
        self.assertTrue(all(x['stop']==19 for x in t if x['family']=='combo156' and x['group']==142))
        self.assertTrue(all(x['checkpoints']==[14,15,17] for x in t if x['family']=='ple134' and x['group']==42))

    def test_checkpoint_saved_before_latest_survives_interruption(self):
        pre,data,target=self.fixture()
        spec=dict(family='ple134',seed=242,stop=2,checkpoints=[1,2])
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            first,_=n.train_one(spec,pre,data,target,folder,device='cpu')
            (folder/'latest.pt').unlink()
            resumed,curve=n.train_one(spec,pre,data,target,folder,device='cpu')
            self.assertEqual(len(curve),2)
            for key,value in first.state_dict().items():
                torch.testing.assert_close(value,resumed.state_dict()[key],atol=0,rtol=0)


if __name__=='__main__':
    unittest.main()
