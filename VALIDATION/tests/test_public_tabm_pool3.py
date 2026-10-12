"""Small exact-update contract for the isolated fixed-seed TabM continuation."""
import sys,tempfile,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import nn_public_tabm_pool3 as production
import tabm_pool3_v79_transport as transport

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(8,32))
    def forward(self,cat,g,plain,*track):return self.net(g)

class Tracks:
    def batch(self,rows):return ()

def probe(function,seed=42):
    torch.manual_seed(seed);np.random.seed(seed);model=Toy()
    data=types.SimpleNamespace(device='cpu',y=np.arange(12.),ty=torch.linspace(-1,1,12),
        train=(torch.zeros(12,1,dtype=torch.long),torch.arange(36.).reshape(12,3)/20,torch.zeros(12,1)))
    namespace=dict(function.__globals__,BATCH=5,MICRO=3)
    small=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    curve=small(model,data,Tracks(),np.arange(12),2,'tabm','synthetic')
    return model.state_dict(),[x['train_mse_std'] for x in curve],torch.get_rng_state(),np.random.get_state()

class Contract(unittest.TestCase):
    def test_exact_cv_updates_partial_batches_and_rng(self):
        torch.set_num_threads(1)
        original_out=production.original.OUT;old_function=production.original.train
        reference=probe(old_function);actual=probe(production.trainer(0))
        self.assertIs(old_function,production.original.train);self.assertEqual(original_out,production.original.OUT)
        for key in reference[0]:torch.testing.assert_close(reference[0][key],actual[0][key],atol=0,rtol=0)
        self.assertEqual(reference[1],actual[1]);torch.testing.assert_close(reference[2],actual[2],atol=0,rtol=0)
        self.assertEqual(reference[3][0],actual[3][0]);np.testing.assert_array_equal(reference[3][1],actual[3][1]);self.assertEqual(reference[3][2:],actual[3][2:])

    def test_fixed_seeds_repeat_and_transport_namespace(self):
        torch.set_num_threads(1)
        for k in [1,2]:
            function=production.trainer(k)
            self.assertEqual(function.__globals__['RUN_SEED'],production.SEEDS[k])
            a,b=probe(function,production.SEEDS[k]),probe(function,production.SEEDS[k])
            for key in a[0]:torch.testing.assert_close(a[0][key],b[0][key],atol=0,rtol=0)
            self.assertEqual(a[1],b[1]);torch.testing.assert_close(a[2],b[2],atol=0,rtol=0)
            np.testing.assert_array_equal(a[3][1],b[3][1]);self.assertEqual(a[3][2:],b[3][2:])
        old=transport.parent.EXPERIMENTS
        fn=transport.implementation('score')
        self.assertIs(fn.__code__,transport.parent.score.__code__)
        self.assertIs(transport.parent.EXPERIMENTS,old)
        self.assertNotIn('tabm_pool3',old)

if __name__=='__main__':unittest.main()
