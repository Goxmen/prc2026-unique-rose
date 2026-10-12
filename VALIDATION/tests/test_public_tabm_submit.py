"""Small exact-update contract for the isolated production TabM trainer."""
import sys,tempfile,types,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import nn_public_tabm_submit as production

class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.net=torch.nn.Sequential(torch.nn.Linear(3,8),torch.nn.ReLU(),torch.nn.Dropout(.1),torch.nn.Linear(8,32))
    def forward(self,cat,g,plain,*track):return self.net(g)

class Tracks:
    def batch(self,rows):return ()

def probe(function):
    torch.manual_seed(42);np.random.seed(42);model=Toy()
    data=types.SimpleNamespace(device='cpu',y=np.arange(12.),ty=torch.linspace(-1,1,12),
        train=(torch.zeros(12,1,dtype=torch.long),torch.arange(36.).reshape(12,3)/20,torch.zeros(12,1)))
    namespace=dict(function.__globals__,BATCH=5,MICRO=3)
    small=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    curve=small(model,data,Tracks(),np.arange(12),2,'tabm','synthetic')
    return model.state_dict(),[x['train_mse_std'] for x in curve],torch.get_rng_state(),np.random.get_state()

class Contract(unittest.TestCase):
    def test_exact_cv_updates_partial_batches_and_rng(self):
        torch.set_num_threads(1)
        original_out=production.cv.OUT;old_function=production.cv.train
        reference=probe(old_function);actual=probe(production.trainer())
        self.assertIs(old_function,production.cv.train);self.assertEqual(original_out,production.cv.OUT)
        for key in reference[0]:torch.testing.assert_close(reference[0][key],actual[0][key],atol=0,rtol=0)
        self.assertEqual(reference[1],actual[1]);torch.testing.assert_close(reference[2],actual[2],atol=0,rtol=0)
        self.assertEqual(reference[3][0],actual[3][0]);np.testing.assert_array_equal(reference[3][1],actual[3][1]);self.assertEqual(reference[3][2:],actual[3][2:])

if __name__=='__main__':unittest.main()
