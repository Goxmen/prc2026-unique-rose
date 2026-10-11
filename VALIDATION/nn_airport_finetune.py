"""Airport-specific copies of a pretrained physical-track network.

The global continuation control uses the same two extra epochs, learning rate,
batch size and starting parameters. No new inputs or label transforms.
"""
import copy
import torch
from torch import nn
import nn_track_physical_scales as physical

SETTINGS=dict(parent=physical.SETTINGS,epochs=2,learning_rate=2e-5,batch=2048,
    optimizer='AdamW with reset state and original weight decay',
    variants=['global','airport'],initialization='saved physical seed42 fold weights',
    unknown_airport='original global network',inference='one complete network per airport')

class AirportFusion(nn.Module):
    def __init__(self,cards,bins,n_plain,airport_column):
        super().__init__();self.airport_column=airport_column
        self.original=physical.SparseFusion(cards,bins,n_plain)
        self.experts=nn.ModuleDict({str(k):copy.deepcopy(self.original) for k in range(1,cards[airport_column])})

    def initialize(self,state):
        self.original.load_state_dict(state)
        for expert in self.experts.values():expert.load_state_dict(state)

    def forward(self,cat,g,plain,points,mask,quality):
        airport=cat[:,self.airport_column];out=torch.empty((len(cat),1),device=cat.device,dtype=g.dtype)
        for key,expert in self.experts.items():
            take=airport==int(key)
            if take.any():out[take]=expert(cat[take],g[take],plain[take],points[take],mask[take],quality[take])
        unknown=airport==0
        if unknown.any():out[unknown]=self.original(cat[unknown],g[unknown],plain[unknown],points[unknown],mask[unknown],quality[unknown])
        return out

def check(device='cpu'):
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,5
    physical.parent.shared.recipe();torch.manual_seed(42)
    old=physical.SparseFusion(cards,bins,plain).to(device)
    ensemble=AirportFusion(cards,bins,plain,0).to(device);ensemble.initialize(old.state_dict())
    tab,track=physical.parent.synthetic(cards,bins,plain,24,device)
    old.eval();ensemble.eval()
    with torch.no_grad():
        expected=old(*tab,*track);actual=ensemble(*tab,*track)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        for key,expert in ensemble.experts.items():expert.fusion[-1].bias.add_(float(key))
        actual=ensemble(*tab,*track)
        for k in range(3):
            rows=tab[0][:,0]==k
            expert=ensemble.original if k==0 else ensemble.experts[str(k)]
            expected=expert(*(v[rows] for v in tab),*(v[rows] for v in track))
            torch.testing.assert_close(actual[rows],expected,rtol=0,atol=0)
    return dict(device=device,initial_predictions_match=True,expert_dispatch_exact=True,unknown_fallback_exact=True)
