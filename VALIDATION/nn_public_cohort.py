"""Fixed paired-flight context residual on the unchanged public/physical model."""
import numpy as np
import pandas as pd
import torch
from torch import nn

import flight_cohort_context as context
import nn_public_scratch as parent

CHANNELS=['relative_takeoff',*context.VALUES[:8],
    'actual_minus_op_ground','actual_minus_op_stand','actual_minus_ab_stand',
    'ground_pair_known','op_stand_pair_known','ab_stand_pair_known',
    'query_minus_peer_actual','query_peer_actual_known',
    'same_stand','stand_keys_known','same_runway','runway_keys_known',
    'stand_distance','stand_coordinates_known','peer_log_raw_count',
    'peer_ground_fraction','peer_ground_known','peer_original_clock_present']
SETTINGS=dict(context=context.SETTINGS,channels=CHANNELS,width=32,
    encoder='27->32ReLU->32ReLU; scalar masked attention plus masked max; pooled64+originalqueryhidden256->32ReLU->1',
    seconds='asinh(value/600)/asinh18 for bounded four clocks and differences',
    time='signed takeoff difference/1800',distance='min(stand_distance_m/40000,1)',
    zero_initialization='original parent parameters and RNG exact; fork_rng for new modules; final linear all zero',
    no_new_dropout=True,no_query_track_gate=True,
    missing_context='separate donor mask; no donors gives exact zero residual',
    query_index='integer-valued final plain slot used only for verified context lookup; stripped before parent')
assert len(CHANNELS)==27


class ContextBank(nn.Module):
    def __init__(self,meta,values,neighbors,ranking_view):
        super().__init__()
        assert meta.dtype==np.int64 and meta.ndim==2 and meta.shape[1]==len(context.META)
        assert values.dtype==np.float32 and values.shape==(len(meta),len(context.VALUES))
        assert neighbors.dtype==np.int32 and neighbors.shape==(len(meta),context.K)
        assert len(meta)>0 and np.isfinite(values).all()
        assert neighbors.min()>=-1 and neighbors.max()<len(meta)
        self.ids=pd.Index(meta[:,0].copy(),name='mvt_id');assert self.ids.is_unique
        self.ranking_view=ranking_view
        self.register_buffer('meta',torch.from_numpy(np.ascontiguousarray(meta[:,[1,3,4]])),persistent=False)
        self.register_buffer('values',torch.from_numpy(np.array(values,copy=True)),persistent=False)
        self.register_buffer('neighbors',torch.from_numpy(np.array(neighbors,copy=True)),persistent=False)

    def align(self,ids):
        rows=self.ids.get_indexer(np.asarray(ids))
        if (rows<0).any():raise ValueError('Query missing from explicit context bank')
        if not np.array_equal(rows.astype('float32').astype(np.int64),rows):raise ValueError('Index not exactly representable')
        return rows

    def tokens(self,query):
        index=self.neighbors[query].long();mask=index>=0;index=index.clamp_min(0)
        peer=self.values[index];own=self.values[query,None,:]
        pm=self.meta[index];qm=self.meta[query,None,:]
        pair=peer[:,:,4:5]*peer[:,:,5:8]
        scale=lambda x:torch.asinh(x/600)/np.arcsinh(18.)
        relative_clock=own[:,:,4:5]*peer[:,:,4:5]
        known_stand=((pm[:,:,1:2]>0)&(qm[:,:,1:2]>0)).to(peer.dtype)
        known_runway=((pm[:,:,2:3]>0)&(qm[:,:,2:3]>0)).to(peer.dtype)
        geometry=own[:,:,11:12]*peer[:,:,11:12]
        delta=peer[:,:,9:11]-own[:,:,9:11]
        distance=torch.sqrt(delta.square().sum(-1,keepdim=True)).div(40000).clamp(max=1)*geometry
        features=torch.cat([(pm[:,:,:1]-qm[:,:,:1]).to(peer.dtype)/1800000,
            scale(peer[:,:,:4]),peer[:,:,4:8],
            scale(peer[:,:,:1]-peer[:,:,1:4])*pair,pair,
            scale(own[:,:,:1]-peer[:,:,:1])*relative_clock,relative_clock,
            (pm[:,:,1:2]==qm[:,:,1:2]).to(peer.dtype)*known_stand,known_stand,
            (pm[:,:,2:3]==qm[:,:,2:3]).to(peer.dtype)*known_runway,known_runway,
            distance,geometry,peer[:,:,12:15],peer[:,:,8:9]],dim=-1)
        return features*mask[:,:,None],mask


class PeerResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(27,32),nn.ReLU(),nn.Linear(32,32),nn.ReLU())
        self.attention=nn.Linear(32,1)
        self.head=nn.Sequential(nn.Linear(64+256,32),nn.ReLU(),nn.Linear(32,1))
        nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)

    def forward(self,tokens,mask,query_hidden):
        # Sanitize masked slots before any network operation, including NaNs.
        x=torch.where(mask[:,:,None],tokens,torch.zeros_like(tokens))
        encoded=self.encoder(x)
        present=mask.any(1,keepdim=True)
        score=self.attention(encoded).squeeze(-1).masked_fill(~mask,-torch.inf)
        maximum_score=torch.where(present,score.max(1,keepdim=True).values,torch.zeros_like(score[:,:1]))
        weight=torch.exp(score-maximum_score)
        weight=weight/weight.sum(1,keepdim=True).clamp_min(1)
        mean=(encoded*weight[:,:,None]).sum(1)
        maximum=encoded.masked_fill(~mask[:,:,None],-10000.).max(1).values
        maximum=torch.where(present,maximum,torch.zeros_like(maximum))
        return self.head(torch.cat([mean,maximum,query_hidden],dim=1))*present


class SparseFusion(parent.SparseFusion):
    def __init__(self,cards,bins,n_plain,bank):
        super().__init__(cards,bins,n_plain-1)
        self.context_bank=bank
        with torch.random.fork_rng(devices=[]):self.peer=PeerResidual()

    def forward(self,cat,g,plain,points,mask,quality):
        query=plain[:,-1].long()
        model_plain=plain[:,:-1]
        # Exact original PublicFusion.forward order, evaluated once.
        hidden=self.tabular.inputs(cat,g,model_plain[:,:self.original_plain])
        for layer in list(self.tabular.mlp.children())[:-1]:hidden=layer(hidden)
        base=self.tabular.mlp[-1](hidden)
        representation=self.track(points,mask)
        delta=self.fusion(torch.cat([hidden,representation,quality],dim=1))
        original=base+delta*(mask.sum(1)>=2).unsqueeze(1)+self.public(torch.cat([hidden,model_plain[:,self.original_plain:]],dim=1))
        tokens,peers=self.context_bank.tokens(query)
        return original+self.peer(tokens,peers,hidden)


def synthetic_bank(n=24):
    meta=np.column_stack([np.arange(n)+100,np.arange(n)*30000,np.ones(n),
        np.arange(n)%4+1,np.arange(n)%2+1,np.full(n,-1)]).astype(np.int64)
    values=np.zeros((n,15),np.float32)
    values[:,:4]=np.arange(n)[:,None]*17+np.array([800,700,600,500])[None,:]
    values[:,4:9]=1;values[:,9]=np.arange(n)*13;values[:,10]=np.arange(n)*19
    values[:,11:]=1
    neighbor=np.full((n,context.K),-1,np.int32)
    for i in range(n):
        candidates=[j for j in range(n) if j!=i][:context.K]
        neighbor[i,:len(candidates)]=candidates
    neighbor[0]=-1;neighbor[1,1:]=-1
    return meta,values,neighbor


def check(device='cpu'):
    parent.physical.parent.shared.recipe()
    cards,bins,plain=[3,5],[torch.tensor([-1.,0.,1.,2.])]*3,17
    arrays=synthetic_bank();torch.manual_seed(42)
    old=parent.SparseFusion(cards,bins,plain).to(device);state=torch.get_rng_state().clone()
    torch.manual_seed(42);bank=ContextBank(*arrays,ranking_view='synthetic')
    new=SparseFusion(cards,bins,plain+1,bank).to(device)
    assert torch.equal(state,torch.get_rng_state())
    for name,value in old.state_dict().items():assert torch.equal(value,new.state_dict()[name]),name
    tab,track=parent.physical.parent.synthetic(cards,bins,plain,24,device)
    expanded=(*tab[:2],torch.cat([tab[2],torch.arange(24,device=device)[:,None]],1))
    for training in [False,True]:
        old.train(training);new.train(training);old.zero_grad();new.zero_grad()
        torch.manual_seed(912);expected=old(*tab,*track);expected.sum().backward()
        cpu=torch.get_rng_state().clone();gpu=torch.cuda.get_rng_state(device).clone()if device!='cpu'else None
        torch.manual_seed(912);actual=new(*expanded,*track);actual.sum().backward()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.equal(cpu,torch.get_rng_state())
        if gpu is not None:assert torch.equal(gpu,torch.cuda.get_rng_state(device))
        for name,value in old.named_parameters():
            torch.testing.assert_close(dict(new.named_parameters())[name].grad,value.grad,rtol=0,atol=0)
    old.eval();new.eval()
    with torch.no_grad():new.peer.head[-1].bias.fill_(.2)
    empty=(track[0],torch.zeros_like(track[1]),track[2])
    before=old(*tab,*empty);after=new(*expanded,*empty)
    torch.testing.assert_close(after[0],before[0],rtol=0,atol=0)
    torch.testing.assert_close(after[1:],before[1:]+.2,rtol=0,atol=0)
    tokens,mask=bank.tokens(torch.arange(24,device=device))
    pool=PeerResidual().to(device)
    with torch.no_grad():pool.head[-1].weight.fill_(.03)
    hidden=torch.zeros(24,256,device=device)
    tokens=tokens.detach().requires_grad_();answer=pool(tokens,mask,hidden);answer.sum().backward();grad=tokens.grad.clone()
    permutation=torch.arange(context.K-1,-1,-1,device=device)
    changed=tokens.detach()[:,permutation].requires_grad_();checkmask=mask[:,permutation]
    reordered=pool(changed,checkmask,hidden);reordered.sum().backward()
    torch.testing.assert_close(reordered,answer,rtol=0,atol=1e-6)
    torch.testing.assert_close(changed.grad[:,permutation],grad,rtol=0,atol=1e-7)
    bad=tokens.detach().clone();bad[~mask]=float('nan')
    torch.testing.assert_close(pool(bad,mask,hidden),answer,rtol=0,atol=0)
    # Equal clock and witness marginals, different within-peer pairing.
    fixture=PeerResidual().to(device)
    with torch.no_grad():
        for parameter in fixture.parameters():parameter.zero_()
        fixture.encoder[0].weight[0,9]=1;fixture.encoder[2].weight[0,0]=1
        fixture.head[0].weight[0,0]=1;fixture.head[-1].weight[0,0]=1
    pairtokens=torch.zeros(2,2,27,device=device)
    pairtokens[:,:,1]=torch.tensor([600.,1800.],device=device)/10800
    pairtokens[0,:,2]=torch.tensor([600.,1800.],device=device)/10800
    pairtokens[1,:,2]=torch.tensor([1800.,600.],device=device)/10800
    pairtokens[:,:,9]=(pairtokens[:,:,1]-pairtokens[:,:,2])
    pairanswer=fixture(pairtokens,torch.ones(2,2,dtype=torch.bool,device=device),torch.zeros(2,256,device=device))
    assert pairanswer[1]>pairanswer[0]
    return dict(device=device,initial_parameters_outputs_gradients_rng_exact=True,
        own_track_absent_context_active=True,no_peers_exact_parent=True,
        complete_donor_permutation_and_gradients=True,padding_nan_invariant=True,
        equal_marginal_different_pairing_distinguished=True,bank_nonpersistent=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a=p.parse_args();torch.set_num_threads(1);print(check(a.device))
