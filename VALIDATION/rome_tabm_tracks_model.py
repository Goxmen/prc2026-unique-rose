"""Matched TabM binary classifiers with a trainable physical-track branch.

The control receives the same tabular and track-quality inputs. A zero-initialized
projection makes initial logits identical; only the physical arm learns raw tracks.
"""
import math
import os
import time
os.environ.setdefault('TORCH_DISABLE_NATIVE_JIT','1')

import numpy as np
import rtdl_num_embeddings
import tabm
import torch
from torch import nn

import rome_neural_model as encoding
import nn_track_physical_scales as physical

SETTINGS=dict(tabular_columns=75,encoding='unchanged history-free Rome classifier encoder and32-bin PLE8',
    k=32,n_blocks=2,d_block=512,dropout=.1,activation='ReLU',arch_type='tabm',
    start_scaling_init='normal',batch=2048,micro=512,inference_batch=512,max_epochs=48,
    lr=.002,weight_decay=.0003,gradient_clip=1.,seed=42,
    loss='unweighted BCEWithLogits over rows and all32 individual heads',
    inference='mean of32 sigmoid probabilities, not sigmoid of mean logit',
    stopping='minimum inner-month mean-probability log loss; first epoch on a tie',
    schedule='OneCycle over48 epochs in inner and full fits; refit full at chosen epoch',
    physical='retained physical CNN plus trainable zero-initialized64-to64 projection',
    missing='projected raw-track representation zero for fewer than2points; same9quality inputs in both arms',
    arms=['tabular','physical'],no_search=True)


class Classifier(nn.Module):
    def __init__(self,state,arm):
        super().__init__()
        assert arm in SETTINGS['arms'] and state['settings']==encoding.SETTINGS
        self.arm=arm
        dims=[min(24,max(2,round(1.6*k**.56))) for k in state['cards']]
        self.emb=nn.ModuleList(nn.Embedding(k,d) for k,d in zip(state['cards'],dims))
        self.ple=(rtdl_num_embeddings.PiecewiseLinearEmbeddings(state['bins'],8,activation=False,version='B')
                  if state['bins'] else None)
        chunks=dims+[8]*len(state['bins'])+[1]*len(state['plain'])+[64]+[1]*9
        self.track=physical.PhysicalTrackEncoder(physical.parent.TrackEncoder())
        self.projection=nn.Linear(64,64)
        nn.init.zeros_(self.projection.weight);nn.init.zeros_(self.projection.bias)
        self.view=tabm.EnsembleView(k=32)
        self.backbone=tabm.make_tabm_backbone(d_in=sum(chunks),
            **{k:SETTINGS[k] for k in ['k','n_blocks','d_block','dropout','activation','arch_type','start_scaling_init']},
            start_scaling_init_chunks=chunks)
        self.output=tabm.LinearEnsemble(512,1,k=32)

    def forward(self,cat,g,plain,points,mask,quality):
        pieces=[emb(cat[:,k]) for k,emb in enumerate(self.emb)]
        if self.ple is not None:
            pieces.append(self.ple(g).flatten(1))
        if self.arm=='physical':
            representation=self.projection(self.track(points,mask))*(mask.sum(1)>=2).unsqueeze(1)
        else:
            representation=plain.new_zeros((len(plain),64))
        joined=torch.cat([*pieces,plain,representation,quality],dim=1)
        return self.output(self.backbone(self.view(joined))).squeeze(-1)


def loss(logits,target):
    assert logits.ndim==2 and logits.shape[1]==32 and target.ndim==1
    return nn.functional.binary_cross_entropy_with_logits(logits,target[:,None].expand_as(logits))


def probabilities(model,data,bank,rows):
    training=model.training;model.eval();values=[]
    assert len(data[0])==len(rows)
    with torch.no_grad():
        for start in range(0,len(rows),SETTINGS['inference_batch']):
            end=min(start+SETTINGS['inference_batch'],len(rows))
            logits=model(*(a[start:end] for a in data),*bank.batch(rows[start:end]))
            values.append(logits.sigmoid().mean(1).cpu().numpy().astype(float))
    model.train(training)
    result=np.concatenate(values)
    assert np.isfinite(result).all() and ((result>=0)&(result<=1)).all()
    return result


def fit(state,X,y,bank,rows,arm,epochs,monitor=None,device='cuda'):
    assert 1<=epochs<=SETTINGS['max_epochs'] and len(X)==len(y)==len(rows)
    y=np.asarray(y,np.float32);assert set(np.unique(y))=={0.,1.}
    data=encoding.encode(state,X,device);target=torch.from_numpy(y).to(device)
    torch.manual_seed(42);np.random.seed(42)
    model=Classifier(state,arm).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=SETTINGS['lr'],weight_decay=SETTINGS['weight_decay'])
    batch,micro=SETTINGS['batch'],SETTINGS['micro']
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=SETTINGS['lr'],
        total_steps=SETTINGS['max_epochs']*math.ceil(len(y)/batch))
    generator=torch.Generator(device=device).manual_seed(42)
    encoded_monitor=None if monitor is None else encoding.encode(state,monitor[0],device)
    curve=[]
    for epoch in range(epochs):
        started=time.monotonic();model.train()
        permutation=torch.randperm(len(y),device=device,generator=generator)
        cpu=permutation.cpu().numpy();total=0.
        for start in range(0,len(y),batch):
            end=min(start+batch,len(y));opt.zero_grad(set_to_none=True)
            for at in range(start,end,micro):
                stop=min(at+micro,end);index=permutation[at:stop]
                logits=model(*(a[index] for a in data),*bank.batch(rows[cpu[at:stop]]))
                error=loss(logits,target[index]);assert torch.isfinite(error)
                (error*((stop-at)/(end-start))).backward()
                total+=float(error.detach())*(stop-at)
            nn.utils.clip_grad_norm_(model.parameters(),SETTINGS['gradient_clip'])
            opt.step();schedule.step()
        point=dict(epoch=epoch+1,training_logloss=total/len(y),seconds=time.monotonic()-started)
        if monitor is not None:
            p=np.clip(probabilities(model,encoded_monitor,bank,monitor[2]),1e-12,1-1e-12)
            label=np.asarray(monitor[1],float)
            point['inner_logloss']=float(-np.mean(label*np.log(p)+(1-label)*np.log1p(-p)))
        curve.append(point)
        print(dict(arm=arm,**point),flush=True)
    return model,curve


def checks(device='cpu'):
    import pandas as pd
    frame=pd.DataFrame(dict(category=pd.Categorical(['A']*25+['B']*25+[None]*4),
                            varied=np.linspace(-3,3,54),missing=np.r_[np.arange(50),[np.nan]*4]))
    state=encoding.preprocessing(frame)
    data=encoding.encode(state,frame,device)
    _,track=physical.parent.synthetic(state['cards'],state['bins'],len(state['plain']),len(frame),device)
    torch.manual_seed(42);a=Classifier(state,'tabular').to(device).eval()
    torch.manual_seed(42);b=Classifier(state,'physical').to(device).eval()
    for name,value in a.state_dict().items():
        torch.testing.assert_close(value,b.state_dict()[name],atol=0,rtol=0)
    torch.testing.assert_close(a(*data,*track),b(*data,*track),atol=0,rtol=0)
    target=torch.arange(len(frame),device=device).remainder(2).float()
    optimizer=torch.optim.AdamW(b.parameters(),lr=.002,weight_decay=.0003)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);value=loss(b(*data,*track),target);value.backward()
        assert torch.isfinite(value)
        assert b.projection.weight.grad.abs().sum()>0
        if step:
            assert b.track.stem.weight.grad.abs().sum()>0
            assert b.track.physics_stem.weight.grad.abs().sum()>0
        optimizer.step()
    output=b(*data,*track)
    points=track[0].clone();points[~track[1]]=999
    torch.testing.assert_close(b(*data,points,*track[1:]),output,atol=0,rtol=0)
    missing=track[1].sum(1)<2
    points=track[0].clone();points[missing]=.3
    torch.testing.assert_close(b(*data,points,*track[1:])[missing],output[missing],atol=0,rtol=0)
    # The two head distributions below share a mean logit but not a mean probability.
    logits=torch.tensor([[-5.,1.]],device=device).repeat(1,16)
    assert abs((logits.sigmoid().mean(1)-logits.mean(1).sigmoid()).item())>.1
    # Partial logical batches preserve the intended average individual-head BCE gradient.
    x=torch.arange(35.,device=device).reshape(7,5)/17
    full=nn.Linear(5,32).to(device);split=nn.Linear(5,32).to(device);split.load_state_dict(full.state_dict())
    labels=torch.arange(7,device=device).remainder(2).float()
    loss(full(x),labels).backward()
    for i in range(0,7,3):
        loss(split(x[i:i+3]),labels[i:i+3]).mul(len(x[i:i+3])/7).backward()
    for p,q in zip(full.parameters(),split.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-7,rtol=1e-6)
    return dict(device=device,matched_initial_state_and_logits=True,projection_and_encoder_learn=True,
                padding_and_missing_track_invariance=True,individual_head_BCE_and_mean_probability=True,
                partial_logical_batch_gradient_correct=True)
