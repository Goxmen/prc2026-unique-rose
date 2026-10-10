"""Identical fixed-epoch N09 updates with bounded, unlabeled inference work.

Epochs are frozen before this function is called. The original trainer performs
full ranking inference after every epoch although every ranking label is NaN.
We monitor its first 1,024 rows, then infer every ranking row once after training.
Evaluation consumes no random numbers and never affects any optimization step.
"""
import math
import time

import numpy as np
import torch

import nn_residual_jan_jul as shared

MONITOR_ROWS = 1024
INFERENCE_CHUNK = 16384
SETTINGS = dict(monitor_rows=MONITOR_ROWS, monitor_selection='first rows in unchanged ranking order',
    inference_chunk=INFERENCE_CHUNK, full_inference='once after final frozen epoch',
    training='identical original initialization, batches, RNG, optimizer, schedule and loss',
    stopping='none; original frozen full-year epochs unchanged')


def train(data, rows, seed, stop, monitor):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model=shared.nni.PleNet(data.cards,data.bins,data.train[2].shape[1]).to(data.device)
    opt=torch.optim.AdamW(model.parameters(),lr=shared.nni.NEW['ple']['lr'],
                         weight_decay=shared.nni.NEW['ple']['weight_decay'])
    batch,count=shared.nni.NEW['batch'],len(rows)
    schedule=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=shared.nni.NEW['ple']['lr'],
                  total_steps=shared.nni.NEW['epochs']*math.ceil(count/batch))
    generator=torch.Generator(device=data.device).manual_seed(seed)
    indices=torch.from_numpy(rows).to(data.device)
    small = (tuple(a[:MONITOR_ROWS] for a in monitor[0]), monitor[1][:MONITOR_ROWS], monitor[2][:MONITOR_ROWS])
    curve=[]
    for epoch in range(stop):
        started,total=time.monotonic(),0.
        perm=indices[torch.randperm(count,device=data.device,generator=generator)]
        for i in range(0,count,batch):
            b=perm[i:i+batch]
            output=model(*(a[b] for a in data.train))
            loss=torch.nn.functional.mse_loss(output,data.ty[b,None].expand_as(output))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
            total+=float(loss)*len(b)
        train_seconds=time.monotonic()-started
        pred=shared.predict(model,small[0],data.sd,data.mu,small[2],chunk=INFERENCE_CHUNK)
        elapsed=time.monotonic()-started
        curve.append(dict(epoch=epoch+1,train_mse_std=total/count,
                          rmse=shared.nni.rmse(small[1],pred),seconds=elapsed,
                          train_seconds=train_seconds,monitor_seconds=elapsed-train_seconds))
        shared.log(f'epoch {epoch+1}: train {total/count:.4f}; '
                   f'training {train_seconds:.2f}s; fixed unlabeled monitor {elapsed-train_seconds:.2f}s')
    started=time.monotonic()
    pred=shared.predict(model,monitor[0],data.sd,data.mu,monitor[2],chunk=INFERENCE_CHUNK)
    inference_seconds=time.monotonic()-started
    curve[-1]['final_inference_seconds']=inference_seconds
    shared.log(f'Full ranking inference once: {inference_seconds:.2f}s, chunk {INFERENCE_CHUNK}')
    return model,pred,curve
