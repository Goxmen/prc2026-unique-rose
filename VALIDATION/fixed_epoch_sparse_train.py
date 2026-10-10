"""Production-only sparse fusion execution with exact frozen training updates.

The CV sources stay unchanged. Use the original optimizer, batches, direct loss,
OneCycle24 horizon and model. Monitor at most1024 unlabeled rows each epoch and
infer each ranking view once after the fixed epochs. Old/final sequence stores
are distinct because the same movement ID has different jittered input clocks.
"""
from dataclasses import dataclass
import math
import time
from types import SimpleNamespace

import numpy as np
import torch

import nn_sparse_track_jan_jul as original

MONITOR_ROWS = 1024
INFERENCE_CHUNK = 2048
SETTINGS = dict(training='exact frozen sparse train updates; fixed epochs only',
    monitor_rows=MONITOR_ROWS, monitor_selection='first rows in unchanged first-view order',
    monitor_labels='not used; ranking targets unavailable', inference_chunk=INFERENCE_CHUNK,
    full_inference='once per ranking view after the last fixed epoch',
    persistence='production saves completed weights before full ranking inference; inference-only resume',
    sequence_namespaces='independent train, old and final stores and ID alignment')


@dataclass(frozen=True)
class RankingView:
    name: str
    start: int
    stop: int
    tracks: object
    rows: np.ndarray


def ranking_views(data, ids_by_view, stores):
    """Allow overlapping IDs across views, never share their sequence lookup."""
    if list(ids_by_view) != list(stores) or not ids_by_view:
        raise ValueError('Ranking view order or stores differ')
    result, offset = [], 0
    seen_stores = set()
    for name, ids in ids_by_view.items():
        ids = np.asarray(ids)
        if len(ids) == 0 or len(np.unique(ids)) != len(ids):
            raise ValueError('Each ranking view must have nonempty unique IDs')
        store = stores[name]
        if getattr(store, 'ranking_view', None) != name:
            raise ValueError('Sequence store must be bound to its validated ranking_view manifest')
        if id(store) in seen_stores:
            raise ValueError('Ranking views must use distinct sequence stores')
        seen_stores.add(id(store))
        rows = store.align(ids)
        result.append(RankingView(name, offset, offset+len(ids), store, rows))
        offset += len(ids)
    if any(len(tensor) != offset for tensor in data.held):
        raise ValueError('Explicit ranking slices do not cover encoded tensors')
    return result


def predict_view(model, data, view, limit=None):
    count = len(view.rows) if limit is None else min(len(view.rows), limit)
    if count < 1:
        raise ValueError('Ranking view cannot be empty')
    inputs = SimpleNamespace(held=tuple(value[view.start:view.start+count] for value in data.held),
                             sd=data.sd, mu=data.mu)
    return original.predict(model, inputs, view.tracks, view.rows[:count], chunk=INFERENCE_CHUNK)


def train_updates(data, train_tracks, train_rows, views, seed, stop):
    if not views or stop < 1 or stop > original.shared.nni.NEW['epochs']:
        raise ValueError('Fixed epochs and ranking views are required')
    if len(train_rows) != len(data.y):
        raise ValueError('Training sequence alignment differs from ordinary rows')
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = original.prototype.SparseFusion(data.cards, data.bins, data.train[2].shape[1]).to(data.device)
    opt = torch.optim.AdamW(model.parameters(), lr=original.shared.nni.NEW['ple']['lr'],
                           weight_decay=original.shared.nni.NEW['ple']['weight_decay'])
    count, batch = len(data.y), original.shared.nni.NEW['batch']
    schedule = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=original.shared.nni.NEW['ple']['lr'],
        total_steps=original.shared.nni.NEW['epochs']*math.ceil(count/batch))
    generator = torch.Generator(device=data.device).manual_seed(seed)
    curve = []
    for epoch in range(stop):
        started, total = time.monotonic(), 0.
        perm = torch.randperm(count, device=data.device, generator=generator)
        cpu_perm = perm.cpu().numpy()
        for i in range(0, count, batch):
            rows = perm[i:i+batch]
            track_batch = train_tracks.batch(train_rows[cpu_perm[i:i+batch]])
            output = model(*(value[rows] for value in data.train), *track_batch)
            loss = torch.nn.functional.mse_loss(output, data.ty[rows, None].expand_as(output))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            schedule.step()
            total += float(loss.detach())*len(rows)
        train_seconds = time.monotonic()-started
        monitor = predict_view(model, data, views[0], limit=MONITOR_ROWS)
        if not np.isfinite(monitor).all():
            raise ValueError('Nonfinite unlabeled monitor output')
        elapsed = time.monotonic()-started
        curve.append(dict(epoch=epoch+1, train_mse_std=total/count, seconds=elapsed,
                          train_seconds=train_seconds, monitor_seconds=elapsed-train_seconds))
        original.shared.log(f'epoch {epoch+1}/{stop}: train {total/count:.4f}; '
                            f'updates {train_seconds:.2f}s; unlabeled monitor {elapsed-train_seconds:.2f}s')
    return model, curve


def infer_views(model, data, views):
    predictions = {}
    inference_started = time.monotonic()
    for view in views:
        predictions[view.name] = predict_view(model, data, view)
        if not np.isfinite(predictions[view.name]).all():
            raise ValueError('Nonfinite full ranking output')
    return predictions, time.monotonic()-inference_started


def train(data, train_tracks, train_rows, views, seed, stop):
    """Parity reference convenience; production persists updates before inference."""
    model, curve = train_updates(data, train_tracks, train_rows, views, seed, stop)
    predictions, elapsed = infer_views(model, data, views)
    curve[-1]['final_inference_seconds'] = elapsed
    return model, predictions, curve
