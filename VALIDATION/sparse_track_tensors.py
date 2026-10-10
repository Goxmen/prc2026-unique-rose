"""Vectorized ragged gathering and GPU transforms for frozen sparse tracks.

No source semantics differ from sparse_track_sequence.encoded_batch, which is
retained as the independently audited scalar reference. Geometry is calculated
in float64 before the final float32 cast, retaining raw microdegree precision.
"""
import math
import numpy as np
import torch

import sparse_track_sequence as seq


class RaggedTracks:
    def __init__(self, index, points, device='cuda'):
        if not index.mvt_id.is_unique:
            raise ValueError('Duplicate sequence movement IDs')
        self.index = index.reset_index(drop=True)
        self.points = points
        self.starts = index.start.to_numpy(np.int64)
        self.counts = index['count'].to_numpy(np.int64)
        self.ids = index.mvt_id.to_numpy(np.int64)
        self.id_rows = {int(value): row for row, value in enumerate(self.ids)}
        self.arange = np.arange(seq.MAX_POINTS, dtype=np.int64)[None, :]
        self.device = device
        age = index.to_ms.to_numpy(np.float64)-index.clock_ms.to_numpy(np.float64, na_value=np.nan)
        geo = index[['origin_lat','origin_lon','stand_lat','stand_lon','runway_lat','runway_lon']].to_numpy(np.float64)
        self.geometry = torch.from_numpy(np.column_stack([geo, age])).to(device)
        n = index.raw_count.to_numpy(np.float64)
        quality = np.column_stack([np.log1p(np.minimum(n,4096))/math.log1p(4096),
            self.counts/seq.MAX_POINTS, index.span_ms.to_numpy()/7260000,
            index.raw_ground_count.to_numpy()/np.maximum(n,1), np.isfinite(age),
            np.isfinite(geo[:,2:4]).all(1), np.isfinite(geo[:,4:6]).all(1),
            index.left_source_missing.to_numpy(),index.right_source_missing.to_numpy()]).astype(np.float32)
        self.quality = torch.from_numpy(quality).to(device)

    def align(self, ids):
        try:
            return np.asarray([self.id_rows[int(value)] for value in ids], dtype=np.int64)
        except KeyError as error:
            raise ValueError(f'Missing movement in sequence cache: {error}') from error

    def batch(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        mask = self.arange < self.counts[rows,None]
        offsets = np.where(mask, self.starts[rows,None]+self.arange, 0)
        if not len(self.points):
            if mask.any(): raise ValueError('Nonempty sequence index points to an empty raw array')
            raw = np.zeros((len(rows),seq.MAX_POINTS,6),dtype=np.int32)
        else:
            raw = np.asarray(self.points[offsets], dtype=np.int32)
        raw_tensor = torch.from_numpy(raw).to(self.device)
        present = torch.from_numpy(mask).to(self.device)
        row_tensor = torch.from_numpy(rows).to(self.device)
        return transform(raw_tensor, present, self.geometry[row_tensor]), present, self.quality[row_tensor]


def transform(raw, mask, geometry):
    r = raw.to(torch.float64)
    meta = geometry.unsqueeze(1)
    latitude, longitude = r[:,:,1]/1e6, r[:,:,2]/1e6
    speed_known = r[:,:,4] >= 0
    ground = r[:,:,3] == -123
    altitude_known = (r[:,:,3] >= 0) | ground
    clock_known = torch.isfinite(meta[:,:,6])
    stand_known = torch.isfinite(meta[:,:,2:4]).all(2)
    runway_known = torch.isfinite(meta[:,:,4:6]).all(2)
    # Missing reference coordinates are replaced before arithmetic, preserving
    # explicit masks and preventing nonfinite values from entering the model.
    meta = torch.nan_to_num(meta, nan=0.)
    def distance(first):
        north = (meta[:,:,first]-latitude)*111320
        east = (meta[:,:,first+1]-longitude)*111320*torch.cos((meta[:,:,first]+latitude)*(math.pi/360))
        return torch.sqrt(north.square()+east.square()).div(40000).clamp(0,1)
    columns = [r[:,:,0]/7200000,
        torch.log1p(r[:,:,5].clamp(min=0)/1000)/math.log1p(7260), r[:,:,5]>=0,
        ((longitude-meta[:,:,1])*111320*torch.cos(meta[:,:,0]*(math.pi/180))/20000).clamp(-2,2),
        ((latitude-meta[:,:,0])*111320/20000).clamp(-2,2),
        torch.where(speed_known,(r[:,:,4]/3000).clamp(0,1),0), speed_known,
        torch.where(altitude_known & ~ground,(r[:,:,3]*25/10000).clamp(0,1),0), altitude_known, ground,
        torch.where(stand_known,distance(2),0), stand_known.expand_as(latitude),
        torch.where(runway_known,distance(4),0), runway_known.expand_as(latitude),
        torch.where(clock_known,torch.asinh((meta[:,:,6]-r[:,:,0])/1800000).clamp(-8,8)/8,0),
        clock_known.expand_as(latitude)]
    return torch.stack(columns,dim=2).to(torch.float32)*mask.unsqueeze(2)


def check_pilot(device='cuda'):
    import pandas as pd
    import time
    index = pd.read_parquet(seq.OUT/'pilot_index.parquet')
    points = np.load(seq.OUT/'pilot_points.npy',mmap_mode='r',allow_pickle=False)
    tracks = RaggedTracks(index,points,device)
    # Cover every pilot movement and every stored point against the scalar
    # reference in bounded batches, including all sparse/missing geometries.
    maximum = 0.
    for start in range(0,len(index),1024):
        rows = np.arange(start,min(start+1024,len(index)))
        expected = seq.encoded_batch(index.iloc[rows],points)
        actual = tracks.batch(rows)
        for left,right in zip(actual,expected):
            value = left.cpu().numpy()
            np.testing.assert_allclose(value,right,rtol=2e-6,atol=2e-7)
            if value.dtype != bool:
                maximum = max(maximum,float(np.abs(value-right).max()))
    rows = np.arange(min(2048,len(index)))
    for _ in range(2): tracks.batch(rows)
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(10): tracks.batch(rows)
    if str(device).startswith('cuda'): torch.cuda.synchronize()
    return dict(status='passed',rows=len(index),stored_points=len(points),max_float_error=maximum,
        gather_transfer_transform_ms=1000*(time.perf_counter()-started)/10,batch=len(rows),device=device)
