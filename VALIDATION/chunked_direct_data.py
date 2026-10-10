"""Memory-bounded, exactly equivalent construction of the existing DirectData.

The original globally fitted Encoder and its transform method are unchanged.
Only row batching/storage and per-column PLE quantile evaluation differ. No
random draws, numeric downcasts beyond the original float32 transform, model
changes, target changes, or training changes are introduced.
"""
import gc
from pathlib import Path
import resource

import numpy as np
import torch

import nn_callsign_clocks_jan_jul as original

CHUNK_ROWS = 65536
SETTINGS = dict(chunk_rows=CHUNK_ROWS, encoder='unchanged nn_jan_jul.Encoder',
    transform='unchanged Encoder.transform on row slices',
    quantiles='same numpy quantile probabilities, one PLE input at a time',
    tensor_order='same int64 categories, float32 PLE and plain features',
    target='unchanged clipped direct target, population mean/std', random_draws=0)


def log(stage):
    current_kib = None
    status = Path('/proc/self/status')
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith('VmRSS:'):
                current_kib = int(line.split()[1])
    original.shared.log(f'{stage}; RSS_MiB={None if current_kib is None else round(current_kib/1024,1)}; '
                        f'peak_RSS_MiB={resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f}')


def encode(encoder, frame, directory, name, chunk_rows=CHUNK_ROWS):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if len(frame) == 0:
        raise ValueError('Production encoder requires nonempty train and ranking frames')
    cat_path, num_path = directory / f'{name}_cat.npy', directory / f'{name}_num.npy'
    if cat_path.exists() or num_path.exists():
        raise ValueError('Do not overwrite an existing encoding scratch artifact')
    cats = nums = None
    for start in range(0, len(frame), chunk_rows):
        end = min(start + chunk_rows, len(frame))
        cat, num = encoder.transform(frame.iloc[start:end])
        if cats is None:
            if cat.dtype != np.int64 or num.dtype != np.float32:
                raise ValueError('Original encoder output dtypes changed')
            cats = np.lib.format.open_memmap(cat_path, mode='w+', dtype=cat.dtype, shape=(len(frame), cat.shape[1]))
            nums = np.lib.format.open_memmap(num_path, mode='w+', dtype=num.dtype, shape=(len(frame), num.shape[1]))
        cats[start:end], nums[start:end] = cat, num
        del cat, num
    cats.flush()
    nums.flush()
    return cats, nums


def layout(encoder, num):
    g, plain, j = [], [], 0
    for name in encoder.nums:
        g.append(j)
        plain.append(j + 1)
        j += 2
        if name in encoder.flags:
            plain.append(j)
            j += 1
    if j != num.shape[1]:
        raise ValueError('Original numeric layout changed')
    probabilities = np.linspace(0, 1, original.shared.nni.NEW['bins'] + 1)
    bins, ple = [], []
    for col in g:
        values = np.unique(np.quantile(num[:, col], probabilities).astype(np.float32))
        if len(values) >= 3:
            bins.append(torch.from_numpy(values))
            ple.append(col)
        else:
            plain.append(col)
    return bins, ple, plain


def tensor_columns(array, columns, device, chunk_rows=CHUNK_ROWS):
    width = array.shape[1] if columns is None else len(columns)
    dtype = torch.int64 if array.dtype == np.int64 else torch.float32
    result = torch.empty((len(array), width), dtype=dtype, device=device)
    for start in range(0, len(array), chunk_rows):
        stop = min(start + chunk_rows, len(array))
        part = array[start:stop] if columns is None else array[start:stop, columns]
        source = torch.from_numpy(np.ascontiguousarray(part))
        result[start:stop].copy_(source)
    return result


class ChunkedDirectData(original.DirectData):
    def __init__(self, item, device, directory, chunk_rows=CHUNK_ROWS):
        log('fit original encoder')
        self.encoder = original.shared.nnj.Encoder(item['X'])
        enc = self.encoder
        log('transform train rows in chunks')
        cat, num = encode(enc, item['X'], directory, 'train', chunk_rows)
        log('transform ranking rows in chunks')
        vcat, vnum = encode(enc, item['Vn'], directory, 'held', chunk_rows)
        bins, ple, plain = layout(enc, num)
        self.bins, self.cards, self.ple, self.plain = bins, enc.cards, ple, plain
        self.columns = list(item['X'].columns)
        self.y = np.asarray(item['y'], float)
        self.yv = item['raw'].taxi_sec.to_numpy(float)
        target = np.minimum(self.y, original.shared.nnj.RECIPE['label_clip'])
        self.mu, self.sd = float(target.mean()), float(target.std())
        log('copy encoded tensor partitions to device')
        self.train = (tensor_columns(cat, None, device, chunk_rows),
                      tensor_columns(num, ple, device, chunk_rows),
                      tensor_columns(num, plain, device, chunk_rows))
        self.held = (tensor_columns(vcat, None, device, chunk_rows),
                     tensor_columns(vnum, ple, device, chunk_rows),
                     tensor_columns(vnum, plain, device, chunk_rows))
        self.ty = torch.from_numpy(((target-self.mu)/self.sd).astype(np.float32)).to(device)
        self.offset = np.zeros(len(self.y), float)
        self.val_offset = np.zeros(len(self.yv), float)
        last = max(original.ts.FOLDS[item['meta']['month']])
        self.inner_val = np.flatnonzero(item['tmonth'] == last)
        self.inner_train = np.flatnonzero(item['tmonth'] != last)
        self.device = device
        self.info = dict(numeric=len(enc.nums), ple_inputs=len(ple), plain_inputs=len(plain),
            categoricals=enc.cats, cards=enc.cards, bins_per_input=[len(b)-1 for b in bins],
            inner_month=last, inner_val_rows=len(self.inner_val), inner_train_rows=len(self.inner_train),
            target_mode='direct', offset_nonzero=0, offset_at_cap=0, target_mu=self.mu, target_sd=self.sd)
        # Tensors own their storage on CPU and CUDA. Close/delete scratch mappings
        # only after copies complete, retaining the same portable preprocessing.
        if str(device).startswith('cuda'):
            torch.cuda.synchronize()
        for value in [cat, num, vcat, vnum]:
            value._mmap.close()
        del cat, num, vcat, vnum
        gc.collect()
        log('bounded DirectData ready')
