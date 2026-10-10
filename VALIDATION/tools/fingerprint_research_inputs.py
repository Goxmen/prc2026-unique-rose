"""Compare reconstructed v58 matrices across machines without transferring them."""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = {}
    with threadpool_limits(1):
        for month in ['2025-01', '2025-07']:
            d = pd.read_pickle(args.inputs / f'{month}.pkl')
            names = d['baseline']
            assert len(names) == 156
            entry = {'columns': names}
            for part in ['X', 'V']:
                frame = d[part][names]
                entry[part] = dict(rows=len(frame), dtypes={c: str(frame[c].dtype) for c in names},
                    index=digest(pd.util.hash_pandas_object(frame.index).to_numpy()),
                    categories={c: dict(values=frame[c].cat.categories.astype(str).tolist(),
                                        ordered=frame[c].cat.ordered)
                                for c in names if isinstance(frame[c].dtype, pd.CategoricalDtype)},
                    columns={c: digest(pd.util.hash_pandas_object(frame[c], index=False).to_numpy())
                             for c in names})
            for part in ['y', 'ids', 'train_ids']:
                entry[part] = digest(np.asarray(d[part], dtype='float64' if part == 'y' else 'int64'))
            result[month] = entry
            print(f'{month}: fingerprinted original 156 inputs', flush=True)
            del d, frame
            gc.collect()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
