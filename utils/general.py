import time
from contextlib import contextmanager

import numpy as np


@contextmanager
def timer(name: str):
    s = time.time()
    yield
    elapsed = time.time() - s
    print(f"[{name}] {elapsed:.3f}sec")


def reduce_dtype(df):
    for c in df.columns:
        if df[c].dtype.name == "float64":
            df[c] = df[c].astype(np.float32)
    return df
