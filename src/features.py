import numpy as np
import pandas as pd

def bandpower(psd, freqs, bands):
    bp = {}
    for name, (fmin, fmax) in bands.items():
        idx = (freqs >= fmin) & (freqs <= fmax)
        bp[name] = psd[..., idx].mean(axis=-1)
    return bp

def make_features(psd, freqs, ch_names):
    bands = {
        "delta": (1, 3),
        "theta": (4, 7),
        "alpha": (8, 12),
        "beta": (13, 30),
        "gamma": (31, 45),
    }
    bp = bandpower(psd, freqs, bands)  # each value: (epochs, channels)
    dfs = []
    for bname, arr in bp.items():
        df = pd.DataFrame(arr, columns=[f"{c}_{bname}" for c in ch_names[:arr.shape[1]]])
        dfs.append(df)
    return pd.concat(dfs, axis=1)
