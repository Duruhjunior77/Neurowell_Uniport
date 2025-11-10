import numpy as np, pandas as pd

BANDS = dict(delta=(1,4), theta=(4,8), alpha=(8,13), beta=(13,30), gamma=(30,45))

def _band(psd, freqs, lo, hi):
    idx = (freqs>=lo) & (freqs<hi)
    return psd[..., idx].mean(-1)

def make_features(psd, freqs, chs):
    feats = {}
    for b,(lo,hi) in BANDS.items():
        bp = _band(psd, freqs, lo, hi)
        for i,c in enumerate(chs[:bp.shape[1]]):
            feats[f"{b}_{c}"] = bp[:, i]
    if 'AF7' in chs and 'AF8' in chs:
        a = _band(psd, freqs, 8, 13)
        L, R = a[:, chs.index('AF7')], a[:, chs.index('AF8')]
        feats['FAA'] = np.log(R+1e-8) - np.log(L+1e-8)
    return pd.DataFrame(feats)
