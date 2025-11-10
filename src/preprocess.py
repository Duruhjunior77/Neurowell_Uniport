import os
import numpy as np
import pandas as pd
import mne

def load_csv(path, ch_names, sfreq):
    df = pd.read_csv(path)
    data = df.values.T
    info = mne.create_info(ch_names=list(ch_names[:data.shape[0]]), sfreq=sfreq, ch_types="eeg")
    raw = mne.io.RawArray(data, info)
    return raw

def clean_raw(raw, notch, hp, lp):
    raw.notch_filter(notch)
    raw.filter(hp, lp, fir_design="firwin")
    raw.set_eeg_reference("average")
    return raw

def epochs_psd(raw, epoch_len=2.0, step=1.0, fmin=1, fmax=45):
    events = mne.make_fixed_length_events(raw, duration=step)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_len, baseline=None, preload=True)
    psd, freqs = mne.time_frequency.psd_welch(epochs, fmin=fmin, fmax=fmax, n_fft=512)
    return psd, freqs

def preprocess_folder(in_dir, out_npz, chs, sfreq, notch, hp, lp):
    files = [f for f in os.listdir(in_dir) if f.endswith(".csv")]
    X_all, meta = [], []
    for f in files:
        raw = load_csv(os.path.join(in_dir, f), chs, sfreq)
        raw = clean_raw(raw, notch, hp, lp)
        psd, freqs = epochs_psd(raw)
        X_all.append(psd)
        meta.append({"file": f, "n_epochs": int(psd.shape[0])})
    np.savez_compressed(out_npz, X=X_all, freqs=freqs, meta=meta)
    print(f"[Saved] {out_npz}")
