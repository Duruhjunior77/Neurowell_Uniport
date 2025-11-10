import os, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.preprocess import preprocess_folder
from src.features import make_features
from src.utils import load_cfg, makedirs

def load_npz(npz_path):
    npz = np.load(npz_path, allow_pickle=True)
    return npz["X"], npz["freqs"], npz["meta"]

def build_xy(cfg):
    npz_path = os.path.join(cfg["paths"]["processed_dir"], "psd_all.npz")
    Xs, freqs, meta = load_npz(npz_path)

    if cfg["device"]["type"] == "muse_ble":
        chs = cfg["muse_channels"]
    else:
        chs = cfg["openbci_channels"]

    labels = pd.read_csv(os.path.join(cfg["paths"]["labels_dir"], "labels.csv"))

    X_list, y_list = [], []
    for arr, m in zip(Xs, meta):
        X_feat = make_features(arr, freqs, chs)
        fname = m["file"]
        lab = labels.loc[labels.file == fname, "label"].values[0]
        X_list.append(X_feat)
        y_list.append(np.repeat(lab, len(X_feat)))

    X = pd.concat(X_list, ignore_index=True)
    y = np.concatenate(y_list)
    return X, y

def train_eval_save():
    cfg = load_cfg()
    makedirs(cfg["paths"]["artifacts_dir"])
    makedirs(cfg["paths"]["processed_dir"])

    raw_dir = cfg["paths"]["raw_dir"]
    out_npz = os.path.join(cfg["paths"]["processed_dir"], "psd_all.npz")

    if cfg["device"]["type"] == "muse_ble":
        chs = cfg["muse_channels"]
    else:
        chs = cfg["openbci_channels"]

    preprocess_folder(raw_dir, out_npz, chs, cfg["sampling_rate"], cfg["notch"], cfg["hp"], cfg["lp"])

    X, y = build_xy(cfg)
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=300))])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
    print("CV ROC-AUC:", float(auc.mean()), "+/-", float(auc.std()))

    pipe.fit(X, y)
    joblib.dump(pipe, os.path.join(cfg["paths"]["artifacts_dir"], "model_logreg.pkl"))
    X.to_csv(os.path.join(cfg["paths"]["processed_dir"], "features.csv"), index=False)

if __name__ == "__main__":
    train_eval_save()
