import numpy as np, pandas as pd, joblib, os
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from utils import load_cfg

def train_riemann_from_features():
    cfg = load_cfg()
    feats = pd.read_csv(os.path.join(cfg['paths']['processed_dir'], 'features.csv'))
    y = pd.read_csv('labels/labels.csv')
    X = feats.values
    yy = np.concatenate([np.repeat(lbl, len(feats)//len(y)) for lbl in y['label'].values])
    pipe = make_pipeline(TangentSpace(metric='riemann'), LogisticRegression(max_iter=300))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc = cross_val_score(pipe, X, yy, cv=cv, scoring='roc_auc')
    print('Riemann ROC-AUC:', float(auc.mean()), '+/-', float(auc.std()))

if __name__ == '__main__':
    train_riemann_from_features()
