
# --- path shim so app can find neurowell/src ---
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "src"))
# -----------------------------------------------

import streamlit as st, pandas as pd, joblib, os
from src.utils import load_cfg

# point to the real config.yaml in neurowell/
cfg = load_cfg(os.path.join(ROOT, "config.yaml"))

import streamlit as st, pandas as pd, joblib, os
from src.utils import load_cfg

st.set_page_config(page_title='NeuroWell (Pilot)', layout='centered')
st.title('🧠 NeuroWell – Student Well-being Screening (Pilot)')

cfg = load_cfg(os.path.join(ROOT, "config.yaml"))

model_path = os.path.join(cfg['paths']['artifacts_dir'], 'model_logreg.pkl')

@st.cache_resource
def load_model():
    return joblib.load(model_path)

file = st.file_uploader('Upload features CSV', type=['csv'])

if file:
    model = load_model()
    X = pd.read_csv(file)
    prob = float(model.predict_proba(X).mean(axis=0)[1])
    st.metric('Estimated Risk (0–1)', f'{prob:.2f}')
    st.write('Interpretation:', '🟢 Low' if prob<0.33 else '🟠 Medium' if prob<0.66 else '🔴 High')
    st.caption('Research prototype for screening only. Not a medical device.')
else:
    st.info('Generate features via `python src/train_sklearn.py` (it writes data/processed/features.csv). Then upload here.')
