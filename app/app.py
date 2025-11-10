import os, sys
import streamlit as st, pandas as pd, joblib

# --- Robust path resolution ---
def find_project_root():
    here = os.path.abspath(os.path.dirname(__file__))
    candidates = [
        os.path.abspath(os.path.join(here, "..")),      # repo root when extracted
        os.getcwd(),                                    # current working dir
        os.path.abspath(os.path.join(os.getcwd(), "..")),
    ]
    for root in candidates:
        if os.path.isdir(os.path.join(root, "src")) and os.path.isfile(os.path.join(root, "config.yaml")):
            return root
    # last resort: the directory that contains 'src' anywhere up from __file__
    p = here
    while True:
        if os.path.isdir(os.path.join(p, "src")) and os.path.isfile(os.path.join(p, "config.yaml")):
            return p
        newp = os.path.abspath(os.path.join(p, ".."))
        if newp == p:
            break
        p = newp
    return here  # fallback, may fail later

ROOT = find_project_root()
if os.path.join(ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(ROOT, "src"))

from src.utils import load_cfg

st.set_page_config(page_title="NeuroWell (UNIPORT Pilot)", layout="centered")
st.title("🧠 NeuroWell – UNIPORT Chapter Pilot")

cfg_path = os.path.join(ROOT, "config.yaml")
if not os.path.isfile(cfg_path):
    st.error("config.yaml not found. Please run from the project root or extract the ZIP first.")
    st.stop()

cfg = load_cfg(cfg_path)
model_path = os.path.join(cfg["paths"]["artifacts_dir"], "model_logreg.pkl")

@st.cache_resource
def load_model():
    return joblib.load(model_path)

file = st.file_uploader("Upload features.csv", type=["csv"])

if file:
    try:
        model = load_model()
        X = pd.read_csv(file)
        proba = model.predict_proba(X)
        risk = float(proba[:, 1].mean())
        st.metric("Estimated Risk (0–1)", f"{risk:.2f}")
        st.write("Interpretation:", "🟢 Low" if risk < 0.33 else "🟠 Medium" if risk < 0.66 else "🔴 High")
        st.caption("Research prototype for screening only. Not a medical device.")
    except Exception as e:
        st.exception(e)
        st.info("Tip: ensure the uploaded CSV matches the features produced by `python -m src.train_sklearn`.")
else:
    st.info("Generate features via `python -m src.train_sklearn` (writes data/processed/features.csv), then upload here.")
