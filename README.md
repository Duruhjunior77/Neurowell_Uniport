# NeuroWell – YNAN UNIPORT Pilot

End-to-end prototype: record EEG → preprocess → extract bandpower features → train a simple classifier → visualize risk in a Streamlit app.

## Folder Layout
```
.
├─ config.yaml
├─ src/
│  ├─ __init__.py
│  ├─ utils.py
│  ├─ preprocess.py
│  ├─ features.py
│  ├─ train_sklearn.py
│  └─ stream_record.py
├─ data/
│  ├─ raw/
│  └─ processed/
├─ labels/
│  └─ labels.csv
├─ artifacts/
│  └─ model_logreg.pkl
├─ app/
│  └─ app.py
└─ experiments/
   └─ train_riemann.py  # parked
```

## Quickstart
```bash
# 1) Create environment
conda create -n neurowell python=3.10 -y
conda activate neurowell
pip install -r requirements.txt

# 2) Configure
# Edit config.yaml with device + paths; ensure labels/labels.csv exists

# 3) (Optional) Record EEG with BrainFlow device
python -m src.stream_record

# 4) Train pipeline (preprocess -> features -> CV -> model)
python -m src.train_sklearn

# 5) Run the app
streamlit run app/app.py
```

## Notes
- `experiments/train_riemann.py` is a placeholder for future Riemannian geometry models (requires SPD covariances).
- This prototype is for research/education; not a medical device.
