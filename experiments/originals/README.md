# NeuroWell — EEG + AI for Early Well-being Screening (Pilot)

**What it is.** A research prototype that uses short EEG recordings + quick self-reports
to estimate stress/mental-wellbeing risk for students. It is **not** a medical device.

## Install
```bash
conda create -n neurowell python=3.10 -y
conda activate neurowell
pip install -r requirements.txt
```

## Configure device
Edit `config.yaml`:
- `device.type`: `muse_ble` or `openbci_serial`
- Provide `muse_mac` or `serial_port`
- Adjust channel names to your montage.

## Collect data
```bash
python src/stream_record.py
```
This saves `data/raw/eeg_<task>.csv` for tasks: baseline, nback, stroop, breathing.

## Labeling
Create `labels/labels.csv` like:
```
file,label
eeg_baseline.csv,0
eeg_nback.csv,1
eeg_stroop.csv,1
eeg_breathing.csv,0
```

## Train
```bash
python src/train_sklearn.py
```
Outputs:
- `artifacts/model_logreg.pkl`
- `data/processed/features.csv`
- CV ROC-AUC printed in console.

## App
```bash
streamlit run app/app.py
```
Upload `features.csv` → get a risk score.

## Ethics
- Obtain consent/assent; students can opt out.
- Pseudonymize data; store locally & encrypted.
- Provide school resources/helplines.
