---
title: SENTINEL-15
emoji: 🛡️
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Explainable AI network intrusion detection (UNSW-NB15)
---

# 🛡️ SENTINEL-15 — explainable network intrusion detection

A two-stage XGBoost detector trained on **UNSW-NB15**, with a dark SOC dashboard built in Gradio:
stage 1 decides *is this network flow an attack?*, stage 2 names *which kind*, and
every alert is explained with SHAP.

| Tab | What it does |
|---|---|
| 📡 Live Traffic Simulator | Streams held-out flows; SAFE / INTRUSION banner, confidence, category, top-5 SHAP reasons |
| 🗂 Batch CSV Scanner | Upload a UNSW-NB15 CSV (or 1-click sample), threat charts, downloadable flagged report |
| 🧠 Model & AI Ethics | Real test metrics, confusion matrix, TTL shortcut audit, false-alarm trade-off, privacy |

## Results (16,467 held-out flows)

| Stage | Metric | Score |
|---|---|---|
| 1 · Normal vs Attack | accuracy / F1 | 97.9% / 0.981 |
| 2 · 10-class end-to-end | accuracy / macro-F1 | 87.6% / 0.628 |

Weak classes: Analysis and Backdoor (F1 < 0.1). Their flows are statistically
near-identical in this dataset; both are still detected as attacks by stage 1.

## Run locally

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python app.py                   # http://localhost:7860
```

Models are already in `artifacts/`, so this works with no dataset download.

### Retrain

Put `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
([UNSW](https://research.unsw.edu.au/projects/unsw-nb15-dataset) or the Kaggle mirror) in
`data/`, then `python train.py`. With only one of the two files, it does a stratified
80/20 split. `python train.py --synthetic` checks the pipeline without any dataset.

### Test

```bash
python smoke_test.py   # inference on all 500 sample flows, hostile input, every UI handler
```

## Deploy (Render, free)

`render.yaml` configures everything:

1. https://dashboard.render.com → sign in with GitHub
2. **New → Blueprint** → pick this repo → **Apply**
3. Live at `https://sentinel-15.onrender.com` (or similar) after ~5 min

Footprint: 2.6 MB of models, ~360 MB peak RAM, inside the free 512 MB tier. The free tier
sleeps after 15 idle minutes; the first visit after that takes ~1 minute to wake.

(Hugging Face Spaces also works if you have PRO: the header block above is its config.)

## Files

`train.py` training + evaluation · `detector.py` inference + SHAP · `app.py` Gradio dashboard ·
`smoke_test.py` headless test · `artifacts/` models + `metrics.json` ·
`sample_test_flows.csv` 500 held-out demo flows · `docs/screenshots/`
