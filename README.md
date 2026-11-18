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

A two-stage XGBoost detector trained on **UNSW-NB15**, with a dark SOC dashboard and an
**LLM analyst**: stage 1 decides *is this network flow an attack?*, stage 2 names *which
kind*, SHAP measures *why*, and an LLM turns that evidence into plain English.

**The LLM explains, it never decides.** It only sees the detector's numbers (never the
ground-truth label, IPs or packet contents), and a grounding check flags any field it
cites that isn't in that evidence. Without an API key the dashboard runs fully, with
template explanations instead.

**Any LLM provider works:** Anthropic (Claude), OpenAI, Google Gemini, Groq, OpenRouter,
Mistral, DeepSeek, xAI, or any OpenAI-compatible server (e.g. Ollama). Paste a key in the
sidebar and the provider is detected from its prefix; the model list is fetched live.
Gemini and Groq give free keys with no card.

| Tab | What it does |
|---|---|
| 📡 Live Traffic Simulator | Streams held-out flows; SAFE / INTRUSION banner, confidence, category, top-5 SHAP reasons |
| 🗂 Batch CSV Scanner | Upload a UNSW-NB15 CSV (or 1-click sample), threat charts, downloadable flagged report, LLM incident report |
| 💬 Ask the Analyst | Chat with the LLM about the current flow, the last scan, or the model itself |
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
python app.py                   # Gradio    → http://localhost:7860
streamlit run streamlit_app.py  # Streamlit → http://localhost:8501
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

## Deploy (Streamlit Community Cloud, free, no card)

The repo has two front-ends over the same `detector.py`: `app.py` (Gradio) and
`streamlit_app.py` (Streamlit). Streamlit Community Cloud hosts the Streamlit one free:

1. https://share.streamlit.io/deploy?repository=gnirris/sentinel-15&branch=main&mainModule=streamlit_app.py
2. Sign in with GitHub → **Advanced settings → Python 3.12**
3. Optional, for the LLM analyst: in the same Advanced settings box, paste one key into
   **Secrets** (any of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`,
   `GROQ_API_KEY`, `OPENROUTER_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `XAI_API_KEY`),
   e.g.
   ```toml
   GEMINI_API_KEY = "AIza..."
   LLM_MODEL = "gemini-2.5-flash"   # optional
   ```
   Without one, visitors can paste their own key in the sidebar, or use template mode.
4. **Deploy**

Footprint: 2.6 MB of models, ~320 MB peak RAM.

Other hosts: `render.yaml` (Render, needs a card on file) and the header block above
(Hugging Face Spaces, needs PRO for Gradio) also work.

## Files

`train.py` training + evaluation · `detector.py` inference + SHAP · `analyst.py` LLM analyst + grounding check · `app.py` Gradio dashboard · `streamlit_app.py` Streamlit dashboard ·
`smoke_test.py` headless test · `artifacts/` models + `metrics.json` ·
`sample_test_flows.csv` 500 held-out demo flows · `docs/screenshots/`
