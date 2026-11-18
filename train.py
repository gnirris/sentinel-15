"""Train the two-stage UNSW-NB15 intrusion detector.

Stage 1  binary XGBoost        : is this flow Normal (0) or an Attack (1)?
Stage 2  multi-class XGBoost   : if it is an attack, which of the 9 kinds?

Run:  python train.py              (uses data/UNSW_NB15_{training,testing}-set.csv)
      python train.py --synthetic  (no dataset needed: builds UNSW-schema fake data,
                                    only for checking the pipeline runs)

Outputs in artifacts/: binary.joblib, multiclass.joblib, metrics.json
and sample_test_flows.csv (500 held-out rows) in the repo root.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).parent
DATA = ROOT / "data"
ART = ROOT / "artifacts"
TRAIN_CSV = DATA / "UNSW_NB15_training-set.csv"
TEST_CSV = DATA / "UNSW_NB15_testing-set.csv"
SAMPLE_CSV = ROOT / "sample_test_flows.csv"

CATEGORICAL = ["proto", "service", "state"]
NUMERIC = [
    "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate", "sttl", "dttl", "sload",
    "dload", "sloss", "dloss", "sinpkt", "dinpkt", "sjit", "djit", "swin", "stcpb",
    "dtcpb", "dwin", "tcprtt", "synack", "ackdat", "smean", "dmean", "trans_depth",
    "response_body_len", "ct_srv_src", "ct_state_ttl", "ct_dst_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm", "is_ftp_login",
    "ct_ftp_cmd", "ct_flw_http_mthd", "ct_src_ltm", "ct_srv_dst", "is_sm_ips_ports",
]
FEATURES = CATEGORICAL + NUMERIC
ATTACKS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
           "Analysis", "Backdoor", "Shellcode", "Worms"]
CLASSES = ["Normal"] + ATTACKS
# Fields that identify the capture testbed rather than the attack itself (see audit)
TTL_SHORTCUTS = ["sttl", "dttl", "ct_state_ttl"]
SEED = 42


# ── data ──────────────────────────────────────────────────────────────────
def load_csv(path: Path) -> pd.DataFrame:
    # utf-8-sig: the public CSVs start with a BOM, which turns "id" into "﻿id"
    # and makes a plain drop("id") silently miss the row-number column.
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    return clean(df)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lstrip("﻿") for c in df.columns]
    df = df.drop(columns=[c for c in ("id",) if c in df.columns])
    if "attack_cat" in df.columns:
        cat = df["attack_cat"].astype(str).str.strip()
        df["attack_cat"] = cat.replace({"Backdoors": "Backdoor", "nan": "Normal", "": "Normal"})
    for c in CATEGORICAL:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    for c in NUMERIC:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def synthetic(n: int = 20000) -> pd.DataFrame:
    """UNSW-schema random data with a learnable signal. Pipeline test only."""
    rng = np.random.default_rng(SEED)
    cat = rng.choice(CLASSES, n, p=[.45, .2, .12, .07, .05, .04, .02, .02, .02, .01])
    k = np.array([CLASSES.index(c) for c in cat])
    df = pd.DataFrame({c: rng.exponential(1 + k, n) for c in NUMERIC})
    df["sttl"] = np.where(k == 0, 31, 254) + rng.integers(-2, 3, n)
    df["proto"] = rng.choice(["tcp", "udp", "arp", "ospf"], n)
    df["service"] = rng.choice(["-", "http", "dns", "ftp"], n)
    df["state"] = rng.choice(["FIN", "INT", "CON", "REQ"], n)
    df["attack_cat"] = cat
    df["label"] = (k > 0).astype(int)
    return df


def get_splits(use_synthetic: bool) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    if use_synthetic:
        df = synthetic()
        tr, te = train_test_split(df, test_size=0.2, stratify=df.attack_cat, random_state=SEED)
        return tr, te, "synthetic UNSW-schema data (pipeline test only)"
    if TRAIN_CSV.exists() and TEST_CSV.exists():
        return load_csv(TRAIN_CSV), load_csv(TEST_CSV), "official UNSW-NB15 train/test partition"
    found = [p for p in (TRAIN_CSV, TEST_CSV) if p.exists()]
    if not found:
        raise SystemExit(
            "No dataset in data/. Download UNSW_NB15_training-set.csv and\n"
            "UNSW_NB15_testing-set.csv from https://research.unsw.edu.au/projects/unsw-nb15-dataset\n"
            "(or the Kaggle mirror) into data/, or run: python train.py --synthetic")
    # Only one file: stratified 80/20 split, keeping every attack type in both halves.
    df = load_csv(found[0])
    tr, te = train_test_split(df, test_size=0.2, stratify=df.attack_cat, random_state=SEED)
    return tr, te, f"stratified 80/20 split of {found[0].name} ({len(df):,} flows)"


# ── models ────────────────────────────────────────────────────────────────
def preprocessor(numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                               sparse_output=False), CATEGORICAL),
         ("num", "passthrough", numeric)],
        verbose_feature_names_out=False)


def xgb(**kw) -> XGBClassifier:
    return XGBClassifier(n_estimators=250, max_depth=7, learning_rate=0.1,
                         subsample=0.9, colsample_bytree=0.8, tree_method="hist",
                         n_jobs=4, random_state=SEED, **kw)


def fit_binary(tr: pd.DataFrame, numeric: list[str]) -> Pipeline:
    pipe = Pipeline([("prep", preprocessor(numeric)),
                     ("model", xgb(objective="binary:logistic", eval_metric="logloss"))])
    return pipe.fit(tr[CATEGORICAL + numeric], tr["label"])


def binary_metrics(y, pred) -> dict:
    return {"accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred),
            "recall": recall_score(y, pred), "f1": f1_score(y, pred),
            "confusion": confusion_matrix(y, pred, labels=[0, 1]).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args()
    ART.mkdir(exist_ok=True)
    t0 = time.time()

    tr, te, source = get_splits(args.synthetic)
    print(f"data: {source}\n  train {len(tr):,} flows / test {len(te):,} flows")

    # Stage 1 ─ Normal vs Attack
    binary = fit_binary(tr, NUMERIC)
    p_attack = binary.predict_proba(te[FEATURES])[:, 1]
    pred_bin = (p_attack >= 0.5).astype(int)
    m_bin = binary_metrics(te["label"], pred_bin)
    print(f"stage 1  acc {m_bin['accuracy']:.4f}  f1 {m_bin['f1']:.4f}")

    # Stage 2 ─ which attack (trained on attack rows only)
    atr = tr[tr.label == 1]
    present = [a for a in ATTACKS if a in set(atr.attack_cat)]
    enc = LabelEncoder().fit(present)
    w = atr["attack_cat"].map(len(atr) / (len(present) * atr["attack_cat"].value_counts()))
    multi = Pipeline([("prep", preprocessor(NUMERIC)),
                      ("model", xgb(objective="multi:softprob", eval_metric="mlogloss"))])
    # Inverse-frequency weights, softened (sqrt) so 44 Worms rows can't dominate
    multi.fit(atr[FEATURES], enc.transform(atr["attack_cat"]),
              model__sample_weight=np.sqrt(w.to_numpy()))

    # End-to-end 10-class result: stage 1 decides, stage 2 names the attack
    cat_pred = np.where(pred_bin == 1, enc.inverse_transform(multi.predict(te[FEATURES])), "Normal")
    labels = [c for c in CLASSES if c in set(te.attack_cat) | set(cat_pred)]
    m_multi = {
        "accuracy": accuracy_score(te.attack_cat, cat_pred),
        "macro_f1": f1_score(te.attack_cat, cat_pred, labels=labels, average="macro"),
        "weighted_f1": f1_score(te.attack_cat, cat_pred, labels=labels, average="weighted"),
        "per_class_f1": dict(zip(labels, f1_score(te.attack_cat, cat_pred, labels=labels,
                                                  average=None).round(4).tolist())),
        "labels": labels,
        "confusion": confusion_matrix(te.attack_cat, cat_pred, labels=labels).tolist(),
    }
    print(f"stage 2  10-class acc {m_multi['accuracy']:.4f}  macro-f1 {m_multi['macro_f1']:.4f}")

    # Shortcut audit: retrain stage 1 WITHOUT the TTL fields and compare.
    no_ttl = [c for c in NUMERIC if c not in TTL_SHORTCUTS]
    audit_model = fit_binary(tr, no_ttl)
    m_audit = binary_metrics(te["label"], audit_model.predict(te[CATEGORICAL + no_ttl]))
    gain = binary.named_steps["model"].get_booster().get_score(importance_type="gain")
    names = binary.named_steps["prep"].get_feature_names_out()
    imp = pd.Series({names[int(k[1:])] if k.startswith("f") and k[1:].isdigit() else k: v
                     for k, v in gain.items()}).sort_values(ascending=False)
    imp = (imp / imp.sum()).head(10).round(4)
    print(f"audit    without TTL fields: acc {m_audit['accuracy']:.4f}  f1 {m_audit['f1']:.4f}")

    # Threshold trade-off curve for the ethics tab (false alarms vs missed attacks)
    y = te["label"].to_numpy()
    curve = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        p = p_attack >= t
        curve.append({"threshold": float(t),
                      "false_alarms": int((p & (y == 0)).sum()),
                      "missed_attacks": int((~p & (y == 1)).sum())})

    joblib.dump(binary, ART / "binary.joblib", compress=3)
    joblib.dump({"pipeline": multi, "classes": enc.classes_.tolist()},
                ART / "multiclass.joblib", compress=3)

    # 500 held-out rows for the demo: never seen in training. Mostly realistic mix,
    # but every attack type guaranteed to appear so the demo shows them all.
    parts = [g.sample(min(len(g), 12), random_state=SEED) for _, g in te.groupby("attack_cat")]
    seeded = pd.concat(parts)
    rest = te.drop(seeded.index).sample(500 - len(seeded), random_state=SEED)
    sample = pd.concat([seeded, rest]).sample(frac=1, random_state=SEED)
    sample[FEATURES + ["attack_cat", "label"]].to_csv(SAMPLE_CSV, index=False)

    metrics = {
        "source": source, "n_train": len(tr), "n_test": len(te),
        "binary": m_bin, "multiclass": m_multi,
        "ttl_audit": {"dropped": TTL_SHORTCUTS, "without_ttl": m_audit,
                      "top_gain_features": imp.to_dict()},
        "threshold_curve": curve,
        "features": {"categorical": CATEGORICAL, "numeric": NUMERIC},
        "train_seconds": round(time.time() - t0, 1),
    }
    (ART / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))

    size = sum(p.stat().st_size for p in ART.iterdir()) / 1e6
    print(f"saved {ART}/ ({size:.2f} MB) + {SAMPLE_CSV.name} in {metrics['train_seconds']}s")
    if size > 25:
        raise SystemExit(f"artifacts are {size:.1f} MB, over the 25 MB budget")


if __name__ == "__main__":
    main()
