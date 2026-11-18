"""Inference + explanation for the two-stage detector. No Streamlit here, so the
smoke test and any other script can use it directly."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

from train import CATEGORICAL, FEATURES, NUMERIC, clean

ART = Path(__file__).parent / "artifacts"
SAMPLE_CSV = Path(__file__).parent / "sample_test_flows.csv"
MAX_ROWS = 300_000  # keeps a batch scan inside a 512 MB container


class Detector:
    def __init__(self, art: Path = ART):
        self.binary = joblib.load(art / "binary.joblib")
        m = joblib.load(art / "multiclass.joblib")
        self.multi, self.classes = m["pipeline"], np.array(m["classes"])
        self.metrics = json.loads((art / "metrics.json").read_text())
        self.prep = self.binary.named_steps["prep"]
        self.enc_names = self.prep.get_feature_names_out()
        # Map every encoded column back to the raw field it came from
        # (proto_tcp, proto_udp ... -> proto) so explanations name real fields.
        self.source = np.array([next((c for c in CATEGORICAL if n.startswith(c + "_")), n)
                                for n in self.enc_names])
        self.explainer = shap.TreeExplainer(self.binary.named_steps["model"])

    # ── input validation ──────────────────────────────────────────────────
    @staticmethod
    def prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Clean an arbitrary uploaded table. Returns (frame, missing_columns)."""
        df = clean(df)
        missing = [c for c in FEATURES if c not in df.columns]
        if not missing:
            df[NUMERIC] = df[NUMERIC].fillna(0)
            df[CATEGORICAL] = df[CATEGORICAL].fillna("-")
        return df, missing

    # ── prediction ────────────────────────────────────────────────────────
    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        X = df[FEATURES]
        p = self.binary.predict_proba(X)[:, 1]
        out = pd.DataFrame(index=df.index)
        out["p_attack"] = p
        out["verdict"] = np.where(p >= threshold, "INTRUSION", "SAFE")
        out["category"] = "Normal"
        out["category_conf"] = 1 - p
        hit = p >= threshold
        if hit.any():
            proba = self.multi.predict_proba(X[hit])
            out.loc[hit, "category"] = self.classes[proba.argmax(1)]
            out.loc[hit, "category_conf"] = proba.max(1)
        return out

    # ── explanation ───────────────────────────────────────────────────────
    def explain(self, row: pd.DataFrame, top: int = 5) -> pd.DataFrame:
        """Exact TreeSHAP for one flow, in log-odds of 'attack'. Positive values
        push toward INTRUSION, negative toward SAFE."""
        Z = self.prep.transform(row[FEATURES])
        sv = np.asarray(self.explainer.shap_values(Z)).reshape(-1)
        agg = pd.Series(sv, index=self.source).groupby(level=0).sum()
        agg = agg.reindex(agg.abs().sort_values(ascending=False).index).head(top)
        r = row.iloc[0]
        return pd.DataFrame({"feature": agg.index, "shap": agg.values,
                             "value": [r[f] for f in agg.index]})
