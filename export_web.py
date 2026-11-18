"""Export the trained detector for the React front-end (web/), which runs the
models in the visitor's browser. Writes web/public/model/:

  model.json    both stages: one-hot rules + every tree as packed arrays
  flows.json    the 500 held-out demo flows
  metrics.json  test metrics (copied)

and web/test/reference.json: Python's own margins and SHAP values for every
sample flow, which web/test/verify.ts checks the JavaScript port against.
Run after train.py:  python export_web.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from detector import ART, SAMPLE_CSV, Detector
from train import CATEGORICAL, FEATURES, NUMERIC

WEB = Path(__file__).parent / "web"
OUT = WEB / "public" / "model"


def f32(a) -> list:
    """Float32 values as the shortest decimal that round-trips, to keep JSON small."""
    return [float(np.format_float_positional(np.float32(v), unique=True, trim="-")) for v in a]


def encoder(prep) -> dict:
    """The ColumnTransformer as plain lookup tables: category -> column index."""
    ohe = prep.named_transformers_["cat"]
    names = list(prep.get_feature_names_out())
    cat = {}
    for field, cats, rare in zip(CATEGORICAL, ohe.categories_, ohe.infrequent_categories_):
        rare = set() if rare is None else set(rare)
        cols = {}
        for c in cats:
            col = f"{field}_infrequent_sklearn" if c in rare else f"{field}_{c}"
            cols[str(c)] = names.index(col)
        cat[field] = cols                       # unknown category -> no column set
    return {"n_cols": len(names), "columns": names, "cat": cat,
            "num": {f: names.index(f) for f in NUMERIC}}


def trees(booster: xgb.Booster) -> tuple[list, list[int]]:
    model = json.loads(booster.save_raw("json"))["learner"]["gradient_booster"]["model"]
    out = []
    for t in model["trees"]:
        out.append({"l": t["left_children"], "r": t["right_children"],
                    "f": t["split_indices"], "t": f32(t["split_conditions"]),
                    "d": t["default_left"], "c": f32(t["sum_hessian"])})
    return out, model["tree_info"]


def base_margins(booster: xgb.Booster, objective_logistic: bool) -> list[float]:
    cfg = json.loads(booster.save_config())["learner"]["learner_model_param"]
    base = [float(v) for v in cfg["base_score"].strip("[]").split(",")]
    if objective_logistic:                       # stored as a probability
        base = [float(np.log(b / (1 - b))) for b in base]
    return base


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (WEB / "test").mkdir(exist_ok=True)
    d = Detector()

    bpipe = d.binary
    bbooster = bpipe.named_steps["model"].get_booster()
    btrees, _ = trees(bbooster)
    mpipe = d.multi
    mbooster = mpipe.named_steps["model"].get_booster()
    mtrees, minfo = trees(mbooster)

    model = {
        "features": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC,
        "binary": {**encoder(bpipe.named_steps["prep"]),
                   "base": base_margins(bbooster, True)[0], "trees": btrees,
                   "source": list(d.source)},
        "multi": {**encoder(mpipe.named_steps["prep"]),
                  "base": base_margins(mbooster, False), "trees": mtrees,
                  "tree_class": minfo, "classes": list(d.classes)},
    }
    (OUT / "model.json").write_text(json.dumps(model, separators=(",", ":")))
    shutil.copy(ART / "metrics.json", OUT / "metrics.json")

    # Demo flows: numbers stay numbers, NaN -> null
    raw = pd.read_csv(SAMPLE_CSV)
    flows, missing = Detector.prepare(raw)
    assert not missing
    keep = FEATURES + ["attack_cat", "label"]
    rows = flows[keep].replace({np.nan: None}).to_dict("records")
    (OUT / "flows.json").write_text(json.dumps(rows, separators=(",", ":")))

    # Answer key for the JS port
    Zb = bpipe.named_steps["prep"].transform(flows[FEATURES])
    Zm = mpipe.named_steps["prep"].transform(flows[FEATURES])
    ref = {
        "binary_margin": bbooster.predict(xgb.DMatrix(Zb), output_margin=True).tolist(),
        "binary_contribs": bbooster.predict(xgb.DMatrix(Zb), pred_contribs=True)[:, :-1].tolist(),
        "multi_prob": mbooster.predict(xgb.DMatrix(Zm)).tolist(),
        "explain_top5": [d.explain(flows.iloc[[i]]).assign(shap=lambda e: e.shap.round(6))
                         [["feature", "shap"]].values.tolist() for i in range(0, len(flows), 25)],
    }
    (WEB / "test" / "reference.json").write_text(json.dumps(ref))

    sizes = {p.name: f"{p.stat().st_size / 1e6:.2f} MB" for p in OUT.iterdir()}
    print("exported", sizes)


if __name__ == "__main__":
    main()
