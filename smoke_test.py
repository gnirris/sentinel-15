"""Headless check: inference on the sample + the real app script, all three tabs,
with zero exceptions. Run: python smoke_test.py"""
import pandas as pd
from streamlit.testing.v1 import AppTest

from detector import SAMPLE_CSV, Detector

# 1. Inference + explanation on every sample flow
d = Detector()
df, missing = Detector.prepare(pd.read_csv(SAMPLE_CSV))
assert not missing, missing
res = d.predict(df)
assert len(res) == 500 and res.p_attack.between(0, 1).all()
for i in range(0, 500, 50):
    assert len(d.explain(df.iloc[[i]])) == 5
acc = ((res.p_attack >= .5) == df.label).mean()
print(f"inference ok: 500 flows, {acc:.1%} agree with labels")

# 2. Hostile input: unseen categories + NaNs + a missing column
bad = pd.read_csv(SAMPLE_CSV).head(20)
bad["proto"], bad["service"], bad.loc[3, "dur"] = "quic-new", "grpc", None
x, m = Detector.prepare(bad)
assert not m and len(d.predict(x)) == 20
assert Detector.prepare(bad.drop(columns=["sbytes"]))[1] == ["sbytes"]
print("hostile input ok")

# 3. The actual Streamlit app, headless
at = AppTest.from_file("app.py", default_timeout=60).run()
assert not at.exception, at.exception
next(b for b in at.button if "Step" in b.label).click().run()
assert not at.exception, at.exception
next(b for b in at.button if "Load Sample" in b.label).click().run()
assert not at.exception, at.exception
assert any("Intrusions" in mm.label for mm in at.metric), "batch scan did not render"
at.sidebar.slider[0].set_value(0.2).run()
assert not at.exception, at.exception
print(f"app ok: {len(at.tabs)} tabs, {len(at.metric)} metrics rendered, no exceptions")
