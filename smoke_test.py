"""Headless check: inference on the sample, hostile input, and every Gradio event
handler the dashboard wires up, with zero exceptions. Run: python smoke_test.py"""
import tempfile

import pandas as pd

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

# 3. The Gradio app: build it and drive every handler
import app  # noqa: E402  (builds the Blocks graph at import)

out = app.render(app.new_state(), 0.5)
state = out[0]
for _ in range(30):                       # 30 timer ticks
    out = app.advance(state, 0.5)
    state = out[0]
assert state["idx"] == 30 and len(state["history"]) == 31
assert "banner" in out[1]
app.reset(0.2)
app.toggle(False, 4)

b = app.load_sample(0.5)
assert "500 flows scanned" in b[1] and b[4] is not None and len(b[4]) > 0
report = b[5].value
assert pd.read_csv(report if isinstance(report, str) else report["path"]).shape[0] == len(b[4])
app.scan(b[0], 0.2)                        # threshold slider re-scan

with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
    f.write("this,is\nnot,unsw\n")
r = app.load_upload(f.name, 0.5)
assert "missing" in r[1] and r[4] is None   # wrong schema → friendly error
with tempfile.NamedTemporaryFile("wb", suffix=".csv", delete=False) as f:
    f.write(b"\x00\xff\xfe garbage \x00")
r = app.load_upload(f.name, 0.5)
assert r[0] is None and 'class="err"' in r[1]  # unreadable → friendly error
app.load_upload(None, 0.5)
app.tradeoff_fig(0.3)
print("app ok: live sim, batch scan, bad uploads and threshold changes all handled")

# 4. The Streamlit front-end (streamlit_app.py, for Streamlit Community Cloud)
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
assert not at.exception, at.exception
next(b for b in at.button if "Step" in b.label).click().run()
next(b for b in at.button if "Load Sample" in b.label).click().run()
assert not at.exception, at.exception
assert any("Intrusions" in mm.label for mm in at.metric)
print("streamlit app ok: all tabs render, sample scan works")
