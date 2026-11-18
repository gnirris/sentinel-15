"""SENTINEL-15 — SOC dashboard (Gradio) for the two-stage UNSW-NB15 intrusion detector.
Run: python app.py   →  http://localhost:7860"""
from __future__ import annotations

import html
import tempfile

import gradio as gr
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from detector import MAX_ROWS, SAMPLE_CSV, Detector

det = Detector()
M = det.metrics
SAMPLE_RAW = pd.read_csv(SAMPLE_CSV)
FLOWS, _ = Detector.prepare(SAMPLE_RAW)

RED, GREEN, CYAN, AMBER = "#ff3b5c", "#2bd576", "#00e5ff", "#ffb020"
CAT_COLORS = {"Normal": GREEN, "Generic": RED, "Exploits": "#ff7a45", "Fuzzers": AMBER,
              "DoS": "#d94bff", "Reconnaissance": CYAN, "Analysis": "#5b8cff",
              "Backdoor": "#ff5fa2", "Shellcode": "#c0ff3e", "Worms": "#ff9e9e"}


# ── small render helpers ──────────────────────────────────────────────────
def dark(fig: go.Figure, h: int = 300) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=40, b=10),
                      font=dict(family="JetBrains Mono, monospace", size=12))
    return fig


def human(n: float) -> str:
    for unit in ("", "k", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def cards(items: list[tuple[str, str]]) -> str:
    cells = "".join(f'<div class="card"><div class="k">{html.escape(k)}</div>'
                    f'<div class="v">{html.escape(str(v))}</div></div>' for k, v in items)
    return f'<div class="cards">{cells}</div>'


def empty_fig(msg: str, h: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(color="#7d8b99"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return dark(fig, h)


# ── TAB 1: live simulator ─────────────────────────────────────────────────
def new_state() -> dict:
    return {"idx": 0, "history": []}


def shap_fig(exp: pd.DataFrame) -> go.Figure:
    e = exp.iloc[::-1]
    labels = [f"{f} = {v:.4g}" if isinstance(v, (int, float, np.number)) else f"{f} = {v}"
              for f, v in zip(e.feature, e.value)]
    fig = go.Figure(go.Bar(x=e.shap, y=labels, orientation="h",
                           marker_color=[RED if s > 0 else GREEN for s in e.shap],
                           hovertemplate="%{y}<br>SHAP %{x:+.3f}<extra></extra>"))
    fig.update_layout(title="Why: top 5 features  (red → pushes toward INTRUSION)",
                      xaxis_title="SHAP contribution (log-odds of attack)")
    return dark(fig, 300)


def history_fig(hist: list[dict], thr: float) -> go.Figure:
    h = pd.DataFrame(hist)
    fig = go.Figure(go.Scatter(x=list(range(len(h))), y=h.p, mode="lines+markers",
                               line=dict(color=CYAN, width=2), customdata=h.cat,
                               marker=dict(color=[RED if p >= thr else GREEN for p in h.p], size=7),
                               hovertemplate="p=%{y:.2f} %{customdata}<extra></extra>"))
    fig.add_hline(y=thr, line_dash="dot", line_color=AMBER, annotation_text="threshold")
    fig.update_layout(title="Attack probability — live stream",
                      yaxis=dict(range=[-0.03, 1.03]), xaxis_title="recent flows")
    return dark(fig, 300)


def render(state: dict, thr: float):
    idx = state["idx"]
    row = FLOWS.iloc[[idx]]
    r = row.iloc[0]
    pred = det.predict(row, thr).iloc[0]
    hist = state["history"]
    if not hist or hist[-1]["n"] != idx:
        hist.append({"n": idx, "p": float(pred.p_attack), "cat": pred.category})
        state["history"] = hist[-80:]

    truth = html.escape(str(r.get("attack_cat", "?")))
    if pred.verdict == "INTRUSION":
        unsure = pred.category_conf < .5
        name = pred.category.upper() + ("?" if unsure else "")
        banner = (f'<div class="banner alert">⚠ INTRUSION DETECTED — {name}<small>flow #{idx} · '
                  f'attack probability {pred.p_attack:.1%} · category confidence '
                  f'{pred.category_conf:.1%}{" (low — type uncertain)" if unsure else ""} · '
                  f'ground truth: {truth}</small></div>')
    else:
        banner = (f'<div class="banner safe">✓ SAFE — NORMAL TRAFFIC<small>flow #{idx} · '
                  f'attack probability {pred.p_attack:.1%} · ground truth: {truth}</small></div>')

    flow = cards([("Protocol", f"{r.proto}/{r.service}"), ("TCP state", r.state),
                  ("Duration", f"{r.dur:.3f}s"),
                  ("Packets ↑/↓", f"{int(r.spkts)}/{int(r.dpkts)}"),
                  ("Bytes ↑/↓", f"{human(r.sbytes)}/{human(r.dbytes)}"),
                  ("Src TTL", int(r.sttl))])
    ps = np.array([x["p"] for x in state["history"]])
    alerts = int((ps >= thr).sum())
    counters = cards([("Flows inspected", len(ps)), ("Alerts raised", alerts),
                      ("Alert rate", f"{alerts / len(ps):.0%}")])
    return (state, banner, flow, shap_fig(det.explain(row)),
            history_fig(state["history"], thr), counters)


def advance(state: dict, thr: float):
    state = dict(state)
    state["idx"] = (state["idx"] + 1) % len(FLOWS)
    return render(state, thr)


def reset(thr: float):
    return render(new_state(), thr)


def toggle(playing: bool, speed: float):
    playing = not playing
    return (playing, gr.Timer(value=1 / speed, active=playing),
            gr.Button(value="⏸ Pause" if playing else "▶ Play"))


# ── TAB 2: batch scanner ──────────────────────────────────────────────────
def scan(batch: tuple[str, pd.DataFrame] | None, thr: float):
    blank = ("", empty_fig("no log loaded"), empty_fig(""), None,
             gr.DownloadButton(visible=False))
    if batch is None:
        return blank
    name, raw = batch
    df, missing = Detector.prepare(raw)
    if missing:
        msg = (f'<div class="err">`{html.escape(name)}` is missing {len(missing)} UNSW-NB15 '
               f'columns: {html.escape(", ".join(missing))}</div>')
        return (msg,) + blank[1:]

    res = det.predict(df, thr)
    flagged = res.verdict == "INTRUSION"
    last = (("Accuracy vs. labels", f"{((res.p_attack >= thr) == df.label).mean():.1%}")
            if "label" in df.columns else
            ("Top threat", res.category[flagged].mode().iat[0] if flagged.any() else "—"))
    summary = (f'<div class="hdr2">{html.escape(name)} — {len(df):,} flows scanned</div>'
               + cards([("Flows", f"{len(df):,}"), ("Intrusions", f"{flagged.sum():,}"),
                        ("Threat rate", f"{flagged.mean():.1%}"), last]))

    counts = res.category.value_counts().reset_index()
    counts.columns = ["category", "flows"]
    pie = dark(px.pie(counts, names="category", values="flows", hole=.55, color="category",
                      color_discrete_map=CAT_COLORS, title="Threat distribution"), 360)
    hist = px.histogram(res, x="p_attack", nbins=40, title="Attack-probability distribution",
                        color_discrete_sequence=[CYAN])
    hist.add_vline(x=thr, line_dash="dot", line_color=AMBER)

    shown = res.copy()
    shown[["p_attack", "category_conf"]] = shown[["p_attack", "category_conf"]].astype("float64").round(4)
    report = pd.concat([shown, raw.loc[df.index]], axis=1)[flagged]
    report = report.sort_values("p_attack", ascending=False)
    path = tempfile.NamedTemporaryFile(prefix="flagged_", suffix=".csv", delete=False).name
    report.to_csv(path, index=False)
    return (summary, pie, dark(hist, 360), report.head(500),
            gr.DownloadButton(value=path, visible=True,
                              label=f"⬇ Download flagged report ({len(report):,} flows)"))


def load_sample(thr: float):
    batch = ("sample_test_flows.csv", SAMPLE_RAW.copy())
    return (batch,) + scan(batch, thr)


def load_upload(path: str | None, thr: float):
    if not path:
        return (None,) + scan(None, thr)
    try:
        raw = pd.read_csv(path, encoding="utf-8-sig", low_memory=False, nrows=MAX_ROWS)
    except Exception as e:  # a malformed upload must never crash the app
        out = scan(None, thr)
        return (None, f'<div class="err">Could not read that file as CSV: '
                      f'{html.escape(str(e))}</div>') + out[1:]
    batch = (path.rsplit("/", 1)[-1], raw)
    return (batch,) + scan(batch, thr)


# ── TAB 3: model & ethics (static figures + threshold-dependent curve) ────
def confusion_fig() -> go.Figure:
    mc = M["multiclass"]
    cm = np.array(mc["confusion"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(z=norm, x=mc["labels"], y=mc["labels"],
                               colorscale=[[0, "#0a0e14"], [1, CYAN]], text=cm.astype(int),
                               texttemplate="%{text}",
                               hovertemplate="true %{y} → predicted %{x}: %{text}<extra></extra>"))
    fig.update_layout(title="Confusion matrix (rows = truth, colour = row share)",
                      yaxis=dict(autorange="reversed"), xaxis_title="predicted")
    return dark(fig, 480)


def f1_fig() -> go.Figure:
    f1 = pd.Series(M["multiclass"]["per_class_f1"]).sort_values()
    fig = go.Figure(go.Bar(x=f1.values, y=f1.index, orientation="h",
                           marker_color=[RED if v < .3 else AMBER if v < .7 else GREEN for v in f1]))
    fig.update_layout(title="F1 per category", xaxis=dict(range=[0, 1]))
    return dark(fig, 480)


def importance_fig() -> go.Figure:
    a = M["ttl_audit"]
    imp = pd.Series(a["top_gain_features"])[::-1]
    fig = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h",
                           marker_color=[AMBER if i in a["dropped"] else CYAN for i in imp.index]))
    fig.update_layout(title="Stage-1 feature importance (share of gain, amber = TTL)")
    return dark(fig, 360)


def tradeoff_fig(thr: float) -> go.Figure:
    c = pd.DataFrame(M["threshold_curve"])
    fig = go.Figure()
    fig.add_scatter(x=c.threshold, y=c.false_alarms, name="false alarms (normal flagged)",
                    line=dict(color=AMBER))
    fig.add_scatter(x=c.threshold, y=c.missed_attacks, name="missed attacks",
                    line=dict(color=RED))
    fig.add_vline(x=thr, line_dash="dot", line_color=CYAN, annotation_text=f"current {thr:.2f}")
    fig.update_layout(title=f"Errors on {M['n_test']:,} test flows vs. alert threshold",
                      xaxis_title="threshold", yaxis_title="flows", legend=dict(orientation="h"))
    return dark(fig, 340)


def audit_md() -> str:
    b, a = M["binary"], M["ttl_audit"]
    wo = a["without_ttl"]
    share = sum(v for k, v in a["top_gain_features"].items() if k in a["dropped"])
    return (
        "In UNSW-NB15, the testbed's attack machines used different **TTL values** than "
        "normal hosts, so TTL fields could be a *shortcut*: true in this lab, meaningless on "
        f"a real network.\n\nThe TTL fields carry **{share:.0%}** of the model's decision "
        "weight. So we retrained without them:\n\n"
        "| | with TTL | without |\n|---|---|---|\n"
        f"| accuracy | {b['accuracy']:.2%} | {wo['accuracy']:.2%} |\n"
        f"| F1 | {b['f1']:.3f} | {wo['f1']:.3f} |\n\n"
        "Performance barely moves: the attack signal is **redundant**, also carried by "
        "connection-count and protocol features. The model is not *dependent* on the "
        "shortcut, but it *prefers* it when it is available.")


PIPELINE = """raw flow (45 fields)
  └─ drop `id` (row number → would leak)        BOM-safe CSV reading
  └─ ColumnTransformer
       ├─ proto / service / state → OneHotEncoder(handle_unknown='ignore')
       └─ 39 numeric flow statistics → passthrough
  └─ Stage 1  XGBoost binary      : Normal vs Attack   (threshold adjustable)
  └─ Stage 2  XGBoost 9-class     : which attack       (only runs on alerts)
  └─ TreeSHAP                     : per-alert explanation, one-hot folded back"""

ETHICS_ERRORS = """**⚖ Errors have different costs**

- A **missed attack** can mean a breach.
- A **false alarm** wastes an analyst's time, and real SOCs see millions of flows a day:
  a 1% false-alarm rate buries the real alerts (*alert fatigue*).
- So the threshold is a human policy choice, not a model fact. That is why it is a slider.
- Alerts are for **a human to investigate**, never for automatic blocking or punishing a user."""

ETHICS_PRIVACY = """**🔒 Packet privacy**

- The model reads **flow metadata only**: sizes, timings, counts, protocol. It never sees
  packet *contents*, so it works on encrypted traffic and reads no messages.
- No IP addresses are used as features, so it cannot learn to distrust a person or a country.
- **Limits:** UNSW-NB15 was generated in a 2015 lab. Attacks change, and accuracy on a real
  network would be lower than shown here. Retrain and re-audit before any real use."""


# ── layout ────────────────────────────────────────────────────────────────
CSS = """
.gradio-container {max-width: 1400px !important; font-family: 'JetBrains Mono', monospace;}
.hdr {font-size:.75rem; color:#7d8b99; letter-spacing:.16em; text-transform:uppercase;}
#title h1 {font-size:2.4rem; letter-spacing:.04em; margin:0;}
.banner {border-radius:6px; padding:18px 22px; font-weight:700; font-size:1.5rem;
         letter-spacing:.08em; border:1px solid;}
.safe  {background:rgba(43,213,118,.10); border-color:#2bd576; color:#2bd576;}
.alert {background:rgba(255,59,92,.13); border-color:#ff3b5c; color:#ff3b5c;
        animation:pulse 1.1s ease-in-out infinite;}
@keyframes pulse {50% {box-shadow:0 0 22px rgba(255,59,92,.55);}}
.banner small {display:block; font-size:.8rem; font-weight:400; letter-spacing:.02em;
               opacity:.85; margin-top:4px;}
.cards {display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr)); gap:10px;}
.card {background:#111823; border:1px solid #1f2a37; border-radius:6px; padding:10px 14px;}
.card .k {font-size:.72rem; color:#7d8b99; letter-spacing:.06em;}
.card .v {font-size:1.25rem; color:#e6edf3; margin-top:2px; white-space:nowrap;
          overflow:hidden; text-overflow:ellipsis;}
.hdr2 {font-size:1.05rem; margin:4px 0 10px; color:#e6edf3;}
.err {border:1px solid #ff3b5c; color:#ff3b5c; border-radius:6px; padding:12px;}
footer {display:none !important;}
.modebar-container {display:none !important;}
"""

# Always dark, whatever the viewer's OS theme is.
FORCE_DARK = """() => {
  const u = new URL(window.location);
  if (u.searchParams.get('__theme') !== 'dark') {
    u.searchParams.set('__theme', 'dark'); window.location.replace(u.href);
  }
}"""

THEME = gr.themes.Base(
    primary_hue="cyan", neutral_hue="slate",
    font=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
).set(body_background_fill_dark="#0a0e14", block_background_fill_dark="#0d131b",
      background_fill_primary_dark="#0a0e14", background_fill_secondary_dark="#111823",
      border_color_primary_dark="#1f2a37", button_primary_background_fill_dark="#00e5ff",
      button_primary_text_color_dark="#0a0e14")


def build() -> gr.Blocks:
    b = M["binary"]
    with gr.Blocks(title="SENTINEL-15 · Intrusion Detection") as demo:
        gr.HTML('<div class="hdr">security operations center · unsw-nb15</div>')
        gr.Markdown("# 🛡️ SENTINEL-15", elem_id="title")
        with gr.Row():
            gr.Markdown("Two-stage XGBoost network intrusion detector: stage 1 flags attacks, "
                        "stage 2 names them, SHAP explains every alert.  \n"
                        f"Test accuracy **{b['accuracy']:.2%}** · attack recall "
                        f"**{b['recall']:.2%}** · {M['n_test']:,} unseen flows")
            thr = gr.Slider(0.05, 0.95, 0.5, step=0.05, label="Alert threshold",
                            info="Lower = fewer missed attacks, more false alarms (see tab 3)")

        with gr.Tab("📡 Live Traffic Simulator"):
            state = gr.State(new_state())
            playing = gr.State(False)
            with gr.Row():
                play = gr.Button("▶ Play", variant="primary")
                step = gr.Button("⏭ Step")
                rst = gr.Button("↺ Reset")
                speed = gr.Slider(0.5, 8, 2, step=0.5, label="Speed (flows / second)", scale=3)
            banner = gr.HTML()
            flow = gr.HTML()
            with gr.Row():
                shap_plot = gr.Plot(show_label=False)
                hist_plot = gr.Plot(show_label=False)
            counters = gr.HTML()
            timer = gr.Timer(0.5, active=False)

            outs = [state, banner, flow, shap_plot, hist_plot, counters]
            timer.tick(advance, [state, thr], outs, show_progress="hidden")
            step.click(advance, [state, thr], outs, show_progress="hidden")
            rst.click(reset, [thr], outs)
            play.click(toggle, [playing, speed], [playing, timer, play])
            speed.change(lambda s: gr.Timer(value=1 / s), speed, timer)
            demo.load(render, [state, thr], outs)

        with gr.Tab("🗂 Batch CSV Scanner"):
            batch = gr.State(None)
            gr.Markdown("Upload any CSV in **UNSW-NB15 format**, or load the bundled sample log.")
            with gr.Row():
                up = gr.File(label="Flow log (.csv)", file_types=[".csv"], type="filepath",
                             scale=3)
                sample_btn = gr.Button("📂 Load Sample Log", variant="primary", scale=1)
            summary = gr.HTML()
            with gr.Row():
                pie = gr.Plot(show_label=False)
                phist = gr.Plot(show_label=False)
            table = gr.Dataframe(label="Flagged flows (top 500 by attack probability)",
                                 max_height=380, interactive=False)
            dl = gr.DownloadButton(visible=False)

            bouts = [batch, summary, pie, phist, table, dl]
            sample_btn.click(load_sample, [thr], bouts)
            up.change(load_upload, [up, thr], bouts)

        with gr.Tab("🧠 Model & AI Ethics"):
            mc = M["multiclass"]
            gr.Markdown(f"### Evaluation on held-out flows\nData: {M['source']}. "
                        "Test flows were never seen during training.")
            gr.HTML(cards([("Accuracy", f"{b['accuracy']:.2%}"),
                           ("Precision", f"{b['precision']:.2%}"),
                           ("Recall", f"{b['recall']:.2%}"), ("F1", f"{b['f1']:.3f}"),
                           ("10-class macro-F1", f"{mc['macro_f1']:.3f}")]))
            with gr.Row():
                gr.Plot(confusion_fig(), show_label=False, scale=3)
                gr.Plot(f1_fig(), show_label=False, scale=2)
            gr.Markdown("> **Analysis** and **Backdoor** are the weak spots: in UNSW-NB15 their "
                        "flows have near-identical statistics, so flow metadata alone cannot "
                        "reliably tell them apart. Both are still caught as *attacks* by stage 1; "
                        "the error is only in naming them.")
            gr.Markdown("### Pipeline")
            gr.Code(PIPELINE, language=None, show_label=False)
            gr.Markdown("### Shortcut audit — does it just memorise TTL values?")
            with gr.Row():
                gr.Plot(importance_fig(), show_label=False, scale=3)
                gr.Markdown(audit_md())
            gr.Markdown("### False-positive trade-off")
            curve = gr.Plot(tradeoff_fig(0.5), show_label=False)
            with gr.Row():
                gr.Markdown(ETHICS_ERRORS)
                gr.Markdown(ETHICS_PRIVACY)

        # The threshold is global: re-score the batch and move the trade-off marker.
        thr.release(scan, [batch, thr], bouts[1:])
        thr.release(tradeoff_fig, thr, curve)
    return demo


demo = build()

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, js=FORCE_DARK)
