"""SENTINEL-15 — SOC dashboard for the two-stage UNSW-NB15 intrusion detector.
Run: streamlit run streamlit_app.py"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from detector import MAX_ROWS, SAMPLE_CSV, Detector

st.set_page_config(page_title="SENTINEL-15 · Intrusion Detection", page_icon="🛡️",
                   layout="wide")

PLOTLY_CFG = {"displayModeBar": False}
RED, GREEN, CYAN, AMBER, MUTED = "#ff3b5c", "#2bd576", "#00e5ff", "#ffb020", "#7d8b99"
CAT_COLORS = {"Normal": GREEN, "Generic": "#ff3b5c", "Exploits": "#ff7a45",
              "Fuzzers": AMBER, "DoS": "#d94bff", "Reconnaissance": CYAN,
              "Analysis": "#5b8cff", "Backdoor": "#ff5fa2", "Shellcode": "#c0ff3e",
              "Worms": "#ff9e9e"}

st.markdown("""
<style>
.block-container {padding-top: 1.6rem; max-width: 1400px;}
.banner {border-radius: 6px; padding: 18px 22px; font-weight: 700; font-size: 1.6rem;
         letter-spacing: .08em; border: 1px solid; margin-bottom: .6rem;}
.safe  {background: rgba(43,213,118,.10); border-color: #2bd576; color: #2bd576;}
.alert {background: rgba(255,59,92,.13); border-color: #ff3b5c; color: #ff3b5c;
        animation: pulse 1.1s ease-in-out infinite;}
@keyframes pulse {50% {box-shadow: 0 0 22px rgba(255,59,92,.55);}}
.banner small {display:block; font-size:.8rem; font-weight:400; letter-spacing:.02em;
               opacity:.85; margin-top:4px;}
.hdr {font-size: .75rem; color: #7d8b99; letter-spacing: .16em; text-transform: uppercase;}
div[data-testid="stMetricValue"] {font-size: 1.35rem;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading models…")
def load_detector() -> Detector:
    return Detector()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


def human(n: float) -> str:
    for unit in ("", "k", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.0f}{unit}" if unit == "" else f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def dark(fig: go.Figure, h: int = 300) -> go.Figure:
    fig.update_layout(template="plotly_dark", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(l=10, r=10, t=36, b=10),
                      font=dict(family="monospace", size=12))
    return fig


def shap_chart(exp: pd.DataFrame) -> go.Figure:
    e = exp.iloc[::-1]
    labels = [f"{f} = {v:.4g}" if isinstance(v, (int, float, np.number)) else f"{f} = {v}"
              for f, v in zip(e.feature, e.value)]
    fig = go.Figure(go.Bar(x=e.shap, y=labels, orientation="h",
                           marker_color=[RED if s > 0 else GREEN for s in e.shap],
                           hovertemplate="%{y}<br>SHAP %{x:+.3f}<extra></extra>"))
    fig.update_layout(title="Why: top 5 features (→ red pushes toward INTRUSION)",
                      xaxis_title="SHAP contribution (log-odds of attack)")
    return dark(fig, 280)


det = load_detector()
M = det.metrics

# ── header + sidebar ──────────────────────────────────────────────────────
st.markdown('<div class="hdr">security operations center · unsw-nb15</div>',
            unsafe_allow_html=True)
st.title("🛡️ SENTINEL-15")
st.caption("Two-stage XGBoost network intrusion detector — stage 1 flags attacks, "
           "stage 2 names them, SHAP explains every alert.")

with st.sidebar:
    st.markdown("### ⚙ Detector")
    threshold = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Probability of attack needed to raise an alert. Lower = "
                               "fewer missed attacks but more false alarms (see tab 3).")
    b = M["binary"]
    st.metric("Test accuracy", f"{b['accuracy']:.2%}")
    st.metric("Attack recall", f"{b['recall']:.2%}")
    st.caption(f"Trained on {M['n_train']:,} flows · tested on {M['n_test']:,} unseen flows")

tab1, tab2, tab3 = st.tabs(["📡 Live Traffic Simulator", "🗂 Batch CSV Scanner",
                            "🧠 Model & AI Ethics"])

# ── TAB 1: live simulator ─────────────────────────────────────────────────
with tab1:
    sample = load_sample()
    flows, _ = Detector.prepare(sample)
    ss = st.session_state
    ss.setdefault("idx", 0)
    ss.setdefault("playing", False)
    ss.setdefault("history", [])

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("▶ Play" if not ss.playing else "⏸ Pause", width="stretch"):
        ss.playing = not ss.playing
    if c2.button("⏭ Step", width="stretch"):
        ss.idx = (ss.idx + 1) % len(flows)
        ss.playing = False
    if c3.button("↺ Reset", width="stretch"):
        ss.idx, ss.history, ss.playing = 0, [], False
    speed = c4.select_slider("Speed (flows / second)", [0.5, 1, 2, 4, 8], value=2)

    @st.fragment(run_every=(1 / speed) if ss.playing else None)
    def live() -> None:
        if ss.playing:
            ss.idx = (ss.idx + 1) % len(flows)
        row = flows.iloc[[ss.idx]]
        r = row.iloc[0]
        pred = det.predict(row, threshold).iloc[0]
        if not ss.history or ss.history[-1]["n"] != ss.idx:
            ss.history.append({"n": ss.idx, "p": pred.p_attack, "cat": pred.category})
            ss.history = ss.history[-80:]

        alert = pred.verdict == "INTRUSION"
        truth = r.get("attack_cat", "?")
        if alert:
            name = pred.category.upper() + ("?" if pred.category_conf < .5 else "")
            st.markdown(f'<div class="banner alert">⚠ INTRUSION DETECTED — {name}'
                        f'<small>flow #{ss.idx} · attack probability {pred.p_attack:.1%} · '
                        f'category confidence {pred.category_conf:.1%}'
                        f'{" (low — type uncertain)" if pred.category_conf < .5 else ""}'
                        f' · ground truth: {truth}'
                        f'</small></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="banner safe">✓ SAFE — NORMAL TRAFFIC'
                        f'<small>flow #{ss.idx} · attack probability {pred.p_attack:.1%} · '
                        f'ground truth: {truth}</small></div>', unsafe_allow_html=True)

        m = st.columns(6)
        m[0].metric("Protocol", f"{r.proto}/{r.service}")
        m[1].metric("TCP state", r.state)
        m[2].metric("Duration", f"{r.dur:.3f}s")
        m[3].metric("Packets ↑/↓", f"{int(r.spkts)}/{int(r.dpkts)}")
        m[4].metric("Bytes ↑/↓", f"{human(r.sbytes)}/{human(r.dbytes)}")
        m[5].metric("Src TTL", int(r.sttl))

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(shap_chart(det.explain(row)), width="stretch", config=PLOTLY_CFG,
                            key=f"shap{ss.idx}")
        with right:
            h = pd.DataFrame(ss.history)
            fig = go.Figure(go.Scatter(x=list(range(len(h))), y=h.p, mode="lines+markers",
                                       line=dict(color=CYAN, width=2),
                                       marker=dict(color=[RED if p >= threshold else GREEN
                                                          for p in h.p], size=7),
                                       customdata=h.cat,
                                       hovertemplate="p=%{y:.2f} %{customdata}<extra></extra>"))
            fig.add_hline(y=threshold, line_dash="dot", line_color=AMBER,
                          annotation_text="threshold")
            fig.update_layout(title="Attack probability — live stream",
                              yaxis=dict(range=[-0.03, 1.03]), xaxis_title="recent flows")
            st.plotly_chart(dark(fig, 280), width="stretch", config=PLOTLY_CFG, key=f"hist{ss.idx}")

        h = pd.DataFrame(ss.history)
        k = st.columns(3)
        k[0].metric("Flows inspected", len(h))
        k[1].metric("Alerts raised", int((h.p >= threshold).sum()))
        k[2].metric("Alert rate", f"{(h.p >= threshold).mean():.0%}")

    live()

# ── TAB 2: batch scanner ──────────────────────────────────────────────────
with tab2:
    st.markdown("Upload any CSV in **UNSW-NB15 format**, or load the bundled sample log.")
    u1, u2 = st.columns([3, 1])
    up = u1.file_uploader("Flow log (.csv)", type="csv", label_visibility="collapsed")
    if u2.button("📂 Load Sample Log", width="stretch", type="primary"):
        st.session_state.batch = ("sample_test_flows.csv", load_sample())
    if up is not None:
        try:
            st.session_state.batch = (up.name, pd.read_csv(up, encoding="utf-8-sig",
                                                           low_memory=False, nrows=MAX_ROWS))
        except Exception as e:  # malformed upload must never crash the app
            st.error(f"Could not read that file as CSV: {e}")

    if "batch" in st.session_state:
        name, raw = st.session_state.batch
        df, missing = Detector.prepare(raw)
        if missing:
            st.error(f"`{name}` is missing {len(missing)} UNSW-NB15 columns: "
                     + ", ".join(f"`{c}`" for c in missing))
        else:
            with st.spinner(f"Scanning {len(df):,} flows…"):
                res = det.predict(df, threshold)
            flagged = res.verdict == "INTRUSION"
            st.markdown(f"##### `{name}` — {len(df):,} flows scanned")
            k = st.columns(4)
            k[0].metric("Flows", f"{len(df):,}")
            k[1].metric("Intrusions", f"{flagged.sum():,}")
            k[2].metric("Threat rate", f"{flagged.mean():.1%}")
            if "label" in df.columns:
                k[3].metric("Accuracy vs. labels", f"{((res.p_attack >= threshold) == df.label).mean():.1%}")
            else:
                k[3].metric("Top threat", res.category[flagged].mode().iat[0] if flagged.any() else "—")

            a, b2 = st.columns(2)
            counts = res.category.value_counts().reset_index()
            counts.columns = ["category", "flows"]
            fig = px.pie(counts, names="category", values="flows", hole=.55,
                         color="category", color_discrete_map=CAT_COLORS,
                         title="Threat distribution")
            a.plotly_chart(dark(fig, 330), width="stretch", config=PLOTLY_CFG)
            fig = px.histogram(res, x="p_attack", nbins=40, title="Attack-probability distribution",
                               color_discrete_sequence=[CYAN])
            fig.add_vline(x=threshold, line_dash="dot", line_color=AMBER)
            b2.plotly_chart(dark(fig, 330), width="stretch", config=PLOTLY_CFG)

            report = pd.concat([raw.loc[df.index], res.round(4)], axis=1)[flagged]
            report = report.sort_values("p_attack", ascending=False)
            st.markdown(f"##### Flagged flows ({len(report):,})")
            st.dataframe(report, width="stretch", height=300)
            st.download_button("⬇ Download flagged report (CSV)",
                               report.to_csv(index=False).encode(),
                               file_name=f"flagged_{name}", mime="text/csv")

# ── TAB 3: model + ethics ─────────────────────────────────────────────────
with tab3:
    b, mc, audit = M["binary"], M["multiclass"], M["ttl_audit"]
    st.markdown("#### Evaluation on held-out flows")
    st.caption(f"Data: {M['source']}. Test flows were never seen during training.")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{b['accuracy']:.2%}")
    k[1].metric("Precision", f"{b['precision']:.2%}")
    k[2].metric("Recall", f"{b['recall']:.2%}")
    k[3].metric("F1", f"{b['f1']:.3f}")
    k[4].metric("10-class macro-F1", f"{mc['macro_f1']:.3f}")

    a, b2 = st.columns([3, 2])
    labels = mc["labels"]
    cm = np.array(mc["confusion"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels, colorscale=[[0, "#0a0e14"], [1, CYAN]],
                               text=cm.astype(int), texttemplate="%{text}",
                               hovertemplate="true %{y} → predicted %{x}: %{text}<extra></extra>"))
    fig.update_layout(title="Confusion matrix (rows = truth, colour = row share)",
                      yaxis=dict(autorange="reversed"), xaxis_title="predicted")
    a.plotly_chart(dark(fig, 460), width="stretch", config=PLOTLY_CFG)
    f1 = pd.Series(mc["per_class_f1"]).sort_values()
    fig = go.Figure(go.Bar(x=f1.values, y=f1.index, orientation="h",
                           marker_color=[RED if v < .3 else AMBER if v < .7 else GREEN for v in f1]))
    fig.update_layout(title="F1 per category", xaxis=dict(range=[0, 1]))
    b2.plotly_chart(dark(fig, 460), width="stretch", config=PLOTLY_CFG)
    st.info("**Analysis** and **Backdoor** are the weak spots: in UNSW-NB15 their flows "
            "have near-identical statistics, so flow metadata alone cannot reliably tell them "
            "apart. Both are still caught as *attacks* by stage 1; the error is in naming them.")

    st.markdown("#### Pipeline")
    st.code(
        "raw flow (45 fields)\n"
        "  └─ drop `id` (row number → would leak)        BOM-safe CSV reading\n"
        "  └─ ColumnTransformer\n"
        "       ├─ proto / service / state → OneHotEncoder(handle_unknown='ignore')\n"
        "       └─ 39 numeric flow statistics → passthrough\n"
        "  └─ Stage 1  XGBoost binary      : Normal vs Attack   (threshold adjustable)\n"
        "  └─ Stage 2  XGBoost 9-class     : which attack       (only runs on alerts)\n"
        "  └─ TreeSHAP                     : per-alert explanation, one-hot folded back",
        language="text")

    st.markdown("#### Shortcut audit — does it just memorise TTL values?")
    imp = pd.Series(audit["top_gain_features"])
    wo = audit["without_ttl"]
    c1, c2 = st.columns([3, 2])
    fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h",
                           marker_color=[AMBER if i in audit["dropped"] else CYAN
                                         for i in imp.index[::-1]]))
    fig.update_layout(title="Stage-1 feature importance (share of gain)")
    c1.plotly_chart(dark(fig, 340), width="stretch", config=PLOTLY_CFG)
    ttl_share = sum(v for k2, v in imp.items() if k2 in audit["dropped"])
    c2.markdown(
        f"In UNSW-NB15, the testbed's attack machines used different **TTL values** than "
        f"normal hosts, so TTL fields could be a shortcut: true in this lab, meaningless "
        f"on a real network.\n\nThe TTL fields carry **{ttl_share:.0%}** of the model's "
        f"decision weight. So we retrained without them:\n\n"
        f"| | with TTL | without |\n|---|---|---|\n"
        f"| accuracy | {b['accuracy']:.2%} | {wo['accuracy']:.2%} |\n"
        f"| F1 | {b['f1']:.3f} | {wo['f1']:.3f} |\n\n"
        f"Performance barely moves: the attack signal is **redundant**, carried by "
        f"connection-count and protocol features too. The model is not *dependent* on "
        f"the shortcut, but it *prefers* it when available.")

    st.markdown("#### False-positive trade-off")
    curve = pd.DataFrame(M["threshold_curve"])
    fig = go.Figure()
    fig.add_scatter(x=curve.threshold, y=curve.false_alarms, name="false alarms (normal flagged)",
                    line=dict(color=AMBER))
    fig.add_scatter(x=curve.threshold, y=curve.missed_attacks, name="missed attacks",
                    line=dict(color=RED))
    fig.add_vline(x=threshold, line_dash="dot", line_color=CYAN,
                  annotation_text=f"current {threshold:.2f}")
    fig.update_layout(title=f"Errors on {M['n_test']:,} test flows vs. alert threshold",
                      xaxis_title="threshold", yaxis_title="flows", legend=dict(orientation="h"))
    st.plotly_chart(dark(fig, 320), width="stretch", config=PLOTLY_CFG)

    e1, e2 = st.columns(2)
    e1.markdown(
        "**⚖ Errors have different costs**\n\n"
        "- A **missed attack** can mean a breach.\n"
        "- A **false alarm** wastes an analyst's time, and real SOCs see millions of flows "
        "a day: a 1% false-alarm rate buries the real alerts (*alert fatigue*).\n"
        "- So the threshold is a human policy choice, not a model fact. That is why it is "
        "a slider here.\n"
        "- Alerts are for **a human to investigate**, never for automatic blocking or "
        "punishing a user.")
    e2.markdown(
        "**🔒 Packet privacy**\n\n"
        "- The model reads **flow metadata only**: sizes, timings, counts, protocol. It never "
        "sees packet *contents*, so it works on encrypted traffic and reads no messages.\n"
        "- No IP addresses are used as features, so it cannot learn to distrust a person or "
        "a country.\n"
        "- **Limits:** UNSW-NB15 was generated in a 2015 lab. Attacks change, and accuracy "
        "on a real network would be lower than shown here. It should be retrained and "
        "re-audited before any real use.")
