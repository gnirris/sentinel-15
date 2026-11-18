"""SENTINEL-15 — SOC dashboard for the two-stage UNSW-NB15 intrusion detector,
with an LLM analyst (Claude) that explains its decisions in plain English.
Run: streamlit run streamlit_app.py"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analyst as A
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


@st.cache_resource(show_spinner=False)
def llm_client(provider: str, key: str, base_url: str | None):
    return A.make_client(provider, key, base_url)


@st.cache_data(ttl=3600, show_spinner=False)
def llm_models(provider: str, key: str, base_url: str | None) -> list[str]:
    return A.list_models(provider, llm_client(provider, key, base_url))


def secret(name: str):
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:  # no secrets file configured
        return os.environ.get(name)


def llm_sidebar() -> A.LLM | None:
    """Provider + key + model picker. Server keys come from Streamlit secrets or
    env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ...);
    a visitor's typed key is used only for their session."""
    st.markdown("### 🤖 LLM analyst")
    env_pid, env_key = A.key_from_env(secret)
    names = {pid: p.name + (" · free tier" if p.free else "") for pid, p in A.PROVIDERS.items()}
    choice = st.selectbox("Provider", ["auto", *A.PROVIDERS],
                          format_func=lambda k: "Auto-detect from key" if k == "auto" else names[k])
    typed = st.text_input("API key", type="password",
                          placeholder="uses the server key" if env_key else "paste any provider's key",
                          help="Used only for this browser session, never stored. "
                               "Gemini and Groq give free keys with no card.").strip()
    if typed:
        pid = A.detect(typed) if choice == "auto" else choice
        key = typed
        if pid is None:
            st.warning("Couldn't tell which provider that key is for. Pick it above.")
            return None
    elif choice not in ("auto", env_pid):
        pid = choice
        key = next((secret(v) for v in A.PROVIDERS[pid].env if secret(v)), None)
    else:
        pid, key = env_pid, env_key
    if not key:
        st.info("Offline: explanations use a template. Add a key for the LLM analyst.",
                icon="⚪")
        return None

    base = None
    if pid == "custom":
        base = st.text_input("Base URL", value=secret("LLM_BASE_URL") or "",
                             placeholder="http://localhost:11434/v1") or None
        if not base:
            st.warning("Enter the server's OpenAI-compatible base URL.")
            return None
    prov = A.PROVIDERS[pid]
    models = llm_models(pid, key, base)
    default = secret("LLM_MODEL") or prov.default
    if models:
        if default not in models:
            models = [default, *models] if default else models
        model = st.selectbox("Model", models, index=models.index(default) if default else 0)
    else:
        model = st.text_input("Model", value=default)
    if not model:
        return None
    llm = A.LLM(pid, model, llm_client(pid, key, base))
    st.success(llm.label, icon="🟢")
    return llm


def show_llm(text: str, evidence: str) -> None:
    """Render an LLM answer with its hallucination check and the evidence it saw."""
    ok, bad = A.grounding_check(text, evidence)
    if ok:
        st.caption("✅ Grounding check passed: every field the LLM cited is in the "
                   "detector's evidence.")
    else:
        st.warning("⚠ Grounding check failed: the LLM cited "
                   + ", ".join(f"`{b}`" for b in bad)
                   + ", which the detector never gave it. Treat that part as unreliable.")
    with st.expander("Evidence the LLM was given"):
        st.code(evidence, language="text")


det = load_detector()
M = det.metrics
ss = st.session_state
ss.setdefault("explained", {})
ss.setdefault("reports", {})
ss.setdefault("chat", [])

# ── header + sidebar ──────────────────────────────────────────────────────
st.markdown('<div class="hdr">security operations center · unsw-nb15</div>',
            unsafe_allow_html=True)
st.title("🛡️ SENTINEL-15")
st.caption("Two-stage XGBoost network intrusion detector — stage 1 flags attacks, "
           "stage 2 names them, SHAP measures why, and an LLM analyst explains it "
           "in plain English.")

with st.sidebar:
    st.markdown("### ⚙ Detector")
    threshold = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                          help="Probability of attack needed to raise an alert. Lower = "
                               "fewer missed attacks but more false alarms (see tab 3).")
    b = M["binary"]
    st.metric("Test accuracy", f"{b['accuracy']:.2%}")
    st.metric("Attack recall", f"{b['recall']:.2%}")
    st.caption(f"Trained on {M['n_train']:,} flows · tested on {M['n_test']:,} unseen flows")

    client = llm_sidebar()

tab1, tab2, tab4, tab3 = st.tabs(["📡 Live Traffic Simulator", "🗂 Batch CSV Scanner",
                                  "💬 Ask the Analyst", "🧠 Model & AI Ethics"])

# ── TAB 1: live simulator ─────────────────────────────────────────────────
with tab1:
    sample = load_sample()
    flows, _ = Detector.prepare(sample)
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

        exp = det.explain(row)
        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(shap_chart(exp), width="stretch", config=PLOTLY_CFG,
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

        # ── LLM analyst ──
        ev = A.flow_evidence(r, pred, exp, threshold)
        ss.flow_ev = f"current live flow #{ss.idx}:\n{ev}"
        st.markdown("##### 🤖 Analyst")
        key = (ss.idx, threshold)
        if key in ss.explained:
            st.markdown(ss.explained[key])
            show_llm(ss.explained[key], ev)
        elif client and not ss.playing and st.button("🤖 Explain this flow with the LLM",
                                                     key=f"explain{ss.idx}", type="primary"):
            ss.explained[key] = st.write_stream(A.explain_flow(client, ev))
            show_llm(ss.explained[key], ev)
        else:
            st.markdown(A.offline_flow(pred, exp))
            if client and ss.playing:
                st.caption("Pause the stream to ask the LLM about this flow.")

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

            # ── LLM incident report ──
            top = report.head(30)
            reasons = pd.Series(
                [f for i in top.index for f in det.explain(df.loc[[i]]).feature],
                dtype=object).value_counts().head(8)
            bev = A.batch_evidence(name, df, res, threshold, reasons)
            ss.batch_ev = bev
            st.markdown("##### 🤖 Incident report")
            rkey = (name, len(df), threshold)
            if rkey in ss.reports:
                st.markdown(ss.reports[rkey])
                show_llm(ss.reports[rkey], bev)
            elif not client:
                st.info("Add an LLM API key in the sidebar and the LLM analyst will "
                        "write an incident report for this scan.")
            elif st.button("🤖 Write incident report", type="primary"):
                ss.reports[rkey] = st.write_stream(A.incident_report(client, bev))
                show_llm(ss.reports[rkey], bev)
            if rkey in ss.reports:
                st.download_button("⬇ Download incident report (Markdown)",
                                   ss.reports[rkey].encode(),
                                   file_name=f"incident_{name}.md", mime="text/markdown")

# ── TAB 4: chat with the analyst ──────────────────────────────────────────
with tab4:
    st.markdown("Ask about the **current live flow**, the **last batch scan**, or how the "
                "model works. The analyst answers only from the detector's own numbers.")
    if not client:
        st.info("The chat needs an LLM API key (sidebar). Everything else on this "
                "dashboard works without one.")
    else:
        suggestions = ["Why was the current flow flagged (or not)?",
                       "Summarise the last batch scan for my manager.",
                       "Which attack types does this model struggle with, and why?",
                       "What would happen if I lowered the alert threshold to 0.2?"]
        cols = st.columns(len(suggestions))
        picked = next((q for c, q in zip(cols, suggestions)
                       if c.button(q, width="stretch")), None)
        for msg in ss.chat:
            with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
        q = st.chat_input("Ask the analyst…") or picked
        if q:
            context = "\n\n".join(x for x in (
                "MODEL:\n" + A.model_evidence(M),
                ss.get("flow_ev"),
                ("LAST BATCH SCAN:\n" + ss.batch_ev) if ss.get("batch_ev") else None) if x)
            with st.chat_message("user"):
                st.markdown(q)
            with st.chat_message("assistant", avatar="🛡️"):
                reply = st.write_stream(A.answer(client, ss.chat, q, context))
                show_llm(reply, context)
            ss.chat += [{"role": "user", "content": q}, {"role": "assistant", "content": reply}]
        if ss.chat and st.button("Clear chat"):
            ss.chat = []
            st.rerun()

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
        "  └─ TreeSHAP                     : per-alert explanation, one-hot folded back\n"
        f"  └─ LLM analyst ({client.label if client else 'offline'})\n"
        "       SHAP + numbers → plain English, grounding-checked",
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

    st.markdown("#### 🤖 The LLM layer: explains, never decides")
    l1, l2 = st.columns(2)
    l1.markdown(
        "- **XGBoost decides, the LLM explains.** The verdict, probability and category "
        "come from the trained models. The LLM is only handed those numbers and the SHAP "
        "values, and is told to use nothing else.\n"
        "- **Why not let the LLM detect attacks?** It is slower, costs money per flow, "
        "can't be audited like a tree model, and has no measured accuracy on this data. "
        "A 98%-accurate tree model plus an LLM translator gets the best of both.\n"
        "- **It never sees the answer key.** Ground-truth labels are removed from what it reads.")
    l2.markdown(
        "- **Hallucination guard.** Every field the LLM names is checked against the "
        "evidence it was given; an invented reason is flagged in yellow, not hidden.\n"
        "- **Calibrated language.** It must call low-confidence alerts uncertain and "
        "recommend human investigation, never automatic blocking.\n"
        "- **Privacy.** Only flow metadata is sent to the LLM: no IPs, no packet "
        "contents. Without an API key the dashboard falls back to a template, so it "
        "never depends on a third-party service to detect anything.")

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
