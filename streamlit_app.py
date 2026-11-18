"""SENTINEL-15 — dashboard for the two-stage UNSW-NB15 intrusion detector, with an
LLM analyst that explains its decisions in plain English.

Visual language: a Wireshark-style flow log (pastel rows by protocol, dark rows
with red text for alerts, blue for the selected row) on a light console, accented
with the four T568B Ethernet pair colours.
Run: streamlit run streamlit_app.py"""
from __future__ import annotations

import html
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analyst as A
from detector import MAX_ROWS, SAMPLE_CSV, Detector

st.set_page_config(page_title="SENTINEL-15", page_icon="🛡️", layout="wide")

# ── tokens ────────────────────────────────────────────────────────────────
INK, PAPER, RULE = "#15191E", "#FFFFFF", "#D3DAE0"
BLUE, GREEN, ORANGE, BROWN = "#2462A8", "#2F8F46", "#E07020", "#7B4B2A"
BAD_BG, BAD_FG, RED = "#12272E", "#F78787", "#C8102E"
FONT = "Archivo, system-ui, sans-serif"
PLOTLY_CFG = {"displayModeBar": False}
CAT_COLORS = {"Normal": GREEN, "Generic": RED, "Exploits": ORANGE, "Fuzzers": "#B8860B",
              "DoS": BROWN, "Reconnaissance": BLUE, "Analysis": "#6A5ACD",
              "Backdoor": "#A0306E", "Shellcode": "#2A9D8F", "Worms": "#5C6770"}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,100..900&display=swap');

.block-container {{padding-top: 3.2rem; max-width: 1320px;}}
.stApp :is(h1, h2, h3, h4, h5, p, li, td, th, label) {{font-family: {FONT};}}
.stApp {{font-variant-numeric: tabular-nums;}}

/* masthead: the product name set wide, one plain sentence beside it */
.mast {{display: flex; align-items: baseline; gap: 1.5rem; flex-wrap: wrap;
        border-bottom: 2px solid {INK}; padding-bottom: .6rem; margin-bottom: 1rem;}}
.mast h1 {{font: 900 2.4rem/1.15 {FONT}; font-stretch: 125%; letter-spacing: -.01em;
           margin: 0; padding: .1em 0 0; color: {INK}; overflow: visible;}}
.mast p {{margin: 0; color: #4A5560; font-size: 1rem; max-width: 60ch;}}

/* verdict strip: the selected flow, stated as plainly and as large as possible */
.verdict {{padding: 1.1rem 1.4rem 1rem; margin: .4rem 0 1rem; border: 1px solid {RULE};
           background: {PAPER}; color: {INK};}}
.verdict .word {{font: 850 clamp(2.6rem, 7vw, 5.2rem)/.95 {FONT}; font-stretch: 125%;
                 letter-spacing: -.02em; margin: 0;}}
.verdict .line {{font-size: 1.15rem; margin: .55rem 0 .15rem; font-weight: 600;}}
.verdict .flow {{font-size: .95rem; margin: 0; opacity: .8; font-stretch: 87%;}}
.verdict.safe .word {{color: {GREEN};}}
.verdict.bad {{background: {BAD_BG}; color: {BAD_FG}; border-color: {BAD_BG};}}
.verdict.bad .word {{color: {BAD_FG};}}
.verdict.flash {{animation: land .7s ease-out 1;}}
@keyframes land {{from {{background: {RED}; color: #fff;}} to {{background: {BAD_BG};}}}}
@media (prefers-reduced-motion: reduce) {{.verdict.flash {{animation: none;}}}}

/* the flow log, coloured like Wireshark's default rules */
.log {{width: 100%; border-collapse: collapse; font: 500 .86rem/1.25 {FONT};
       font-stretch: 87%; background: {PAPER}; border: 1px solid {RULE};}}
.log th {{text-align: left; font-weight: 700; padding: .35rem .5rem;
          border-bottom: 1px solid {INK}; background: {PAPER}; color: {INK};}}
.log td {{padding: .22rem .5rem; white-space: nowrap; color: {INK};}}
.log td.n {{text-align: right;}}
.log tr.tcp td {{background: #E7E6FF;}}
.log tr.udp td {{background: #DAEEFF;}}
.log tr.oth td {{background: #FAF0D7;}}
.log tr.bad td {{background: {BAD_BG}; color: {BAD_FG};}}
.log tr.sel td {{background: {BLUE}; color: #fff;}}
.logcap {{font-size: .9rem; color: #4A5560; margin: .35rem 0 0;}}

.replay {{margin: 0 0 .9rem; padding: .55rem .9rem; max-width: 80ch; font-size: .95rem;
           background: #FAF0D7; border: 1px solid #E6D3A3; color: {INK};}}
.note {{background: {PAPER}; border: 1px solid {RULE}; padding: .9rem 1.1rem;}}
.err {{color: {RED}; font-weight: 600;}}
div[data-testid="stMetricValue"] {{font-size: 1.4rem; font-weight: 700;}}
[role="tab"] p {{font-size: 1.05rem !important; font-weight: 650;}}
</style>""", unsafe_allow_html=True)


# ── cached resources ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models…")
def load_detector() -> Detector:
    return Detector()


@st.cache_data
def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_CSV)


@st.cache_resource(show_spinner=False)
def llm_client(provider: str, key: str, base_url: str | None):
    return A.make_client(provider, key, base_url)


@st.cache_data(ttl=3600, show_spinner=False)
def llm_models(provider: str, key: str, base_url: str | None) -> list[str]:
    return A.list_models(provider, llm_client(provider, key, base_url))


# ── helpers ───────────────────────────────────────────────────────────────
def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def styled(fig: go.Figure, h: int = 300, title: str | None = None) -> go.Figure:
    fig.update_layout(template="plotly_white", height=h, paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor=PAPER, margin=dict(l=10, r=10, t=40 if title else 12, b=10),
                      font=dict(family=FONT, size=13, color=INK),
                      title=dict(text=title, font=dict(size=15)) if title else None)
    fig.update_xaxes(gridcolor="#E4E9ED", zerolinecolor=RULE)
    fig.update_yaxes(gridcolor="#E4E9ED", zerolinecolor=RULE)
    return fig


def shap_chart(exp: pd.DataFrame) -> go.Figure:
    e = exp.iloc[::-1]
    labels = [f"{f} = {v:.4g}" if isinstance(v, (int, float, np.number)) else f"{f} = {v}"
              for f, v in zip(e.feature, e.value)]
    fig = go.Figure(go.Bar(x=e.shap, y=labels, orientation="h",
                           marker_color=[RED if s > 0 else GREEN for s in e.shap],
                           hovertemplate="%{y}<br>push %{x:+.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="← toward normal      toward attack →")
    return styled(fig, 250, "What drove this verdict")


def secret(name: str):
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:  # no secrets file configured
        return os.environ.get(name)


def show_llm(text: str, evidence: str) -> None:
    """Render an LLM answer's hallucination check and the evidence it saw."""
    ok, bad = A.grounding_check(text, evidence)
    if ok:
        st.caption("✅ Grounding check passed: every field the LLM cited is in the "
                   "detector's evidence.")
    else:
        st.warning("Grounding check failed: the LLM cited "
                   + ", ".join(f"`{b}`" for b in bad)
                   + ", which the detector never gave it. Treat that part as unreliable.")
    with st.expander("Show the evidence the LLM was given"):
        st.code(evidence, language="text")


def llm_sidebar() -> A.LLM | None:
    """Provider + key + model picker. Server keys come from Streamlit secrets or
    env vars (ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY, ...);
    a visitor's typed key is used only for their session."""
    st.markdown("#### LLM analyst")
    env_pid, env_key = A.key_from_env(secret)
    names = {pid: p.name + (", free tier" if p.free else "") for pid, p in A.PROVIDERS.items()}
    choice = st.selectbox("Provider", ["auto", *A.PROVIDERS],
                          format_func=lambda k: "Detect from key" if k == "auto" else names[k])
    typed = st.text_input("API key", type="password",
                          placeholder="using the server's key" if env_key else "paste any provider's key",
                          help="Used only for this browser session, never stored. "
                               "Gemini and Groq give free keys with no card.").strip()
    if typed:
        pid = A.detect(typed) if choice == "auto" else choice
        key = typed
        if pid is None:
            st.warning("This key doesn't match a known provider. Choose the provider above.")
            return None
    elif choice not in ("auto", env_pid):
        pid = choice
        key = next((secret(v) for v in A.PROVIDERS[pid].env if secret(v)), None)
    else:
        pid, key = env_pid, env_key
    if not key:
        st.info("No API key, so explanations use a built-in template. Paste a key to "
                "turn on the LLM analyst.")
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
    st.success(f"Connected to {llm.label}")
    return llm


def verdict_html(idx: int, r: pd.Series, pred: pd.Series, flash: bool) -> str:
    flow = (f"{r.proto}/{r.service}, {r.state} state, {r.dur:.2f} s, "
            f"{human(r.sbytes)} out in {int(r.spkts)} packets, "
            f"{human(r.dbytes)} back in {int(r.dpkts)}, TTL {int(r.sttl)}. "
            f"Dataset label: {r.get('attack_cat', 'unknown')}.")
    if pred.verdict == "INTRUSION":
        unsure = pred.category_conf < .5
        word = pred.category + ("?" if unsure else "")
        line = (f"Flow {idx} looks like an attack ({pred.p_attack:.0%} likely). "
                + (f"The type is a guess: only {pred.category_conf:.0%} sure it is {pred.category}."
                   if unsure else f"{pred.category_conf:.0%} sure it is {pred.category}."))
        cls = "verdict bad" + (" flash" if flash else "")
    else:
        word, cls = "Normal", "verdict safe"
        line = f"Flow {idx} looks like normal traffic ({pred.p_attack:.0%} chance of attack)."
    return (f'<section class="{cls}" aria-live="polite"><p class="word">{html.escape(word)}</p>'
            f'<p class="line">{html.escape(line)}</p><p class="flow">{html.escape(flow)}</p></section>')


def log_html(history: list[dict], current: int, threshold: float) -> str:
    rows = []
    for h in reversed(history[-14:]):
        alert = h["p"] >= threshold
        cls = "sel" if h["n"] == current else "bad" if alert else \
              {"tcp": "tcp", "udp": "udp"}.get(h["proto"], "oth")
        rows.append(
            f'<tr class="{cls}"><td class="n">{h["n"]}</td><td>{html.escape(h["proto"])}</td>'
            f'<td>{html.escape(h["service"])}</td><td>{html.escape(h["state"])}</td>'
            f'<td class="n">{h["dur"]:.2f}</td><td class="n">{human(h["out"])}</td>'
            f'<td class="n">{human(h["back"])}</td><td class="n">{h["p"]:.0%}</td>'
            f'<td>{html.escape(h["cat"] if alert else "Normal")}</td></tr>')
    return ('<table class="log"><thead><tr><th>No.</th><th>Proto</th><th>Service</th>'
            '<th>State</th><th>Secs</th><th>Out</th><th>Back</th><th>Attack</th>'
            '<th>Verdict</th></tr></thead><tbody>' + "".join(rows) + "</tbody></table>")


# ── page ──────────────────────────────────────────────────────────────────
det = load_detector()
M = det.metrics
ss = st.session_state
ss.setdefault("explained", {})
ss.setdefault("reports", {})
ss.setdefault("chat", [])

st.markdown('<header class="mast"><h1>SENTINEL-15</h1><p>Watches network traffic, '
            'flags intrusions, and explains every decision in plain English.</p></header>',
            unsafe_allow_html=True)

with st.sidebar:
    st.markdown("#### Detector")
    threshold = st.slider("Alert threshold", 0.05, 0.95, 0.50, 0.05,
                          help="How likely a flow must be to be an attack before it raises an "
                               "alert. Lower catches more attacks but raises more false alarms.")
    b = M["binary"]
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{b['accuracy']:.1%}")
    c2.metric("Attacks caught", f"{b['recall']:.1%}")
    st.caption(f"Tested on {M['n_test']:,} flows it never saw while training.")
    client = llm_sidebar()

tab1, tab2, tab4, tab3 = st.tabs(["Live traffic", "Scan a log file", "Ask the analyst",
                                  "How it works"])

# ── TAB 1: live traffic ───────────────────────────────────────────────────
with tab1:
    flows, _ = Detector.prepare(load_sample())
    ss.setdefault("idx", 0)
    ss.setdefault("playing", False)
    ss.setdefault("history", [])
    ss.setdefault("flashed", -1)

    # Callbacks run before the page redraws, so the Play/Pause label is never stale.
    def toggle() -> None:
        ss.playing = not ss.playing

    def step() -> None:
        ss.idx, ss.playing = (ss.idx + 1) % len(flows), False

    def reset() -> None:
        ss.idx, ss.history, ss.playing = 0, [], False

    st.markdown('<p class="replay"><strong>This is a replay, not your network.</strong> '
                'The flows are 500 recorded connections from the UNSW-NB15 test set, which '
                'the model never saw in training, played back as if they were arriving '
                'live. Nothing is captured from your computer or connection.</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    c1.button("⏸ Pause" if ss.playing else "▶ Play", width="stretch", type="primary",
              on_click=toggle)
    c2.button("⏭ Step", width="stretch", on_click=step)
    c3.button("↺ Reset", width="stretch", on_click=reset)
    speed = c4.select_slider("Flows per second", [0.5, 1, 2, 4, 8], value=2)

    @st.fragment(run_every=(1 / speed) if ss.playing else None)
    def live() -> None:
        if ss.playing:
            ss.idx = (ss.idx + 1) % len(flows)
        row = flows.iloc[[ss.idx]]
        r = row.iloc[0]
        pred = det.predict(row, threshold).iloc[0]
        if not ss.history or ss.history[-1]["n"] != ss.idx:
            ss.history.append({"n": ss.idx, "p": float(pred.p_attack), "cat": pred.category,
                               "proto": str(r.proto), "service": str(r.service),
                               "state": str(r.state), "dur": float(r.dur),
                               "out": float(r.sbytes), "back": float(r.dbytes)})
            ss.history = ss.history[-80:]

        flash = pred.verdict == "INTRUSION" and ss.flashed != ss.idx
        ss.flashed = ss.idx
        st.markdown(verdict_html(ss.idx, r, pred, flash), unsafe_allow_html=True)

        exp = det.explain(row)
        left, right = st.columns([11, 10], gap="large")
        with left:
            st.markdown(log_html(ss.history, ss.idx, threshold), unsafe_allow_html=True)
            h = pd.DataFrame(ss.history)
            n_alert = int((h.p >= threshold).sum())
            st.markdown(f'<p class="logcap">{len(h)} flows inspected, {n_alert} raised an '
                        f'alert ({n_alert / len(h):.0%}). Newest first; blue is the flow '
                        f'shown above.</p>', unsafe_allow_html=True)
        with right:
            st.plotly_chart(shap_chart(exp), width="stretch", config=PLOTLY_CFG,
                            key=f"shap{ss.idx}")
            ev = A.flow_evidence(r, pred, exp, threshold)
            ss.flow_ev = f"current live flow #{ss.idx}:\n{ev}"
            key = (ss.idx, threshold)
            if key in ss.explained:
                st.markdown(ss.explained[key])
                show_llm(ss.explained[key], ev)
            elif client and not ss.playing and st.button("Explain this verdict",
                                                         key=f"explain{ss.idx}", type="primary"):
                ss.explained[key] = st.write_stream(A.explain_flow(client, ev))
                show_llm(ss.explained[key], ev)
            else:
                st.markdown(A.offline_flow(pred, exp, hint=not client))
                if client and ss.playing:
                    st.caption("Pause the stream to have the LLM explain this flow.")

    live()

# ── TAB 2: scan a log file ────────────────────────────────────────────────
with tab2:
    st.markdown("Upload a flow log in UNSW-NB15 format, or scan the 500-flow sample.")
    u1, u2 = st.columns([3, 1])
    up = u1.file_uploader("Flow log (.csv)", type="csv", label_visibility="collapsed")
    if u2.button("Scan the sample log", width="stretch", type="primary"):
        st.session_state.batch = ("sample_test_flows.csv", load_sample())
    if up is not None:
        try:
            st.session_state.batch = (up.name, pd.read_csv(up, encoding="utf-8-sig",
                                                           low_memory=False, nrows=MAX_ROWS))
        except Exception as e:  # malformed upload must never crash the app
            st.error(f"This file couldn't be read as a CSV ({e}). Export it as "
                     f"comma-separated text and upload it again.")

    if "batch" in st.session_state:
        name, raw = st.session_state.batch
        df, missing = Detector.prepare(raw)
        if missing:
            st.error(f"`{name}` is missing {len(missing)} UNSW-NB15 columns: "
                     + ", ".join(f"`{c}`" for c in missing)
                     + ". Add them, or scan the sample log to see the expected format.")
        else:
            with st.spinner(f"Scanning {len(df):,} flows…"):
                res = det.predict(df, threshold)
            flagged = res.verdict == "INTRUSION"
            st.markdown(f"#### {name}")
            k = st.columns(4)
            k[0].metric("Flows", f"{len(df):,}")
            k[1].metric("Intrusions", f"{flagged.sum():,}")
            k[2].metric("Share flagged", f"{flagged.mean():.1%}")
            if "label" in df.columns:
                k[3].metric("Agrees with labels",
                            f"{((res.p_attack >= threshold) == df.label).mean():.1%}")
            else:
                k[3].metric("Most common attack",
                            res.category[flagged].mode().iat[0] if flagged.any() else "None")

            a, b2 = st.columns(2)
            counts = res.category.value_counts().reset_index()
            counts.columns = ["category", "flows"]
            fig = px.bar(counts.sort_values("flows"), x="flows", y="category", orientation="h",
                         color="category", color_discrete_map=CAT_COLORS)
            fig.update_layout(showlegend=False, yaxis_title=None)
            a.plotly_chart(styled(fig, 340, "What the scan found"), width="stretch",
                           config=PLOTLY_CFG)
            fig = px.histogram(res, x="p_attack", nbins=40, color_discrete_sequence=[BLUE])
            fig.add_vline(x=threshold, line_dash="dot", line_color=ORANGE, line_width=2,
                          annotation_text="alert threshold")
            fig.update_layout(xaxis_title="chance of attack", yaxis_title="flows")
            b2.plotly_chart(styled(fig, 340, "How sure the model was"), width="stretch",
                            config=PLOTLY_CFG)

            report = pd.concat([raw.loc[df.index], res.round(4)], axis=1)[flagged]
            report = report.sort_values("p_attack", ascending=False)
            st.markdown(f"#### Flagged flows ({len(report):,})")
            st.dataframe(report, width="stretch", height=300)
            st.download_button("Download flagged flows (CSV)",
                               report.to_csv(index=False).encode(),
                               file_name=f"flagged_{name}", mime="text/csv")

            # ── LLM incident report ──
            top = report.head(30)
            reasons = pd.Series(
                [f for i in top.index for f in det.explain(df.loc[[i]]).feature],
                dtype=object).value_counts().head(8)
            bev = A.batch_evidence(name, df, res, threshold, reasons)
            ss.batch_ev = bev
            st.markdown("#### Incident report")
            rkey = (name, len(df), threshold)
            if rkey in ss.reports:
                st.markdown(ss.reports[rkey])
                show_llm(ss.reports[rkey], bev)
            elif not client:
                st.info("Paste an LLM API key in the sidebar and the analyst will write an "
                        "incident report for this scan.")
            elif st.button("Write incident report", type="primary"):
                ss.reports[rkey] = st.write_stream(A.incident_report(client, bev))
                show_llm(ss.reports[rkey], bev)
            if rkey in ss.reports:
                st.download_button("Download incident report (Markdown)",
                                   ss.reports[rkey].encode(),
                                   file_name=f"incident_{name}.md", mime="text/markdown")

# ── TAB 4: ask the analyst ────────────────────────────────────────────────
with tab4:
    st.markdown("Ask about the flow on the live tab, your last scan, or how the model "
                "works. The analyst answers only from the detector's own numbers.")
    if not client:
        st.info("The chat needs an LLM API key. Paste one in the sidebar; the rest of "
                "the dashboard works without it.")
    else:
        suggestions = ["Why was the current flow flagged, or not?",
                       "Summarise my last scan for a manager.",
                       "Which attacks does this model struggle with?",
                       "What happens if I lower the threshold to 0.2?"]
        cols = st.columns(len(suggestions))
        picked = next((q for c, q in zip(cols, suggestions)
                       if c.button(q, width="stretch")), None)
        for msg in ss.chat:
            with st.chat_message(msg["role"], avatar="🛡️" if msg["role"] == "assistant" else None):
                st.markdown(msg["content"])
        q = st.chat_input("Ask the analyst a question") or picked
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
        if ss.chat and st.button("Clear the conversation"):
            ss.chat = []
            st.rerun()

# ── TAB 3: how it works ───────────────────────────────────────────────────
with tab3:
    b, mc, audit = M["binary"], M["multiclass"], M["ttl_audit"]
    st.markdown("#### How well it works")
    st.caption(f"Data: {M['source']}. None of these test flows were seen during training.")
    k = st.columns(5)
    k[0].metric("Accuracy", f"{b['accuracy']:.2%}")
    k[1].metric("Precision", f"{b['precision']:.2%}")
    k[2].metric("Attacks caught", f"{b['recall']:.2%}")
    k[3].metric("F1", f"{b['f1']:.3f}")
    k[4].metric("Attack-type macro-F1", f"{mc['macro_f1']:.3f}")

    a, b2 = st.columns([3, 2])
    labels = mc["labels"]
    cm = np.array(mc["confusion"], dtype=float)
    norm = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig = go.Figure(go.Heatmap(z=norm, x=labels, y=labels,
                               colorscale=[[0, PAPER], [1, BLUE]], showscale=False,
                               text=cm.astype(int), texttemplate="%{text}",
                               hovertemplate="true %{y}, predicted %{x}: %{text}<extra></extra>"))
    fig.update_layout(yaxis=dict(autorange="reversed", title="true type"),
                      xaxis_title="predicted type")
    a.plotly_chart(styled(fig, 470, "Where it gets attack types right and wrong"),
                   width="stretch", config=PLOTLY_CFG)
    f1 = pd.Series(mc["per_class_f1"]).sort_values()
    fig = go.Figure(go.Bar(x=f1.values, y=f1.index, orientation="h",
                           marker_color=[RED if v < .3 else ORANGE if v < .7 else GREEN for v in f1]))
    fig.update_layout(xaxis=dict(range=[0, 1], title="F1 score"))
    b2.plotly_chart(styled(fig, 470, "Score per attack type"), width="stretch", config=PLOTLY_CFG)
    st.info("**Analysis** and **Backdoor** are the weak spots: in UNSW-NB15 their flows "
            "have near-identical statistics, so flow metadata alone cannot reliably tell them "
            "apart. Both are still caught as *attacks*; the mistake is in naming them.")

    st.markdown("#### The pipeline")
    st.code(
        "raw flow (45 fields)\n"
        "  └─ drop `id` (a row number, would leak)      BOM-safe CSV reading\n"
        "  └─ ColumnTransformer\n"
        "       ├─ proto / service / state → OneHotEncoder(handle_unknown='ignore')\n"
        "       └─ 39 numeric flow statistics → passthrough\n"
        "  └─ Stage 1  XGBoost binary      : normal or attack   (threshold adjustable)\n"
        "  └─ Stage 2  XGBoost 9-class     : which attack       (runs only on alerts)\n"
        "  └─ TreeSHAP                     : why, per flow, one-hot folded back\n"
        f"  └─ LLM analyst: {client.label if client else 'offline template'}\n"
        "       SHAP + numbers → plain English, grounding-checked",
        language="text")

    st.markdown("#### The LLM explains; it never decides")
    l1, l2 = st.columns(2)
    l1.markdown(
        "- **XGBoost decides, the LLM explains.** The verdict, probability and type come "
        "from the trained models. The LLM is handed those numbers and the SHAP values, and "
        "told to use nothing else.\n"
        "- **Why not let the LLM detect attacks?** It is slower, costs money per flow, "
        "can't be audited like a tree model, and has no measured accuracy on this data. A "
        "98%-accurate tree model with an LLM translator gets the best of both.\n"
        "- **It never sees the answer key.** Ground-truth labels are removed from what it reads.")
    l2.markdown(
        "- **Hallucination guard.** Every field the LLM names is checked against the "
        "evidence it was given; an invented reason is flagged, not hidden.\n"
        "- **Calibrated language.** It must call low-confidence alerts uncertain and "
        "recommend human investigation, never automatic blocking.\n"
        "- **Privacy.** Only flow metadata goes to the LLM: no IPs, no packet contents. "
        "Without an API key the dashboard uses a template, so detection never depends on "
        "a third-party service.")

    st.markdown("#### Does it just memorise TTL values?")
    imp = pd.Series(audit["top_gain_features"])
    wo = audit["without_ttl"]
    c1, c2 = st.columns([3, 2])
    fig = go.Figure(go.Bar(x=imp.values[::-1], y=imp.index[::-1], orientation="h",
                           marker_color=[ORANGE if i in audit["dropped"] else BLUE
                                         for i in imp.index[::-1]]))
    fig.update_layout(xaxis_title="share of the model's decision weight")
    c1.plotly_chart(styled(fig, 340, "Most important fields (TTL fields in orange)"),
                    width="stretch", config=PLOTLY_CFG)
    ttl_share = sum(v for k2, v in imp.items() if k2 in audit["dropped"])
    c2.markdown(
        f"In UNSW-NB15, the testbed's attack machines used different **TTL values** than "
        f"normal hosts, so TTL could be a shortcut: true in this lab, meaningless on a real "
        f"network.\n\nThe TTL fields carry **{ttl_share:.0%}** of the model's decision "
        f"weight, so we retrained without them:\n\n"
        f"| | with TTL | without |\n|---|---|---|\n"
        f"| accuracy | {b['accuracy']:.2%} | {wo['accuracy']:.2%} |\n"
        f"| F1 | {b['f1']:.3f} | {wo['f1']:.3f} |\n\n"
        f"Performance barely moves: other fields carry the same signal. The model doesn't "
        f"*depend* on the shortcut, but it *prefers* it when it's there.")

    st.markdown("#### Missed attacks versus false alarms")
    curve = pd.DataFrame(M["threshold_curve"])
    fig = go.Figure()
    fig.add_scatter(x=curve.threshold, y=curve.false_alarms, name="false alarms",
                    line=dict(color=ORANGE, width=3))
    fig.add_scatter(x=curve.threshold, y=curve.missed_attacks, name="missed attacks",
                    line=dict(color=RED, width=3))
    fig.add_vline(x=threshold, line_dash="dot", line_color=BLUE, line_width=2,
                  annotation_text=f"your threshold, {threshold:.2f}")
    fig.update_layout(xaxis_title="alert threshold", yaxis_title="flows",
                      legend=dict(orientation="h", y=1.12))
    st.plotly_chart(styled(fig, 330, f"Errors on {M['n_test']:,} test flows"),
                    width="stretch", config=PLOTLY_CFG)

    e1, e2 = st.columns(2)
    e1.markdown(
        "**Errors have different costs**\n\n"
        "- A **missed attack** can mean a breach.\n"
        "- A **false alarm** wastes an analyst's time, and real networks see millions of "
        "flows a day: a 1% false-alarm rate buries the real alerts (*alert fatigue*).\n"
        "- So the threshold is a human policy choice, not a model fact. That's why it's "
        "a slider.\n"
        "- Alerts are for **a person to investigate**, never for automatic blocking or "
        "punishing a user.")
    e2.markdown(
        "**Privacy and limits**\n\n"
        "- The model reads **flow metadata only**: sizes, timings, counts, protocol. It never "
        "sees packet *contents*, so it works on encrypted traffic and reads no messages.\n"
        "- No IP addresses are used, so it can't learn to distrust a person or a country.\n"
        "- **Limits:** UNSW-NB15 was generated in a 2015 lab. Attacks change, and accuracy "
        "on a real network would be lower than shown here. Retrain and re-audit before any "
        "real use.")
