"""LLM "SOC analyst" layered on top of the detector.

The XGBoost models make every decision. The LLM (Claude, GPT, Gemini, Llama via
Groq, ... — any provider in PROVIDERS) only *explains* them: it
turns a flow's prediction + SHAP values into plain English, writes incident reports
for batch scans, and answers questions. It is handed the exact numbers and told to
use nothing else; `grounding_check` then verifies every field it names really is in
the evidence, so a hallucinated reason is caught and shown, not hidden.

Without an API key everything still works: `offline_*` builds the same explanation
from a template, so the demo never breaks."""
from __future__ import annotations

import os
import re
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd

from train import FEATURES

# Plain-English names for the UNSW-NB15 fields the analyst is most likely to cite.
GLOSSARY = {
    "proto": "transport protocol", "service": "application service (http, dns, ...)",
    "state": "connection state (FIN, CON, INT, ...)", "dur": "flow duration (s)",
    "sbytes": "bytes sent by source", "dbytes": "bytes sent by destination",
    "spkts": "packets from source", "dpkts": "packets from destination",
    "sttl": "source time-to-live", "dttl": "destination time-to-live",
    "rate": "packets per second", "sload": "source bits per second",
    "dload": "destination bits per second", "smean": "mean source packet size",
    "dmean": "mean destination packet size", "ct_state_ttl": "connections with same state+TTL",
    "ct_srv_src": "recent connections, same service + source",
    "ct_dst_src_ltm": "recent connections, same source + destination",
    "ct_src_dport_ltm": "recent connections, same source + destination port",
    "ct_dst_sport_ltm": "recent connections, same destination + source port",
    "ct_srv_dst": "recent connections, same service + destination",
    "synack": "SYN→SYN-ACK time", "ackdat": "SYN-ACK→ACK time", "tcprtt": "TCP round-trip time",
    "sinpkt": "source inter-packet time", "dinpkt": "destination inter-packet time",
    "sjit": "source jitter", "djit": "destination jitter",
    "is_sm_ips_ports": "same source/destination IP and port",
}

SYSTEM = """You are the explanation layer of SENTINEL-15, a network intrusion detector \
used by a security operations center (SOC). Two XGBoost models already made the \
decision; you never re-decide it. You translate their output for a human analyst.

Rules:
- Use ONLY the numbers in the evidence block. Never invent a field, value, IP address, \
host name, CVE or attacker. If the evidence does not say something, say it is unknown.
- When you cite a field, write its exact name in backticks (e.g. `sbytes`), then what it means.
- SHAP values are in log-odds: positive pushes toward INTRUSION, negative toward SAFE.
- The model sees flow metadata only (sizes, timings, counts, protocol), never packet \
contents. Do not claim to know payloads or intent.
- Be calibrated. If attack probability or category confidence is below 70%, say the call \
is uncertain. Mention that UNSW-NB15 is a 2015 lab dataset when it matters.
- Recommend investigation steps for a human. Never recommend automatic blocking or \
punishing a user based on this alert alone.
- Write for a smart 15-year-old: short sentences, no jargon without a gloss. \
Markdown, no headings above ###."""


# ── providers ────────────────────────────────────────────────────────────────
# Anthropic uses its own SDK; every other provider exposes an OpenAI-compatible
# endpoint, so one OpenAI SDK client with a different base_url covers them all.
# `default` is only a starting suggestion: the sidebar lists the provider's live
# models, so new model names work without a code change.
@dataclass(frozen=True)
class Provider:
    name: str
    env: tuple[str, ...]          # env vars / Streamlit secrets holding its key
    prefix: tuple[str, ...]       # key prefixes for auto-detection
    base_url: str | None          # None = SDK default
    default: str
    free: bool = False            # has a free tier without a card


PROVIDERS = {
    "anthropic": Provider("Anthropic (Claude)", ("ANTHROPIC_API_KEY",), ("sk-ant-",), None,
                          "claude-opus-5-5"),
    "openai": Provider("OpenAI (GPT)", ("OPENAI_API_KEY",), ("sk-proj-", "sk-svcacct-"), None,
                       "gpt-4o-mini"),
    "gemini": Provider("Google Gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY"), ("AIza", "AQ."),
                       "https://generativelanguage.googleapis.com/v1beta/openai/",
                       "gemini-3.5-flash-lite", free=True),
    "groq": Provider("Groq (Llama, Qwen…)", ("GROQ_API_KEY",), ("gsk_",),
                     "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", free=True),
    "openrouter": Provider("OpenRouter (any model)", ("OPENROUTER_API_KEY",), ("sk-or-",),
                           "https://openrouter.ai/api/v1", "openrouter/auto", free=True),
    "mistral": Provider("Mistral", ("MISTRAL_API_KEY",), (), "https://api.mistral.ai/v1",
                        "mistral-small-latest", free=True),
    "deepseek": Provider("DeepSeek", ("DEEPSEEK_API_KEY",), (), "https://api.deepseek.com",
                         "deepseek-chat"),
    "xai": Provider("xAI (Grok)", ("XAI_API_KEY",), ("xai-",), "https://api.x.ai/v1", "grok-3-mini"),
    "custom": Provider("Custom (OpenAI-compatible)", ("LLM_API_KEY",), (), None, ""),
}


@dataclass
class LLM:
    provider: str
    model: str
    client: object

    @property
    def label(self) -> str:
        return f"{PROVIDERS[self.provider].name} ({self.model})"


def detect(key: str) -> str | None:
    """Guess the provider from a key's prefix. Plain `sk-` keys are treated as OpenAI."""
    key = key.strip()
    for pid, p in PROVIDERS.items():
        if any(key.startswith(x) for x in p.prefix):
            return pid
    return "openai" if key.startswith("sk-") else None


def key_from_env(get=os.environ.get) -> tuple[str, str] | tuple[None, None]:
    """First (provider, key) configured in env vars / secrets, in PROVIDERS order."""
    for pid, p in PROVIDERS.items():
        for var in p.env:
            if v := get(var):
                return pid, v
    return None, None


def make_client(provider: str, key: str, base_url: str | None = None):
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=key, max_retries=1, timeout=45)
    import openai
    return openai.OpenAI(api_key=key, base_url=base_url or PROVIDERS[provider].base_url,
                         max_retries=1, timeout=45)


def list_models(provider: str, client) -> list[str]:
    """Live model ids from the provider; empty list if listing isn't supported."""
    try:
        ids = [m.id for m in client.with_options(timeout=10, max_retries=0).models.list()]
    except Exception:
        return []
    if provider == "gemini":
        ids = [i.removeprefix("models/") for i in ids]
    skip = ("embed", "tts", "whisper", "dall-e", "image", "audio", "realtime", "moderation",
            "transcribe", "search", "guard", "veo", "imagen", "aqa")
    return sorted(i for i in ids if not any(w in i.lower() for w in skip))


def _stream(llm: LLM, messages: list[dict], effort: str, max_tokens: int) -> Iterator[str]:
    """Stream text from whichever provider; errors become readable text, never exceptions."""
    if llm.provider == "anthropic":
        yield from _stream_anthropic(llm, messages, effort, max_tokens)
    else:
        yield from _stream_openai(llm, messages, max_tokens)


def _stream_anthropic(llm: LLM, messages, effort, max_tokens) -> Iterator[str]:
    import anthropic
    try:
        with llm.client.beta.messages.stream(
            model=llm.model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=messages,
            output_config={"effort": effort},
            # If a safety classifier declines (network-attack wording can trip the
            # cyber classifier even for defensive analysis), the API re-runs the
            # request on a fallback model inside the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        ) as stream:
            yield from stream.text_stream
            final = stream.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_The language model declined this request. The detector's verdict above is unaffected._"
        elif final.stop_reason == "max_tokens":
            yield " …"
    except anthropic.BadRequestError as e:
        # Older Claude models reject effort / fallbacks; retry once with the basics.
        if "effort" in str(e.message) or "fallback" in str(e.message):
            yield from _stream_anthropic_plain(llm, messages, max_tokens)
        else:
            yield f"⚠ Request rejected: {e.message}"
    except anthropic.AuthenticationError:
        yield "⚠ The API key was rejected. Check it in the sidebar or in Streamlit secrets."
    except anthropic.NotFoundError:
        yield f"⚠ Model `{llm.model}` was not found for this key. Pick another in the sidebar."
    except anthropic.PermissionDeniedError:
        yield "⚠ This API key is not allowed to use this model."
    except anthropic.RateLimitError:
        yield "⚠ Rate limited by the API. Wait a few seconds and try again."
    except anthropic.APIStatusError as e:
        yield f"⚠ API error {e.status_code}: {e.message}"
    except anthropic.APIConnectionError:
        yield "⚠ Could not reach the API (network error)."


def _stream_anthropic_plain(llm: LLM, messages, max_tokens) -> Iterator[str]:
    import anthropic
    try:
        with llm.client.messages.stream(model=llm.model, max_tokens=max_tokens,
                                        system=SYSTEM, messages=messages) as stream:
            yield from stream.text_stream
    except anthropic.APIError as e:
        yield f"⚠ API error: {getattr(e, 'message', e)}"


def _stream_openai(llm: LLM, messages, max_tokens) -> Iterator[str]:
    import openai
    try:
        stream = llm.client.chat.completions.create(
            model=llm.model, stream=True,
            messages=[{"role": "system", "content": SYSTEM}, *messages])
        got = False
        for chunk in stream:
            if chunk.choices and (t := chunk.choices[0].delta.content):
                got = True
                yield t
        if not got:
            yield "_The model returned no text. Try a different model in the sidebar._"
    except openai.AuthenticationError:
        yield "⚠ The API key was rejected. Check the key and the provider in the sidebar."
    except openai.NotFoundError:
        yield f"⚠ Model `{llm.model}` was not found for this key. Pick another in the sidebar."
    except openai.PermissionDeniedError:
        yield "⚠ This API key is not allowed to use this model."
    except openai.RateLimitError:
        yield "⚠ Rate limited (or out of free quota). Wait a bit, or switch provider/model."
    except openai.APIStatusError as e:
        if "api key" in str(e.body).lower() or "api_key" in str(e.body).lower():
            yield "⚠ The API key was rejected. Check the key and the provider in the sidebar."
        else:
            yield f"⚠ API error {e.status_code}: {e.body or e.message}"[:400]
    except openai.APIConnectionError:
        yield "⚠ Could not reach the provider (network error, or wrong base URL)."


# ── evidence builders (the only facts the LLM may use) ──────────────────────
def _fmt(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:.6g}"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    return str(v)


def flow_evidence(row: pd.Series, pred: pd.Series, exp: pd.DataFrame, threshold: float) -> str:
    """Everything known about one flow, as a compact text block. Ground-truth labels
    are deliberately left out: the analyst must not see the answer key."""
    fields = ["proto", "service", "state", "dur", "spkts", "dpkts", "sbytes", "dbytes",
              "rate", "sttl", "dttl", "sload", "dload", "smean", "dmean", "ct_srv_src",
              "ct_dst_src_ltm", "ct_state_ttl"]
    lines = [
        f"verdict: {pred.verdict}  (alert threshold {threshold:.2f})",
        f"attack_probability: {pred.p_attack:.3f}",
    ]
    if pred.verdict == "INTRUSION":
        lines.append(f"predicted_category: {pred.category}  (stage-2 confidence {pred.category_conf:.3f})")
    lines.append("top SHAP contributions (field = value -> log-odds push):")
    for f, s, v in zip(exp.feature, exp.shap, exp.value):
        lines.append(f"  {f} = {_fmt(v)} -> {s:+.3f} ({'toward INTRUSION' if s > 0 else 'toward SAFE'})"
                     + (f"  [{GLOSSARY[f]}]" if f in GLOSSARY else ""))
    lines.append("raw flow fields:")
    lines.append("  " + ", ".join(f"{f}={_fmt(row[f])}" for f in fields if f in row.index))
    return "\n".join(lines)


def batch_evidence(name: str, df: pd.DataFrame, res: pd.DataFrame, threshold: float,
                   reasons: pd.Series) -> str:
    """Summary of a whole scan. `reasons` counts how often each field appeared in
    the top-5 SHAP list of the highest-scoring flagged flows."""
    flagged = res.verdict == "INTRUSION"
    lines = [f"file: {name}", f"flows scanned: {len(df)}", f"alert threshold: {threshold:.2f}",
             f"flagged as intrusion: {int(flagged.sum())} ({flagged.mean():.1%})"]
    if flagged.any():
        cats = res.category[flagged].value_counts()
        lines.append("flagged by predicted category: "
                     + ", ".join(f"{c}={n}" for c, n in cats.items()))
        low = (res.category_conf[flagged] < .5).sum()
        lines.append(f"flagged flows with category confidence < 50%: {int(low)}")
        lines.append("p_attack of flagged flows: min {:.2f}, median {:.2f}, max {:.2f}".format(
            *res.p_attack[flagged].quantile([0, .5, 1])))
        near = ((res.p_attack >= threshold - .1) & (res.p_attack < threshold + .1)).sum()
        lines.append(f"borderline flows (within ±0.10 of threshold): {int(near)}")
        lines.append("protocol/service of flagged flows: " + ", ".join(
            f"{p}/{s}={n}" for (p, s), n in
            df.loc[flagged, ["proto", "service"]].value_counts().head(6).items()))
        lines.append("fields most often in the top-5 SHAP reasons of the top flagged flows: "
                     + ", ".join(f"{f} ({n}x)" for f, n in reasons.items()))
    return "\n".join(lines)


def model_evidence(m: dict) -> str:
    b, mc, a = m["binary"], m["multiclass"], m["ttl_audit"]
    f1 = ", ".join(f"{k}={v:.2f}" for k, v in sorted(mc["per_class_f1"].items(), key=lambda kv: kv[1]))
    return (f"training data: {m['source']}; {m['n_train']} train flows, {m['n_test']} held-out test flows\n"
            f"stage 1 (normal vs attack): accuracy {b['accuracy']:.4f}, precision {b['precision']:.4f}, "
            f"recall {b['recall']:.4f}, F1 {b['f1']:.4f}\n"
            f"stage 2 (10-class end-to-end): macro-F1 {mc['macro_f1']:.3f}; per-class F1: {f1}\n"
            f"TTL shortcut audit: retrained without {', '.join(a['dropped'])}: accuracy "
            f"{a['without_ttl']['accuracy']:.4f}, F1 {a['without_ttl']['f1']:.4f}\n"
            f"features: 42 flow-metadata fields (3 categorical one-hot, 39 numeric); no IPs, no payloads")


# ── the three analyst jobs ───────────────────────────────────────────────────
def explain_flow(llm: LLM, evidence: str) -> Iterator[str]:
    msg = (f"<evidence>\n{evidence}\n</evidence>\n\n"
           "Explain this verdict to the analyst on shift in at most 120 words: "
           "what the flow looks like, the 2–3 reasons that mattered most, how sure the "
           "model is, and one concrete thing a human should check next.")
    return _stream(llm, [{"role": "user", "content": msg}], "low", 2000)


def incident_report(llm: LLM, evidence: str) -> Iterator[str]:
    msg = (f"<evidence>\n{evidence}\n</evidence>\n\n"
           "Write a short incident report for this scan (at most 250 words) with these "
           "parts in bold: **Summary**, **What was found**, **Confidence and caveats**, "
           "**Recommended next steps** (numbered, for a human). Quote counts exactly.")
    return _stream(llm, [{"role": "user", "content": msg}], "medium", 4000)


def answer(llm: LLM, history: list[dict], question: str, context: str) -> Iterator[str]:
    """Chat turn. Context (model metrics + current flow + current scan) rides along
    with the newest question only, so earlier turns stay byte-identical."""
    msgs = [{"role": h["role"], "content": h["content"]} for h in history[-10:]]
    msgs.append({"role": "user", "content":
                 f"<evidence>\n{context}\n</evidence>\n\nAnalyst question: {question}\n\n"
                 "Answer in at most 150 words. If the evidence cannot answer it, say so."})
    return _stream(llm, msgs, "low", 3000)


# ── offline fallback (no API key) ───────────────────────────────────────────
def offline_flow(pred: pd.Series, exp: pd.DataFrame, hint: bool = True) -> str:
    up = exp[exp.shap > 0].head(3)
    down = exp[exp.shap < 0].head(2)
    sure = "confident" if max(pred.p_attack, 1 - pred.p_attack) >= .9 else "unsure"
    if pred.verdict == "INTRUSION":
        head = (f"**Flagged as {pred.category}** with attack probability "
                f"{pred.p_attack:.0%}; the model is {sure}.")
        if pred.category_conf < .5:
            head += f" The attack *type* is uncertain ({pred.category_conf:.0%})."
    else:
        head = f"**Looks normal** (attack probability {pred.p_attack:.0%}); the model is {sure}."
    def bullet(r):
        g = f" ({GLOSSARY[r.feature]})" if r.feature in GLOSSARY else ""
        return f"- `{r.feature}` = {_fmt(r.value)}{g}"
    parts = [head]
    if len(up):
        parts.append("Pushed toward intrusion:\n" + "\n".join(bullet(r) for r in up.itertuples()))
    if len(down):
        parts.append("Pushed toward safe:\n" + "\n".join(bullet(r) for r in down.itertuples()))
    if hint:
        parts.append("_Offline template. Add an LLM API key in the sidebar for the "
                     "LLM analyst's plain-English write-up._")
    return "\n\n".join(parts)


# ── hallucination guard ─────────────────────────────────────────────────────
_FIELD = re.compile(r"`([a-z_]+)`")


def grounding_check(text: str, evidence: str) -> tuple[bool, list[str]]:
    """Every dataset field the LLM cites in backticks must appear in its evidence.
    Returns (ok, unsupported_fields)."""
    cited = {f for f in _FIELD.findall(text) if f in FEATURES}
    bad = sorted(f for f in cited if not re.search(rf"\b{f}\b", evidence))
    return not bad, bad
