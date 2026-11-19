// LLM "SOC analyst" — port of analyst.py. The XGBoost models make every
// decision; the LLM only explains them from the exact numbers handed to it,
// and groundingCheck() flags any dataset field it cites that wasn't in that
// evidence. Without a key, offlineFlow() gives a template explanation.
import type { Flow, ModelData, Prediction, Reason } from "./model";

export const GLOSSARY: Record<string, string> = {
  proto: "transport protocol", service: "application service (http, dns, …)",
  state: "connection state (FIN, CON, INT, …)", dur: "flow duration (s)",
  sbytes: "bytes sent by source", dbytes: "bytes sent by destination",
  spkts: "packets from source", dpkts: "packets from destination",
  sttl: "source time-to-live", dttl: "destination time-to-live",
  rate: "packets per second", sload: "source bits per second",
  dload: "destination bits per second", smean: "mean source packet size",
  dmean: "mean destination packet size", sloss: "source packets retransmitted or dropped",
  dloss: "destination packets retransmitted or dropped",
  ct_state_ttl: "connections with the same state and TTL",
  ct_srv_src: "recent connections, same service and source",
  ct_dst_src_ltm: "recent connections, same source and destination",
  ct_src_dport_ltm: "recent connections, same source and destination port",
  ct_dst_sport_ltm: "recent connections, same destination and source port",
  ct_srv_dst: "recent connections, same service and destination",
  synack: "SYN to SYN-ACK time", ackdat: "SYN-ACK to ACK time", tcprtt: "TCP round-trip time",
  sinpkt: "source inter-packet time", dinpkt: "destination inter-packet time",
  sjit: "source jitter", djit: "destination jitter",
  is_sm_ips_ports: "same source/destination IP and port",
};

export const SYSTEM = `You are the explanation layer of SENTINEL-15, a network intrusion detector used by a security operations center (SOC). Two XGBoost models already made the decision; you never re-decide it. You translate their output for a human analyst.

Rules:
- Use ONLY the numbers in the evidence block. Never invent a field, value, IP address, host name, CVE or attacker. If the evidence does not say something, say it is unknown.
- When you cite a field, write its exact name in backticks (e.g. \`sbytes\`), then what it means.
- SHAP values are in log-odds: positive pushes toward INTRUSION, negative toward SAFE.
- The model sees flow metadata only (sizes, timings, counts, protocol), never packet contents. Do not claim to know payloads or intent.
- Be calibrated. If attack probability or category confidence is below 70%, say the call is uncertain. Mention that UNSW-NB15 is a 2015 lab dataset when it matters.
- Recommend investigation steps for a human. Never recommend automatic blocking or punishing a user based on this alert alone.
- Write for a smart 15-year-old: short sentences, no jargon without a gloss. Markdown, no headings above ###.`;

const fmt = (v: unknown): string =>
  typeof v === "number" ? (Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6)))) : String(v);

// ── evidence (the only facts the LLM may use) ───────────────────────────────
export function flowEvidence(row: Flow, pred: Prediction, reasons: Reason[], threshold: number): string {
  // Ground-truth labels are deliberately left out: the analyst must not see the answer key.
  const fields = ["proto", "service", "state", "dur", "spkts", "dpkts", "sbytes", "dbytes", "rate",
    "sttl", "dttl", "sload", "dload", "smean", "dmean", "ct_srv_src", "ct_dst_src_ltm", "ct_state_ttl"];
  const lines = [
    `verdict: ${pred.intrusion ? "INTRUSION" : "SAFE"}  (alert threshold ${threshold.toFixed(2)})`,
    `attack_probability: ${pred.pAttack.toFixed(3)}`,
  ];
  if (pred.intrusion && !Number.isNaN(pred.categoryConf))
    lines.push(`predicted_category: ${pred.category}  (stage-2 confidence ${pred.categoryConf.toFixed(3)})`);
  else if (pred.intrusion) lines.push("predicted_category: unknown (attack-type model still loading)");
  lines.push("top SHAP contributions (field = value -> log-odds push):");
  for (const r of reasons)
    lines.push(`  ${r.feature} = ${fmt(r.value)} -> ${r.shap >= 0 ? "+" : ""}${r.shap.toFixed(3)} ` +
      `(${r.shap > 0 ? "toward INTRUSION" : "toward SAFE"})` + (GLOSSARY[r.feature] ? `  [${GLOSSARY[r.feature]}]` : ""));
  lines.push("raw flow fields:");
  lines.push("  " + fields.filter((f) => f in row).map((f) => `${f}=${fmt(row[f])}`).join(", "));
  return lines.join("\n");
}

export interface ScanSummary {
  name: string; n: number; threshold: number; flagged: number;
  byCategory: [string, number][]; lowConf: number; pMin: number; pMed: number; pMax: number;
  borderline: number; protoService: [string, number][]; reasons: [string, number][];
}

export function scanEvidence(s: ScanSummary): string {
  const lines = [`file: ${s.name}`, `flows scanned: ${s.n}`, `alert threshold: ${s.threshold.toFixed(2)}`,
    `flagged as intrusion: ${s.flagged} (${((100 * s.flagged) / Math.max(s.n, 1)).toFixed(1)}%)`];
  if (s.flagged) {
    lines.push("flagged by predicted category: " + s.byCategory.map(([c, n]) => `${c}=${n}`).join(", "));
    lines.push(`flagged flows with category confidence < 50%: ${s.lowConf}`);
    lines.push(`p_attack of flagged flows: min ${s.pMin.toFixed(2)}, median ${s.pMed.toFixed(2)}, max ${s.pMax.toFixed(2)}`);
    lines.push(`borderline flows (within ±0.10 of threshold): ${s.borderline}`);
    lines.push("protocol/service of flagged flows: " + s.protoService.map(([k, n]) => `${k}=${n}`).join(", "));
    lines.push("fields most often in the top-5 SHAP reasons of the top flagged flows: " +
      s.reasons.map(([f, n]) => `${f} (${n}x)`).join(", "));
  }
  return lines.join("\n");
}

export function modelEvidence(m: Metrics): string {
  const b = m.binary, mc = m.multiclass, a = m.ttl_audit;
  const f1 = Object.entries(mc.per_class_f1).sort((x, y) => x[1] - y[1]).map(([k, v]) => `${k}=${v.toFixed(2)}`).join(", ");
  return `training data: ${m.source}; ${m.n_train} train flows, ${m.n_test} held-out test flows
stage 1 (normal vs attack): accuracy ${b.accuracy.toFixed(4)}, precision ${b.precision.toFixed(4)}, recall ${b.recall.toFixed(4)}, F1 ${b.f1.toFixed(4)}
stage 2 (10-class end-to-end): macro-F1 ${mc.macro_f1.toFixed(3)}; per-class F1: ${f1}
TTL shortcut audit: retrained without ${a.dropped.join(", ")}: accuracy ${a.without_ttl.accuracy.toFixed(4)}, F1 ${a.without_ttl.f1.toFixed(4)}
features: 42 flow-metadata fields (3 categorical one-hot, 39 numeric); no IPs, no payloads
the model runs in the visitor's browser; no traffic data is uploaded`;
}

export interface Metrics {
  source: string; n_train: number; n_test: number;
  binary: { accuracy: number; precision: number; recall: number; f1: number };
  multiclass: { labels: string[]; confusion: number[][]; per_class_f1: Record<string, number>; macro_f1: number };
  ttl_audit: { dropped: string[]; without_ttl: { accuracy: number; f1: number }; top_gain_features: Record<string, number> };
  threshold_curve: { threshold: number; false_alarms: number; missed_attacks: number }[];
}

// ── prompts ─────────────────────────────────────────────────────────────────
export type Msg = { role: "user" | "assistant"; content: string };

export const explainPrompt = (ev: string): Msg[] => [{ role: "user", content:
  `<evidence>\n${ev}\n</evidence>\n\nExplain this verdict to the analyst on shift in at most 120 words: ` +
  "what the flow looks like, the 2–3 reasons that mattered most, how sure the model is, and one " +
  "concrete thing a human should check next." }];

export const reportPrompt = (ev: string): Msg[] => [{ role: "user", content:
  `<evidence>\n${ev}\n</evidence>\n\nWrite a short incident report for this scan (at most 250 words) ` +
  "with these parts in bold: **Summary**, **What was found**, **Confidence and caveats**, " +
  "**Recommended next steps** (numbered, for a human). Quote counts exactly." }];

export const chatPrompt = (history: Msg[], q: string, context: string): Msg[] => [
  ...history.slice(-10),
  { role: "user", content: `<evidence>\n${context}\n</evidence>\n\nAnalyst question: ${q}\n\n` +
    "Answer in at most 150 words. If the evidence cannot answer it, say so." },
];

// ── offline template ────────────────────────────────────────────────────────
export function offlineFlow(pred: Prediction, reasons: Reason[]): string {
  const sure = Math.max(pred.pAttack, 1 - pred.pAttack) >= 0.9 ? "confident" : "unsure";
  const pct = (v: number) => `${Math.round(v * 100)}%`;
  let head = pred.intrusion
    ? `**Flagged as ${pred.category}** with attack probability ${pct(pred.pAttack)}; the model is ${sure}.`
    : `**Looks normal** (attack probability ${pct(pred.pAttack)}); the model is ${sure}.`;
  if (pred.intrusion && pred.categoryConf < 0.5) head += ` The attack *type* is uncertain (${pct(pred.categoryConf)}).`;
  const bullet = (r: Reason) => `- \`${r.feature}\` = ${fmt(r.value)}${GLOSSARY[r.feature] ? ` (${GLOSSARY[r.feature]})` : ""}`;
  const up = reasons.filter((r) => r.shap > 0).slice(0, 3), down = reasons.filter((r) => r.shap < 0).slice(0, 2);
  const parts = [head];
  if (up.length) parts.push("Pushed toward intrusion:\n" + up.map(bullet).join("\n"));
  if (down.length) parts.push("Pushed toward safe:\n" + down.map(bullet).join("\n"));
  return parts.join("\n\n");
}

// ── hallucination guard ─────────────────────────────────────────────────────
export function groundingCheck(text: string, evidence: string, model: ModelData): { ok: boolean; bad: string[] } {
  const feats = new Set(model.features);
  const cited = new Set([...text.matchAll(/`([a-z_]+)`/g)].map((m) => m[1]).filter((f) => feats.has(f)));
  const bad = [...cited].filter((f) => !new RegExp(`\\b${f}\\b`).test(evidence)).sort();
  return { ok: bad.length === 0, bad };
}
