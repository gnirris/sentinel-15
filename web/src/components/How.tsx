import type { Metrics } from "../lib/analyst";
import { HBars, Heatmap, TradeOff } from "./Charts";

const pct = (v: number, d = 2) => `${(v * 100).toFixed(d)}%`;

export default function How({ m, threshold, llmLabel }: { m: Metrics; threshold: number; llmLabel: string }) {
  const b = m.binary, mc = m.multiclass, a = m.ttl_audit;
  const f1 = Object.entries(mc.per_class_f1).sort((x, y) => y[1] - x[1]);
  const imp = Object.entries(a.top_gain_features);
  const ttlShare = imp.filter(([k]) => a.dropped.includes(k)).reduce((s, [, v]) => s + v, 0);
  return (
    <>
      <h2 style={{ marginTop: 0 }}>How well it works</h2>
      <p className="hint">Data: {m.source}. None of these {m.n_test.toLocaleString()} test flows were seen during training.</p>
      <div className="kpis">
        <div><b>{pct(b.accuracy)}</b><span>accuracy</span></div>
        <div><b>{pct(b.precision)}</b><span>precision (alerts that were real)</span></div>
        <div><b>{pct(b.recall)}</b><span>attacks caught</span></div>
        <div><b>{b.f1.toFixed(3)}</b><span>F1 score</span></div>
        <div><b>{mc.macro_f1.toFixed(3)}</b><span>attack-type macro-F1</span></div>
      </div>
      <div className="charts">
        <div className="chart"><h3>Where it gets attack types right and wrong</h3>
          <Heatmap labels={mc.labels} matrix={mc.confusion} />
          <p className="hint">Rows are the true type, columns the prediction. Darker means a bigger share of that row.</p>
        </div>
        <div className="chart"><h3>Score per attack type (F1)</h3>
          <HBars data={f1} max={1} fmt={(v) => v.toFixed(2)} label="F1 per type"
                 color={(_, v) => (v < .3 ? "#C8102E" : v < .7 ? "#E07020" : "#2F8F46")} />
        </div>
      </div>
      <p className="note"><b>Analysis</b> and <b>Backdoor</b> are the weak spots: in UNSW-NB15 their flows have
        near-identical statistics, so flow metadata alone can't reliably tell them apart. Both are still caught
        as <i>attacks</i>; the mistake is in naming them.</p>

      <h2>The pipeline</h2>
      <pre className="pipe">{`raw flow (45 fields)
  └─ drop \`id\` (a row number, would leak)        BOM-safe CSV reading
  └─ ColumnTransformer
       ├─ proto / service / state → one-hot (unknown values → all zero)
       └─ 39 numeric flow statistics → passthrough
  └─ Stage 1  XGBoost, 250 trees  : normal or attack   (threshold adjustable)
  └─ Stage 2  XGBoost, 9 classes  : which attack       (runs only on alerts)
  └─ TreeSHAP                     : why, per flow, one-hot folded back
  └─ LLM analyst: ${llmLabel}
       SHAP + numbers → plain English, grounding-checked

Trained in Python (scikit-learn, XGBoost). Runs here in your browser:
the trees are exported to JSON and re-implemented in TypeScript, checked
against Python on all 500 sample flows (max difference 0.00001).`}</pre>

      <h2>The LLM explains; it never decides</h2>
      <div className="cols">
        <ul>
          <li><b>XGBoost decides, the LLM explains.</b> The verdict, probability and type come from the trained
            models. The LLM is handed those numbers and the SHAP values, and told to use nothing else.</li>
          <li><b>Why not let the LLM detect attacks?</b> It's slower, costs money per flow, can't be audited like
            a tree model, and has no measured accuracy on this data. A 98%-accurate tree model with an LLM
            translator gets the best of both.</li>
          <li><b>It never sees the answer key.</b> Ground-truth labels are removed from what it reads.</li>
        </ul>
        <ul>
          <li><b>Hallucination guard.</b> Every field the LLM names is checked against the evidence it was given;
            an invented reason is flagged, not hidden.</li>
          <li><b>Calibrated language.</b> It must call low-confidence alerts uncertain and recommend human
            investigation, never automatic blocking.</li>
          <li><b>Privacy.</b> Detection runs on your device; traffic data never leaves the browser. Only when you
            ask the LLM does it receive that one flow's metadata: no IPs, no packet contents.</li>
        </ul>
      </div>

      <h2>Does it just memorise TTL values?</h2>
      <div className="cols">
        <div className="chart"><h3>Most important fields (TTL fields in orange)</h3>
          <HBars data={imp} fmt={(v) => pct(v, 1)} label="Share of decision weight"
                 color={(k) => (a.dropped.includes(k) ? "#E07020" : "#2462A8")} />
        </div>
        <div>
          <p>In UNSW-NB15, the testbed's attack machines used different <b>TTL values</b> than normal hosts, so
            TTL could be a shortcut: true in this lab, meaningless on a real network.</p>
          <p>The TTL fields carry <b>{pct(ttlShare, 0)}</b> of the model's decision weight, so we retrained without them:</p>
          <table className="plain">
            <thead><tr><th></th><th>with TTL</th><th>without</th></tr></thead>
            <tbody>
              <tr><td>accuracy</td><td>{pct(b.accuracy)}</td><td>{pct(a.without_ttl.accuracy)}</td></tr>
              <tr><td>F1</td><td>{b.f1.toFixed(3)}</td><td>{a.without_ttl.f1.toFixed(3)}</td></tr>
            </tbody>
          </table>
          <p>Performance barely moves: other fields carry the same signal. The model doesn't <i>depend</i> on the
            shortcut, but it <i>prefers</i> it when it's there.</p>
        </div>
      </div>

      <h2>Missed attacks versus false alarms</h2>
      <div className="chart"><TradeOff curve={m.threshold_curve} threshold={threshold} /></div>
      <div className="cols" style={{ marginTop: "1.2rem" }}>
        <div>
          <h3>Errors have different costs</h3>
          <ul>
            <li>A <b>missed attack</b> can mean a breach.</li>
            <li>A <b>false alarm</b> wastes an analyst's time, and real networks see millions of flows a day: a 1%
              false-alarm rate buries the real alerts (<i>alert fatigue</i>).</li>
            <li>So the threshold is a human policy choice, not a model fact. That's why it's a slider.</li>
            <li>Alerts are for <b>a person to investigate</b>, never for automatic blocking or punishing a user.</li>
          </ul>
        </div>
        <div>
          <h3>Privacy and limits</h3>
          <ul>
            <li>The model reads <b>flow metadata only</b>: sizes, timings, counts, protocol. It never sees packet
              contents, so it works on encrypted traffic and reads no messages.</li>
            <li>No IP addresses are used, so it can't learn to distrust a person or a country.</li>
            <li><b>Limits:</b> UNSW-NB15 was generated in a 2015 lab. Attacks change, and accuracy on a real network
              would be lower than shown here. Retrain and re-audit before any real use.</li>
          </ul>
        </div>
      </div>
    </>
  );
}
