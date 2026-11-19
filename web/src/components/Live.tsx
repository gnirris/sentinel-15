import { useEffect, useMemo, useState } from "react";
import { explainPrompt, flowEvidence, offlineFlow } from "../lib/analyst";
import type { LLMConfig } from "../lib/llm";
import { clean, explain, predict, type Flow, type ModelData } from "../lib/model";
import { AnalystOutput, useStream } from "./Analyst";
import { ShapBars } from "./Charts";
import Md from "./Md";

export const human = (n: number) => {
  for (const u of ["B", "kB", "MB", "GB"]) {
    if (Math.abs(n) < 1000) return u === "B" ? `${Math.round(n)} B` : `${n.toFixed(1)} ${u}`;
    n /= 1000;
  }
  return `${n.toFixed(1)} TB`;
};
const pct = (v: number) => `${Math.round(v * 100)}%`;
const SPEEDS = [0.5, 1, 2, 4, 8];

export default function Live({ model, flows, threshold, llm, onEvidence }: {
  model: ModelData; flows: Flow[]; threshold: number; llm: LLMConfig | null;
  onEvidence: (ev: string) => void;
}) {
  const [idx, setIdx] = useState(0);
  const [seen, setSeen] = useState<number[]>([0]);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const s = useStream();

  useEffect(() => {
    if (!playing) return;
    const t = setInterval(() => setIdx((i) => (i + 1) % flows.length), 1000 / speed);
    return () => clearInterval(t);
  }, [playing, speed, flows.length]);

  useEffect(() => {
    setSeen((h) => (h[h.length - 1] === idx ? h : [...h, idx].slice(-80)));
    s.reset();
  }, [idx]); // eslint-disable-line react-hooks/exhaustive-deps

  const raw = flows[idx];
  const row = useMemo(() => clean(raw, model), [raw, model]);
  const pred = useMemo(() => predict(model, raw, threshold), [raw, model, threshold]);
  const reasons = useMemo(() => explain(model, raw), [raw, model]);
  const evidence = useMemo(() => flowEvidence(row, pred, reasons, threshold), [row, pred, reasons, threshold]);
  useEffect(() => onEvidence(`current live flow #${idx}:\n${evidence}`), [evidence, idx, onEvidence]);

  const pAll = useMemo(() => seen.map((i) => predict(model, flows[i], threshold)), [seen, model, flows, threshold]);
  const alerts = pAll.filter((p) => p.intrusion).length;

  // The strip remounts per flow (key={idx}), so the one-shot flash plays once per new alert.
  const flash = pred.intrusion;

  const step = () => { setPlaying(false); setIdx((i) => (i + 1) % flows.length); };
  const reset = () => { setPlaying(false); setIdx(0); setSeen([0]); };

  const naming = Number.isNaN(pred.categoryConf);   // stage 2 still downloading
  const unsure = pred.categoryConf < 0.5;
  const flowLine = `${row.proto}/${row.service}, ${row.state} state, ${(row.dur as number).toFixed(2)} s, ` +
    `${human(row.sbytes as number)} out in ${row.spkts} packets, ${human(row.dbytes as number)} back in ${row.dpkts}, ` +
    `TTL ${row.sttl}. Dataset label: ${raw.attack_cat ?? "unknown"}.`;

  return (
    <>
      <p className="replay"><strong>This is a replay, not your network.</strong> The flows are 500 recorded
        connections from the UNSW-NB15 test set, which the model never saw in training, played back as if
        they were arriving live. Nothing is captured from your computer or connection.</p>

      <div className="controls">
        <button className="btn primary" onClick={() => setPlaying((p) => !p)} aria-pressed={playing}>
          {playing ? "⏸ Pause" : "▶ Play"}
        </button>
        <button className="btn" onClick={step}>⏭ Step</button>
        <button className="btn" onClick={reset}>↺ Reset</button>
        <label className="speed">
          <span className="hint" style={{ margin: 0 }}>Flows per second: <b>{speed}</b></span>
          <input type="range" min={0} max={SPEEDS.length - 1} value={SPEEDS.indexOf(speed)}
                 onChange={(e) => setSpeed(SPEEDS[+e.target.value])} aria-label="Flows per second" />
        </label>
      </div>

      <section key={idx} className={`verdict ${pred.intrusion ? "bad" : "safe"}${flash ? " flash" : ""}`} aria-live="polite">
        <p className="word">{pred.intrusion ? pred.category + (unsure ? "?" : "") : "Normal"}</p>
        <p className="line">{pred.intrusion
          ? `Flow ${idx} looks like an attack (${pct(pred.pAttack)} likely). ` +
            (naming ? "Loading the attack-type model…"
              : unsure ? `The type is a guess: only ${pct(pred.categoryConf)} sure it is ${pred.category}.`
                       : `${pct(pred.categoryConf)} sure it is ${pred.category}.`)
          : `Flow ${idx} looks like normal traffic (${pct(pred.pAttack)} chance of attack).`}</p>
        <p className="flow">{flowLine}</p>
      </section>

      <div className="split">
        <div>
          <div className="tablewrap">
            <table className="log">
              <thead><tr><th>No.</th><th>Proto</th><th>Service</th><th>State</th><th className="n">Secs</th>
                <th className="n">Out</th><th className="n">Back</th><th className="n">Attack</th><th>Verdict</th></tr></thead>
              <tbody>
                {seen.slice(-14).map((i, k, arr) => {
                  const p = pAll[pAll.length - arr.length + k], f = clean(flows[i], model);
                  const cls = i === idx ? "sel" : p.intrusion ? "bad" : f.proto === "tcp" ? "tcp" : f.proto === "udp" ? "udp" : "oth";
                  return (
                    <tr key={`${i}-${k}`} className={cls} onClick={() => { setPlaying(false); setIdx(i); }}
                        title="Show this flow">
                      <td className="n">{i}</td><td>{String(f.proto)}</td><td>{String(f.service)}</td><td>{String(f.state)}</td>
                      <td className="n">{(f.dur as number).toFixed(2)}</td><td className="n">{human(f.sbytes as number)}</td>
                      <td className="n">{human(f.dbytes as number)}</td><td className="n">{pct(p.pAttack)}</td>
                      <td>{p.intrusion ? p.category : "Normal"}</td>
                    </tr>
                  );
                }).reverse()}
              </tbody>
            </table>
          </div>
          <p className="cap">{seen.length} flows inspected, {alerts} raised an alert ({pct(alerts / seen.length)}).
            Newest first; blue is the flow shown above. Click a row to go back to it.</p>
        </div>

        <div>
          <div className="chart">
            <h3>What drove this verdict</h3>
            <ShapBars reasons={reasons} />
          </div>
          <div className="analyst">
            <h3>Analyst</h3>
            {s.text || s.busy
              ? <AnalystOutput text={s.text} busy={s.busy} evidence={evidence} model={model} />
              : <Md text={offlineFlow(pred, reasons)} />}
            {llm && !s.text && !s.busy && (
              playing
                ? <p className="hint">Pause to have the LLM explain this flow.</p>
                : <button className="btn primary" style={{ marginTop: ".6rem" }}
                          onClick={() => s.run(llm, explainPrompt(evidence), "low", 2000)}>Explain this verdict</button>
            )}
            {!llm && <p className="hint">Template text. Turn on the LLM analyst in Settings for a plain-English write-up.</p>}
          </div>
        </div>
      </div>
    </>
  );
}
