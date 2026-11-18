import Papa from "papaparse";
import { useEffect, useMemo, useRef, useState } from "react";
import { reportPrompt, scanEvidence, type ScanSummary } from "../lib/analyst";
import type { LLMConfig } from "../lib/llm";
import { clean, explain, missingColumns, predict, type Flow, type ModelData, type Prediction } from "../lib/model";
import { AnalystOutput, useStream } from "./Analyst";
import { CAT_COLORS, HBars, Histogram } from "./Charts";

const MAX_ROWS = 100_000;   // keeps a scan under ~a minute on a laptop
const MIN_THRESHOLD = 0.05; // stage 2 runs for anything the slider could ever flag

interface Scored { raw: Flow; pAttack: number; category: string; categoryConf: number }

export default function Scan({ model, sample, threshold, llm, onEvidence }: {
  model: ModelData; sample: Flow[]; threshold: number; llm: LLMConfig | null;
  onEvidence: (ev: string | null) => void;
}) {
  const [name, setName] = useState<string | null>(null);
  const [rows, setRows] = useState<Scored[] | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const [truncated, setTruncated] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const job = useRef(0);
  const s = useStream();

  async function score(fname: string, data: Flow[]) {
    const id = ++job.current;
    setName(fname); setRows(null); setError(null); s.reset();
    setTruncated(data.length > MAX_ROWS);
    data = data.slice(0, MAX_ROWS);
    const out: Scored[] = [];
    for (let i = 0; i < data.length; i += 500) {
      for (const r of data.slice(i, i + 500)) {
        const p: Prediction = predict(model, r, MIN_THRESHOLD);
        out.push({ raw: r, pAttack: p.pAttack, category: p.category, categoryConf: p.categoryConf });
      }
      setProgress(Math.min(1, (i + 500) / data.length));
      await new Promise((res) => setTimeout(res, 0));   // let the page breathe
      if (id !== job.current) return;
    }
    setProgress(null); setRows(out);
  }

  function loadFile(f: File) {
    setError(null); setProgress(0);
    Papa.parse<Flow>(f, {
      header: true, dynamicTyping: true, skipEmptyLines: true, worker: true,
      transformHeader: (h) => h.replace(/^﻿/, "").trim(),
      complete: (res) => {
        const missing = missingColumns(res.meta.fields ?? [], model);
        if (missing.length) {
          setProgress(null); setRows(null); setName(f.name);
          setError(`${f.name} is missing ${missing.length} UNSW-NB15 columns: ${missing.join(", ")}. ` +
            "Add them, or scan the sample log to see the expected format.");
          return;
        }
        score(f.name, res.data);
      },
      error: (e) => { setProgress(null); setError(`This file couldn't be read as a CSV (${e.message}). Export it as comma-separated text and try again.`); },
    });
  }

  // Everything below depends on the threshold, but none of it re-runs the models.
  const view = useMemo(() => {
    if (!rows) return null;
    const res = rows.map((r) => ({ ...r, intrusion: r.pAttack >= threshold }));
    const flagged = res.filter((r) => r.intrusion).sort((a, b) => b.pAttack - a.pAttack);
    const byCat = new Map<string, number>();
    for (const r of res) { const c = r.intrusion ? r.category : "Normal"; byCat.set(c, (byCat.get(c) ?? 0) + 1); }
    const hasLabel = rows.length > 0 && rows[0].raw.label !== undefined && rows[0].raw.label !== null;
    const agree = hasLabel ? res.filter((r) => (r.intrusion ? 1 : 0) === Number(r.raw.label)).length / res.length : null;
    return { res, flagged, byCat: [...byCat.entries()].sort((a, b) => a[1] - b[1]), agree };
  }, [rows, threshold]);

  const evidence = useMemo(() => {
    if (!view || !name) return null;
    const f = view.flagged, ps = f.map((r) => r.pAttack).sort((a, b) => a - b);
    const ctr = (xs: string[]) => {
      const m = new Map<string, number>(); for (const x of xs) m.set(x, (m.get(x) ?? 0) + 1);
      return [...m.entries()].sort((a, b) => b[1] - a[1]);
    };
    const summary: ScanSummary = {
      name, n: view.res.length, threshold, flagged: f.length,
      byCategory: ctr(f.map((r) => r.category)), lowConf: f.filter((r) => r.categoryConf < .5).length,
      pMin: ps[0] ?? 0, pMed: ps[Math.floor(ps.length / 2)] ?? 0, pMax: ps[ps.length - 1] ?? 0,
      borderline: view.res.filter((r) => Math.abs(r.pAttack - threshold) < .1).length,
      protoService: ctr(f.map((r) => { const c = clean(r.raw, model); return `${c.proto}/${c.service}`; })).slice(0, 6),
      reasons: ctr(f.slice(0, 30).flatMap((r) => explain(model, r.raw).map((e) => e.feature))).slice(0, 8),
    };
    return scanEvidence(summary);
  }, [view, name, threshold, model]);
  useEffect(() => { onEvidence(evidence); s.reset(); }, [evidence]); // eslint-disable-line react-hooks/exhaustive-deps

  function download() {
    if (!view) return;
    const out = view.flagged.map((r) => ({ ...r.raw, p_attack: +r.pAttack.toFixed(4), category: r.category,
      category_conf: +r.categoryConf.toFixed(4) }));
    const url = URL.createObjectURL(new Blob([Papa.unparse(out)], { type: "text/csv" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `flagged_${name}` });
    a.click(); URL.revokeObjectURL(url);
  }

  function downloadReport() {
    const url = URL.createObjectURL(new Blob([s.text], { type: "text/markdown" }));
    const a = Object.assign(document.createElement("a"), { href: url, download: `incident_${name}.md` });
    a.click(); URL.revokeObjectURL(url);
  }

  const cols = ["proto", "service", "state", "dur", "sbytes", "dbytes", "sttl"];

  return (
    <>
      <p>Upload a flow log in UNSW-NB15 format, or scan the 500-flow sample. The file is read and scored
        in your browser; it is never uploaded anywhere.</p>
      <div className={`drop${over ? " over" : ""}`}
           onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
           onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) loadFile(f); }}>
        <span>Drop a <b>.csv</b> file here, or</span>
        <div className="row">
          <button className="btn" onClick={() => fileInput.current?.click()}>Choose a file</button>
          <button className="btn primary" onClick={() => score("sample_test_flows.csv", sample)}>Scan the sample log</button>
        </div>
        <input ref={fileInput} type="file" accept=".csv,text/csv" hidden
               onChange={(e) => { const f = e.target.files?.[0]; if (f) loadFile(f); e.target.value = ""; }} />
      </div>

      {progress !== null && (
        <div className="progress" role="progressbar" aria-valuenow={Math.round(progress * 100)} aria-label="Scan progress">
          <div style={{ width: `${progress * 100}%` }} />
        </div>
      )}
      {error && <p className="err" role="alert">{error}</p>}

      {view && name && (
        <>
          <h2>{name}</h2>
          {truncated && <p className="hint">Only the first {MAX_ROWS.toLocaleString()} rows were scanned, to keep it fast in a browser.</p>}
          <div className="kpis">
            <div><b>{view.res.length.toLocaleString()}</b><span>flows scanned</span></div>
            <div className="alert"><b>{view.flagged.length.toLocaleString()}</b><span>intrusions</span></div>
            <div><b>{(100 * view.flagged.length / Math.max(view.res.length, 1)).toFixed(1)}%</b><span>share flagged</span></div>
            {view.agree !== null
              ? <div><b>{(100 * view.agree).toFixed(1)}%</b><span>agrees with the file's labels</span></div>
              : <div><b>{view.byCat.filter(([c]) => c !== "Normal").at(-1)?.[0] ?? "None"}</b><span>most common attack</span></div>}
          </div>

          <div className="charts">
            <div className="chart"><h3>What the scan found</h3>
              <HBars data={view.byCat} color={(k) => CAT_COLORS[k] ?? "#5C6770"} label="Flows per category" fmt={(v) => v.toLocaleString()} />
            </div>
            <div className="chart"><h3>How sure the model was</h3>
              <Histogram values={view.res.map((r) => r.pAttack)} threshold={threshold} />
            </div>
          </div>

          <h2>Flagged flows ({view.flagged.length.toLocaleString()})</h2>
          {view.flagged.length > 0 && (
            <>
              <div className="tablewrap" style={{ maxHeight: 360 }}>
                <table className="log">
                  <thead><tr>{cols.map((c) => <th key={c}>{c}</th>)}<th className="n">Attack</th><th>Type</th><th className="n">Sure</th></tr></thead>
                  <tbody>
                    {view.flagged.slice(0, 200).map((r, i) => {
                      const c = clean(r.raw, model);
                      return (
                        <tr key={i} className="bad">
                          {cols.map((k) => <td key={k}>{typeof c[k] === "number" ? Number((c[k] as number).toPrecision(5)) : String(c[k])}</td>)}
                          <td className="n">{Math.round(r.pAttack * 100)}%</td><td>{r.category}</td>
                          <td className="n">{Math.round(r.categoryConf * 100)}%</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
              <p className="cap">{view.flagged.length > 200 ? "Showing the 200 most likely attacks. " : ""}
                <button className="btn" onClick={download}>Download all flagged flows (CSV)</button></p>
            </>
          )}

          <h2>Incident report</h2>
          {!llm && <p className="hint">Turn on the LLM analyst in Settings and it will write an incident report for this scan.</p>}
          {llm && !s.text && !s.busy && (
            <button className="btn primary" onClick={() => evidence && s.run(llm, reportPrompt(evidence), "medium", 4000)}>
              Write incident report</button>
          )}
          {(s.text || s.busy) && evidence && (
            <div className="analyst">
              <AnalystOutput text={s.text} busy={s.busy} evidence={evidence} model={model} />
              {!s.busy && <button className="btn" style={{ marginTop: ".6rem" }} onClick={downloadReport}>Download incident report (Markdown)</button>}
            </div>
          )}
        </>
      )}
    </>
  );
}
