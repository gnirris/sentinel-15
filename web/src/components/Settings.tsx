import { useEffect, useState } from "react";
import type { Metrics } from "../lib/analyst";
import { detect, listModels, PROVIDERS, type LLMConfig, type ProviderId } from "../lib/llm";

// The typed key lives only in this tab's sessionStorage: gone when the tab closes.
const store = {
  get: (k: string) => { try { return sessionStorage.getItem(k) ?? ""; } catch { return ""; } },
  set: (k: string, v: string) => { try { sessionStorage.setItem(k, v); } catch { /* private mode */ } },
};

export default function Settings({ m, threshold, setThreshold, serverModel, onLLM }: {
  m: Metrics; threshold: number; setThreshold: (t: number) => void;
  serverModel: string | null; onLLM: (c: LLMConfig | null) => void;
}) {
  const [choice, setChoice] = useState<"auto" | ProviderId>(() => (store.get("provider") as ProviderId) || "auto");
  const [key, setKey] = useState(() => store.get("key"));
  const [model, setModel] = useState(() => store.get("model"));
  const [baseURL, setBaseURL] = useState(() => store.get("baseURL"));
  const [models, setModels] = useState<string[]>([]);

  const pid: ProviderId | null = key.trim()
    ? (choice === "auto" || choice === "server" ? detect(key) : choice)
    : serverModel && (choice === "auto" || choice === "server") ? "server" : null;
  const prov = pid ? PROVIDERS[pid] : null;
  const effModel = pid === "server" ? serverModel ?? "" : model || prov?.model || "";

  useEffect(() => { store.set("provider", choice); store.set("key", key); store.set("model", model); store.set("baseURL", baseURL); },
    [choice, key, model, baseURL]);

  // Live model list for the chosen provider (debounced while typing a key)
  useEffect(() => {
    setModels([]);
    if (!pid || pid === "server" || !key.trim() || (pid === "custom" && !baseURL)) return;
    const t = setTimeout(() => {
      listModels({ provider: pid, key: key.trim(), model: "", baseURL }).then(setModels);
    }, 500);
    return () => clearTimeout(t);
  }, [pid, key, baseURL]);

  const ready = !!pid && !!effModel && (pid !== "custom" || !!baseURL);
  useEffect(() => {
    onLLM(ready ? { provider: pid!, key: key.trim(), model: effModel, baseURL: baseURL || undefined } : null);
  }, [ready, pid, key, effModel, baseURL, onLLM]);

  const options = models.length ? (models.includes(effModel) ? models : [effModel, ...models]) : [];

  return (
    <>
      <section>
        <h2>Detector</h2>
        <label className="field" htmlFor="thr">Alert threshold: {threshold.toFixed(2)}</label>
        <input id="thr" type="range" min={0.05} max={0.95} step={0.05} value={threshold}
               onChange={(e) => setThreshold(+e.target.value)} />
        <p className="hint">How likely a flow must be to be an attack before it raises an alert. Lower catches more
          attacks but raises more false alarms.</p>
        <div className="stat">
          <div><b>{(m.binary.accuracy * 100).toFixed(1)}%</b><span>accuracy</span></div>
          <div><b>{(m.binary.recall * 100).toFixed(1)}%</b><span>attacks caught</span></div>
        </div>
        <p className="hint">Tested on {m.n_test.toLocaleString()} flows it never saw in training.</p>
      </section>

      <section id="llm-settings">
        <h2>LLM analyst</h2>
        <p className="hint">Explains verdicts in plain English. Optional: detection works without it.</p>
        <label className="field" htmlFor="prov">Provider</label>
        <select id="prov" className="input" value={choice} onChange={(e) => { setChoice(e.target.value as ProviderId); setModel(""); }}>
          <option value="auto">{serverModel ? "This site's analyst, or detect from my key" : "Detect from my key"}</option>
          {(Object.entries(PROVIDERS) as [ProviderId, typeof PROVIDERS[ProviderId]][])
            .filter(([id]) => id !== "server" || serverModel)
            .map(([id, p]) => <option key={id} value={id}>{p.name}{p.free ? ", free tier" : ""}{p.note ? `, ${p.note}` : ""}</option>)}
        </select>

        {choice !== "server" && (
          <>
            <label className="field" htmlFor="key">Your API key{serverModel ? " (optional)" : ""}</label>
            <input id="key" className="input" type="password" autoComplete="off" spellCheck={false} value={key}
                   placeholder={serverModel ? "leave empty to use this site's analyst" : "paste any provider's key"}
                   onChange={(e) => { setKey(e.target.value); setModel(""); }} />
            <p className="hint">Sent only to that provider, straight from your browser. Kept in this tab until you close it.
              Gemini and Groq give free keys with no card.</p>
          </>
        )}
        {key.trim() && !pid && <p className="err">This key doesn't match a known provider. Choose the provider above.</p>}

        {pid === "custom" && (
          <>
            <label className="field" htmlFor="base">Base URL</label>
            <input id="base" className="input" value={baseURL} placeholder="http://localhost:11434/v1"
                   onChange={(e) => setBaseURL(e.target.value)} />
          </>
        )}

        {pid && pid !== "server" && (
          <>
            <label className="field" htmlFor="model">Model</label>
            {options.length
              ? <select id="model" className="input" value={effModel} onChange={(e) => setModel(e.target.value)}>
                  {options.map((o) => <option key={o}>{o}</option>)}</select>
              : <input id="model" className="input" value={effModel} onChange={(e) => setModel(e.target.value)} />}
          </>
        )}

        <div className={`status${ready ? " on" : ""}`} role="status">
          {ready ? <>Connected: <b>{prov!.name}</b> ({effModel})</>
                 : "Off. Explanations use a built-in template."}
        </div>
      </section>
    </>
  );
}
