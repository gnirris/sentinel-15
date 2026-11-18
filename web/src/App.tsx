import { useCallback, useEffect, useMemo, useState } from "react";
import Ask from "./components/Ask";
import How from "./components/How";
import Live from "./components/Live";
import Scan from "./components/Scan";
import Settings from "./components/Settings";
import { modelEvidence, type Metrics } from "./lib/analyst";
import { PROVIDERS, serverStatus, type LLMConfig } from "./lib/llm";
import { loadModel, type Flow, type ModelData } from "./lib/model";

const TABS = [
  ["live", "Live traffic"], ["scan", "Scan a log file"], ["ask", "Ask the analyst"], ["how", "How it works"],
] as const;
type Tab = (typeof TABS)[number][0];

interface Data { model: ModelData; flows: Flow[]; metrics: Metrics }

export default function App() {
  const [data, setData] = useState<Data | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("live");
  const [threshold, setThreshold] = useState(0.5);
  const [llm, setLLM] = useState<LLMConfig | null>(null);
  const [serverModel, setServerModel] = useState<string | null>(null);
  const [flowEv, setFlowEv] = useState("");
  const [scanEv, setScanEv] = useState<string | null>(null);

  useEffect(() => {
    const get = (p: string) => fetch(new URL(`model/${p}`, document.baseURI)).then((r) => {
      if (!r.ok) throw new Error(`${p}: HTTP ${r.status}`);
      return r.json();
    });
    Promise.all([get("model.json"), get("flows.json"), get("metrics.json")])
      .then(([m, f, x]) => setData({ model: loadModel(m), flows: f, metrics: x }))
      .catch((e) => setLoadError(String(e.message ?? e)));
    serverStatus().then(setServerModel);
  }, []);

  const onFlowEv = useCallback((ev: string) => setFlowEv(ev), []);
  const onScanEv = useCallback((ev: string | null) => setScanEv(ev), []);
  const onLLM = useCallback((c: LLMConfig | null) => setLLM(c), []);

  const context = useMemo(() => data ? [
    "MODEL:\n" + modelEvidence(data.metrics), flowEv, scanEv ? "LAST BATCH SCAN:\n" + scanEv : "",
  ].filter(Boolean).join("\n\n") : "", [data, flowEv, scanEv]);

  const openSettings = () => document.getElementById("llm-settings")?.scrollIntoView({ behavior: "smooth" });
  const llmLabel = llm ? `${PROVIDERS[llm.provider].name} (${llm.model})` : "off (template text)";

  if (loadError) return <p className="loading err">The detector couldn't load ({loadError}). Reload the page to try again.</p>;
  if (!data) return <p className="loading">Loading the detector into your browser (about 1.5 MB)…</p>;

  return (
    <div className="app">
      <aside className="rail" aria-label="Settings">
        <Settings m={data.metrics} threshold={threshold} setThreshold={setThreshold}
                  serverModel={serverModel} onLLM={onLLM} />
      </aside>
      <main className="main">
        <header className="mast">
          <h1>SENTINEL-15</h1>
          <p>Watches network traffic, flags intrusions, and explains every decision in plain English.
            The detector runs entirely in your browser.</p>
        </header>
        <nav className="tabs" role="tablist" aria-label="Sections">
          {TABS.map(([id, label]) => (
            <button key={id} role="tab" aria-selected={tab === id} aria-controls={`panel-${id}`}
                    id={`tab-${id}`} onClick={() => setTab(id)}>{label}</button>
          ))}
        </nav>
        {/* Live stays mounted so the replay keeps its place while you look at other tabs. */}
        <section id="panel-live" role="tabpanel" aria-labelledby="tab-live" hidden={tab !== "live"}>
          <Live model={data.model} flows={data.flows} threshold={threshold} llm={llm} onEvidence={onFlowEv} />
        </section>
        <section id="panel-scan" role="tabpanel" aria-labelledby="tab-scan" hidden={tab !== "scan"}>
          <Scan model={data.model} sample={data.flows} threshold={threshold} llm={llm} onEvidence={onScanEv} />
        </section>
        {tab === "ask" && (
          <section id="panel-ask" role="tabpanel" aria-labelledby="tab-ask">
            <Ask model={data.model} llm={llm} context={context} onOpenSettings={openSettings} />
          </section>
        )}
        {tab === "how" && (
          <section id="panel-how" role="tabpanel" aria-labelledby="tab-how">
            <How m={data.metrics} threshold={threshold} llmLabel={llmLabel} />
          </section>
        )}
      </main>
    </div>
  );
}
