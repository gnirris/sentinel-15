import { useState } from "react";
import { chatPrompt, type Msg } from "../lib/analyst";
import type { LLMConfig } from "../lib/llm";
import type { ModelData } from "../lib/model";
import { AnalystOutput, useStream } from "./Analyst";
import Md from "./Md";

const SUGGESTIONS = [
  "Why was the current flow flagged, or not?",
  "Summarise my last scan for a manager.",
  "Which attacks does this model struggle with?",
  "What happens if I lower the threshold to 0.2?",
];

export default function Ask({ model, llm, context, onOpenSettings }: {
  model: ModelData; llm: LLMConfig | null; context: string; onOpenSettings: () => void;
}) {
  const [history, setHistory] = useState<Msg[]>([]);
  const [pending, setPending] = useState<{ q: string; ctx: string } | null>(null);
  const [input, setInput] = useState("");
  const s = useStream();

  async function ask(q: string) {
    if (!llm || !q.trim() || s.busy) return;
    setInput(""); setPending({ q, ctx: context });
    const reply = await s.run(llm, chatPrompt(history, q, context), "low", 3000);
    setHistory((h) => [...h, { role: "user", content: q }, { role: "assistant", content: reply }]);
    setPending(null); s.reset();
  }

  if (!llm) return (
    <>
      <p>Ask about the flow on the live tab, your last scan, or how the model works.</p>
      <p className="note">The chat needs the LLM analyst. <button className="btn" onClick={onOpenSettings}>Open Settings</button>{" "}
        The rest of the site works without it.</p>
    </>
  );

  return (
    <>
      <p>Ask about the flow on the live tab, your last scan, or how the model works. The analyst answers
        only from the detector's own numbers.</p>
      {history.length === 0 && !pending && (
        <div className="chips">{SUGGESTIONS.map((q) => <button key={q} onClick={() => ask(q)}>{q}</button>)}</div>
      )}
      {history.map((m, i) => (
        <div key={i} className={`msg ${m.role}`}>{m.role === "user" ? m.content : <Md text={m.content} />}</div>
      ))}
      {pending && (
        <>
          <div className="msg user">{pending.q}</div>
          <div className="msg assistant"><AnalystOutput text={s.text} busy={s.busy} evidence={pending.ctx} model={model} /></div>
        </>
      )}
      <form className="ask" onSubmit={(e) => { e.preventDefault(); ask(input); }}>
        <input className="input" value={input} onChange={(e) => setInput(e.target.value)}
               placeholder="Ask the analyst a question" aria-label="Question for the analyst" />
        <button className="btn primary" disabled={!input.trim() || s.busy}>Ask</button>
        {history.length > 0 && <button type="button" className="btn" onClick={() => setHistory([])}>Clear</button>}
      </form>
    </>
  );
}
