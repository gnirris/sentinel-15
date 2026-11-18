import { useCallback, useEffect, useRef, useState } from "react";
import { groundingCheck, type Msg } from "../lib/analyst";
import { stream, type LLMConfig } from "../lib/llm";
import type { ModelData } from "../lib/model";
import Md from "./Md";

/** Streams one LLM answer into state; cancels if the component goes away. */
export function useStream() {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const ctl = useRef<AbortController | null>(null);
  useEffect(() => () => ctl.current?.abort(), []);
  const run = useCallback(async (llm: LLMConfig, msgs: Msg[], effort: "low" | "medium", max: number) => {
    ctl.current?.abort();
    const c = new AbortController();
    ctl.current = c;
    setText(""); setBusy(true);
    let acc = "";
    for await (const t of stream(llm, msgs, effort, max, c.signal)) {
      if (c.signal.aborted) return acc;
      acc += t; setText(acc);
    }
    setBusy(false);
    return acc;
  }, []);
  const reset = useCallback(() => { ctl.current?.abort(); setText(""); setBusy(false); }, []);
  return { text, busy, run, reset };
}

/** LLM answer + hallucination check + the evidence it was given. */
export function AnalystOutput({ text, busy, evidence, model }:
  { text: string; busy: boolean; evidence: string; model: ModelData }) {
  const g = busy ? null : groundingCheck(text, evidence, model);
  return (
    <>
      <Md text={text || "…"} streaming={busy} />
      {g && (g.ok
        ? <p className="ground">✓ Grounding check passed: every field the LLM cited is in the detector's evidence.</p>
        : <p className="ground bad">Grounding check failed: the LLM cited {g.bad.map((b) => `\`${b}\``).join(", ")},
            which the detector never gave it. Treat that part as unreliable.</p>)}
      {!busy && (
        <details>
          <summary>Show the evidence the LLM was given</summary>
          <pre>{evidence}</pre>
        </details>
      )}
    </>
  );
}
