// Provider layer. Every call runs from the visitor's browser with their own key,
// except "server", which goes through gnirris.xyz/api/llm.php so the site's own
// key never reaches the browser. Anthropic and Gemini use their official SDKs;
// the rest expose OpenAI-compatible endpoints that allow browser (CORS) calls.
import { SYSTEM, type Msg } from "./analyst";

// SDKs load on first use, so visitors who never turn on the LLM don't download them.
const sdk = {
  anthropic: async () => (await import("@anthropic-ai/sdk")).default,
  gemini: async () => (await import("@google/genai")).GoogleGenAI,
  openai: async () => (await import("openai")).default,
};

export type ProviderId =
  | "server" | "anthropic" | "openai" | "gemini" | "groq" | "openrouter" | "mistral" | "deepseek" | "xai" | "custom";

export interface Provider {
  name: string; prefix: string[]; baseURL?: string; model: string; free?: boolean; note?: string;
}

export const PROVIDERS: Record<ProviderId, Provider> = {
  server: { name: "This site's analyst", prefix: [], model: "", note: "free, limited per hour" },
  anthropic: { name: "Anthropic (Claude)", prefix: ["sk-ant-"], model: "claude-opus-5-5" },
  openai: { name: "OpenAI (GPT)", prefix: ["sk-proj-", "sk-svcacct-"], model: "gpt-4o-mini" },
  gemini: { name: "Google Gemini", prefix: ["AIza", "AQ."], model: "gemini-3.5-flash-lite", free: true },
  groq: { name: "Groq (Llama, Qwen…)", prefix: ["gsk_"], baseURL: "https://api.groq.com/openai/v1", model: "llama-3.3-70b-versatile", free: true },
  openrouter: { name: "OpenRouter (any model)", prefix: ["sk-or-"], baseURL: "https://openrouter.ai/api/v1", model: "openrouter/auto", free: true },
  mistral: { name: "Mistral", prefix: [], baseURL: "https://api.mistral.ai/v1", model: "mistral-small-latest", free: true },
  deepseek: { name: "DeepSeek", prefix: [], baseURL: "https://api.deepseek.com", model: "deepseek-chat" },
  xai: { name: "xAI (Grok)", prefix: ["xai-"], baseURL: "https://api.x.ai/v1", model: "grok-3-mini" },
  custom: { name: "Custom (OpenAI-compatible)", prefix: [], model: "" },
};

export interface LLMConfig { provider: ProviderId; key: string; model: string; baseURL?: string }

export const SERVER_URL = new URL("api/llm.php", window.location.href).href;

export function detect(key: string): ProviderId | null {
  const k = key.trim();
  for (const [id, p] of Object.entries(PROVIDERS) as [ProviderId, Provider][])
    if (p.prefix.some((x) => k.startsWith(x))) return id;
  return k.startsWith("sk-") ? "openai" : null;
}

/** Does this site have a server-side key? Returns its model label, or null. */
export async function serverStatus(): Promise<string | null> {
  try {
    const r = await fetch(`${SERVER_URL}/status`, { cache: "no-store" });
    if (!r.ok) return null;
    const j = await r.json();
    return j.enabled ? String(j.model) : null;
  } catch {
    return null;
  }
}

const openaiClient = async (c: LLMConfig) => new (await sdk.openai())({
  apiKey: c.provider === "server" ? "server" : c.key,
  baseURL: c.provider === "server" ? SERVER_URL : c.baseURL || PROVIDERS[c.provider].baseURL,
  dangerouslyAllowBrowser: true, maxRetries: 1, timeout: 45_000,
});

export async function listModels(c: LLMConfig): Promise<string[]> {
  const skip = /embed|tts|whisper|dall-e|image|audio|realtime|moderation|transcribe|search|guard|veo|imagen|aqa|live/i;
  try {
    let ids: string[] = [];
    if (c.provider === "server") return [];
    if (c.provider === "anthropic") {
      const a = new (await sdk.anthropic())({ apiKey: c.key, dangerouslyAllowBrowser: true, maxRetries: 0, timeout: 10_000 });
      for await (const m of a.models.list()) ids.push(m.id);
    } else if (c.provider === "gemini") {
      const g = new (await sdk.gemini())({ apiKey: c.key });
      for await (const m of await g.models.list()) {
        if (m.name && (m.supportedActions ?? ["generateContent"]).includes("generateContent"))
          ids.push(m.name.replace(/^models\//, ""));
      }
    } else {
      for await (const m of (await openaiClient(c)).models.list()) ids.push(m.id);
    }
    return ids.filter((i) => !skip.test(i)).sort();
  } catch {
    return [];
  }
}

// ── streaming ───────────────────────────────────────────────────────────────
/** Stream an answer. Errors become readable text, never exceptions. */
export async function* stream(c: LLMConfig, messages: Msg[], effort: "low" | "medium", maxTokens: number,
                              signal?: AbortSignal): AsyncGenerator<string> {
  try {
    if (c.provider === "anthropic") yield* anthropicStream(c, messages, effort, maxTokens, signal);
    else if (c.provider === "gemini") yield* geminiStream(c, messages, maxTokens, signal);
    else yield* openaiStream(c, messages, signal);
  } catch (e) {
    if (signal?.aborted) return;
    yield explainError(e);
  }
}

async function* anthropicStream(c: LLMConfig, messages: Msg[], effort: "low" | "medium", maxTokens: number,
                                signal?: AbortSignal) {
  const client = new (await sdk.anthropic())({ apiKey: c.key, dangerouslyAllowBrowser: true, maxRetries: 1, timeout: 60_000 });
  const s = client.beta.messages.stream({
    model: c.model, max_tokens: maxTokens, system: SYSTEM, messages,
    output_config: { effort },
    // A safety classifier can misread defensive network-attack analysis; on a
    // decline the API re-runs the request on a fallback model in the same call.
    betas: ["server-side-fallback-2026-07-01"], fallbacks: "default",
  }, { signal });
  for await (const ev of s)
    if (ev.type === "content_block_delta" && ev.delta.type === "text_delta") yield ev.delta.text;
  const final = await s.finalMessage();
  if (final.stop_reason === "refusal") yield "\n\n_The language model declined this request. The detector's verdict is unaffected._";
}

async function* geminiStream(c: LLMConfig, messages: Msg[], maxTokens: number, signal?: AbortSignal) {
  const g = new (await sdk.gemini())({ apiKey: c.key });
  const it = await g.models.generateContentStream({
    model: c.model,
    contents: messages.map((m) => ({ role: m.role === "assistant" ? "model" : "user", parts: [{ text: m.content }] })),
    config: { systemInstruction: SYSTEM, maxOutputTokens: maxTokens, abortSignal: signal },
  });
  let got = false;
  for await (const chunk of it) {
    const t = chunk.text;
    if (t) { got = true; yield t; }
  }
  if (!got) yield "_The model returned no text. Try a different model._";
}

async function* openaiStream(c: LLMConfig, messages: Msg[], signal?: AbortSignal) {
  const s = await (await openaiClient(c)).chat.completions.create({
    model: c.provider === "server" ? "server" : c.model, stream: true,
    messages: [{ role: "system", content: SYSTEM }, ...messages],
  }, { signal });
  let got = false;
  for await (const chunk of s) {
    const t = chunk.choices?.[0]?.delta?.content;
    if (t) { got = true; yield t; }
  }
  if (!got) yield "_The model returned no text. Try a different model._";
}

function explainError(e: unknown): string {
  const err = e as { status?: number; message?: string };
  const msg = String(err?.message ?? e);
  if (err?.status === 401 || /api key|api_key|unauthori[sz]ed|invalid.*key/i.test(msg))
    return "The API key was rejected. Check the key and the provider in Settings.";
  if (err?.status === 404 || /not found/i.test(msg))
    return "That model isn't available for this key. Pick another model in Settings.";
  if (err?.status === 429 || /rate|quota|exhausted/i.test(msg))
    return "Rate limited or out of free quota. Wait a minute, or switch provider in Settings.";
  if (err?.status === 503 || /overloaded|unavailable/i.test(msg))
    return "The model is overloaded right now. Try again, or pick a lighter model in Settings.";
  if (/fetch|network|cors/i.test(msg))
    return "Couldn't reach the provider. Check your connection, or the base URL for a custom provider.";
  return `The LLM request failed: ${msg.slice(0, 300)}`;
}
