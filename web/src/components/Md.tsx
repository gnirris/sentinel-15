// Minimal Markdown → React elements for LLM output. Never uses innerHTML, so
// anything an LLM writes (including HTML or script tags) renders as plain text.
import { Fragment, type ReactNode } from "react";

function inline(text: string, key: string): ReactNode[] {
  const out: ReactNode[] = [];
  const re = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|_[^_\s][^_]*_)/g;
  let last = 0, i = 0;
  for (const m of text.matchAll(re)) {
    if (m.index! > last) out.push(text.slice(last, m.index));
    const t = m[0], k = `${key}-${i++}`;
    if (t.startsWith("`")) out.push(<code key={k}>{t.slice(1, -1)}</code>);
    else if (t.startsWith("**")) out.push(<strong key={k}>{inline(t.slice(2, -2), k)}</strong>);
    else out.push(<em key={k}>{inline(t.slice(1, -1), k)}</em>);
    last = m.index! + t.length;
  }
  if (last < text.length) out.push(text.slice(last));
  return out;
}

export default function Md({ text, streaming = false }: { text: string; streaming?: boolean }) {
  const blocks: ReactNode[] = [];
  const lines = text.replace(/\r/g, "").split("\n");
  let list: { ordered: boolean; items: string[] } | null = null;
  let para: string[] = [];
  const flushPara = () => {
    if (para.length) blocks.push(<p key={`p${blocks.length}`}>{inline(para.join(" "), `p${blocks.length}`)}</p>);
    para = [];
  };
  const flushList = () => {
    if (!list) return;
    const items = list.items.map((it, j) => <li key={j}>{inline(it, `l${blocks.length}-${j}`)}</li>);
    blocks.push(list.ordered ? <ol key={`l${blocks.length}`}>{items}</ol> : <ul key={`l${blocks.length}`}>{items}</ul>);
    list = null;
  };
  for (const raw of lines) {
    const line = raw.trimEnd();
    const bullet = line.match(/^\s*[-*•]\s+(.*)$/), num = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const head = line.match(/^#{1,4}\s+(.*)$/);
    if (bullet || num) {
      flushPara();
      const ordered = !!num;
      if (!list || list.ordered !== ordered) { flushList(); list = { ordered, items: [] }; }
      list.items.push((bullet ?? num)![1]);
    } else if (head) {
      flushPara(); flushList();
      blocks.push(<h4 key={`h${blocks.length}`}>{inline(head[1], `h${blocks.length}`)}</h4>);
    } else if (!line.trim()) {
      flushPara(); flushList();
    } else {
      flushList(); para.push(line.trim());
    }
  }
  flushPara(); flushList();
  return <div className={`md${streaming ? " cursor" : ""}`}>{blocks.map((b, i) => <Fragment key={i}>{b}</Fragment>)}</div>;
}
