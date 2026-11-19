// Small hand-drawn SVG charts. Every chart scales to its container (viewBox)
// and carries a text summary for screen readers.
import type { Reason } from "../lib/model";

const C = { ink: "#15191E", ink2: "#4A5560", rule: "#D3DAE0", grid: "#E4E9ED", blue: "#2462A8",
  green: "#2F8F46", orange: "#E07020", red: "#C8102E", panel: "#FFFFFF" };

export const CAT_COLORS: Record<string, string> = {
  Normal: C.green, Generic: C.red, Exploits: C.orange, Fuzzers: "#B8860B", DoS: "#7B4B2A",
  Reconnaissance: C.blue, Analysis: "#6A5ACD", Backdoor: "#A0306E", Shellcode: "#2A9D8F", Worms: "#5C6770",
};

const fmtVal = (v: unknown) => (typeof v === "number" ? String(Number(v.toPrecision(4))) : String(v));

/** Diverging bars: red pushes toward attack, green toward normal. */
export function ShapBars({ reasons }: { reasons: Reason[] }) {
  const W = 460, rowH = 36, labelW = 190, H = reasons.length * rowH + 42;
  const max = Math.max(...reasons.map((r) => Math.abs(r.shap)), 0.01);
  const plotW = W - labelW - 16, zero = labelW + plotW / 2;
  const x = (v: number) => (v / max) * (plotW / 2 - 46);
  const summary = reasons.map((r) => `${r.feature} ${r.shap > 0 ? "toward attack" : "toward normal"} ${Math.abs(r.shap).toFixed(2)}`).join("; ");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`Top reasons: ${summary}`}>
      <line x1={zero} x2={zero} y1={0} y2={reasons.length * rowH} stroke={C.ink2} />
      {reasons.map((r, i) => {
        const y = i * rowH, w = x(r.shap);
        return (
          <g key={r.feature}>
            <text x={labelW - 8} y={y + rowH / 2 + 5} textAnchor="end" fontSize="14">
              <tspan fontWeight="650" fill={C.ink}>{r.feature}</tspan> = {fmtVal(r.value)}
            </text>
            <rect x={w >= 0 ? zero : zero + w} y={y + 7} width={Math.max(Math.abs(w), 1.5)} height={rowH - 14}
                  fill={r.shap > 0 ? C.red : C.green} />
            <text x={w >= 0 ? zero + w + 5 : zero + w - 5} y={y + rowH / 2 + 5} fontSize="12"
                  textAnchor={w >= 0 ? "start" : "end"}>{r.shap > 0 ? "+" : ""}{r.shap.toFixed(2)}</text>
          </g>
        );
      })}
      <text x={zero - 8} y={H - 8} textAnchor="end" fontSize="12">← toward normal</text>
      <text x={zero + 8} y={H - 8} fontSize="12">toward attack →</text>
    </svg>
  );
}

/** Horizontal bars, one per label. */
export function HBars({ data, color, max, fmt = (v) => String(v), label }:
  { data: [string, number][]; color: (k: string, v: number) => string; max?: number; fmt?: (v: number) => string; label: string }) {
  const W = 520, rowH = 26, labelW = 130, H = data.length * rowH + 6;
  const top = max ?? Math.max(...data.map((d) => d[1]), 1e-9);
  const plotW = W - labelW - 60;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${label}: ${data.map(([k, v]) => `${k} ${fmt(v)}`).join(", ")}`}>
      {data.map(([k, v], i) => (
        <g key={k}>
          <text x={labelW - 8} y={i * rowH + 18} textAnchor="end" fontSize="13">{k}</text>
          <rect x={labelW} y={i * rowH + 5} width={Math.max((v / top) * plotW, 1)} height={rowH - 10} fill={color(k, v)} />
          <text x={labelW + (v / top) * plotW + 6} y={i * rowH + 18} fontSize="12" fill={C.ink}>{fmt(v)}</text>
        </g>
      ))}
    </svg>
  );
}

/** Histogram of probabilities with the alert threshold marked. */
export function Histogram({ values, threshold, bins = 40 }: { values: number[]; threshold: number; bins?: number }) {
  const W = 520, H = 230, pad = { l: 44, r: 10, t: 18, b: 34 };
  const counts = new Array(bins).fill(0);
  for (const v of values) counts[Math.min(bins - 1, Math.floor(v * bins))]++;
  const top = Math.max(...counts, 1), pw = W - pad.l - pad.r, ph = H - pad.t - pad.b, bw = pw / bins;
  const y = (c: number) => pad.t + ph - (c / top) * ph, tx = pad.l + threshold * pw;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img"
         aria-label={`${values.length} flows; ${values.filter((v) => v >= threshold).length} above the threshold of ${threshold}`}>
      {[0, .5, 1].map((f) => (
        <g key={f}>
          <line x1={pad.l} x2={W - pad.r} y1={y(top * f)} y2={y(top * f)} stroke={C.grid} />
          <text x={pad.l - 6} y={y(top * f) + 4} textAnchor="end" fontSize="11">{Math.round(top * f)}</text>
        </g>
      ))}
      {counts.map((c, i) => (
        <rect key={i} x={pad.l + i * bw + .5} y={y(c)} width={bw - 1} height={pad.t + ph - y(c)}
              fill={i / bins >= threshold ? C.red : C.blue} />
      ))}
      <line x1={tx} x2={tx} y1={pad.t - 6} y2={pad.t + ph} stroke={C.orange} strokeWidth="2" strokeDasharray="4 3" />
      <text x={tx + 5} y={pad.t + 4} fontSize="12" fill={C.ink}>alert threshold</text>
      {[0, .25, .5, .75, 1].map((f) => (
        <text key={f} x={pad.l + f * pw} y={H - 14} textAnchor="middle" fontSize="11">{f}</text>
      ))}
      <text x={pad.l + pw / 2} y={H} textAnchor="middle" fontSize="12">chance of attack</text>
    </svg>
  );
}

/** Confusion matrix, rows = truth, shade = share of the row. */
export function Heatmap({ labels, matrix }: { labels: string[]; matrix: number[][] }) {
  const n = labels.length, cell = 46, left = 118, top = 96, W = left + n * cell + 4, H = top + n * cell + 4;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="Confusion matrix of true versus predicted attack type">
      {labels.map((l, j) => (
        <text key={`c${l}`} transform={`translate(${left + j * cell + cell / 2 + 4},${top - 8}) rotate(-50)`} fontSize="12">{l}</text>
      ))}
      {matrix.map((row, i) => {
        const total = row.reduce((a, b) => a + b, 0) || 1;
        return (
          <g key={labels[i]}>
            <text x={left - 8} y={top + i * cell + cell / 2 + 4} textAnchor="end" fontSize="12">{labels[i]}</text>
            {row.map((v, j) => {
              const share = v / total;
              return (
                <g key={j}>
                  <rect x={left + j * cell} y={top + i * cell} width={cell - 2} height={cell - 2}
                        fill={C.blue} fillOpacity={0.06 + share * 0.94} stroke={i === j ? C.ink : "none"} />
                  <text x={left + j * cell + cell / 2 - 1} y={top + i * cell + cell / 2 + 4} textAnchor="middle"
                        fontSize="11" style={{ fill: share > .5 ? "#fff" : C.ink }}>{v}</text>
                  <title>{`true ${labels[i]}, predicted ${labels[j]}: ${v}`}</title>
                </g>
              );
            })}
          </g>
        );
      })}
    </svg>
  );
}

/** Two error curves against the alert threshold. */
export function TradeOff({ curve, threshold }:
  { curve: { threshold: number; false_alarms: number; missed_attacks: number }[]; threshold: number }) {
  const W = 760, H = 270, pad = { l: 52, r: 16, t: 30, b: 36 };
  const top = Math.max(...curve.flatMap((c) => [c.false_alarms, c.missed_attacks]), 1);
  const pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
  const x = (t: number) => pad.l + t * pw, y = (v: number) => pad.t + ph - (v / top) * ph;
  const path = (k: "false_alarms" | "missed_attacks") => curve.map((c, i) => `${i ? "L" : "M"}${x(c.threshold)},${y(c[k])}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label="False alarms fall and missed attacks rise as the threshold increases">
      {[0, .25, .5, .75, 1].map((f) => (
        <g key={f}>
          <line x1={pad.l} x2={W - pad.r} y1={y(top * f)} y2={y(top * f)} stroke={C.grid} />
          <text x={pad.l - 6} y={y(top * f) + 4} textAnchor="end" fontSize="11">{Math.round(top * f)}</text>
          <text x={x(f)} y={H - 16} textAnchor="middle" fontSize="11">{f}</text>
        </g>
      ))}
      <path d={path("false_alarms")} fill="none" stroke={C.orange} strokeWidth="3" />
      <path d={path("missed_attacks")} fill="none" stroke={C.red} strokeWidth="3" />
      <line x1={x(threshold)} x2={x(threshold)} y1={pad.t} y2={pad.t + ph} stroke={C.blue} strokeWidth="2" strokeDasharray="4 3" />
      <text x={x(threshold) + 6} y={pad.t + 12} fontSize="12" fill={C.ink}>your threshold, {threshold.toFixed(2)}</text>
      <rect x={pad.l} y={6} width="14" height="4" fill={C.orange} /><text x={pad.l + 20} y={12} fontSize="12" fill={C.ink}>false alarms</text>
      <rect x={pad.l + 120} y={6} width="14" height="4" fill={C.red} /><text x={pad.l + 140} y={12} fontSize="12" fill={C.ink}>missed attacks</text>
      <text x={pad.l + pw / 2} y={H} textAnchor="middle" fontSize="12">alert threshold</text>
    </svg>
  );
}
