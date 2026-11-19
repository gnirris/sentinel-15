// Checks the JavaScript port against Python on all 500 sample flows.
// Run: node --experimental-strip-types test/verify.ts   (from web/)
import { readFileSync } from "node:fs";
import { gunzipSync } from "node:zlib";
import {
  binaryMargin, clean, explain, loadModel, multiProba, predict, shapColumns, withMulti, type Flow, type ModelData,
} from "../src/lib/model.ts";

const load = (p: string) => JSON.parse(readFileSync(new URL(p, import.meta.url), "utf8"));
const loadGz = (p: string) => JSON.parse(gunzipSync(readFileSync(new URL(p, import.meta.url))).toString("utf8"));
const model: ModelData = withMulti(loadModel(loadGz("../public/model/binary.json.gz")), loadGz("../public/model/multi.json.gz"));
const flows: Flow[] = loadGz("../public/model/flows.json.gz");
const ref = load("./reference.json");

// encode() is internal; reach it through the exported pieces
const enc = (raw: Flow, stage: "binary" | "multi") => {
  const s = model[stage]!, row = clean(raw, model), x = new Float32Array(s.n_cols);
  for (const c of model.categorical) { const col = s.cat[c][String(row[c])]; if (col !== undefined) x[col] = 1; }
  for (const n of model.numeric) x[s.num[n]] = row[n] as number;
  return x;
};

let worstMargin = 0, worstProb = 0, worstShap = 0, flips = 0;
flows.forEach((f, i) => {
  worstMargin = Math.max(worstMargin, Math.abs(binaryMargin(model, enc(f, "binary")) - ref.binary_margin[i]));
  const p = multiProba(model, enc(f, "multi"));
  p.forEach((v, k) => { worstProb = Math.max(worstProb, Math.abs(v - ref.multi_prob[i][k])); });
  const phi = shapColumns(model, f);
  phi.forEach((v, k) => { worstShap = Math.max(worstShap, Math.abs(v - ref.binary_contribs[i][k])); });
  const py = 1 / (1 + Math.exp(-ref.binary_margin[i])) >= 0.5;
  if (predict(model, f).intrusion !== py) flips++;
});

let topMismatch = 0;
ref.explain_top5.forEach((want: [string, number][], j: number) => {
  const got = explain(model, flows[j * 25]);
  want.forEach(([feat, s], k) => {
    if (got[k].feature !== feat || Math.abs(got[k].shap - s) > 1e-4) topMismatch++;
  });
});

const t0 = performance.now();
for (const f of flows.slice(0, 100)) { predict(model, f); explain(model, f); }
const ms = (performance.now() - t0) / 100;

console.log(`stage-1 margin   max |diff| ${worstMargin.toExponential(2)}`);
console.log(`stage-2 probs    max |diff| ${worstProb.toExponential(2)}`);
console.log(`SHAP per column  max |diff| ${worstShap.toExponential(2)}`);
console.log(`verdict flips vs Python: ${flips}/500, top-5 reason mismatches: ${topMismatch}`);
console.log(`speed: ${ms.toFixed(1)} ms per flow (predict + explain)`);
const ok = worstMargin < 1e-4 && worstProb < 1e-5 && worstShap < 1e-4 && flips === 0 && topMismatch === 0;
console.log(ok ? "PASS" : "FAIL");
process.exit(ok ? 0 : 1);
