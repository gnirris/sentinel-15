// In-browser port of the two-stage detector (detector.py). Loads the trees
// exported by export_web.py and reproduces, exactly:
//   - the sklearn ColumnTransformer (one-hot with infrequent buckets, unknown -> all zero)
//   - XGBoost prediction (float32 comparisons, default direction for missing values)
//   - TreeSHAP (Lundberg et al., the same algorithm as xgboost pred_contribs)
// web/test/verify.ts checks all of it against Python on the 500 sample flows.

export type Flow = Record<string, string | number | null>;

interface Tree {
  l: number[]; r: number[]; f: number[]; d: (0 | 1 | boolean)[];
  t: ArrayLike<number>;   // split thresholds (internal nodes) / leaf values (leaves)
  c: ArrayLike<number>;   // node cover (sum of hessians), for TreeSHAP
}

interface Stage {
  n_cols: number;
  columns: string[];
  cat: Record<string, Record<string, number>>;
  num: Record<string, number>;
  trees: Tree[];
}

export interface ModelData {
  features: string[];
  categorical: string[];
  numeric: string[];
  binary: Stage & { base: number; source: string[] };
  /** Stage 2 arrives after the page is usable; null until then. */
  multi: (Stage & { base: number[]; tree_class: number[]; classes: string[] }) | null;
}

export interface Prediction {
  pAttack: number;
  intrusion: boolean;
  category: string;          // "Normal" when not an intrusion
  categoryConf: number;
}

export interface Reason { feature: string; shap: number; value: string | number | null }

/** Call once after loading model.json. XGBoost compares float32 inputs against
 * float32 thresholds; JSON decimals parse as float64, which sends values that sit
 * exactly on a split boundary down the wrong branch. Float32Array restores the
 * exact float32 values. */
function toFloat32(stage: Stage) {
  for (const t of stage.trees) {
    t.t = Float32Array.from(t.t);
    t.c = Float32Array.from(t.c);
  }
}

/** Stage 1 + shared fields (binary.json.gz). */
export function loadModel(json: unknown): ModelData {
  const m = json as ModelData;
  toFloat32(m.binary);
  m.multi = null;
  return m;
}

/** Add stage 2 (multi.json.gz) once it has downloaded. Returns a new object so React re-renders. */
export function withMulti(m: ModelData, json: unknown): ModelData {
  const multi = json as NonNullable<ModelData["multi"]>;
  toFloat32(multi);
  return { ...m, multi };
}

// ── preprocessing ──────────────────────────────────────────────────────────
export function clean(row: Flow, model: ModelData): Flow {
  // Same normalisation as train.clean(): trimmed lower-case categoricals,
  // numeric coercion, NaN -> 0 / "-".
  const out: Flow = { ...row };
  for (const c of model.categorical) {
    const v = row[c];
    out[c] = v === null || v === undefined || v === "" ? "-" : String(v).trim().toLowerCase();
  }
  for (const n of model.numeric) {
    const v = typeof row[n] === "number" ? (row[n] as number) : parseFloat(String(row[n] ?? ""));
    out[n] = Number.isFinite(v) ? v : 0;
  }
  return out;
}

export function missingColumns(header: string[], model: ModelData): string[] {
  const have = new Set(header.map((h) => h.replace(/^﻿/, "").trim()));
  return model.features.filter((f) => !have.has(f));
}

function encode(row: Flow, stage: Stage, model: ModelData): Float32Array {
  const x = new Float32Array(stage.n_cols);
  for (const c of model.categorical) {
    const col = stage.cat[c][String(row[c])];
    if (col !== undefined) x[col] = 1;   // unseen category: no column set (handle_unknown='ignore')
  }
  for (const n of model.numeric) x[stage.num[n]] = row[n] as number;   // Float32Array rounds like xgboost
  return x;
}

// ── trees ──────────────────────────────────────────────────────────────────
function leafValue(tree: Tree, x: Float32Array): number {
  let n = 0;
  while (tree.l[n] !== -1) {
    const v = x[tree.f[n]];
    n = Number.isNaN(v) ? (tree.d[n] ? tree.l[n] : tree.r[n]) : v < tree.t[n] ? tree.l[n] : tree.r[n];
  }
  return tree.t[n];
}

export function binaryMargin(model: ModelData, x: Float32Array): number {
  let m = model.binary.base;
  for (const t of model.binary.trees) m += leafValue(t, x);
  return m;
}

export function multiProba(model: ModelData, x: Float32Array): number[] {
  const mc = model.multi!;
  const m = mc.base.slice();
  mc.trees.forEach((t, i) => { m[mc.tree_class[i]] += leafValue(t, x); });
  const top = Math.max(...m);
  const e = m.map((v) => Math.exp(v - top));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / s);
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

export function predict(model: ModelData, raw: Flow, threshold = 0.5): Prediction {
  const row = clean(raw, model);
  const pAttack = sigmoid(binaryMargin(model, encode(row, model.binary, model)));
  if (pAttack < threshold) return { pAttack, intrusion: false, category: "Normal", categoryConf: 1 - pAttack };
  // Stage 2 still downloading: the alert stands, the attack type comes a moment later.
  if (!model.multi) return { pAttack, intrusion: true, category: "Attack", categoryConf: NaN };
  const p = multiProba(model, encode(row, model.multi, model));
  const k = p.indexOf(Math.max(...p));
  return { pAttack, intrusion: true, category: model.multi.classes[k], categoryConf: p[k] };
}

// ── TreeSHAP ───────────────────────────────────────────────────────────────
interface PathEl { f: number; zero: number; one: number; w: number }

function extend(path: PathEl[], depth: number, zero: number, one: number, f: number) {
  path[depth] = { f, zero, one, w: depth === 0 ? 1 : 0 };
  for (let i = depth - 1; i >= 0; i--) {
    path[i + 1].w += (one * path[i].w * (i + 1)) / (depth + 1);
    path[i].w = (zero * path[i].w * (depth - i)) / (depth + 1);
  }
}

function unwind(path: PathEl[], depth: number, idx: number) {
  const { one, zero } = path[idx];
  let next = path[depth].w;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = path[i].w;
      path[i].w = (next * (depth + 1)) / ((i + 1) * one);
      next = tmp - (path[i].w * zero * (depth - i)) / (depth + 1);
    } else {
      path[i].w = (path[i].w * (depth + 1)) / (zero * (depth - i));
    }
  }
  for (let i = idx; i < depth; i++) {
    path[i].f = path[i + 1].f; path[i].zero = path[i + 1].zero; path[i].one = path[i + 1].one;
  }
}

function unwoundSum(path: PathEl[], depth: number, idx: number): number {
  const { one, zero } = path[idx];
  let next = path[depth].w, total = 0;
  for (let i = depth - 1; i >= 0; i--) {
    if (one !== 0) {
      const tmp = (next * (depth + 1)) / ((i + 1) * one);
      total += tmp;
      next = path[i].w - tmp * zero * ((depth - i) / (depth + 1));
    } else if (zero !== 0) {
      total += path[i].w / zero / ((depth - i) / (depth + 1));
    }
  }
  return total;
}

function shapTree(tree: Tree, x: Float32Array, phi: Float64Array) {
  const rec = (node: number, parent: PathEl[], depth: number, zero: number, one: number, f: number) => {
    const path = parent.slice(0, depth).map((p) => ({ ...p }));
    extend(path, depth, zero, one, f);
    if (tree.l[node] === -1) {
      for (let i = 1; i <= depth; i++) {
        const el = path[i];
        phi[el.f] += unwoundSum(path, depth, i) * (el.one - el.zero) * tree.t[node];
      }
      return;
    }
    const split = tree.f[node], v = x[split];
    const goLeft = Number.isNaN(v) ? !!tree.d[node] : v < tree.t[node];
    const hot = goLeft ? tree.l[node] : tree.r[node];
    const cold = goLeft ? tree.r[node] : tree.l[node];
    const w = tree.c[node];
    let inZero = 1, inOne = 1, d = depth;
    let k = 0;
    while (k <= d && path[k].f !== split) k++;
    if (k <= d) {
      inZero = path[k].zero; inOne = path[k].one;
      unwind(path, d, k);
      d -= 1;
    }
    rec(hot, path, d + 1, (tree.c[hot] / w) * inZero, inOne, split);
    rec(cold, path, d + 1, (tree.c[cold] / w) * inZero, 0, split);
  };
  rec(0, [], 0, 1, 1, -1);
}

/** SHAP values of stage 1 per encoded column, in log-odds of "attack". */
export function shapColumns(model: ModelData, raw: Flow): Float64Array {
  const x = encode(clean(raw, model), model.binary, model);
  const phi = new Float64Array(model.binary.n_cols);
  for (const t of model.binary.trees) shapTree(t, x, phi);
  return phi;
}

/** Top reasons, with one-hot columns folded back into their raw field (like detector.explain). */
export function explain(model: ModelData, raw: Flow, top = 5): Reason[] {
  const phi = shapColumns(model, raw);
  const sums = new Map<string, number>();
  model.binary.source.forEach((src, i) => sums.set(src, (sums.get(src) ?? 0) + phi[i]));
  const row = clean(raw, model);
  return [...sums.entries()]
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    .slice(0, top)
    .map(([feature, shap]) => ({ feature, shap, value: row[feature] }));
}
