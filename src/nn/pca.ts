/**
 * Two-dimensional PCA and 2-D Procrustes alignment, for the letter map.
 *
 * Pure maths, no DOM: the worker runs the PCA, the UI aligns each new frame
 * to the previous one so the map turns smoothly instead of flipping.
 */

export interface Pca2 {
  /** [n, 2] coordinates, row-major (x0, y0, x1, y1, …). */
  coords: Float32Array;
  /** Share of the total variance explained by each of the two axes. */
  explained: [number, number];
}

/**
 * Project the given rows of a [*, d] matrix onto its top two principal
 * components. Deterministic (fixed start vectors), so the same weights give
 * the same picture.
 */
export function pca2(data: Float32Array, d: number, rows: readonly number[]): Pca2 {
  const n = rows.length;
  const mean = new Float64Array(d);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += data[r * d + j] / n;
  const X = new Float64Array(n * d);
  rows.forEach((r, i) => { for (let j = 0; j < d; j++) X[i * d + j] = data[r * d + j] - mean[j]; });
  // Covariance C = Xᵀ X / n  (d × d).
  const C = new Float64Array(d * d);
  for (let i = 0; i < n; i++)
    for (let a = 0; a < d; a++) {
      const xa = X[i * d + a];
      if (xa === 0) continue;
      for (let b = a; b < d; b++) C[a * d + b] += (xa * X[i * d + b]) / n;
    }
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a * d + b] = C[b * d + a];
  let trace = 0;
  for (let a = 0; a < d; a++) trace += C[a * d + a];

  const v1 = topEigenvector(C, d, null);
  const l1 = rayleigh(C, d, v1);
  const v2 = topEigenvector(C, d, v1);
  const l2 = rayleigh(C, d, v2);

  const coords = new Float32Array(n * 2);
  for (let i = 0; i < n; i++) {
    let x = 0, y = 0;
    for (let j = 0; j < d; j++) { x += X[i * d + j] * v1[j]; y += X[i * d + j] * v2[j]; }
    coords[i * 2] = x;
    coords[i * 2 + 1] = y;
  }
  return { coords, explained: trace > 0 ? [l1 / trace, l2 / trace] : [0, 0] };
}

/** Power iteration; `orth` (if given) is projected out every step. */
function topEigenvector(C: Float64Array, d: number, orth: Float64Array | null): Float64Array {
  const v = new Float64Array(d);
  for (let j = 0; j < d; j++) v[j] = 1 + ((j * 7919) % 13) / 13; // fixed, not aligned with any axis
  const next = new Float64Array(d);
  for (let it = 0; it < 300; it++) {
    if (orth) removeComponent(v, orth);
    normalise(v);
    next.fill(0);
    for (let a = 0; a < d; a++) {
      let s = 0;
      for (let b = 0; b < d; b++) s += C[a * d + b] * v[b];
      next[a] = s;
    }
    if (orth) removeComponent(next, orth);
    if (normalise(next) === 0) break;
    let diff = 0;
    for (let j = 0; j < d; j++) diff += Math.abs(next[j] - v[j]);
    v.set(next);
    if (diff < 1e-10) break;
  }
  // Sign convention: the largest component is positive.
  let big = 0;
  for (let j = 1; j < d; j++) if (Math.abs(v[j]) > Math.abs(v[big])) big = j;
  if (v[big] < 0) for (let j = 0; j < d; j++) v[j] = -v[j];
  return v;
}

function removeComponent(v: Float64Array, u: Float64Array): void {
  let dot = 0;
  for (let j = 0; j < v.length; j++) dot += v[j] * u[j];
  for (let j = 0; j < v.length; j++) v[j] -= dot * u[j];
}

function normalise(v: Float64Array): number {
  let s = 0;
  for (let j = 0; j < v.length; j++) s += v[j] * v[j];
  const n = Math.sqrt(s);
  if (n > 0) for (let j = 0; j < v.length; j++) v[j] /= n;
  return n;
}

function rayleigh(C: Float64Array, d: number, v: Float64Array): number {
  let s = 0;
  for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) s += v[a] * C[a * d + b] * v[b];
  return s;
}

/**
 * Rotate (and if that fits better, mirror) `next` so it lines up with `prev`
 * as well as possible (orthogonal Procrustes in 2-D). Both are [n, 2] with
 * the same points in the same order. Scale is kept: the map may grow as the
 * letters spread out.
 */
export function alignTo(prev: Float32Array, next: Float32Array): Float32Array {
  if (prev.length !== next.length || next.length === 0) return next;
  const n = next.length / 2;
  const cp = centroid(prev), cn = centroid(next);
  let best: Float32Array = next;
  let bestErr = Infinity;
  for (const mirror of [1, -1]) {
    // Optimal rotation angle for 2-D point sets: atan2(Σ cross, Σ dot).
    let sDot = 0, sCross = 0;
    for (let i = 0; i < n; i++) {
      const qx = next[2 * i] - cn[0], qy = mirror * (next[2 * i + 1] - cn[1]);
      const px = prev[2 * i] - cp[0], py = prev[2 * i + 1] - cp[1];
      sDot += qx * px + qy * py;
      sCross += qx * py - qy * px;
    }
    const th = Math.atan2(sCross, sDot);
    const c = Math.cos(th), s = Math.sin(th);
    const out = new Float32Array(next.length);
    let err = 0;
    for (let i = 0; i < n; i++) {
      const qx = next[2 * i] - cn[0], qy = mirror * (next[2 * i + 1] - cn[1]);
      out[2 * i] = c * qx - s * qy + cn[0];
      out[2 * i + 1] = s * qx + c * qy + cn[1];
      err += (out[2 * i] - cn[0] - (prev[2 * i] - cp[0])) ** 2 + (out[2 * i + 1] - cn[1] - (prev[2 * i + 1] - cp[1])) ** 2;
    }
    if (err < bestErr) { bestErr = err; best = out; }
  }
  return best;
}

function centroid(p: Float32Array): [number, number] {
  const n = p.length / 2;
  let x = 0, y = 0;
  for (let i = 0; i < n; i++) { x += p[2 * i]; y += p[2 * i + 1]; }
  return [x / n, y / n];
}

/**
 * For each of the given rows of a [*, d] matrix, the `k` other rows that point
 * the most the same way, using all `d` numbers, not the flat 2-D picture.
 * "Same way" is cosine similarity measured from the average candidate row,
 * so what all letters share does not count. Only rows with `candidate[i]`
 * can be neighbours. Indices refer to positions in `rows`; rows that are not
 * candidates get no list.
 */
export function nearestRows(data: Float32Array, d: number, rows: readonly number[], candidate: readonly boolean[], k: number): number[][] {
  const n = rows.length;
  const mean = new Float64Array(d);
  const m = candidate.filter(Boolean).length || 1;
  rows.forEach((r, i) => { if (candidate[i]) for (let j = 0; j < d; j++) mean[j] += data[r * d + j] / m; });
  const X = rows.map((r) => Float64Array.from({ length: d }, (_, j) => data[r * d + j] - mean[j]));
  const norm = X.map((x) => Math.sqrt(x.reduce((s, q) => s + q * q, 0)) || 1);
  const cos = (a: number, b: number) => {
    let s = 0;
    for (let j = 0; j < d; j++) s += X[a][j] * X[b][j];
    return s / (norm[a] * norm[b]);
  };
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    if (!candidate[i]) { out.push([]); continue; }
    const others: [number, number][] = [];
    for (let j = 0; j < n; j++) if (j !== i && candidate[j]) others.push([j, cos(i, j)]);
    others.sort((a, b) => b[1] - a[1]);
    out.push(others.slice(0, k).map(([j]) => j));
  }
  return out;
}
