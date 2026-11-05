/**
 * The operations the models are built from. Each op:
 *   1. computes its output (the "forward" pass), and
 *   2. records how to pass gradients back to its inputs (the "backward" pass).
 *
 * All loops are plain JavaScript so they are easy to read.
 */
import { recordOp } from './autograd';
import { Tensor, assertShape, numel, sameShape } from './tensor';

// ---------------------------------------------------------------------------
// Raw matrix-multiply kernels on flat arrays (C += ...). Used by matmul and bmm.
// ---------------------------------------------------------------------------

/** C[n×m] += A[n×k] · B[k×m] */
export function mmNN(
  A: Float32Array, ao: number, B: Float32Array, bo: number, C: Float32Array, co: number,
  n: number, k: number, m: number,
): void {
  for (let i = 0; i < n; i++) {
    const aRow = ao + i * k;
    const cRow = co + i * m;
    for (let p = 0; p < k; p++) {
      const a = A[aRow + p];
      if (a === 0) continue;
      const bRow = bo + p * m;
      for (let j = 0; j < m; j++) C[cRow + j] += a * B[bRow + j];
    }
  }
}

/** C[n×m] += A[n×k] · B[m×k]ᵀ */
export function mmNT(
  A: Float32Array, ao: number, B: Float32Array, bo: number, C: Float32Array, co: number,
  n: number, k: number, m: number,
): void {
  for (let i = 0; i < n; i++) {
    const aRow = ao + i * k;
    const cRow = co + i * m;
    for (let j = 0; j < m; j++) {
      const bRow = bo + j * k;
      let s = 0;
      for (let p = 0; p < k; p++) s += A[aRow + p] * B[bRow + p];
      C[cRow + j] += s;
    }
  }
}

/** C[k×m] += A[n×k]ᵀ · B[n×m] */
export function mmTN(
  A: Float32Array, ao: number, B: Float32Array, bo: number, C: Float32Array, co: number,
  n: number, k: number, m: number,
): void {
  for (let i = 0; i < n; i++) {
    const aRow = ao + i * k;
    const bRow = bo + i * m;
    for (let p = 0; p < k; p++) {
      const a = A[aRow + p];
      if (a === 0) continue;
      const cRow = co + p * m;
      for (let j = 0; j < m; j++) C[cRow + j] += a * B[bRow + j];
    }
  }
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/**
 * Matrix product. a: [N, K]. b: [K, M], or [M, K] when `transB` is true.
 * Result: [N, M].
 */
export function matmul(a: Tensor, b: Tensor, transB = false): Tensor {
  assertShape(a, 2, 'matmul a');
  assertShape(b, 2, 'matmul b');
  const [N, K] = a.shape;
  const M = transB ? b.shape[0] : b.shape[1];
  const bK = transB ? b.shape[1] : b.shape[0];
  if (bK !== K) throw new Error(`matmul: [${a.shape}] x [${b.shape}]${transB ? 'ᵀ' : ''}`);
  const out = Tensor.zeros([N, M]);
  if (transB) mmNT(a.data, 0, b.data, 0, out.data, 0, N, K, M);
  else mmNN(a.data, 0, b.data, 0, out.data, 0, N, K, M);

  recordOp([a, b], out, () => {
    const g = out.grad;
    if (!g) return;
    if (transB) {
      if (a.requiresGrad) mmNN(g, 0, b.data, 0, a.ensureGrad(), 0, N, M, K);
      if (b.requiresGrad) mmTN(g, 0, a.data, 0, b.ensureGrad(), 0, N, M, K);
    } else {
      if (a.requiresGrad) mmNT(g, 0, b.data, 0, a.ensureGrad(), 0, N, M, K);
      if (b.requiresGrad) mmTN(a.data, 0, g, 0, b.ensureGrad(), 0, N, K, M);
    }
  });
  return out;
}

/**
 * Batched matrix product. a: [B, N, K]. b: [B, K, M], or [B, M, K] when `transB`.
 * Result: [B, N, M].
 */
export function bmm(a: Tensor, b: Tensor, transB = false): Tensor {
  assertShape(a, 3, 'bmm a');
  assertShape(b, 3, 'bmm b');
  const [Bt, N, K] = a.shape;
  const M = transB ? b.shape[1] : b.shape[2];
  const bK = transB ? b.shape[2] : b.shape[1];
  if (b.shape[0] !== Bt || bK !== K) throw new Error(`bmm: [${a.shape}] x [${b.shape}]`);
  const out = Tensor.zeros([Bt, N, M]);
  const aS = N * K, bS = K * M, cS = N * M;
  for (let s = 0; s < Bt; s++) {
    if (transB) mmNT(a.data, s * aS, b.data, s * bS, out.data, s * cS, N, K, M);
    else mmNN(a.data, s * aS, b.data, s * bS, out.data, s * cS, N, K, M);
  }

  recordOp([a, b], out, () => {
    const g = out.grad;
    if (!g) return;
    const ga = a.requiresGrad ? a.ensureGrad() : null;
    const gb = b.requiresGrad ? b.ensureGrad() : null;
    for (let s = 0; s < Bt; s++) {
      if (transB) {
        if (ga) mmNN(g, s * cS, b.data, s * bS, ga, s * aS, N, M, K);
        if (gb) mmTN(g, s * cS, a.data, s * aS, gb, s * bS, N, M, K);
      } else {
        if (ga) mmNT(g, s * cS, b.data, s * bS, ga, s * aS, N, M, K);
        if (gb) mmTN(a.data, s * aS, g, s * cS, gb, s * bS, N, K, M);
      }
    }
  });
  return out;
}

/** Swap the last two dimensions: [B, N, M] -> [B, M, N]. */
export function transpose(x: Tensor): Tensor {
  assertShape(x, 3, 'transpose');
  const [B, N, M] = x.shape;
  const out = Tensor.zeros([B, M, N]);
  for (let b = 0; b < B; b++) {
    const o = b * N * M;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < M; j++) out.data[o + j * N + i] = x.data[o + i * M + j];
    }
  }
  recordOp([x], out, () => {
    const g = out.grad;
    if (!g) return;
    const gx = x.ensureGrad();
    for (let b = 0; b < B; b++) {
      const o = b * N * M;
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < M; j++) gx[o + i * M + j] += g[o + j * N + i];
      }
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// Element-wise
// ---------------------------------------------------------------------------

/**
 * a + b. Either the same shape, or b is 1-D and matches a's last dimension
 * (adding a bias to every row).
 */
export function add(a: Tensor, b: Tensor): Tensor {
  const out = Tensor.zeros(a.shape.slice());
  const n = a.size;
  if (sameShape(a.shape, b.shape)) {
    for (let i = 0; i < n; i++) out.data[i] = a.data[i] + b.data[i];
    recordOp([a, b], out, () => {
      const g = out.grad;
      if (!g) return;
      if (a.requiresGrad) { const ga = a.ensureGrad(); for (let i = 0; i < n; i++) ga[i] += g[i]; }
      if (b.requiresGrad) { const gb = b.ensureGrad(); for (let i = 0; i < n; i++) gb[i] += g[i]; }
    });
    return out;
  }
  const m = b.size;
  if (b.shape.length !== 1 || a.shape[a.shape.length - 1] !== m) {
    throw new Error(`add: cannot broadcast [${b.shape}] onto [${a.shape}]`);
  }
  for (let i = 0; i < n; i++) out.data[i] = a.data[i] + b.data[i % m];
  recordOp([a, b], out, () => {
    const g = out.grad;
    if (!g) return;
    if (a.requiresGrad) { const ga = a.ensureGrad(); for (let i = 0; i < n; i++) ga[i] += g[i]; }
    if (b.requiresGrad) { const gb = b.ensureGrad(); for (let i = 0; i < n; i++) gb[i % m] += g[i]; }
  });
  return out;
}

/** Element-wise product of two same-shaped tensors. */
export function mul(a: Tensor, b: Tensor): Tensor {
  if (!sameShape(a.shape, b.shape)) throw new Error(`mul: [${a.shape}] vs [${b.shape}]`);
  const n = a.size;
  const out = Tensor.zeros(a.shape.slice());
  for (let i = 0; i < n; i++) out.data[i] = a.data[i] * b.data[i];
  recordOp([a, b], out, () => {
    const g = out.grad;
    if (!g) return;
    if (a.requiresGrad) { const ga = a.ensureGrad(); for (let i = 0; i < n; i++) ga[i] += g[i] * b.data[i]; }
    if (b.requiresGrad) { const gb = b.ensureGrad(); for (let i = 0; i < n; i++) gb[i] += g[i] * a.data[i]; }
  });
  return out;
}

/** Multiply every element by a constant. */
export function scale(a: Tensor, s: number): Tensor {
  const n = a.size;
  const out = Tensor.zeros(a.shape.slice());
  for (let i = 0; i < n; i++) out.data[i] = a.data[i] * s;
  recordOp([a], out, () => {
    const g = out.grad;
    if (!g) return;
    const ga = a.ensureGrad();
    for (let i = 0; i < n; i++) ga[i] += g[i] * s;
  });
  return out;
}

/** Sum of all elements -> 1-element tensor. */
export function sum(a: Tensor): Tensor {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i];
  const out = Tensor.from([s], [1]);
  recordOp([a], out, () => {
    const g = out.grad;
    if (!g) return;
    const ga = a.ensureGrad();
    for (let i = 0; i < a.size; i++) ga[i] += g[0];
  });
  return out;
}

const GELU_C = Math.sqrt(2 / Math.PI);

/** GELU activation (tanh approximation), a smooth version of max(0, x). */
export function gelu(x: Tensor): Tensor {
  const n = x.size;
  const out = Tensor.zeros(x.shape.slice());
  const th = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const v = x.data[i];
    const t = Math.tanh(GELU_C * (v + 0.044715 * v * v * v));
    th[i] = t;
    out.data[i] = 0.5 * v * (1 + t);
  }
  recordOp([x], out, () => {
    const g = out.grad;
    if (!g) return;
    const gx = x.ensureGrad();
    for (let i = 0; i < n; i++) {
      const v = x.data[i];
      const t = th[i];
      const dInner = GELU_C * (1 + 3 * 0.044715 * v * v);
      gx[i] += g[i] * (0.5 * (1 + t) + 0.5 * v * (1 - t * t) * dInner);
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// Lookup, normalisation, softmax
// ---------------------------------------------------------------------------

/** Embedding lookup: picks rows `ids` from table [V, D] -> [ids.length, D]. */
export function gather(table: Tensor, ids: Int32Array): Tensor {
  assertShape(table, 2, 'gather');
  const [V, D] = table.shape;
  const out = Tensor.zeros([ids.length, D]);
  for (let i = 0; i < ids.length; i++) {
    const id = ids[i];
    if (id < 0 || id >= V) throw new Error(`gather: id ${id} out of range ${V}`);
    out.data.set(table.data.subarray(id * D, id * D + D), i * D);
  }
  recordOp([table], out, () => {
    const g = out.grad;
    if (!g) return;
    const gt = table.ensureGrad();
    for (let i = 0; i < ids.length; i++) {
      const o = ids[i] * D;
      for (let j = 0; j < D; j++) gt[o + j] += g[i * D + j];
    }
  });
  return out;
}

/**
 * Layer normalisation over the last dimension: each row is shifted to mean 0
 * and scaled to variance 1, then multiplied by `gamma` and shifted by `beta`.
 */
export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor, eps = 1e-5): Tensor {
  const D = x.shape[x.shape.length - 1];
  const R = x.size / D;
  const out = Tensor.zeros(x.shape.slice());
  const xhat = new Float32Array(x.size);
  const rstd = new Float32Array(R);
  for (let r = 0; r < R; r++) {
    const o = r * D;
    let mean = 0;
    for (let j = 0; j < D; j++) mean += x.data[o + j];
    mean /= D;
    let v = 0;
    for (let j = 0; j < D; j++) { const d = x.data[o + j] - mean; v += d * d; }
    const rs = 1 / Math.sqrt(v / D + eps);
    rstd[r] = rs;
    for (let j = 0; j < D; j++) {
      const h = (x.data[o + j] - mean) * rs;
      xhat[o + j] = h;
      out.data[o + j] = h * gamma.data[j] + beta.data[j];
    }
  }
  recordOp([x, gamma, beta], out, () => {
    const g = out.grad;
    if (!g) return;
    const gg = gamma.requiresGrad ? gamma.ensureGrad() : null;
    const gb = beta.requiresGrad ? beta.ensureGrad() : null;
    const gx = x.requiresGrad ? x.ensureGrad() : null;
    for (let r = 0; r < R; r++) {
      const o = r * D;
      let sumD = 0, sumDX = 0;
      for (let j = 0; j < D; j++) {
        const dy = g[o + j];
        if (gg) gg[j] += dy * xhat[o + j];
        if (gb) gb[j] += dy;
        const dh = dy * gamma.data[j];
        sumD += dh;
        sumDX += dh * xhat[o + j];
      }
      if (!gx) continue;
      const rs = rstd[r];
      for (let j = 0; j < D; j++) {
        const dh = g[o + j] * gamma.data[j];
        gx[o + j] += rs * (dh - sumD / D - xhat[o + j] * (sumDX / D));
      }
    }
  });
  return out;
}

/**
 * Softmax over the last dimension: turns any row of numbers into
 * probabilities that are positive and sum to 1.
 *
 * Optional additive `mask` (0 = allowed, a large negative number = blocked)
 * of shape [B, Tq, Tk] for x of shape [B*heads, Tq, Tk]: each group of
 * `heads` consecutive matrices shares one mask.
 */
export function softmax(x: Tensor, mask?: Float32Array, heads = 1): Tensor {
  const T = x.shape[x.shape.length - 1];
  const R = x.size / T;
  const rowsPerMat = x.shape.length === 3 ? x.shape[1] : R;
  const out = Tensor.zeros(x.shape.slice());
  for (let r = 0; r < R; r++) {
    const o = r * T;
    let mo = -1;
    if (mask) {
      const mat = Math.floor(r / rowsPerMat);
      const row = r % rowsPerMat;
      mo = (Math.floor(mat / heads) * rowsPerMat + row) * T;
    }
    let max = -Infinity;
    for (let j = 0; j < T; j++) {
      const v = x.data[o + j] + (mo >= 0 ? mask![mo + j] : 0);
      out.data[o + j] = v;
      if (v > max) max = v;
    }
    let s = 0;
    for (let j = 0; j < T; j++) { const e = Math.exp(out.data[o + j] - max); out.data[o + j] = e; s += e; }
    for (let j = 0; j < T; j++) out.data[o + j] /= s;
  }
  recordOp([x], out, () => {
    const g = out.grad;
    if (!g) return;
    const gx = x.ensureGrad();
    for (let r = 0; r < R; r++) {
      const o = r * T;
      let dot = 0;
      for (let j = 0; j < T; j++) dot += g[o + j] * out.data[o + j];
      for (let j = 0; j < T; j++) gx[o + j] += out.data[o + j] * (g[o + j] - dot);
    }
  });
  return out;
}

// ---------------------------------------------------------------------------
// Reshaping
// ---------------------------------------------------------------------------

/** Same numbers, new shape. */
export function reshape(x: Tensor, shape: number[]): Tensor {
  if (numel(shape) !== x.size) throw new Error(`reshape: [${x.shape}] -> [${shape}]`);
  const out = new Tensor(x.data, shape);
  recordOp([x], out, () => {
    const g = out.grad;
    if (!g) return;
    const gx = x.ensureGrad();
    for (let i = 0; i < g.length; i++) gx[i] += g[i];
  });
  return out;
}

/**
 * [B*T, H*hd] -> [B*H, T, hd]: cut each row into `heads` pieces so every
 * attention head gets its own slice.
 */
export function splitHeads(x: Tensor, B: number, T: number, heads: number): Tensor {
  const D = x.shape[1];
  const hd = D / heads;
  const out = Tensor.zeros([B * heads, T, hd]);
  const copy = (src: Float32Array, dst: Float32Array, forward: boolean) => {
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++)
        for (let h = 0; h < heads; h++) {
          const xo = (b * T + t) * D + h * hd;
          const oo = ((b * heads + h) * T + t) * hd;
          for (let e = 0; e < hd; e++) {
            if (forward) dst[oo + e] = src[xo + e];
            else dst[xo + e] += src[oo + e];
          }
        }
  };
  copy(x.data, out.data, true);
  recordOp([x], out, () => {
    if (out.grad) copy(out.grad, x.ensureGrad(), false);
  });
  return out;
}

/** Inverse of splitHeads: [B*H, T, hd] -> [B*T, H*hd]. */
export function mergeHeads(x: Tensor, B: number, heads: number): Tensor {
  const [, T, hd] = x.shape;
  const D = heads * hd;
  const out = Tensor.zeros([B * T, D]);
  const copy = (src: Float32Array, dst: Float32Array, forward: boolean) => {
    for (let b = 0; b < B; b++)
      for (let t = 0; t < T; t++)
        for (let h = 0; h < heads; h++) {
          const oo = (b * T + t) * D + h * hd;
          const xo = ((b * heads + h) * T + t) * hd;
          for (let e = 0; e < hd; e++) {
            if (forward) dst[oo + e] = src[xo + e];
            else dst[xo + e] += src[oo + e];
          }
        }
  };
  copy(x.data, out.data, true);
  recordOp([x], out, () => {
    if (out.grad) copy(out.grad, x.ensureGrad(), false);
  });
  return out;
}

/** Stack tensors along the first dimension. All other dimensions must match. */
export function concat(parts: Tensor[]): Tensor {
  const rest = parts[0].shape.slice(1);
  for (const p of parts) {
    if (!sameShape(p.shape.slice(1), rest)) throw new Error('concat: shapes differ');
  }
  const rows = parts.reduce((n, p) => n + p.shape[0], 0);
  const out = Tensor.zeros([rows, ...rest]);
  let o = 0;
  const offsets: number[] = [];
  for (const p of parts) {
    offsets.push(o);
    out.data.set(p.data, o);
    o += p.size;
  }
  recordOp(parts, out, () => {
    const g = out.grad;
    if (!g) return;
    parts.forEach((p, i) => {
      if (!p.requiresGrad) return;
      const gp = p.ensureGrad();
      for (let j = 0; j < p.size; j++) gp[j] += g[offsets[i] + j];
    });
  });
  return out;
}

// ---------------------------------------------------------------------------
// Loss
// ---------------------------------------------------------------------------

/**
 * Cross-entropy loss: on average, how surprised was the model by the right
 * answer? logits [N, V], targets[N] (class ids). Rows whose target is
 * `ignoreIndex` (padding) are skipped. Uses a fused, numerically stable
 * log-softmax. Returns a 1-element tensor (the mean over counted rows).
 */
export function crossEntropy(logits: Tensor, targets: Int32Array, ignoreIndex = -1): Tensor {
  assertShape(logits, 2, 'crossEntropy');
  const [N, V] = logits.shape;
  if (targets.length !== N) throw new Error('crossEntropy: targets length mismatch');
  const probs = new Float32Array(N * V);
  let total = 0;
  let count = 0;
  for (let i = 0; i < N; i++) {
    if (targets[i] === ignoreIndex) continue;
    const o = i * V;
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (logits.data[o + j] > max) max = logits.data[o + j];
    let s = 0;
    for (let j = 0; j < V; j++) { const e = Math.exp(logits.data[o + j] - max); probs[o + j] = e; s += e; }
    for (let j = 0; j < V; j++) probs[o + j] /= s;
    total += -(logits.data[o + targets[i]] - max - Math.log(s));
    count++;
  }
  const out = Tensor.from([count > 0 ? total / count : 0], [1]);
  recordOp([logits], out, () => {
    const g = out.grad;
    if (!g || count === 0) return;
    const gl = logits.ensureGrad();
    const k = g[0] / count;
    for (let i = 0; i < N; i++) {
      if (targets[i] === ignoreIndex) continue;
      const o = i * V;
      for (let j = 0; j < V; j++) gl[o + j] += k * probs[o + j];
      gl[o + targets[i]] -= k;
    }
  });
  return out;
}
