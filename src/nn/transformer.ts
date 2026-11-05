/**
 * Transformer building blocks: parameters, linear layers, attention, the
 * feed-forward network and a "block" that combines them.
 */
import {
  add, bmm, gelu, layerNorm, matmul, mergeHeads, scale, softmax, splitHeads,
} from './ops';
import { Rng } from './rng';
import { Tensor } from './tensor';

/** What the rows/columns of a weight matrix stand for (used by the visualisations). */
export type AxisKind = 'token' | 'position' | 'dim' | 'hidden' | 'none';

export interface ParamMeta {
  rows: AxisKind;
  cols: AxisKind;
  /** Which kind of weight this is, e.g. "Wq", "embed", "ln.gamma". Used to pick a plain-language label. */
  role: string;
}

/** All trainable numbers of a model, in a fixed order, with names. */
export class ParamSet {
  readonly list: Tensor[] = [];
  readonly meta = new Map<string, ParamMeta>();
  private byName = new Map<string, Tensor>();

  constructor(private rng: Rng) {}

  private add(name: string, t: Tensor, meta: ParamMeta): Tensor {
    if (this.byName.has(name)) throw new Error(`Duplicate parameter ${name}`);
    t.name = name;
    t.requiresGrad = true;
    this.list.push(t);
    this.byName.set(name, t);
    this.meta.set(name, meta);
    return t;
  }

  normal(name: string, shape: number[], std: number, meta: ParamMeta): Tensor {
    const t = Tensor.zeros(shape);
    for (let i = 0; i < t.size; i++) t.data[i] = this.rng.normal() * std;
    return this.add(name, t, meta);
  }

  fill(name: string, shape: number[], value: number, meta: ParamMeta): Tensor {
    const t = Tensor.zeros(shape);
    t.data.fill(value);
    return this.add(name, t, meta);
  }

  get(name: string): Tensor {
    const t = this.byName.get(name);
    if (!t) throw new Error(`No parameter ${name}`);
    return t;
  }

  names(): string[] {
    return this.list.map((t) => t.name);
  }

  count(): number {
    return this.list.reduce((n, t) => n + t.size, 0);
  }

  /** Copy all weights out (a snapshot). */
  snapshot(): Float32Array[] {
    return this.list.map((t) => t.data.slice());
  }

  /** Load weights from a snapshot taken from a model with the same shapes. */
  load(weights: readonly Float32Array[]): void {
    if (weights.length !== this.list.length) throw new Error('Snapshot does not match model');
    this.list.forEach((t, i) => {
      if (weights[i].length !== t.size) throw new Error(`Snapshot size mismatch for ${t.name}`);
      t.data.set(weights[i]);
    });
  }

  zeroGrad(): void {
    for (const t of this.list) t.zeroGrad();
  }
}

/** Optional recorder for intermediate values (attention maps, activations). */
export type Trace = Map<string, Tensor>;

// ---------------------------------------------------------------------------

export class Linear {
  readonly W: Tensor;
  readonly b: Tensor;

  constructor(ps: ParamSet, name: string, dIn: number, dOut: number, opts: { std?: number; rows?: AxisKind; cols?: AxisKind; role: string }) {
    const std = opts.std ?? 1 / Math.sqrt(dIn);
    this.W = ps.normal(`${name}.W`, [dIn, dOut], std, { rows: opts.rows ?? 'dim', cols: opts.cols ?? 'dim', role: opts.role });
    this.b = ps.fill(`${name}.b`, [dOut], 0, { rows: 'none', cols: opts.cols ?? 'dim', role: `${opts.role}.bias` });
  }

  forward(x: Tensor): Tensor {
    return add(matmul(x, this.W), this.b);
  }
}

export class LayerNorm {
  readonly gamma: Tensor;
  readonly beta: Tensor;

  constructor(ps: ParamSet, name: string, d: number) {
    this.gamma = ps.fill(`${name}.gamma`, [d], 1, { rows: 'none', cols: 'dim', role: 'ln.gamma' });
    this.beta = ps.fill(`${name}.beta`, [d], 0, { rows: 'none', cols: 'dim', role: 'ln.beta' });
  }

  forward(x: Tensor): Tensor {
    return layerNorm(x, this.gamma, this.beta);
  }
}

/** Cached keys/values (already split into heads) for fast step-by-step decoding. */
export interface KV {
  k: Tensor; // [B*H, T, hd]
  v: Tensor; // [B*H, T, hd]
}

/**
 * Multi-head attention. Every position asks a question (query), every
 * position offers a label (key) and some content (value). Each position
 * collects the values of the positions whose keys best match its query.
 */
export class MultiHeadAttention {
  readonly q: Linear;
  readonly k: Linear;
  readonly v: Linear;
  readonly o: Linear;
  readonly headDim: number;

  constructor(ps: ParamSet, name: string, readonly d: number, readonly heads: number, outStd: number) {
    if (d % heads !== 0) throw new Error('d_model must be divisible by heads');
    this.headDim = d / heads;
    this.q = new Linear(ps, `${name}.q`, d, d, { role: 'Wq' });
    this.k = new Linear(ps, `${name}.k`, d, d, { role: 'Wk' });
    this.v = new Linear(ps, `${name}.v`, d, d, { role: 'Wv' });
    this.o = new Linear(ps, `${name}.o`, d, d, { role: 'Wo', std: outStd });
  }

  /** Keys and values for a whole sequence [B*T, d]. */
  keysValues(x: Tensor, B: number, T: number): KV {
    return {
      k: splitHeads(this.k.forward(x), B, T, this.heads),
      v: splitHeads(this.v.forward(x), B, T, this.heads),
    };
  }

  /**
   * xq: [B*Tq, d] (who is asking). kv: keys/values of [B*Tk] positions.
   * mask: additive [B, Tq, Tk] or undefined. Returns [B*Tq, d].
   */
  attend(xq: Tensor, kv: KV, B: number, Tq: number, mask?: Float32Array, trace?: Trace, traceName?: string): Tensor {
    const q = splitHeads(this.q.forward(xq), B, Tq, this.heads);
    const scores = scale(bmm(q, kv.k, true), 1 / Math.sqrt(this.headDim));
    const probs = softmax(scores, mask, this.heads); // [B*H, Tq, Tk]
    if (trace && traceName) trace.set(traceName, probs);
    const ctx = bmm(probs, kv.v); // [B*H, Tq, hd]
    return this.o.forward(mergeHeads(ctx, B, this.heads));
  }
}

export class FeedForward {
  readonly up: Linear;
  readonly down: Linear;

  constructor(ps: ParamSet, name: string, d: number, hidden: number, outStd: number) {
    this.up = new Linear(ps, `${name}.up`, d, hidden, { role: 'ffnUp', cols: 'hidden' });
    this.down = new Linear(ps, `${name}.down`, hidden, d, { role: 'ffnDown', rows: 'hidden', std: outStd });
  }

  forward(x: Tensor): Tensor {
    return this.down.forward(gelu(this.up.forward(x)));
  }
}

/**
 * One pre-LayerNorm transformer block:
 *   x = x + SelfAttention(LN(x))
 *   x = x + CrossAttention(LN(x), memory)   (decoder of the translator only)
 *   x = x + FeedForward(LN(x))
 */
export class Block {
  readonly ln1: LayerNorm;
  readonly self: MultiHeadAttention;
  readonly lnCross: LayerNorm | null;
  readonly cross: MultiHeadAttention | null;
  readonly ln2: LayerNorm;
  readonly ffn: FeedForward;

  constructor(ps: ParamSet, readonly name: string, d: number, heads: number, hidden: number, withCross: boolean, nLayers: number) {
    const outStd = 1 / Math.sqrt(d) / Math.sqrt(2 * nLayers);
    this.ln1 = new LayerNorm(ps, `${name}.ln1`, d);
    this.self = new MultiHeadAttention(ps, `${name}.self`, d, heads, outStd);
    this.lnCross = withCross ? new LayerNorm(ps, `${name}.lnX`, d) : null;
    this.cross = withCross ? new MultiHeadAttention(ps, `${name}.cross`, d, heads, outStd) : null;
    this.ln2 = new LayerNorm(ps, `${name}.ln2`, d);
    this.ffn = new FeedForward(ps, `${name}.ffn`, d, hidden, outStd);
  }

  /** Full-sequence forward (training, visualisation). x: [B*T, d]. */
  forward(
    x: Tensor, B: number, T: number, selfMask: Float32Array | undefined,
    memory?: { kv: KV; mask: Float32Array | undefined }, trace?: Trace,
  ): Tensor {
    const h1 = this.ln1.forward(x);
    x = add(x, this.self.attend(h1, this.self.keysValues(h1, B, T), B, T, selfMask, trace, `${this.name}.self`));
    if (this.cross && this.lnCross && memory) {
      x = add(x, this.cross.attend(this.lnCross.forward(x), memory.kv, B, T, memory.mask, trace, `${this.name}.cross`));
    }
    x = add(x, this.ffn.forward(this.ln2.forward(x)));
    trace?.set(`${this.name}.out`, x);
    return x;
  }

  /**
   * One decoding step with a key/value cache. x: [B, d] for the newest
   * position only. `cache` holds this block's self-attention keys/values of
   * all earlier positions and is extended in place.
   */
  step(x: Tensor, B: number, cache: { kv: KV | null }, memory?: { kv: KV; mask: Float32Array | undefined }): Tensor {
    const h1 = this.ln1.forward(x);
    const fresh = this.self.keysValues(h1, B, 1);
    cache.kv = cache.kv ? { k: appendTime(cache.kv.k, fresh.k), v: appendTime(cache.kv.v, fresh.v) } : fresh;
    x = add(x, this.self.attend(h1, cache.kv, B, 1));
    if (this.cross && this.lnCross && memory) {
      x = add(x, this.cross.attend(this.lnCross.forward(x), memory.kv, B, 1, memory.mask));
    }
    return add(x, this.ffn.forward(this.ln2.forward(x)));
  }
}

/** [BH, T, hd] + [BH, 1, hd] -> [BH, T+1, hd]. Inference only (no gradient). */
function appendTime(a: Tensor, b: Tensor): Tensor {
  const [BH, T, hd] = a.shape;
  const out = Tensor.zeros([BH, T + 1, hd]);
  for (let i = 0; i < BH; i++) {
    out.data.set(a.data.subarray(i * T * hd, (i + 1) * T * hd), i * (T + 1) * hd);
    out.data.set(b.data.subarray(i * hd, (i + 1) * hd), (i * (T + 1) + T) * hd);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Masks (additive: 0 = may look, NEG = must not look)
// ---------------------------------------------------------------------------

export const NEG = -1e9;

/** [B, Tq, Tk] mask that hides padded keys. `valid[b*Tk + k]` is 1 for real tokens. */
export function keyPaddingMask(valid: Uint8Array, B: number, Tq: number, Tk: number): Float32Array {
  const m = new Float32Array(B * Tq * Tk);
  for (let b = 0; b < B; b++)
    for (let q = 0; q < Tq; q++)
      for (let k = 0; k < Tk; k++) if (!valid[b * Tk + k]) m[(b * Tq + q) * Tk + k] = NEG;
  return m;
}

/** [B, T, T] mask that stops each position from looking at later positions. */
export function causalMask(B: number, T: number): Float32Array {
  const m = new Float32Array(B * T * T);
  for (let b = 0; b < B; b++)
    for (let q = 0; q < T; q++)
      for (let k = q + 1; k < T; k++) m[(b * T + q) * T + k] = NEG;
  return m;
}

/** Index of the largest value in arr[offset .. offset+n). */
export function argmax(arr: Float32Array, offset: number, n: number): number {
  let best = 0;
  for (let j = 1; j < n; j++) if (arr[offset + j] > arr[offset + best]) best = j;
  return best;
}

/** Probabilities from logits at row `row` of a [N, V] array, with optional temperature. */
export function rowSoftmax(logits: Float32Array, row: number, V: number, temperature = 1): Float32Array {
  const p = new Float32Array(V);
  let max = -Infinity;
  for (let j = 0; j < V; j++) max = Math.max(max, logits[row * V + j]);
  let s = 0;
  for (let j = 0; j < V; j++) { p[j] = Math.exp((logits[row * V + j] - max) / temperature); s += p[j]; }
  for (let j = 0; j < V; j++) p[j] /= s;
  return p;
}
