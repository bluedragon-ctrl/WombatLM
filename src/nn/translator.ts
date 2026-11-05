/**
 * The translator: a tiny character-level Transformer encoder–decoder.
 *
 * Encoder input:  <target-token> s o u r c e      e.g.  <en> t y g r
 * Decoder input:  <bos> t i g e r
 * Decoder output:       t i g e r <eos>
 *
 * The target token says what to write (which language / form). The source
 * language is never given: the model works it out from the letters.
 *
 * One embedding table is shared by encoder input, decoder input and the
 * output layer ("tied"), so there is a single "letter map" to look at.
 */
import { add, crossEntropy, dropout, gather, matmul } from './ops';
import { Rng } from './rng';
import { LetterExplain, explainLetter } from './explain';
import { growVocab } from './serialize';
import { Tensor } from './tensor';
import {
  Block, KV, LayerNorm, ParamSet, Trace, argmax, causalMask, keyPaddingMask, rowSoftmax,
} from './transformer';

export interface TranslatorConfig {
  vocabSize: number;
  dModel: number;
  heads: number;
  ffn: number;
  encLayers: number;
  decLayers: number;
  /** Maximum sequence length (positions), including special tokens. */
  maxLen: number;
  /** Dropout probability used while training (0 = off). */
  dropout?: number;
  /**
   * Start the output with the target token (e.g. <cs:female>) instead of
   * <bos>, so the decoder always sees what it was asked to write.
   */
  targetStart?: boolean;
}

export const TRANSLATOR_DEFAULTS: Omit<TranslatorConfig, 'vocabSize'> = {
  dModel: 32,
  heads: 2,
  ffn: 64,
  encLayers: 1,
  decLayers: 1,
  maxLen: 22,
};

/** Special token ids the batching code needs. */
export interface SpecialIds {
  pad: number;
  bos: number;
  eos: number;
}

/** One training example, already turned into token ids. */
export interface Example {
  /** Target token id, e.g. id of <en>. */
  target: number;
  /** Source word characters. */
  src: number[];
  /** Target word characters (without <bos>/<eos>). */
  tgt: number[];
}

export interface Batch {
  B: number;
  Ts: number;
  Tt: number;
  /** [B*Ts] encoder tokens: target token + source characters + padding. */
  src: Int32Array;
  /** [B*Ts] 1 for real tokens, 0 for padding. */
  srcValid: Uint8Array;
  /** [B*Tt] decoder input: <bos> + target characters + padding. */
  tgtIn: Int32Array;
  /** [B*Tt] what the decoder should predict: target characters + <eos> + padding. */
  tgtOut: Int32Array;
}

/** Pad a list of examples into one rectangular batch. */
export function makeBatch(examples: readonly Example[], ids: SpecialIds): Batch {
  const B = examples.length;
  const Ts = Math.max(...examples.map((e) => e.src.length + 1));
  const Tt = Math.max(...examples.map((e) => e.tgt.length + 1));
  const src = new Int32Array(B * Ts).fill(ids.pad);
  const srcValid = new Uint8Array(B * Ts);
  const tgtIn = new Int32Array(B * Tt).fill(ids.pad);
  const tgtOut = new Int32Array(B * Tt).fill(ids.pad);
  examples.forEach((e, b) => {
    const s = [e.target, ...e.src];
    s.forEach((t, i) => { src[b * Ts + i] = t; srcValid[b * Ts + i] = 1; });
    tgtIn[b * Tt] = ids.bos;
    e.tgt.forEach((t, i) => { tgtIn[b * Tt + i + 1] = t; tgtOut[b * Tt + i] = t; });
    tgtOut[b * Tt + e.tgt.length] = ids.eos;
  });
  return { B, Ts, Tt, src, srcValid, tgtIn, tgtOut };
}

/**
 * For each row of `logits` ([N, V]), the probability of the target token
 * (NaN where the target is padding).
 */
export function rightTokenProbs(logits: Tensor, targets: Int32Array, padId: number, out: Float32Array): void {
  const V = logits.shape[1];
  const d = logits.data;
  for (let i = 0; i < targets.length; i++) {
    if (targets[i] === padId) { out[i] = NaN; continue; }
    const o = i * V;
    let max = -Infinity;
    for (let j = 0; j < V; j++) if (d[o + j] > max) max = d[o + j];
    let sum = 0;
    for (let j = 0; j < V; j++) sum += Math.exp(d[o + j] - max);
    out[i] = Math.exp(d[o + targets[i]] - max) / sum;
  }
}

/** Everything the visualisations need about one translation. */
export interface TranslationTrace {
  /** Encoder input tokens (target token + source). */
  input: number[];
  /** Generated characters (without <eos>). */
  output: number[];
  /** Did generation stop with <eos> (as opposed to hitting the length limit)? */
  finished: boolean;
  /** Per output step (including the final <eos> step): full probability distribution. */
  probs: Float32Array[];
  /**
   * Attention maps: name -> [heads][queries][keys] flattened as Float32Array
   * of length heads*Tq*Tk, plus its dimensions.
   */
  attention: Record<string, { heads: number; Tq: number; Tk: number; data: Float32Array }>;
  /**
   * Activations, [T, d] each, in the order the signal flows (for "follow the
   * signal"): `enc.in` (letters + positions), `enc0.out`, …, `dec.in`, `dec0.out`, ….
   */
  activations: Record<string, { T: number; d: number; data: Float32Array }>;
}

export class Translator {
  readonly config: TranslatorConfig;
  readonly params: ParamSet;
  readonly tokEmb: Tensor;
  readonly encPos: Tensor;
  readonly decPos: Tensor;
  readonly enc: Block[];
  readonly dec: Block[];
  readonly encNorm: LayerNorm;
  readonly decNorm: LayerNorm;
  readonly outBias: Tensor;
  /** Random numbers for dropout (separate from initialisation so runs stay reproducible). */
  private dropRng: Rng;
  /** True only inside `loss()`: dropout is active. */
  private training = false;

  constructor(config: TranslatorConfig, seed: number) {
    this.config = { ...config };
    this.dropRng = new Rng(seed ^ 0xd209);
    const { vocabSize: V, dModel: d, heads, ffn, encLayers, decLayers, maxLen } = config;
    const ps = new ParamSet(new Rng(seed));
    this.params = ps;
    this.tokEmb = ps.normal('tokEmb', [V, d], 0.3, { rows: 'token', cols: 'dim', role: 'embed' });
    this.encPos = ps.normal('encPos', [maxLen, d], 0.1, { rows: 'position', cols: 'dim', role: 'pos' });
    this.decPos = ps.normal('decPos', [maxLen, d], 0.1, { rows: 'position', cols: 'dim', role: 'pos' });
    this.enc = Array.from({ length: encLayers }, (_, i) => new Block(ps, `enc${i}`, d, heads, ffn, false, encLayers + decLayers));
    this.encNorm = new LayerNorm(ps, 'encNorm', d);
    this.dec = Array.from({ length: decLayers }, (_, i) => new Block(ps, `dec${i}`, d, heads, ffn, true, encLayers + decLayers));
    this.decNorm = new LayerNorm(ps, 'decNorm', d);
    this.outBias = ps.fill('outBias', [V], 0, { rows: 'none', cols: 'token', role: 'outBias' });
  }

  get vocabSize(): number {
    return this.tokEmb.shape[0];
  }

  /** Parameters whose rows are vocabulary tokens. */
  vocabParams(): Tensor[] {
    return [this.tokEmb, this.outBias];
  }

  /** Grow (or reorder) the vocabulary, keeping every existing weight. */
  growVocab(oldTokens: readonly string[], newTokens: readonly string[], rng: Rng): void {
    growVocab(this.vocabParams(), oldTokens, newTokens, rng);
    this.config.vocabSize = newTokens.length;
  }

  /** Dropout while training, identity otherwise. */
  private readonly drop = (t: Tensor): Tensor =>
    this.training && this.config.dropout ? dropout(t, this.config.dropout, () => this.dropRng.next()) : t;

  /** Token + position embeddings for [B, T] token ids. */
  private embed(tokens: Int32Array, B: number, T: number, posTable: Tensor): Tensor {
    const pos = new Int32Array(B * T);
    for (let b = 0; b < B; b++) for (let t = 0; t < T; t++) pos[b * T + t] = t;
    return this.drop(add(gather(this.tokEmb, tokens), gather(posTable, pos)));
  }

  /** Encoder: [B*Ts] tokens -> memory [B*Ts, d]. */
  encode(src: Int32Array, srcValid: Uint8Array, B: number, Ts: number, trace?: Trace): Tensor {
    let x = this.embed(src, B, Ts, this.encPos);
    trace?.set('enc.in', x);
    const mask = keyPaddingMask(srcValid, B, Ts, Ts);
    for (const block of this.enc) x = block.forward(x, B, Ts, mask, undefined, trace, this.drop);
    const memory = this.encNorm.forward(x);
    trace?.set('memory', memory);
    return memory;
  }

  /** Decoder with teacher forcing. Returns output logits [B*Tt, V]. */
  decode(memory: Tensor, srcValid: Uint8Array, tgtIn: Int32Array, B: number, Ts: number, Tt: number, trace?: Trace): Tensor {
    let y = this.embed(tgtIn, B, Tt, this.decPos);
    trace?.set('dec.in', y);
    const selfMask = causalMask(B, Tt);
    const crossMask = keyPaddingMask(srcValid, B, Tt, Ts);
    for (const block of this.dec) {
      const kv = block.cross!.keysValues(memory, B, Ts);
      y = block.forward(y, B, Tt, selfMask, { kv, mask: crossMask }, trace, this.drop);
    }
    const final = this.decNorm.forward(y);
    trace?.set('dec.final', final);
    return this.project(final);
  }

  /** Hidden states -> scores for every token (tied with the embedding table). */
  private project(h: Tensor): Tensor {
    return add(matmul(h, this.tokEmb, true), this.outBias);
  }

  /** Output logits [B*Tt, V]. */
  logits(batch: Batch, trace?: Trace): Tensor {
    const memory = this.encode(batch.src, batch.srcValid, batch.B, batch.Ts, trace);
    return this.decode(memory, batch.srcValid, this.decoderInput(batch), batch.B, batch.Ts, batch.Tt, trace);
  }

  /** Decoder input; with `targetStart` the first token is the target token instead of <bos>. */
  private decoderInput(batch: Batch): Int32Array {
    if (!this.config.targetStart) return batch.tgtIn;
    const tgtIn = batch.tgtIn.slice();
    for (let b = 0; b < batch.B; b++) tgtIn[b * batch.Tt] = batch.src[b * batch.Ts];
    return tgtIn;
  }

  /**
   * The training loss: the average mistake over every letter of the batch.
   * If `tokenP` is given ([B*Tt]), it receives for each position the
   * probability the model gave the right token (NaN for padding), read from
   * the same forward pass: the mistake of one letter is −ln of it.
   */
  loss(batch: Batch, padId: number, tokenP?: Float32Array): Tensor {
    this.training = true;
    try {
      const logits = this.logits(batch);
      if (tokenP) rightTokenProbs(logits, batch.tgtOut, padId, tokenP);
      return crossEntropy(logits, batch.tgtOut, padId);
    } finally {
      this.training = false;
    }
  }

  /**
   * Greedy decoding for many inputs at once, using a key/value cache so each
   * step only computes the newest position. Words that have ended (<eos>) are
   * dropped from the batch, so later steps only compute the words still being
   * written. Returns generated ids (no <eos>).
   */
  greedy(inputs: readonly { target: number; src: number[] }[], ids: SpecialIds, maxSteps = this.config.maxLen - 1): number[][] {
    if (inputs.length === 0) return [];
    const batch = makeBatch(inputs.map((i) => ({ ...i, tgt: [] })), ids);
    const { B, Ts } = batch;
    const H = this.config.heads;
    const memory = this.encode(batch.src, batch.srcValid, B, Ts);
    let crossMask = keyPaddingMask(batch.srcValid, B, 1, Ts);
    let crossKV: KV[] = this.dec.map((blk) => blk.cross!.keysValues(memory, B, Ts));
    const caches = this.dec.map(() => ({ kv: null as KV | null }));
    const V = this.vocabSize;
    const out: number[][] = Array.from({ length: B }, () => []);
    /** Which inputs are still being written (row i of the current batch is input alive[i]). */
    let alive = Array.from({ length: B }, (_, b) => b);
    let current = this.decoderInput({ ...batch, Tt: 1, tgtIn: new Int32Array(B).fill(ids.bos) });
    const steps = Math.min(maxSteps, this.config.maxLen);
    for (let t = 0; t < steps; t++) {
      const n = alive.length;
      let y = add(gather(this.tokEmb, current), gather(this.decPos, new Int32Array(n).fill(t)));
      this.dec.forEach((blk, l) => { y = blk.step(y, n, caches[l], { kv: crossKV[l], mask: crossMask }); });
      const logits = this.project(this.decNorm.forward(y));
      const keep: number[] = [];
      const next: number[] = [];
      for (let i = 0; i < n; i++) {
        const id = argmax(logits.data, i * V, V);
        if (id === ids.eos) continue;
        out[alive[i]].push(id);
        keep.push(i);
        next.push(id);
      }
      if (keep.length === 0) break;
      if (keep.length < n) {
        alive = keep.map((i) => alive[i]);
        crossKV = crossKV.map((kv) => ({ k: keepRows(kv.k, keep, H), v: keepRows(kv.v, keep, H) }));
        for (const c of caches) c.kv = { k: keepRows(c.kv!.k, keep, H), v: keepRows(c.kv!.v, keep, H) };
        crossMask = keepMaskRows(crossMask, keep, Ts);
      }
      current = Int32Array.from(next);
    }
    return out;
  }

  /**
   * Translate one word and record everything needed to visualise it:
   * per-step probabilities, attention maps and activations.
   */
  inspect(target: number, src: number[], ids: SpecialIds): TranslationTrace {
    const [output] = this.greedy([{ target, src }], ids);
    const finished = output.length < this.config.maxLen - 1;
    // One full teacher-forced pass over the generated answer gives exactly the
    // same per-step probabilities and attention as step-by-step decoding.
    const tgt = finished ? output : output.slice(0, this.config.maxLen - 1);
    const batch = makeBatch([{ target, src, tgt }], ids);
    const trace: Trace = new Map();
    const logits = this.logits(batch, trace);
    const V = this.vocabSize;
    const nSteps = finished ? batch.Tt : batch.Tt - 1;
    const probs = Array.from({ length: nSteps }, (_, t) => rowSoftmax(logits.data, t, V));
    const attention: TranslationTrace['attention'] = {};
    const activations: TranslationTrace['activations'] = {};
    for (const [name, t] of trace) {
      if (t.shape.length === 3) attention[name] = { heads: t.shape[0], Tq: t.shape[1], Tk: t.shape[2], data: t.data.slice() };
      else if (SIGNAL.test(name)) activations[name] = { T: t.shape[0], d: t.shape[1], data: t.data.slice() };
    }
    return { input: [target, ...src], output, finished, probs, attention, activations };
  }

  /**
   * "Look inside one letter": every number on the way to output letter
   * `position` of this word's answer, and where each one comes from.
   */
  explain(target: number, src: number[], ids: SpecialIds, position: number): LetterExplain {
    const [output] = this.greedy([{ target, src }], ids);
    const tgt = output.slice(0, this.config.maxLen - 1);
    const batch = makeBatch([{ target, src, tgt }], ids);
    const trace: Trace = new Map();
    const logits = this.logits(batch, trace);
    const p = Math.max(0, Math.min(position, batch.Tt - 1));
    return explainLetter(this, trace, Array.from(this.decoderInput(batch)), p, logits.data.subarray(p * this.vocabSize, (p + 1) * this.vocabSize), output);
  }
}

/**
 * Keep only some batch rows of a [B*group, ...] tensor (rows of the same
 * input come in groups, e.g. one per attention head). Inference only.
 */
function keepRows(t: Tensor, keep: readonly number[], group: number): Tensor {
  const per = (t.size / t.shape[0]) * group;
  const out = Tensor.zeros([keep.length * group, ...t.shape.slice(1)]);
  keep.forEach((k, j) => out.data.set(t.data.subarray(k * per, (k + 1) * per), j * per));
  return out;
}

/** Keep only some batch rows of a [B, 1, Tk] attention mask. */
function keepMaskRows(mask: Float32Array, keep: readonly number[], Tk: number): Float32Array {
  const out = new Float32Array(keep.length * Tk);
  keep.forEach((k, j) => out.set(mask.subarray(k * Tk, (k + 1) * Tk), j * Tk));
  return out;
}

/** Trace entries that "follow the signal" shows (the rest are for `explain`). */
const SIGNAL = /^(enc|dec)\d*\.(in|out)$/;
