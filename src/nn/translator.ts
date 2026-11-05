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
import { add, bmm, crossEntropy, dropout, gather, matmul, nllFromProbs, pointerMix, reshape, scale, softmax } from './ops';
import { Rng } from './rng';
import { growVocab } from './serialize';
import { Tensor } from './tensor';
import {
  Block, KV, LayerNorm, Linear, NEG, ParamSet, Trace, argmax, causalMask, keyPaddingMask, rowSoftmax,
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
  /** Copy mechanism: the model may point at an input letter and copy it. */
  copy?: boolean;
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
  /** Copy mechanism only: per step, how much the model wrote itself (1) vs copied (0). */
  writeShare?: number[];
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
  /** Copy mechanism (optional): query, key and the write-or-copy gate. */
  readonly copyQ: Linear | null = null;
  readonly copyK: Linear | null = null;
  readonly copyGate: Linear | null = null;
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
    if (config.copy) {
      this.copyQ = new Linear(ps, 'copy.q', d, d, { role: 'copyQ' });
      this.copyK = new Linear(ps, 'copy.k', d, d, { role: 'copyK' });
      this.copyGate = new Linear(ps, 'copy.gate', d, 1, { role: 'copyGate', cols: 'none' });
    }
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
    return this.encNorm.forward(x);
  }

  /**
   * Decoder with teacher forcing. Returns output scores [B*Tt, V]: logits
   * without the copy mechanism, probabilities with it. Either way the
   * highest score is the model's choice.
   */
  decode(memory: Tensor, src: Int32Array, srcValid: Uint8Array, tgtIn: Int32Array, B: number, Ts: number, Tt: number, trace?: Trace): Tensor {
    let y = this.embed(tgtIn, B, Tt, this.decPos);
    trace?.set('dec.in', y);
    const selfMask = causalMask(B, Tt);
    const crossMask = keyPaddingMask(srcValid, B, Tt, Ts);
    for (const block of this.dec) {
      const kv = block.cross!.keysValues(memory, B, Ts);
      y = block.forward(y, B, Tt, selfMask, { kv, mask: crossMask }, trace, this.drop);
    }
    return this.output(this.decNorm.forward(y), this.copyKeys(memory, B, Ts), src, srcValid, B, Tt, Ts, trace);
  }

  /** Hidden states -> scores for every token (tied with the embedding table). */
  private project(h: Tensor): Tensor {
    return add(matmul(h, this.tokEmb, true), this.outBias);
  }

  /** Keys the copy mechanism points at: [B, Ts, d] (null without copying). */
  private copyKeys(memory: Tensor, B: number, Ts: number): Tensor | null {
    return this.copyK ? reshape(this.copyK.forward(memory), [B, Ts, this.config.dModel]) : null;
  }

  /**
   * Final hidden states h [B*Tq, d] -> output scores. With the copy
   * mechanism: attend over the input letters (never the target token or
   * padding), then mix "write" and "copy" with a learned gate.
   */
  private output(h: Tensor, keys: Tensor | null, src: Int32Array, srcValid: Uint8Array, B: number, Tq: number, Ts: number, trace?: Trace): Tensor {
    const logits = this.project(h);
    if (!keys || !this.copyQ || !this.copyGate) return logits;
    const d = this.config.dModel;
    const q = reshape(this.copyQ.forward(h), [B, Tq, d]);
    const attn = softmax(scale(bmm(q, keys, true), 1 / Math.sqrt(d)), copyMask(srcValid, B, Tq, Ts)); // [B, Tq, Ts]
    const gate = this.copyGate.forward(h); // [B*Tq, 1]
    trace?.set('copy', attn);
    trace?.set('copyGate', gate);
    return pointerMix(logits, reshape(attn, [B * Tq, Ts]), src, Tq, gate);
  }

  /** Output scores [B*Tt, V] (logits, or probabilities with the copy mechanism). */
  logits(batch: Batch, trace?: Trace): Tensor {
    const memory = this.encode(batch.src, batch.srcValid, batch.B, batch.Ts, trace);
    return this.decode(memory, batch.src, batch.srcValid, this.decoderInput(batch), batch.B, batch.Ts, batch.Tt, trace);
  }

  /** Decoder input; with `targetStart` the first token is the target token instead of <bos>. */
  private decoderInput(batch: Batch): Int32Array {
    if (!this.config.targetStart) return batch.tgtIn;
    const tgtIn = batch.tgtIn.slice();
    for (let b = 0; b < batch.B; b++) tgtIn[b * batch.Tt] = batch.src[b * batch.Ts];
    return tgtIn;
  }

  /**
   * Mean cross-entropy over all real target characters. If `keep` is given,
   * the output scores are stored in it (no extra work: the UI reads the
   * model's teacher-forced guesses from them).
   */
  loss(batch: Batch, padId: number, keep?: { scores?: Tensor }): Tensor {
    this.training = true;
    try {
      const scores = this.logits(batch);
      if (keep) keep.scores = scores;
      return this.config.copy ? nllFromProbs(scores, batch.tgtOut, padId) : crossEntropy(scores, batch.tgtOut, padId);
    } finally {
      this.training = false;
    }
  }

  /**
   * Greedy decoding for many inputs at once, using a key/value cache so each
   * step only computes the newest position. Returns generated ids (no <eos>).
   */
  greedy(inputs: readonly { target: number; src: number[] }[], ids: SpecialIds, maxSteps = this.config.maxLen - 1): number[][] {
    if (inputs.length === 0) return [];
    const batch = makeBatch(inputs.map((i) => ({ ...i, tgt: [] })), ids);
    const { B, Ts } = batch;
    const memory = this.encode(batch.src, batch.srcValid, B, Ts);
    const crossMask = keyPaddingMask(batch.srcValid, B, 1, Ts);
    const crossKV: KV[] = this.dec.map((blk) => blk.cross!.keysValues(memory, B, Ts));
    const keys = this.copyKeys(memory, B, Ts);
    const caches = this.dec.map(() => ({ kv: null as KV | null }));
    const V = this.vocabSize;
    const out: number[][] = Array.from({ length: B }, () => []);
    const done = new Uint8Array(B);
    let current = this.decoderInput({ ...batch, Tt: 1, tgtIn: new Int32Array(B).fill(ids.bos) });
    const steps = Math.min(maxSteps, this.config.maxLen);
    for (let t = 0; t < steps; t++) {
      let y = add(gather(this.tokEmb, current), gather(this.decPos, new Int32Array(B).fill(t)));
      this.dec.forEach((blk, l) => { y = blk.step(y, B, caches[l], { kv: crossKV[l], mask: crossMask }); });
      const logits = this.output(this.decNorm.forward(y), keys, batch.src, batch.srcValid, B, 1, Ts);
      const next = new Int32Array(B);
      let allDone = true;
      for (let b = 0; b < B; b++) {
        const id = argmax(logits.data, b * V, V);
        next[b] = id;
        if (done[b]) continue;
        if (id === ids.eos) done[b] = 1;
        else out[b].push(id);
        if (!done[b]) allDone = false;
      }
      if (allDone) break;
      current = next;
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
    const probs = Array.from({ length: nSteps }, (_, t) =>
      this.config.copy ? logits.data.slice(t * V, (t + 1) * V) : rowSoftmax(logits.data, t, V));
    const attention: TranslationTrace['attention'] = {};
    const activations: TranslationTrace['activations'] = {};
    let writeShare: number[] | undefined;
    for (const [name, t] of trace) {
      if (name === 'copyGate') writeShare = Array.from(t.data.subarray(0, nSteps), (x) => 1 / (1 + Math.exp(-x)));
      else if (t.shape.length === 3) attention[name] = { heads: t.shape[0], Tq: t.shape[1], Tk: t.shape[2], data: t.data.slice() };
      else activations[name] = { T: t.shape[0], d: t.shape[1], data: t.data.slice() };
    }
    return { input: [target, ...src], output, finished, probs, attention, activations, writeShare };
  }
}

/**
 * [B, Tq, Ts] mask for the copy mechanism: it may point at input letters
 * only, never at the target token (position 0) or padding.
 */
function copyMask(srcValid: Uint8Array, B: number, Tq: number, Ts: number): Float32Array {
  const m = keyPaddingMask(srcValid, B, Tq, Ts);
  for (let b = 0; b < B; b++) for (let q = 0; q < Tq; q++) m[(b * Tq + q) * Ts] = NEG;
  return m;
}
