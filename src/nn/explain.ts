/**
 * "Look inside one letter": every number on the way from one letter going
 * into the decoder to the guess for the next letter, and where each number
 * comes from.
 *
 * The decoder keeps one list of d numbers per position (the "residual
 * stream"). Each step only *adds* something to it:
 *
 *   x = letter + position
 *   x = x + look back at the answer so far   (self-attention)
 *   x = x + look at the animal               (cross-attention)
 *   x = x + think                            (feed-forward)
 *   guess scores = volumeKnobs(x) · letter table + favourite letters
 *
 * so each added list can be split exactly into contributions: per earlier
 * letter (attention), per thinking unit (feed-forward) or per number (the
 * guess scores), plus a fixed bias.
 *
 * "What would it write now?" (the logit lens): the guess scores the model
 * would give if the letter's numbers went straight to the last volume knobs
 * and the letter table after any stage, not only the last one. It turns each
 * stage's list of numbers into letters a beginner can read, and shows which
 * stage makes the decision.
 */
import type { Translator } from './translator';
import type { Trace } from './transformer';
import type { Tensor } from './tensor';

/** One step that adds a list of numbers to the letter's numbers. */
export interface ExplainStage {
  /** "attend": self- or cross-attention; "think": the feed-forward network. */
  kind: 'attend' | 'think';
  /** Model part, e.g. "dec0.self", "dec0.cross", "dec0.ffn". */
  name: string;
  /** Attention only: number of heads and of positions it could look at. */
  heads: number;
  /** Sources: positions looked at (attention) or thinking units (feed-forward). */
  sources: number;
  /** Attention only: [heads × sources] how much each head looked at each position (rows sum to 1). */
  weights: Float32Array;
  /** Feed-forward only: the thinking units after GELU [sources]. */
  hidden: Float32Array;
  /** What this step adds [d]. */
  add: Float32Array;
  /** [sources × d] how much each source adds to each number; with `bias`, sums exactly to `add`. */
  contrib: Float32Array;
  /** [d] the part of `add` that does not depend on the input (bias). */
  bias: Float32Array;
  /** The letter's numbers after this step (before + add) [d]. */
  x: Float32Array;
}

export interface LetterExplain {
  /** Output position (= guess-bar column); the decoder input at this position is `decIn[position]`. */
  position: number;
  /** Decoder input tokens 0..position (task token or <bos>, then the letters written so far). */
  decIn: number[];
  /** The full answer (token ids, without <eos>). */
  output: number[];
  /** The letter going in: its row of the letter table, its position row, and their sum [d each]. */
  start: { token: Float32Array; pos: Float32Array; x: Float32Array };
  stages: ExplainStage[];
  /** The numbers after the last volume knobs, used for guessing [d]. */
  final: Float32Array;
  /** Guess scores for every token [V]; score(t) = Σ final[i] · tokEmb[t, i] + outBias[t]. */
  logits: Float32Array;
  /**
   * "What would it write now?": guess scores [V] from the letter's numbers
   * after the start and after each stage (1 + stages.length lists). The last
   * one equals `logits`.
   */
  lens: Float32Array[];
}

function row(t: Tensor, r: number): Float32Array {
  const d = t.shape[t.shape.length - 1];
  return t.data.slice(r * d, (r + 1) * d);
}

function need(trace: Trace, name: string): Tensor {
  const t = trace.get(name);
  if (!t) throw new Error(`explain: missing trace ${name}`);
  return t;
}

/** v = x · W + b for one row x. */
function linearRow(x: Float32Array, W: Tensor, b: Tensor): Float32Array {
  const [dIn, dOut] = W.shape;
  const out = b.data.slice();
  for (let i = 0; i < dIn; i++) {
    const xi = x[i];
    if (xi === 0) continue;
    for (let j = 0; j < dOut; j++) out[j] += xi * W.data[i * dOut + j];
  }
  return out;
}

/**
 * Attention for query position `p`, split by the position looked at:
 * contrib[t, j] = Σ_heads weight[h, t] · (V[t] restricted to head h) · Wo[:, j].
 */
function attendStage(model: Translator, trace: Trace, name: string, p: number, keysIn: Tensor, nKeys: number, before: Float32Array): ExplainStage {
  const [blockName, which] = name.split('.');
  const block = model.dec.find((b) => b.name === blockName)!;
  const att = which === 'self' ? block.self : block.cross!;
  const probs = need(trace, name); // [H, Tq, Tk]
  const [H, Tq, Tk] = probs.shape;
  const d = att.d, hd = att.headDim;
  const weights = new Float32Array(H * nKeys);
  for (let h = 0; h < H; h++) for (let t = 0; t < nKeys; t++) weights[h * nKeys + t] = probs.data[(h * Tq + p) * Tk + t];
  const Wo = att.o.W.data;
  const contrib = new Float32Array(nKeys * d);
  for (let t = 0; t < nKeys; t++) {
    const v = linearRow(row(keysIn, t), att.v.W, att.v.b);
    for (let h = 0; h < H; h++) {
      const a = weights[h * nKeys + t];
      if (a === 0) continue;
      for (let c = h * hd; c < (h + 1) * hd; c++) {
        const av = a * v[c];
        for (let j = 0; j < d; j++) contrib[t * d + j] += av * Wo[c * d + j];
      }
    }
  }
  const add = row(need(trace, `${name}.add`), p);
  const x = before.map((v, i) => v + add[i]);
  return { kind: 'attend', name, heads: H, sources: nKeys, weights, hidden: new Float32Array(0), add, contrib, bias: att.o.b.data.slice(), x };
}

function thinkStage(model: Translator, trace: Trace, name: string, p: number, before: Float32Array): ExplainStage {
  const block = model.dec.find((b) => b.name === name.split('.')[0])!;
  const hidden = row(need(trace, `${name}.hidden`), p);
  const W = block.ffn.down.W;
  const [units, d] = W.shape;
  const contrib = new Float32Array(units * d);
  for (let k = 0; k < units; k++) for (let j = 0; j < d; j++) contrib[k * d + j] = hidden[k] * W.data[k * d + j];
  const add = row(need(trace, `${name}.add`), p);
  const x = before.map((v, i) => v + add[i]);
  return { kind: 'think', name, heads: 0, sources: units, weights: new Float32Array(0), hidden, add, contrib, bias: block.ffn.down.b.data.slice(), x };
}

/** Guess scores from one list of numbers: last volume knobs, then the letter table and favourite letters (logit lens). */
export function lensLogits(model: Translator, x: Float32Array): Float32Array {
  const d = x.length;
  const { gamma, beta } = model.decNorm;
  let mean = 0;
  for (let i = 0; i < d; i++) mean += x[i];
  mean /= d;
  let v = 0;
  for (let i = 0; i < d; i++) v += (x[i] - mean) ** 2;
  const rs = 1 / Math.sqrt(v / d + 1e-5);
  const y = new Float32Array(d);
  for (let i = 0; i < d; i++) y[i] = (x[i] - mean) * rs * gamma.data[i] + beta.data[i];
  const V = model.tokEmb.shape[0];
  const out = model.outBias.data.slice(0, V);
  for (let t = 0; t < V; t++) {
    let s = 0;
    for (let i = 0; i < d; i++) s += y[i] * model.tokEmb.data[t * d + i];
    out[t] += s;
  }
  return out;
}

/** Build the explanation from a traced teacher-forced pass of one word (batch of 1). */
export function explainLetter(model: Translator, trace: Trace, decIn: number[], p: number, logits: Float32Array, output: number[]): LetterExplain {
  const token = row(model.tokEmb, decIn[p]);
  const pos = row(model.decPos, p);
  let x: Float32Array = token.map((v, i) => v + pos[i]);
  const start = { token, pos, x };
  const memory = need(trace, 'memory');
  const stages: ExplainStage[] = [];
  for (const block of model.dec) {
    const n = block.name;
    let s = attendStage(model, trace, `${n}.self`, p, need(trace, `${n}.ln1`), p + 1, x);
    stages.push(s); x = s.x;
    if (block.cross) { s = attendStage(model, trace, `${n}.cross`, p, memory, memory.shape[0], x); stages.push(s); x = s.x; }
    s = thinkStage(model, trace, `${n}.ffn`, p, x);
    stages.push(s); x = s.x;
  }
  const lens = [lensLogits(model, start.x), ...stages.map((s) => lensLogits(model, s.x))];
  return { position: p, decIn: decIn.slice(0, p + 1), output, start, stages, final: row(need(trace, 'dec.final'), p), logits: logits.slice(), lens };
}
