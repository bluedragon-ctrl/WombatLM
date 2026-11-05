/**
 * Everything about ONE learning step, recorded only when asked (Step ×1
 * while paused): it costs about as much as 20 ordinary steps, so it is never
 * done while training runs.
 *
 * A learning step, exactly:
 *   1. measure: the batch goes through the model (with dropout: 1 in 10
 *      numbers switched off at random) and every letter's mistake is measured;
 *   2. push back: backpropagation gives every number its gradient ("push"):
 *      how much the batch mistake would grow if the number grew a little;
 *   3. limit: if all pushes together are bigger than 1 (their length, √Σg²),
 *      every push is shrunk by the same factor (gradient clipping);
 *   4. nudge: Adam mixes the push into its memory of earlier pushes
 *      (momentum m, typical size v) and moves the number by
 *      learning rate × m̂ / √v̂, against the push.
 *
 * Because the batch mistake is an average over all letters, the push of a
 * number is exactly the sum of what each example pushed; `perExample`
 * holds those shares (computed with the same switched-off numbers).
 */
import type { Task } from '../data/dataset';

/** The recorded step (kept in the worker until the next step). */
export interface StepRecord {
  step: number;
  tasks: Task[];
  /** Weights before the step, per parameter. */
  before: Float32Array[];
  /** The gradient of the batch mistake (before clipping), per parameter. */
  grad: Float32Array[];
  /** [example][parameter] each example's share of `grad` (null when the batch is too big to split). */
  perExample: Float32Array[][] | null;
  /** Adam's memory before the step (zeros before its first step), per parameter. */
  mPrev: Float32Array[];
  vPrev: Float32Array[];
  /** Learning rate used (after warm-up), the one set, and the warm-up length. */
  lr: number;
  baseLr: number;
  warmupSteps: number;
  /** Adam's update count after this step (for the start corrections). */
  t: number;
  beta1: number;
  beta2: number;
  eps: number;
  weightDecay: number;
  /** Length of all pushes together (before clipping), the limit and the factor applied. */
  gradNorm: number;
  clipNorm: number;
  clip: number;
  /** [B*Tt] chance of the right token, measured the ordinary way (no dropout), before and after the step. */
  pBefore: Float32Array;
  pAfter: Float32Array;
  /** Rows of the padded batch (positions per example). */
  Tt: number;
  /** The batch mistake measured the ordinary way before and after the step. */
  lossBefore: number;
  lossAfter: number;
  /** Greedy answers of every example after the step. */
  answersAfter: string[];
}

/** Why one number changed: the whole calculation of its nudge. */
export interface NumberWhy {
  before: number;
  /** The push (gradient) and the push used (after clipping). */
  grad: number;
  clip: number;
  gradUsed: number;
  mPrev: number;
  vPrev: number;
  m: number;
  v: number;
  /** Start corrections 1 − β^t and the corrected averages. */
  bc1: number;
  bc2: number;
  mHat: number;
  vHat: number;
  lr: number;
  /** Weight decay factor (1 = none). */
  decay: number;
  /** How much the number moved down: lr × m̂ / (√v̂ + ε). */
  move: number;
  /** before × decay − move, recomputed here. */
  after: number;
  /** Each example's share of `grad` (null if not split). */
  shares: number[] | null;
}

/** Recompute the nudge of number `index` of parameter `param`. */
export function explainNumber(r: StepRecord, param: number, index: number, isMatrix: boolean): NumberWhy {
  const before = r.before[param][index];
  const grad = r.grad[param][index];
  const gradUsed = grad * r.clip;
  const mPrev = r.mPrev[param][index], vPrev = r.vPrev[param][index];
  const m = r.beta1 * mPrev + (1 - r.beta1) * gradUsed;
  const v = r.beta2 * vPrev + (1 - r.beta2) * gradUsed * gradUsed;
  const bc1 = 1 - Math.pow(r.beta1, r.t), bc2 = 1 - Math.pow(r.beta2, r.t);
  const mHat = m / bc1, vHat = v / bc2;
  const decay = isMatrix ? 1 - r.lr * r.weightDecay : 1;
  const move = (r.lr * mHat) / (Math.sqrt(vHat) + r.eps);
  return {
    before, grad, clip: r.clip, gradUsed, mPrev, vPrev, m, v, bc1, bc2, mHat, vHat, lr: r.lr, decay, move,
    after: before * decay - move,
    shares: r.perExample ? r.perExample.map((ex) => ex[param][index]) : null,
  };
}

/** A number of the model: parameter, position, and its value before / after / push. */
export interface NumberChange {
  param: number;
  index: number;
  before: number;
  after: number;
  grad: number;
}

/** The `k` numbers with the largest |value(i)|, biggest first. */
export function topNumbers(arrays: readonly Float32Array[], value: (p: number, i: number) => number, k: number): { param: number; index: number }[] {
  const best: { param: number; index: number; a: number }[] = [];
  arrays.forEach((arr, p) => {
    for (let i = 0; i < arr.length; i++) {
      const a = Math.abs(value(p, i));
      if (best.length < k || a > best[best.length - 1].a) {
        best.push({ param: p, index: i, a });
        best.sort((x, y) => y.a - x.a);
        if (best.length > k) best.pop();
      }
    }
  });
  return best.map(({ param, index }) => ({ param, index }));
}

/** How many numbers moved, the biggest move and the median move. */
export function changeStats(before: readonly Float32Array[], after: readonly Float32Array[]): { total: number; moved: number; max: number; median: number } {
  const all: number[] = [];
  let moved = 0;
  before.forEach((b, p) => {
    const a = after[p];
    for (let i = 0; i < b.length; i++) {
      const d = Math.abs(a[i] - b[i]);
      if (d > 0) moved++;
      all.push(d);
    }
  });
  all.sort((x, y) => x - y);
  return { total: all.length, moved, max: all.length ? all[all.length - 1] : 0, median: all.length ? all[all.length >> 1] : 0 };
}
