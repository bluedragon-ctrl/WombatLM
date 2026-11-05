/**
 * Adam optimiser with global gradient-norm clipping and a linear warm-up.
 *
 * After each backward pass every weight knows its gradient. Adam nudges each
 * weight a small step against its gradient, using running averages of the
 * gradient (momentum) and of its square (to adapt the step size per weight).
 */
import { Tensor } from './tensor';

export interface AdamOptions {
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  clipNorm: number;
  warmupSteps: number;
  /** Decoupled weight decay (AdamW), applied to matrices only. 0 = off. */
  weightDecay: number;
}

export const DEFAULT_ADAM: AdamOptions = {
  lr: 3e-3,
  beta1: 0.9,
  beta2: 0.98,
  eps: 1e-9,
  clipNorm: 1.0,
  warmupSteps: 100,
  weightDecay: 0,
};

export interface StepInfo {
  /** Learning rate used for this step (after warm-up). */
  lr: number;
  /** Global gradient norm before clipping. */
  gradNorm: number;
}

export class Adam {
  opts: AdamOptions;
  /** Number of updates done so far. */
  t = 0;
  private m = new Map<Tensor, Float32Array>();
  private v = new Map<Tensor, Float32Array>();

  constructor(readonly params: Tensor[], opts: Partial<AdamOptions> = {}) {
    this.opts = { ...DEFAULT_ADAM, ...opts };
  }

  /** Learning rate for update number `t` (1-based): linear warm-up, then constant. */
  lrAt(t: number): number {
    const w = this.opts.warmupSteps;
    return w > 0 && t < w ? (this.opts.lr * t) / w : this.opts.lr;
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }

  step(): StepInfo {
    this.t++;
    let sq = 0;
    for (const p of this.params) {
      if (!p.grad) continue;
      for (let i = 0; i < p.grad.length; i++) sq += p.grad[i] * p.grad[i];
    }
    const gradNorm = Math.sqrt(sq);
    const clip = gradNorm > this.opts.clipNorm ? this.opts.clipNorm / gradNorm : 1;
    const lr = this.lrAt(this.t);
    const { beta1, beta2, eps } = this.opts;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    for (const p of this.params) {
      if (!p.grad) continue;
      let m = this.m.get(p);
      let v = this.v.get(p);
      // A parameter can change size when the vocabulary grows: restart its state.
      if (!m || m.length !== p.size) { m = new Float32Array(p.size); this.m.set(p, m); }
      if (!v || v.length !== p.size) { v = new Float32Array(p.size); this.v.set(p, v); }
      const g = p.grad;
      const w = p.data;
      const decay = p.shape.length > 1 ? 1 - lr * this.opts.weightDecay : 1;
      for (let i = 0; i < w.length; i++) {
        if (decay !== 1) w[i] *= decay;
        const gi = g[i] * clip;
        m[i] = beta1 * m[i] + (1 - beta1) * gi;
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        w[i] -= (lr * (m[i] / bc1)) / (Math.sqrt(v[i] / bc2) + eps);
      }
    }
    return { lr, gradNorm };
  }

  /**
   * Adam's memory of one parameter: running average of its gradient (`m`)
   * and of the gradient squared (`v`), as used by the next step. Undefined
   * before its first step.
   */
  moments(p: Tensor): { m: Float32Array; v: Float32Array } | undefined {
    const m = this.m.get(p), v = this.v.get(p);
    return m && v && m.length === p.size ? { m, v } : undefined;
  }

  /** Forget momentum (e.g. after loading different weights). */
  reset(): void {
    this.t = 0;
    this.m.clear();
    this.v.clear();
  }
}
