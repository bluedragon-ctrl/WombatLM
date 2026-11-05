import { Tape } from '../../src/nn/autograd';
import { Rng } from '../../src/nn/rng';
import { Tensor } from '../../src/nn/tensor';

export function randTensor(rng: Rng, shape: number[], std = 1, requiresGrad = true): Tensor {
  const t = Tensor.zeros(shape, requiresGrad);
  for (let i = 0; i < t.size; i++) t.data[i] = rng.normal() * std;
  return t;
}

export interface GradCheckResult {
  maxError: number;
  worst: string;
}

/**
 * Compare analytic gradients (from the tape) with central finite differences.
 * `lossFn` must build a scalar loss from the current values of `inputs`.
 */
export function gradCheck(
  lossFn: () => Tensor, inputs: Tensor[], opts: { eps?: number; atol?: number; rtol?: number; maxPerInput?: number; rng?: Rng } = {},
): GradCheckResult {
  const eps = opts.eps ?? 1e-3;
  const atol = opts.atol ?? 1e-3;
  const rtol = opts.rtol ?? 1e-2;
  const rng = opts.rng ?? new Rng(123);
  for (const t of inputs) t.grad = null;
  const tape = new Tape();
  const loss = tape.run(lossFn);
  tape.backward(loss);
  let maxError = 0;
  let worst = '';
  for (const [n, t] of inputs.entries()) {
    const analytic = t.grad ? t.grad.slice() : new Float32Array(t.size);
    const idx = Array.from({ length: t.size }, (_, i) => i);
    const sample = opts.maxPerInput && t.size > opts.maxPerInput ? rng.shuffle(idx).slice(0, opts.maxPerInput) : idx;
    for (const i of sample) {
      const orig = t.data[i];
      t.data[i] = orig + eps;
      const lp = lossFn().item();
      t.data[i] = orig - eps;
      const lm = lossFn().item();
      t.data[i] = orig;
      const numeric = (lp - lm) / (2 * eps);
      const err = Math.abs(numeric - analytic[i]) / (atol + rtol * Math.max(Math.abs(numeric), Math.abs(analytic[i])));
      if (err > maxError) {
        maxError = err;
        worst = `input ${n}${t.name ? ` (${t.name})` : ''}[${i}]: analytic ${analytic[i]} numeric ${numeric}`;
      }
    }
  }
  return { maxError, worst };
}
