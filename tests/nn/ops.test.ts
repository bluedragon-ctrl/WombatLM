import { describe, expect, it } from 'vitest';
import {
  add, bmm, concat, crossEntropy, gather, gelu, layerNorm, matmul, mergeHeads, mul, nllFromProbs, pointerMix, reshape, scale,
  softmax, splitHeads, sum, transpose,
} from '../../src/nn/ops';
import { Rng } from '../../src/nn/rng';
import { Tensor } from '../../src/nn/tensor';
import { NEG } from '../../src/nn/transformer';
import { gradCheck, randTensor } from './gradcheck';

const rng = new Rng(42);

/** Scalar loss = sum(out * R) for a fixed random R, so every output element matters differently. */
function project(out: Tensor, seed = 7): Tensor {
  const r = randTensor(new Rng(seed), out.shape, 1, false);
  return sum(mul(out, r));
}

function check(name: string, fn: () => Tensor, inputs: Tensor[]) {
  it(`${name}: gradient matches finite differences`, () => {
    const { maxError, worst } = gradCheck(() => project(fn()), inputs);
    expect(maxError, worst).toBeLessThan(1);
  });
}

describe('op gradients', () => {
  { const a = randTensor(rng, [3, 4]), b = randTensor(rng, [4, 5]); check('matmul', () => matmul(a, b), [a, b]); }
  { const a = randTensor(rng, [3, 4]), b = randTensor(rng, [5, 4]); check('matmul transB', () => matmul(a, b, true), [a, b]); }
  { const a = randTensor(rng, [2, 3, 4]), b = randTensor(rng, [2, 4, 3]); check('bmm', () => bmm(a, b), [a, b]); }
  { const a = randTensor(rng, [2, 3, 4]), b = randTensor(rng, [2, 5, 4]); check('bmm transB', () => bmm(a, b, true), [a, b]); }
  { const a = randTensor(rng, [2, 3, 4]); check('transpose', () => transpose(a), [a]); }
  { const a = randTensor(rng, [3, 4]), b = randTensor(rng, [3, 4]); check('add', () => add(a, b), [a, b]); }
  { const a = randTensor(rng, [3, 4]), b = randTensor(rng, [4]); check('add (bias broadcast)', () => add(a, b), [a, b]); }
  { const a = randTensor(rng, [3, 4]), b = randTensor(rng, [3, 4]); check('mul', () => mul(a, b), [a, b]); }
  { const a = randTensor(rng, [3, 4]); check('scale', () => scale(a, -0.7), [a]); }
  { const a = randTensor(rng, [5, 3]); check('gelu', () => gelu(a), [a]); }
  { const t = randTensor(rng, [6, 3]); check('gather', () => gather(t, Int32Array.from([0, 2, 2, 5])), [t]); }
  {
    const x = randTensor(rng, [4, 6]), g = randTensor(rng, [6]), b = randTensor(rng, [6]);
    check('layerNorm', () => layerNorm(x, g, b), [x, g, b]);
  }
  { const x = randTensor(rng, [3, 5]); check('softmax', () => softmax(x), [x]); }
  {
    // [B*H, Tq, Tk] with B=2, H=2 and a mask shared by the heads.
    const x = randTensor(rng, [4, 3, 3]);
    const mask = new Float32Array(2 * 3 * 3);
    mask[2] = NEG; mask[5] = NEG; mask[9 + 1] = NEG;
    check('softmax (mask, heads)', () => softmax(x, mask, 2), [x]);
  }
  { const x = randTensor(rng, [2, 6]); check('reshape', () => reshape(x, [3, 4]), [x]); }
  { const x = randTensor(rng, [2 * 3, 4]); check('splitHeads', () => splitHeads(x, 2, 3, 2), [x]); }
  { const x = randTensor(rng, [4, 3, 2]); check('mergeHeads', () => mergeHeads(x, 2, 2), [x]); }
  { const a = randTensor(rng, [2, 3]), b = randTensor(rng, [1, 3]); check('concat', () => concat([a, b]), [a, b]); }
  {
    const logits = randTensor(rng, [4, 5]);
    const targets = Int32Array.from([1, 0, 4, 2]);
    it('crossEntropy (with ignore index): gradient matches', () => {
      const { maxError, worst } = gradCheck(() => crossEntropy(logits, targets, 0), [logits]);
      expect(maxError, worst).toBeLessThan(1);
    });
  }
});

describe('op values', () => {
  it('matmul computes the right numbers', () => {
    const a = Tensor.from([1, 2, 3, 4], [2, 2]);
    const b = Tensor.from([5, 6, 7, 8], [2, 2]);
    expect(Array.from(matmul(a, b).data)).toEqual([19, 22, 43, 50]);
    expect(Array.from(matmul(a, b, true).data)).toEqual([17, 23, 39, 53]);
  });

  it('softmax rows sum to 1 and masked entries are 0', () => {
    const x = Tensor.from([1, 2, 3, 0, 0, 0], [1, 2, 3]);
    const mask = Float32Array.from([0, 0, NEG, 0, NEG, NEG]);
    const p = softmax(x, mask).data;
    expect(p[0] + p[1] + p[2]).toBeCloseTo(1, 6);
    expect(p[2]).toBe(0);
    expect(p[3]).toBeCloseTo(1, 6);
  });

  it('crossEntropy equals -log(p) of the right answer and ignores padding', () => {
    const logits = Tensor.from([0, 0, 0, 0, 5, 5], [2, 3]);
    const loss = crossEntropy(logits, Int32Array.from([1, 0]), 0).item();
    expect(loss).toBeCloseTo(Math.log(3), 5);
  });

  it('layerNorm output has mean 0 and variance 1 per row', () => {
    const x = Tensor.from([1, 2, 3, 4, 10, 0, -10, 5], [2, 4]);
    const y = layerNorm(x, Tensor.from([1, 1, 1, 1], [4]), Tensor.zeros([4])).data;
    for (let r = 0; r < 2; r++) {
      const row = Array.from(y.subarray(r * 4, r * 4 + 4));
      const mean = row.reduce((a, b) => a + b) / 4;
      const v = row.reduce((a, b) => a + (b - mean) ** 2, 0) / 4;
      expect(mean).toBeCloseTo(0, 5);
      expect(v).toBeCloseTo(1, 3);
    }
  });
});

describe('copy mechanism ops', () => {
  const r = new Rng(9);
  const logits = randTensor(r, [4, 6]);
  const raw = randTensor(r, [4, 3]);
  const gate = randTensor(r, [4, 1]);
  const src = Int32Array.from([2, 5, 2, 1, 0, 3]); // 2 items × 3 positions
  it('pointerMix: gradient matches finite differences', () => {
    const { maxError, worst } = gradCheck(() => project(pointerMix(logits, softmax(raw), src, 2, gate)), [logits, raw, gate]);
    expect(maxError, worst).toBeLessThan(1);
  });
  it('pointerMix rows are probabilities and copy mass lands on input tokens', () => {
    const p = pointerMix(logits, softmax(raw), src, 2, Tensor.from([-50, -50, -50, -50], [4, 1])).data;
    for (let i = 0; i < 4; i++) {
      const row = Array.from(p.subarray(i * 6, i * 6 + 6));
      expect(row.reduce((a, b) => a + b)).toBeCloseTo(1, 5);
      const allowed = i < 2 ? [2, 5] : [1, 0, 3];
      row.forEach((v, j) => { if (!allowed.includes(j)) expect(v).toBeLessThan(1e-6); });
    }
  });
  it('nllFromProbs: gradient matches finite differences', () => {
    const probs = randTensor(r, [3, 4]);
    const { maxError, worst } = gradCheck(() => nllFromProbs(softmax(probs), Int32Array.from([1, 0, 3]), 0), [probs]);
    expect(maxError, worst).toBeLessThan(1);
  });
});
