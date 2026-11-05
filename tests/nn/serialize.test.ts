import { describe, expect, it } from 'vitest';
import { Rng } from '../../src/nn/rng';
import { decodeWeights, encodeWeights, fromHalf, loadInto, toHalf } from '../../src/nn/serialize';
import { TRANSLATOR_DEFAULTS, Translator } from '../../src/nn/translator';

describe('serialize', () => {
  const model = new Translator({ ...TRANSLATOR_DEFAULTS, vocabSize: 49 }, 11);

  it('f32 round trip is exact', () => {
    const buf = encodeWeights(model.params.list, { kind: 'translator', step: 7 }, 'f32');
    const { header, tensors } = decodeWeights(buf);
    expect(header.meta.step).toBe(7);
    const copy = new Translator({ ...TRANSLATOR_DEFAULTS, vocabSize: 49 }, 12);
    loadInto(copy.params, tensors);
    copy.params.list.forEach((t, i) => expect(Array.from(t.data)).toEqual(Array.from(model.params.list[i].data)));
  });

  it('f16 and int8 round trips are close', () => {
    for (const enc of ['f16', 'int8'] as const) {
      const { tensors } = decodeWeights(encodeWeights(model.params.list, {}, enc));
      for (const t of model.params.list) {
        const d = tensors.get(t.name)!;
        let max = 0;
        for (let i = 0; i < t.size; i++) max = Math.max(max, Math.abs(t.data[i]));
        const tol = enc === 'f16' ? 2e-3 * Math.max(1, max) : max / 127;
        for (let i = 0; i < t.size; i++) expect(Math.abs(d[i] - t.data[i])).toBeLessThanOrEqual(tol + 1e-7);
      }
    }
  });

  it('int8 file is about a quarter of f32', () => {
    const f32 = encodeWeights(model.params.list, {}, 'f32').byteLength;
    const i8 = encodeWeights(model.params.list, {}, 'int8').byteLength;
    expect(i8).toBeLessThan(f32 * 0.4);
  });

  it('half-precision conversion handles common values', () => {
    for (const v of [0, 1, -2, 0.5, 0.1, 65504, 1e-5, -3.14159]) {
      expect(Math.abs(fromHalf(toHalf(v)) - v)).toBeLessThanOrEqual(Math.abs(v) * 1e-3 + 1e-7);
    }
  });
});

describe('rng', () => {
  it('is deterministic per seed', () => {
    const a = new Rng(5), b = new Rng(5), c = new Rng(6);
    const sa = Array.from({ length: 5 }, () => a.next());
    expect(Array.from({ length: 5 }, () => b.next())).toEqual(sa);
    expect(Array.from({ length: 5 }, () => c.next())).not.toEqual(sa);
  });

  it('normal samples have roughly mean 0 and sd 1', () => {
    const r = new Rng(1);
    const xs = Array.from({ length: 20000 }, () => r.normal());
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length);
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(Math.abs(sd - 1)).toBeLessThan(0.03);
  });
});
