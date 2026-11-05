import { describe, expect, it } from 'vitest';
import { Tape } from '../../src/nn/autograd';
import { Adam } from '../../src/nn/optim';
import { Rng } from '../../src/nn/rng';
import { Example, TRANSLATOR_DEFAULTS, Translator, makeBatch } from '../../src/nn/translator';
import { argmax } from '../../src/nn/transformer';
import { gradCheck } from './gradcheck';

// A toy vocabulary: 0 pad, 1 bos, 2 eos, 3 <x>, 4 <y>, 5.. letters.
const ids = { pad: 0, bos: 1, eos: 2 };
const V = 12;

const tinyConfig = { vocabSize: V, dModel: 8, heads: 1, ffn: 16, encLayers: 1, decLayers: 1, maxLen: 8 };

const pairs: Example[] = [
  { target: 3, src: [5, 6, 7], tgt: [7, 6, 5] },
  { target: 4, src: [7, 6, 5], tgt: [5, 6, 7] },
  { target: 3, src: [8, 9], tgt: [10, 11, 8] },
  { target: 4, src: [10, 11, 8], tgt: [8, 9] },
];

function train(model: Translator, examples: Example[], steps: number, lr = 1e-2) {
  const opt = new Adam(model.params.list, { lr, warmupSteps: 10 });
  const batch = makeBatch(examples, ids);
  let loss = Infinity;
  for (let s = 0; s < steps; s++) {
    opt.zeroGrad();
    const tape = new Tape();
    const l = tape.run(() => model.loss(batch, ids.pad));
    tape.backward(l);
    opt.step();
    loss = l.item();
  }
  return loss;
}

describe('translator', () => {
  it('full tiny model: gradients match finite differences', () => {
    const model = new Translator({ ...tinyConfig, heads: 1 }, 3);
    const batch = makeBatch(pairs.slice(0, 2), ids);
    const { maxError, worst } = gradCheck(() => model.loss(batch, ids.pad), model.params.list, {
      maxPerInput: 12, eps: 1e-2, atol: 2e-3, rtol: 3e-2,
    });
    expect(maxError, worst).toBeLessThan(1);
  });

  it('overfits 4 pairs to loss < 0.05', () => {
    const model = new Translator({ ...tinyConfig, dModel: 16, heads: 2, ffn: 32 }, 1);
    const loss = train(model, pairs, 400);
    expect(loss).toBeLessThan(0.05);
    const out = model.greedy(pairs.map((p) => ({ target: p.target, src: p.src })), ids);
    expect(out).toEqual(pairs.map((p) => p.tgt));
  });

  it('same seed gives identical weights after 50 steps', () => {
    const a = new Translator(tinyConfig, 99);
    const b = new Translator(tinyConfig, 99);
    train(a, pairs, 50);
    train(b, pairs, 50);
    a.params.list.forEach((t, i) => expect(Array.from(t.data)).toEqual(Array.from(b.params.list[i].data)));
    const c = new Translator(tinyConfig, 100);
    expect(Array.from(c.params.list[0].data)).not.toEqual(Array.from(a.params.list[0].data));
  });

  it('cached greedy decoding matches full recomputation', () => {
    const model = new Translator({ ...tinyConfig, dModel: 16, heads: 2 }, 5);
    train(model, pairs, 60);
    const fast = model.greedy([{ target: 3, src: [5, 6, 7] }, { target: 4, src: [8, 9] }], ids);
    for (const [k, input] of [{ target: 3, src: [5, 6, 7] }, { target: 4, src: [8, 9] }].entries()) {
      const out: number[] = [];
      for (let t = 0; t < tinyConfig.maxLen - 1; t++) {
        const batch = makeBatch([{ ...input, tgt: out }], ids);
        const logits = model.logits(batch);
        const next = argmax(logits.data, out.length * V, V);
        if (next === ids.eos) break;
        out.push(next);
      }
      expect(fast[k]).toEqual(out);
    }
  });

  it('inspect returns one probability row per output step and attention maps', () => {
    const model = new Translator({ ...tinyConfig, dModel: 16, heads: 2, ffn: 32 }, 1);
    train(model, pairs, 400);
    const tr = model.inspect(3, [5, 6, 7], ids);
    expect(tr.output).toEqual([7, 6, 5]);
    expect(tr.probs.length).toBe(4); // 3 letters + <eos>
    expect(argmax(tr.probs[3], 0, V)).toBe(ids.eos);
    const cross = tr.attention['dec0.cross'];
    expect(cross.heads).toBe(2);
    expect(cross.Tq).toBe(4);
    expect(cross.Tk).toBe(4);
  });

  it('default model has about 25k parameters', () => {
    const model = new Translator({ ...TRANSLATOR_DEFAULTS, vocabSize: 49 }, 1);
    const n = model.params.count();
    console.log(`Translator parameter count (vocab 49): ${n}`);
    expect(n).toBeGreaterThan(20_000);
    expect(n).toBeLessThan(30_000);
  });

  it('vocabulary growth keeps old weights and old behaviour', () => {
    const model = new Translator({ ...tinyConfig, dModel: 16, heads: 2, ffn: 32 }, 1);
    train(model, pairs, 400);
    const oldTokens = Array.from({ length: V }, (_, i) => `t${i}`);
    const before = model.greedy(pairs.map((p) => ({ target: p.target, src: p.src })), ids);
    const emb = model.tokEmb.data.slice();
    model.growVocab(oldTokens, [...oldTokens, 'new1', 'new2'], new Rng(1));
    expect(model.tokEmb.shape).toEqual([V + 2, 16]);
    expect(Array.from(model.tokEmb.data.subarray(0, emb.length))).toEqual(Array.from(emb));
    const after = model.greedy(pairs.map((p) => ({ target: p.target, src: p.src })), ids);
    expect(after).toEqual(before);
    // It can keep training at the new size.
    expect(train(model, pairs, 5)).toBeLessThan(0.5);
  });
});
