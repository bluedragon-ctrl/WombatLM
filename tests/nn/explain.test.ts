import { describe, expect, it } from 'vitest';
import { Translator } from '../../src/nn/translator';
import { rowSoftmax } from '../../src/nn/transformer';

const ids = { pad: 0, bos: 1, eos: 2 };
const V = 12;
const config = { vocabSize: V, dModel: 16, heads: 2, ffn: 32, encLayers: 1, decLayers: 1, maxLen: 8, targetStart: true };

describe('explain (look inside one letter)', () => {
  const model = new Translator(config, 5);
  const src = [5, 6, 7];
  const tr = model.inspect(3, src, ids);

  for (const position of [0, 1, 2]) {
    it(`position ${position}: every added list splits exactly into its contributions`, () => {
      const ex = model.explain(3, src, ids, position);
      const d = config.dModel;
      expect(ex.position).toBe(position);
      expect(ex.decIn).toHaveLength(position + 1);
      expect(ex.decIn[0]).toBe(3); // target-token start
      expect(ex.stages.map((s) => s.name)).toEqual(['dec0.self', 'dec0.cross', 'dec0.ffn']);
      for (const s of ex.stages) {
        for (let j = 0; j < d; j++) {
          let sum = s.bias[j];
          for (let k = 0; k < s.sources; k++) sum += s.contrib[k * d + j];
          expect(sum).toBeCloseTo(s.add[j], 4);
        }
        if (s.kind === 'attend') {
          for (let h = 0; h < s.heads; h++) {
            const w = s.weights.subarray(h * s.sources, (h + 1) * s.sources);
            expect(w.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 4);
          }
        }
      }
      expect(ex.stages[0].sources).toBe(position + 1); // looks back only at the answer so far
      expect(ex.stages[1].sources).toBe(src.length + 1); // task token + letters

      // The scores: final · letter table + favourite letters, the same as answering.
      for (let t = 0; t < V; t++) {
        let s = model.outBias.data[t];
        for (let i = 0; i < d; i++) s += ex.final[i] * model.tokEmb.data[t * d + i];
        expect(s).toBeCloseTo(ex.logits[t], 4);
      }
      if (position < tr.probs.length) {
        const p = rowSoftmax(ex.logits, 0, V);
        for (let t = 0; t < V; t++) expect(p[t]).toBeCloseTo(tr.probs[position][t], 5);
      }
    });
  }

  it('"what would it write now" after the last stage is the real guess', () => {
    const ex = model.explain(3, src, ids, 2);
    expect(ex.lens).toHaveLength(1 + ex.stages.length);
    const last = ex.lens[ex.lens.length - 1];
    for (let t = 0; t < V; t++) expect(last[t]).toBeCloseTo(ex.logits[t], 4);
  });

  it('the running numbers after the last step are the decoder output', () => {
    const ex = model.explain(3, src, ids, 1);
    const sig = tr.activations['dec0.out'];
    const last = ex.stages[ex.stages.length - 1].x;
    for (let i = 0; i < config.dModel; i++) expect(last[i]).toBeCloseTo(sig.data[1 * sig.d + i], 4);
    // "Follow the signal" gets the in/out strips, the encoder result and the decoder's final numbers.
    expect(Object.keys(tr.activations).sort()).toEqual(['dec.final', 'dec.in', 'dec0.out', 'enc.in', 'enc0.out', 'memory']);
  });
});
