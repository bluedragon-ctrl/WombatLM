import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { Engine, EngineHost } from '../../src/train/engine';
import type { AnswerMsg, FromWorker } from '../../src/train/protocol';
import { buildMatrixModel, meanAttention } from '../../src/ui/matrixModel';

const ds = new Dataset(baseBundle());

function answer(steps = 30): AnswerMsg {
  const sent: FromWorker[] = [];
  const host: EngineHost = { post: (m) => sent.push(m), schedule: () => {}, now: () => 0 };
  const e = new Engine(host, ds);
  e.handle({ type: 'init', seed: 3 });
  e.handle({ type: 'step', n: steps });
  e.handle({ type: 'ask', id: 1, text: 'tygr', lang: ds.languageCodes[0], form: 'female' });
  return sent.find((m) => m.type === 'answer') as AnswerMsg;
}

describe('matrix model', () => {
  const a = answer();
  const m = buildMatrixModel(a, ds.vocab.tokens);
  const act = (name: string) => a.activations.find((x) => x.name === name)!;

  it('answers carry what every part of a block adds, the letter table and the scores', () => {
    expect(a.adds.map((x) => x.name)).toEqual(['enc0.self.add', 'enc0.ffn.add', 'dec0.self.add', 'dec0.cross.add', 'dec0.ffn.add']);
    expect(a.letterTable.length).toBe(ds.vocab.size * m.d);
    expect(a.logits.length).toBe(a.steps.length * ds.vocab.size);
  });

  it('reading: one station after another, all input rows travel together', () => {
    const read = m.stations.filter((s) => s.lane === 'read');
    expect(read.map((s) => s.kind)).toEqual(['letters', 'position', 'self', 'think', 'knobs']);
    for (const s of read) {
      expect(s.active).toEqual(a.input.map((_, i) => i));
      expect(s.grid.length).toBe(a.input.length * m.d);
    }
    // Letters + position + look + think = the block's output; the volume knobs give the memory.
    const think = read[3].grid, out = act('enc0.out').data;
    for (let i = 0; i < out.length; i++) expect(think[i]).toBeCloseTo(out[i], 4);
    expect(Array.from(read[4].grid)).toEqual(Array.from(act('memory').data));
  });

  it('writing: one pass per token, only the newest row is new, the guess matches the answer', () => {
    expect(m.passStart.length).toBe(1 + a.steps.length);
    const V = ds.vocab.size;
    a.steps.forEach((step, k) => {
      const pass = m.stations.filter((s) => s.pass === k);
      expect(pass.map((s) => s.kind)).toEqual(['letters', 'position', 'self', 'cross', 'think', 'knobs', 'guess']);
      for (const s of pass) {
        expect(s.active).toEqual([k]);
        expect(s.tokens.length).toBe(k + 1);
        expect(s.grid.length).toBe((k + 1) * m.d);
      }
      const think = pass[4].grid, out = act('dec0.out').data;
      for (let i = 0; i < think.length; i++) expect(think[i]).toBeCloseTo(out[i], 4);
      const guess = pass[6].guess!;
      expect(guess.scores.length).toBe(V);
      const best = guess.probs.indexOf(Math.max(...guess.probs));
      expect(ds.vocab.tokens[best]).toBe(step.chosen || '<eos>');
      expect(guess.probs[best]).toBeCloseTo(step.top[0].p, 4);
      // Looking back: the newest row's arrows add up to 100 %, over the rows so far.
      const self = pass[2].attention!;
      expect(self.keys).toBe(k + 1);
      let sum = 0;
      for (let j = 0; j < self.keys; j++) sum += meanAttention(self, 1, 0, j);
      expect(sum).toBeCloseTo(1, 4);
      expect(pass[3].attention!.keys).toBe(a.input.length);
    });
  });

  it('the trail of one number follows it station by station', () => {
    const tr = m.trail('read', 1, 4);
    const read = m.stations.filter((s) => s.lane === 'read');
    expect(tr).toEqual(read.map((s) => s.grid[1 * m.d + 4]));
    expect(m.trail('write', 0, 0).length).toBe(m.laneKinds.write.length);
  });
});
