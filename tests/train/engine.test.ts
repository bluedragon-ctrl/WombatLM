import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { Engine, EngineHost } from '../../src/train/engine';
import type { AnswerMsg, EvalMsg, FromWorker, LetterMapMsg, ReadyMsg, TickMsg, WeightsMsg } from '../../src/train/protocol';

const ds = new Dataset(baseBundle());

/** A host with a manual clock and a manual task queue. */
function fakeHost() {
  const sent: FromWorker[] = [];
  const tasks: { fn: () => void; at: number }[] = [];
  let clock = 0;
  const host: EngineHost = {
    post: (m) => sent.push(m),
    schedule: (fn, ms) => tasks.push({ fn, at: clock + ms }),
    now: () => (clock += 0.5), // every reading advances time a little, so slices end
  };
  const runDue = (n: number) => {
    for (let i = 0; i < n && tasks.length; i++) {
      tasks.sort((a, b) => a.at - b.at);
      const t = tasks.shift()!;
      clock = Math.max(clock, t.at);
      t.fn();
    }
  };
  const of = <T extends FromWorker['type']>(type: T) => sent.filter((m) => m.type === type) as Extract<FromWorker, { type: T }>[];
  return { host, sent, tasks, runDue, of };
}

describe('Engine', () => {
  it('init sends ready, an evaluation with inventions and a tick', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    const ready = h.of('ready')[0] as ReadyMsg;
    expect(ready.params).toBeGreaterThan(20000);
    expect(ready.languages.map((l) => l.code)).toEqual(ds.languageCodes);
    expect(ready.animals.length).toBe(ds.animals.length);
    const ev = h.of('eval')[0] as EvalMsg;
    expect(ev.step).toBe(0);
    expect(ev.inventions).toBe(true);
    expect((h.of('tick')[0] as TickMsg).step).toBe(0);
  });

  it('step trains while paused and reports losses', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'step', n: 3 });
    const tick = h.of('tick').at(-1) as TickMsg;
    expect(tick.step).toBe(3);
    expect(tick.losses.map((l) => l.step)).toEqual([1, 2, 3]);
    expect(h.of('eval').at(-1)!.step).toBe(3);
  });

  it('play runs slices until pause; stale slices do nothing', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3, speed: 'max' });
    e.handle({ type: 'play' });
    h.runDue(3);
    const stepped = e.trainer.step;
    expect(stepped).toBeGreaterThan(3);
    e.handle({ type: 'pause' });
    h.runDue(5);
    expect(e.trainer.step).toBe(stepped);
    expect((h.of('tick').at(-1) as TickMsg).running).toBe(false);
  });

  it('slow speed trains about speed × time steps', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3, speed: 2 });
    e.handle({ type: 'play' });
    h.runDue(21); // first slice right away, then 20 slices 100 ms apart -> 1 + ~2 s × 2 steps/s
    expect(e.trainer.step).toBeGreaterThanOrEqual(4);
    expect(e.trainer.step).toBeLessThanOrEqual(6);
  });

  it('answers with guesses, attention and a badge', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    const lang = ds.languageCodes[0];
    const known = ds.formTasks((a) => a.split === 'train').find((t) => t.to === lang)!;
    const hidden = ds.formTasks((a) => a.split === 'unseen').find((t) => t.to === lang)!;
    const invented = ds.missingFormTasks().find((t) => t.to === lang)!;
    e.handle({ type: 'ask', id: 1, text: known.source, lang, form: known.form });
    e.handle({ type: 'ask', id: 2, text: hidden.source.toUpperCase(), lang, form: hidden.form });
    e.handle({ type: 'ask', id: 3, text: invented.source, lang, form: invented.form });
    e.handle({ type: 'ask', id: 4, text: 'qqq', lang, form: 'young' });
    const [a, b, c, d] = h.of('answer') as AnswerMsg[];
    expect(a.badge).toBe('real');
    expect(a.answer!.accepted[0]).toBe(known.answer);
    expect(b.badge).toBe('hidden');
    expect(b.text).toBe(hidden.source);
    expect(c.badge).toBe('invention');
    expect(d.badge).toBe('free');
    expect(a.steps.length).toBeGreaterThan(0);
    expect(a.steps[0].top.length).toBe(5);
    expect(a.attention.length).toBeGreaterThan(0);
    expect(a.input[0]).toBe(`<${lang}:${known.form}>`);
  });

  it('answers carry the activations in the order the signal flows', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'ask', id: 1, text: 'tygr', lang: ds.languageCodes[0], form: 'female' });
    const a = h.of('answer')[0] as AnswerMsg;
    expect(a.activations.map((x) => x.name)).toEqual(['enc.in', 'enc0.out', 'dec.in', 'dec0.out']);
    const encIn = a.activations[0];
    expect(encIn.T).toBe(a.input.length);
    expect(encIn.data.length).toBe(encIn.T * encIn.d);
  });

  it('ready lists every matrix with its shape and axis kinds', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    const ready = h.of('ready')[0] as ReadyMsg;
    expect(ready.matrices.reduce((n, m) => n + m.rows * m.cols, 0)).toBe(ready.params);
    const emb = ready.matrices.find((m) => m.name === 'tokEmb')!;
    expect([emb.rows, emb.cols, emb.rowKind, emb.colKind]).toEqual([ds.vocab.size, 32, 'token', 'dim']);
    const bias = ready.matrices.find((m) => m.name === 'outBias')!;
    expect([bias.rows, bias.cols, bias.colKind]).toEqual([1, ds.vocab.size, 'token']);
  });

  it('inspect sends weights, their change and the gradient', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    const names = ['tokEmb', 'dec0.cross.q.W'];
    const last = () => h.of('weights').at(-1) as WeightsMsg;

    e.handle({ type: 'inspect', id: 1, matrixIds: names, mode: 'weights' });
    expect(last().matrices.map((m) => m.name)).toEqual(names);
    expect(Array.from(last().matrices[0].data)).toEqual(Array.from(e.trainer.model.tokEmb.data));

    // Gradient before any training: all zeros, steps 0.
    e.handle({ type: 'inspect', id: 2, matrixIds: names, mode: 'grad' });
    expect(last().steps).toBe(0);
    expect(last().matrices[0].data.every((v) => v === 0)).toBe(true);

    // Delta: the first request sets the baseline (no change yet).
    e.handle({ type: 'inspect', id: 3, matrixIds: names, mode: 'delta' });
    expect(last().steps).toBe(0);
    const before = e.trainer.model.tokEmb.data.slice();
    e.handle({ type: 'step', n: 3 });
    e.handle({ type: 'inspect', id: 4, matrixIds: names, mode: 'delta' });
    expect(last().steps).toBe(3);
    const d = last().matrices[0].data;
    const after = e.trainer.model.tokEmb.data;
    for (let i = 0; i < d.length; i += 97) expect(d[i]).toBeCloseTo(after[i] - before[i], 6);
    expect(d.some((v) => v !== 0)).toBe(true);
    // No training in between: the previous change is shown again.
    e.handle({ type: 'inspect', id: 5, matrixIds: names, mode: 'delta' });
    expect(last().steps).toBe(3);
    expect(Array.from(last().matrices[0].data)).toEqual(Array.from(d));

    e.handle({ type: 'inspect', id: 6, matrixIds: names, mode: 'grad' });
    expect(last().steps).toBe(1);
    expect(last().matrices[0].data.some((v) => v !== 0)).toBe(true);
    expect(last().id).toBe(6);
  });

  it('the first change request after training shows the latest step', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'step', n: 2 });
    e.handle({ type: 'inspect', id: 1, matrixIds: ['tokEmb'], mode: 'delta' });
    const w = h.of('weights').at(-1) as WeightsMsg;
    expect(w.steps).toBe(1);
    expect(w.matrices[0].data.some((v) => v !== 0)).toBe(true);
  });

  it('inspect sends the letter map: characters, end of word and task tokens', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'inspect', id: 7, matrixIds: [], mode: 'weights', letterMap: true });
    expect(h.of('weights').length).toBe(0);
    const m = h.of('letterMap')[0] as LetterMapMsg;
    const v = ds.vocab;
    expect(m.ids).toContain(v.eos);
    expect(m.ids).not.toContain(v.pad);
    expect(m.ids).toContain(v.id('<cs:female>'));
    expect(m.coords.length).toBe(m.ids.length * 2);
    expect(m.explained[0]).toBeGreaterThanOrEqual(m.explained[1]);
    expect(m.coords.every(Number.isFinite)).toBe(true);
  });

  it('reports errors instead of throwing', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'ask', id: 1, text: 'abc', lang: 'xx', form: 'young' });
    expect(h.of('error').length).toBe(1);
  });
});
