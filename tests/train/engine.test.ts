import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { EVAL, Engine, EngineHost } from '../../src/train/engine';
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
  /** Run queued work (background tests) until nothing is left. Only while not training. */
  const settle = () => { for (let i = 0; i < 1000 && tasks.length; i++) runDue(1); };
  const of = <T extends FromWorker['type']>(type: T) => sent.filter((m) => m.type === type) as Extract<FromWorker, { type: T }>[];
  return { host, sent, tasks, runDue, settle, of };
}

describe('Engine', () => {
  it('init sends ready and a tick, and no test until asked', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    const ready = h.of('ready')[0] as ReadyMsg;
    expect(ready.params).toBeGreaterThan(20000);
    expect(ready.languages.map((l) => l.code)).toEqual(ds.languageCodes);
    expect(ready.animals.length).toBe(ds.animals.length);
    expect((h.of('tick')[0] as TickMsg).step).toBe(0);
    expect(h.of('eval')).toHaveLength(0);
  });

  it('step trains while paused and reports losses', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'step', n: 3 });
    const tick = h.of('tick').at(-1) as TickMsg;
    expect(tick.step).toBe(3);
    expect(tick.losses.map((l) => l.step)).toEqual([1, 2, 3]);
    expect(h.of('eval')).toHaveLength(0);
  });

  it('tests every animal only while the Test tab is open', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'watchTests', on: true });
    expect(h.of('eval')).toHaveLength(0); // the test runs in the background…
    h.settle();
    const first = h.of('eval')[0] as EvalMsg; // …and reports when done
    expect(first.step).toBe(0);
    expect(first.inventions).toBe(true);
    e.handle({ type: 'watchTests', on: true }); // already tested at this step: no new test
    h.settle();
    expect(h.of('eval')).toHaveLength(1);
    e.handle({ type: 'step', n: 2 });
    h.settle();
    expect(h.of('eval').at(-1)!.step).toBe(2);
    e.handle({ type: 'watchTests', on: false });
    e.handle({ type: 'step', n: 2 });
    h.settle();
    expect(h.of('eval').at(-1)!.step).toBe(2);
    e.handle({ type: 'evaluate' }); // on request
    h.settle();
    expect(h.of('eval').at(-1)!.step).toBe(4);
  });

  it('a test answers with the weights of the step it started at, while training goes on', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    const twin = new Engine(fakeHost().host, ds);
    e.handle({ type: 'init', seed: 4 });
    twin.handle({ type: 'init', seed: 4 });
    e.handle({ type: 'step', n: 30 });
    twin.handle({ type: 'step', n: 30 });
    e.handle({ type: 'evaluate', inventions: true });
    h.runDue(1); // part of the test…
    e.handle({ type: 'step', n: 5 }); // …then more training before it finishes
    h.settle();
    const ev = h.of('eval');
    expect(ev).toHaveLength(1);
    expect(ev[0].step).toBe(30);
    const expected = twin.trainer.evaluate({ inventions: true });
    expect(ev[0].results.map((r) => r.output)).toEqual(expected.results.map((r) => r.output));
    expect(ev[0].scores).toEqual(expected.scores);
    expect(e.trainer.step).toBe(35);
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
    const known = ds.formTasks((a) => !a.unseen).find((t) => t.to === lang)!;
    const hidden = ds.formTasks((a) => a.unseen).find((t) => t.to === lang)!;
    const invented = ds.missingFormTasks().find((t) => t.to === lang)!;
    e.handle({ type: 'ask', id: 1, text: known.source, lang, form: known.form });
    e.handle({ type: 'ask', id: 2, text: hidden.source.toUpperCase(), lang, form: hidden.form });
    e.handle({ type: 'ask', id: 3, text: invented.source, lang, form: invented.form });
    e.handle({ type: 'ask', id: 4, text: 'qqq', lang, form: 'young' });
    const [a, b, c, d] = h.of('answer') as AnswerMsg[];
    expect(a.badge).toBe('real');
    expect(a.answer!.word).toBe(known.answer);
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
    expect(a.activations.map((x) => x.name)).toEqual(['enc.in', 'enc0.out', 'memory', 'dec.in', 'dec0.out', 'dec.final']);
    const encIn = a.activations[0];
    expect(encIn.T).toBe(a.input.length);
    expect(encIn.data.length).toBe(encIn.T * encIn.d);
  });

  it('answers carry the rows each token looks up: letter table + position = what goes in', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'step', n: 20 });
    e.handle({ type: 'ask', id: 1, text: 'tygr', lang: ds.languageCodes[0], form: 'female' });
    const a = h.of('answer')[0] as AnswerMsg;
    expect(a.inputIds.map((id) => ds.vocab.tokens[id])).toEqual(a.input);
    expect(a.vocabSize).toBe(ds.vocab.size);
    const check = (tok: Float32Array, pos: Float32Array, name: string, T: number) => {
      const act = a.activations.find((x) => x.name === name)!;
      expect(tok.length).toBe(T * act.d);
      for (let i = 0; i < T * act.d; i++) expect(tok[i] + pos[i]).toBeCloseTo(act.data[i], 5);
    };
    check(a.lookup.encTok, a.lookup.encPos, 'enc.in', a.input.length);
    // The writer gets one row per pass: the task token, then every written token but the last.
    check(a.lookup.decTok, a.lookup.decPos, 'dec.in', a.steps.length);
  });

  it('a single step while paused is recorded in detail; why explains any number of it', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'step', n: 10 });
    expect(h.of('stepDetail')).toHaveLength(0); // only for single steps
    e.handle({ type: 'step', n: 1 });
    const d = h.of('stepDetail')[0];
    expect(d.step).toBe(11);
    expect(d.numbers.total).toBe((h.of('ready')[0] as ReadyMsg).params);
    expect(d.top.change).toHaveLength(10);
    expect(Math.abs(d.top.change[0].after - d.top.change[0].before)).toBeCloseTo(d.numbers.max, 6);
    expect(d.split).toBe(true);
    const tick = h.of('tick').at(-1) as TickMsg;
    expect(tick.practice?.check).toBeDefined();
    expect(tick.practice!.examples.every((x) => x.check !== undefined)).toBe(true);

    const ref = d.top.push[0];
    e.handle({ type: 'why', id: 5, name: ref.name, index: ref.index });
    const w = h.of('why')[0];
    expect(w.id).toBe(5);
    expect(w.missing).toBeUndefined();
    expect(w.calc!.after).toBeCloseTo(w.now!, 6);
    expect(w.shares!.reduce((s, x) => s + x.share, 0)).toBeCloseTo(w.calc!.grad, 6);

    // After another step it is no longer the latest: nothing to explain.
    e.handle({ type: 'step', n: 10 });
    e.handle({ type: 'why', id: 6, name: ref.name, index: ref.index });
    expect(h.of('why')[1].missing).toBe(true);
    expect((h.of('tick').at(-1) as TickMsg).practice?.check).toBeUndefined();
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

  it('weights carry a change summary of every matrix', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    const last = () => h.of('weights').at(-1) as WeightsMsg;
    e.handle({ type: 'inspect', id: 1, matrixIds: ['tokEmb'], mode: 'weights' });
    expect(last().summary.map((m) => m.name)).toEqual(e.trainer.model.params.names());
    expect(last().summary.every((m) => m.steps === 0 && m.change === 0)).toBe(true);
    expect(last().summary.find((m) => m.name === 'tokEmb')!.norm).toBeGreaterThan(0);
    e.handle({ type: 'step', n: 4 });
    e.handle({ type: 'inspect', id: 2, matrixIds: ['tokEmb'], mode: 'grad' });
    const s = last().summary.find((m) => m.name === 'tokEmb')!;
    expect(s.steps).toBe(4);
    expect(s.change).toBeGreaterThan(0);
    // Nothing trained since: the same summary again.
    e.handle({ type: 'inspect', id: 3, matrixIds: ['tokEmb'], mode: 'delta' });
    expect(last().summary.find((m) => m.name === 'tokEmb')).toEqual(s);
  });

  it('explain looks inside one letter of the answer', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'step', n: 2 });
    e.handle({ type: 'ask', id: 1, text: 'tygr', lang: 'cs', form: 'female' });
    const ans = h.of('answer').at(-1) as AnswerMsg;
    e.handle({ type: 'explain', id: 5, text: 'tygr', lang: 'cs', form: 'female', position: 1 });
    const ex = h.of('explain').at(-1)!;
    expect(ex.id).toBe(5);
    expect(ex.output).toBe(ans.output);
    expect(ex.columns).toBe(ans.steps.length);
    expect(ex.chosen).toBe(ans.steps[1].chosen);
    expect(ex.decIn).toHaveLength(2);
    expect(ex.input).toEqual(ans.input);
    expect(ex.stages.map((st) => st.name)).toEqual(['dec0.self', 'dec0.cross', 'dec0.ffn']);
    expect(ex.guesses).toHaveLength(5);
    // Same guesses as the guess bars.
    ex.guesses.forEach((g, i) => { expect(g.char).toBe(ans.steps[1].top[i].char); expect(g.p).toBeCloseTo(ans.steps[1].top[i].p, 5); });
    const g = ex.guesses[0];
    expect(g.contrib.reduce((a, b) => a + b, g.bias)).toBeCloseTo(g.score, 4);
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
    // Closest letters: only for letters, never itself, never a task token.
    const isTask = (i: number) => v.tokens[m.ids[i]].startsWith('<') && m.ids[i] !== v.eos;
    expect(m.near.length).toBe(m.ids.length);
    m.near.forEach((list, i) => {
      if (isTask(i)) { expect(list).toEqual([]); return; }
      expect(list.length).toBe(4);
      expect(list).not.toContain(i);
      expect(list.some(isTask)).toBe(false);
    });
    // How often each letter appears in the learning data.
    expect(m.counts.length).toBe(m.ids.length);
    expect(m.counts[m.ids.indexOf(v.id('a'))]).toBeGreaterThan(100);
    expect(m.counts[m.ids.indexOf(v.eos)]).toBe(-1);
  });

  it('ticks carry the latest practice batch once per new step', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    expect((h.of('tick').at(-1) as TickMsg).practice).toBeUndefined();
    e.handle({ type: 'step', n: 2 });
    const p = (h.of('tick').at(-1) as TickMsg).practice!;
    expect(p.step).toBe(2);
    expect(p.examples.length).toBe(16); // the whole batch (default batch size 16)
    expect(p.examples[0].target.length).toBeGreaterThan(0);
    e.handle({ type: 'pause' });
    expect((h.of('tick').at(-1) as TickMsg).practice).toBeUndefined();
  });

  it('answers before each step while paused or slow; at max speed only for the open Model learning tab', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3, speed: 'max' });
    e.handle({ type: 'step', n: 1 });
    const stepped = (h.of('tick').at(-1) as TickMsg).practice!;
    expect(stepped.examples.every((x) => typeof x.answer === 'string')).toBe(true);
    expect(stepped.rightCount).toBeDefined();
    // Max speed, Model learning tab closed: no answering (full speed).
    e.handle({ type: 'play' });
    h.runDue(2);
    expect(h.of('tick').filter((t) => (t as TickMsg).practice?.step! > 1).every((t) => (t as TickMsg).practice!.rightCount === undefined)).toBe(true);
    // Model learning tab open: only answered batches are sent.
    e.handle({ type: 'watchPractice', on: true });
    const from = h.sent.length;
    h.runDue(6);
    const ticks = h.sent.slice(from).filter((m) => m.type === 'tick' && m.practice) as TickMsg[];
    expect(ticks.length).toBeGreaterThan(0);
    expect(ticks.every((t) => t.practice!.rightCount !== undefined)).toBe(true);
    e.handle({ type: 'pause' });
  });

  it('quiet learning skips tests and answering first while running, and catches up on pause', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3, speed: 'max' });
    e.handle({ type: 'watchTests', on: true });
    e.handle({ type: 'watchPractice', on: true });
    e.handle({ type: 'quiet', on: true });
    expect((h.of('tick').at(-1) as TickMsg).quiet).toBe(true);
    h.settle();
    const evals = h.of('eval').length;
    e.handle({ type: 'play' });
    while (e.trainer.step < EVAL.everySteps + 5) h.runDue(1);
    expect(h.of('eval')).toHaveLength(evals); // no test while running quietly
    expect(h.of('tick').some((t) => t.practice)).toBe(false);
    e.handle({ type: 'pause' });
    h.settle();
    expect(h.of('eval').at(-1)!.step).toBe(e.trainer.step); // tested on pause
  });

  it('stops by itself at the step goal; Play does nothing there, Step still works', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3, speed: 'max' });
    e.handle({ type: 'watchTests', on: true });
    e.handle({ type: 'setGoal', steps: 7 });
    e.handle({ type: 'play' });
    for (let i = 0; i < 50 && h.tasks.length; i++) h.runDue(1);
    expect(e.trainer.step).toBe(7);
    expect(e.running).toBe(false);
    const last = h.of('tick').at(-1) as TickMsg;
    expect(last).toMatchObject({ step: 7, running: false, goal: 7 });
    expect(h.of('eval').at(-1)!.step).toBe(7); // the final model is tested
    e.handle({ type: 'play' });
    expect(e.running).toBe(false);
    e.handle({ type: 'step', n: 2 });
    expect(e.trainer.step).toBe(9);
    e.handle({ type: 'setGoal', steps: 12 }); // a higher goal: Play works again
    e.handle({ type: 'play' });
    for (let i = 0; i < 50 && h.tasks.length; i++) h.runDue(1);
    expect(e.trainer.step).toBe(12);
    e.handle({ type: 'setGoal', steps: null });
    e.handle({ type: 'play' });
    h.runDue(3);
    expect(e.trainer.step).toBeGreaterThan(12);
    e.handle({ type: 'pause' });
  });

  it('lowering the goal below the current step while running stops it', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3, speed: 'max' });
    e.handle({ type: 'play' });
    h.runDue(2);
    expect(e.running).toBe(true);
    e.handle({ type: 'setGoal', steps: 1 });
    expect(e.running).toBe(false);
    expect((h.of('tick').at(-1) as TickMsg).running).toBe(false);
  });

  it('reports the real (wall-clock) speed while running', () => {
    const sent: FromWorker[] = [];
    const tasks: (() => void)[] = [];
    let clock = 0;
    const e = new Engine({ post: (m) => sent.push(m), schedule: (fn) => tasks.push(fn), now: () => (clock += 20) }, ds);
    e.handle({ type: 'init', seed: 3, speed: 'max' });
    e.handle({ type: 'play' });
    for (let i = 0; i < 40 && tasks.length; i++) tasks.shift()!();
    const ticks = sent.filter((m): m is TickMsg => m.type === 'tick');
    const wall = ticks.at(-1)!.wallStepsPerSec;
    expect(wall).toBeGreaterThan(0);
    // Wall-clock time includes everything, so it is never faster than the clock allows.
    expect(wall).toBeLessThanOrEqual(1000 / 20);
  });

  it('answering and evaluating never change the model', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    const twin = new Engine(fakeHost().host, ds);
    e.handle({ type: 'init', seed: 5 });
    twin.handle({ type: 'init', seed: 5 });
    e.handle({ type: 'step', n: 20 });
    twin.handle({ type: 'step', n: 20 });
    const bits = () => e.trainer.model.params.list.map((p) => Array.from(new Uint32Array(p.data.slice().buffer)));
    const before = bits();
    const lang = ds.languageCodes[0];
    for (let i = 0; i < 5; i++) e.handle({ type: 'ask', id: i, text: ['tygr', 'kočka', 'qqq', 'buvol', 'x'][i], lang, form: 'female' });
    e.handle({ type: 'evaluate', inventions: true });
    e.handle({ type: 'inspect', id: 9, matrixIds: ['tokEmb'], mode: 'delta', letterMap: true });
    expect(h.of('answer').length).toBe(5);
    expect(bits()).toEqual(before);
    expect(e.trainer.step).toBe(20);
    // …and nothing hidden changed either (optimiser, random numbers): the next step is identical.
    e.handle({ type: 'step', n: 1 });
    twin.handle({ type: 'step', n: 1 });
    expect(e.trainer.lastLoss).toBe(twin.trainer.lastLoss);
    expect(bits()).toEqual(twin.trainer.model.params.list.map((p) => Array.from(new Uint32Array(p.data.slice().buffer))));
  });

  it('reports errors instead of throwing', () => {
    const h = fakeHost();
    const e = new Engine(h.host, ds);
    e.handle({ type: 'init', seed: 3 });
    e.handle({ type: 'ask', id: 1, text: 'abc', lang: 'xx', form: 'young' });
    expect(h.of('error').length).toBe(1);
  });
});
