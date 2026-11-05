import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { FormTrainer, SCORE_KEYS, TaskResult, confidence, judge, scoreResults } from '../../src/train/trainer';

const ds = new Dataset(baseBundle());

describe('FormTrainer', () => {
  it('trains on word forms of known animals plus copy practice', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    const forms = tr.trainTasks();
    const copies = tr.copyTasks();
    expect(forms.length).toBeGreaterThan(200);
    expect(forms.every((t) => t.animal.split === 'train' && t.form !== 'name' && t.from === t.to)).toBe(true);
    expect(copies.every((t) => t.form === 'name' && t.source === t.answer)).toBe(true);
    expect(tr.model.config.copy && tr.model.config.targetStart).toBe(true);
  });

  it('is deterministic for a seed', () => {
    const a = new FormTrainer(ds, { seed: 7 });
    const b = new FormTrainer(ds, { seed: 7 });
    for (let i = 0; i < 5; i++) expect(a.trainStep()).toBe(b.trainStep());
  });

  it('reports the latest batch: learning data and, if asked, its answers from before the step', () => {
    const tr = new FormTrainer(ds, { seed: 7 });
    expect(tr.practice()).toBeNull();
    const loss = tr.trainStep();
    const plain = tr.practice(8)!;
    expect(plain.step).toBe(1);
    expect(plain.batchSize).toBe(16);
    expect(plain.loss).toBe(loss);
    expect(plain.examples.length).toBe(8);
    expect(plain.rightCount).toBeUndefined();
    expect(plain.examples[0].answer).toBeUndefined();

    // Answering first: the answers are those of the brain before the step.
    const before = new FormTrainer(ds, { seed: 7 });
    before.trainStep();
    const snapshot = before.model.params.snapshot();
    const expected = before.runTasks([]);
    expect(expected).toEqual([]);
    before.trainStep(16);
    const p = before.practice(16)!;
    expect(p.examples.every((e) => typeof e.answer === 'string')).toBe(true);
    expect(p.rightCount).toBe(p.examples.filter((e) => e.status === 'correct').length);
    before.model.params.load(snapshot);
    expect(before.runTasks(p.examples.map((e) => ({ ...ds.formTasks()[0], source: e.source, to: e.lang, form: e.form })))).toEqual(p.examples.map((e) => e.answer));
  });

  it('answering first does not change training', () => {
    const a = new FormTrainer(ds, { seed: 7 });
    const b = new FormTrainer(ds, { seed: 7 });
    for (let i = 0; i < 5; i++) expect(a.trainStep(16)).toBe(b.trainStep());
  });

  it('answers the learning data right more often once it has learned', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    for (let i = 0; i < 600; i++) tr.trainStep();
    tr.trainStep(16);
    expect(tr.practice(16)!.rightCount).toBeGreaterThan(4);
  });

  it('evaluates every form per language, with inventions on request', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    tr.trainStep();
    const ev = tr.evaluate();
    expect(ev.results.length).toBe(ds.formTasks().length);
    for (const lang of ds.languageCodes) {
      expect(Object.keys(ev.scores[lang]).sort()).toEqual([...SCORE_KEYS].sort());
      expect(ev.scores[lang].known).not.toBeNull();
      expect(ev.scores[lang].fresh).toBeNull();
    }
    const full = tr.evaluate({ inventions: true });
    const inventions = full.results.filter((r) => r.status === 'invented');
    expect(inventions.length).toBe(ds.missingFormTasks().length);
    expect(inventions.length).toBeGreaterThan(0);
    expect(full.scores).toEqual(ev.scores); // inventions are not scored
  });

  it('inspect returns a trace with a confidence between 0 and 1', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    const lang = ds.languageCodes[0];
    const trace = tr.inspect('abc', lang, 'female');
    const c = confidence(trace, tr.vocab.eos);
    expect(c).toBeGreaterThan(0);
    expect(c).toBeLessThanOrEqual(1);
    expect(trace.writeShare).toBeDefined();
  });
});

describe('scoring', () => {
  it('judges correct / close / wrong', () => {
    expect(judge('tygřice', ['tygřice'])).toBe('correct');
    expect(judge('tygrice', ['tygřice'])).toBe('close');
    expect(judge('tygr', ['tygřice'])).toBe('wrong');
  });

  it('splits never-seen forms by kind and counts close answers', () => {
    const r = (split: 'train' | 'unseen', status: TaskResult['status'], kind?: TaskResult['kind']): TaskResult =>
      ({ id: 'x', lang: 'xx', form: 'female', source: 's', answer: status === 'invented' ? '' : 'a', output: 'o', status, split, kind });
    const s = scoreResults([
      r('train', 'correct', 'pattern'), r('train', 'wrong', 'pattern'),
      r('unseen', 'correct', 'pattern'), r('unseen', 'close', 'pattern'), r('unseen', 'wrong', 'irregular'),
      r('unseen', 'invented'),
    ]).xx;
    expect(s.known).toBe(0.5);
    expect(s.pattern).toBe(0.5);
    expect(s.partly).toBeNull();
    expect(s.irregular).toBe(0);
    expect(s.unseen).toBeCloseTo(1 / 3);
    expect(s.close).toBeCloseTo(2 / 3);
    expect(s.old).toBe(0.5);
  });
});
