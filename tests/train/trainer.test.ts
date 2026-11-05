import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { FormTrainer, SCORE_KEYS, TaskResult, confidence, judge, scoreResults } from '../../src/train/trainer';
import { changeStats, explainNumber, topNumbers } from '../../src/train/stepDetail';

const ds = new Dataset(baseBundle());

describe('FormTrainer', () => {
  it('trains on word forms of known animals plus copy practice', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    const forms = tr.trainTasks();
    const copies = tr.copyTasks();
    expect(forms.length).toBeGreaterThan(200);
    expect(forms.every((t) => !t.animal.unseen && t.form !== 'name' && t.from === t.to)).toBe(true);
    expect(copies.every((t) => t.form === 'name' && t.source === t.answer)).toBe(true);
    expect(tr.model.config.targetStart).toBe(true);
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

    // Answering first: the answers are those of the model before the step.
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

  it('gives each example of the batch its mistake, from its letters; together they make the batch loss', () => {
    const tr = new FormTrainer(ds, { seed: 4, batchSize: 8 });
    for (let i = 0; i < 30; i++) tr.trainStep();
    const p = tr.practice(8)!;
    let total = 0;
    let count = 0;
    for (const ex of p.examples) {
      expect(ex.letters.map((l) => l.char).join('')).toBe(ex.target); // its letters, then the end of the word
      expect(ex.letters.at(-1)!.char).toBe('');
      for (const l of ex.letters) { expect(l.p).toBeGreaterThan(0); expect(l.p).toBeLessThanOrEqual(1); }
      const mistakes = ex.letters.map((l) => -Math.log(l.p + 1e-9));
      expect(ex.loss).toBeCloseTo(mistakes.reduce((a, b) => a + b, 0) / mistakes.length, 5);
      total += mistakes.reduce((a, b) => a + b, 0);
      count += mistakes.length;
    }
    // The batch loss is the average over every letter of every example.
    expect(p.letterCount).toBe(count);
    expect(total / count).toBeCloseTo(p.loss, 4);
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
  });
});

describe('scoring', () => {
  it('judges correct / close / wrong', () => {
    expect(judge('tygřice', 'tygřice')).toBe('correct');
    expect(judge('tygrice', 'tygřice')).toBe('close');
    expect(judge('tygr', 'tygřice')).toBe('wrong');
  });

  it('splits never-seen forms by kind and counts close answers', () => {
    const r = (split: 'train' | 'unseen', status: TaskResult['status'], kind?: TaskResult['kind']): TaskResult =>
      ({ id: 'x', lang: 'xx', form: 'female', source: 's', answer: status === 'invented' ? '' : 'a', output: 'o', status, unseen: split === 'unseen', kind });
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

  describe('one step recorded in detail', () => {
    it('trains exactly like an ordinary step', () => {
      const a = new FormTrainer(ds, { seed: 9 });
      const b = new FormTrainer(ds, { seed: 9 });
      for (let i = 0; i < 5; i++) { a.trainStep(); b.trainStep(); }
      expect(b.trainStep(16, true)).toBe(a.trainStep(16));
      for (let i = 0; i < 3; i++) expect(b.trainStep()).toBe(a.trainStep());
      a.model.params.list.forEach((p, i) => expect(Array.from(b.model.params.list[i].data)).toEqual(Array.from(p.data)));
      expect(b.record).toBeNull(); // only the detailed step is recorded
    });

    it("splits every number's gradient exactly into the examples' shares and recomputes its nudge", () => {
      const tr = new FormTrainer(ds, { seed: 2 });
      for (let i = 0; i < 20; i++) tr.trainStep();
      tr.trainStep(0, true);
      const r = tr.record!;
      expect(r.step).toBe(21);
      expect(r.perExample).toHaveLength(16);
      tr.model.params.list.forEach((p, pi) => {
        for (let i = 0; i < p.size; i += 11) {
          let sum = 0;
          for (const ex of r.perExample!) sum += ex[pi][i];
          expect(sum).toBeCloseTo(r.grad[pi][i], 6);
          const w = explainNumber(r, pi, i, p.shape.length > 1);
          expect(w.after).toBeCloseTo(p.data[i], 6);
          expect(w.before).toBe(r.before[pi][i]);
        }
      });
      // Warm-up: the rate used is smaller than the one set.
      expect(r.lr).toBeCloseTo(r.baseLr * 21 / r.warmupSteps, 8);
    });

    it('measures the same batch before and after the step (no dropout) for the practice table', () => {
      const tr = new FormTrainer(ds, { seed: 6 });
      for (let i = 0; i < 40; i++) tr.trainStep();
      tr.trainStep(16, true);
      const p = tr.practice(16)!;
      expect(p.check!.lossAfter).toBeLessThan(p.check!.lossBefore);
      expect(p.check!.lettersUp + p.check!.lettersDown).toBeLessThanOrEqual(p.letterCount);
      for (const ex of p.examples) {
        expect(ex.check!.before).toHaveLength(ex.letters.length);
        expect(ex.check!.after).toHaveLength(ex.letters.length);
        expect(ex.check!.status).toBe(judge(ex.check!.answer, ex.target));
      }
      expect(p.check!.rightAfter).toBe(p.examples.filter((e) => e.check!.status === 'correct').length);
      tr.trainStep();
      expect(tr.practice()!.check).toBeUndefined();
    });

    it('finds the biggest numbers and summarises the change', () => {
      const a = [Float32Array.from([1, -5, 2]), Float32Array.from([4])];
      expect(topNumbers(a, (p, i) => a[p][i], 2)).toEqual([{ param: 0, index: 1 }, { param: 1, index: 0 }]);
      const s = changeStats([Float32Array.from([0, 0, 0])], [Float32Array.from([0, 0.5, -2])]);
      expect(s).toEqual({ total: 3, moved: 2, max: 2, median: 0.5 });
    });
  });
});
