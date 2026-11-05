import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { FormTrainer } from '../../src/train/trainer';
import { countData, dataQuestions } from '../../src/ui/dataTab';

const ds = new Dataset(baseBundle());
const info = {
  forms: ds.forms,
  languages: ds.languages,
  animals: ds.animals.map((a) => ({
    id: a.id, unseen: a.unseen, pack: a.pack, names: a.names, forms: a.forms,
  })),
};
const key = (q: { id?: string; animal?: { id: string }; lang?: string; to?: string; form: string }) =>
  `${q.id ?? q.animal!.id}/${q.lang ?? q.to}/${q.form}`;

describe('Learning data tab counts', () => {
  it('match what the trainer samples from', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    const all = countData(info);
    expect(all.forms).toBe(tr.trainTasks().length);
    expect(all.copies).toBe(tr.copyTasks().length);
    expect(all.train + all.unseen).toBe(ds.animals.length);
    expect(all.unseen).toBe(ds.animals.filter((a) => a.unseen).length);
    expect(all.tests).toBe(ds.formTasks().length);
    expect(all.unseenTests).toBe(ds.formTasks((a) => a.unseen).length);
    expect(all.inventions).toBe(ds.missingFormTasks().length);
    for (const lang of ds.languageCodes) {
      const c = countData(info, lang);
      expect(c.forms).toBe(tr.trainTasks().filter((t) => t.to === lang).length);
      expect(c.copies).toBe(tr.copyTasks().filter((t) => t.to === lang).length);
    }
  });
});

describe('Every question (learning data tab)', () => {
  const qs = dataQuestions(info);

  it('learned questions are exactly what the trainer samples, with the same answers', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    const tasks = [...tr.trainTasks(), ...tr.copyTasks()];
    const learned = qs.filter((q) => q.learn);
    expect(learned.map(key).sort()).toEqual(tasks.map(key).sort());
    const byKey = new Map(tasks.map((t) => [key(t), t]));
    for (const q of learned) {
      const t = byKey.get(key(q))!;
      expect([q.source, q.answer]).toEqual([t.source, t.answer]);
    }
  });

  it('tested questions are exactly what the tests ask, with the same answers', () => {
    const tested = qs.filter((q) => q.test && q.answer);
    const tasks = ds.formTasks();
    expect(tested.map(key).sort()).toEqual(tasks.map(key).sort());
    const byKey = new Map(tasks.map((t) => [key(t), t]));
    for (const q of tested) expect([q.source, q.answer]).toEqual([byKey.get(key(q))!.source, byKey.get(key(q))!.answer]);
    const invented = qs.filter((q) => q.test && !q.answer);
    expect(invented.map(key).sort()).toEqual(ds.missingFormTasks().map(key).sort());
  });

  it('never learns from never-seen animals', () => {
    expect(qs.some((q) => q.unseen && q.learn)).toBe(false);
  });
});
