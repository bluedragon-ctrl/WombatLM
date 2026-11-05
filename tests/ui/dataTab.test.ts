import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { FormTrainer } from '../../src/train/trainer';
import { countData } from '../../src/ui/dataTab';

const ds = new Dataset(baseBundle());

describe('Learning data tab counts', () => {
  it('match what the trainer samples from', () => {
    const tr = new FormTrainer(ds, { seed: 1 });
    const info = { forms: ds.forms, animals: ds.animals.map((a) => ({ id: a.id, split: a.split, pack: a.pack, names: a.names, forms: a.forms })) };
    const all = countData(info);
    expect(all.forms).toBe(tr.trainTasks().length);
    expect(all.copies).toBe(tr.copyTasks().length);
    expect(all.train + all.unseen).toBe(ds.animals.length);
    expect(all.unseen).toBe(ds.animals.filter((a) => a.split === 'unseen').length);
    for (const lang of ds.languageCodes) {
      const c = countData(info, lang);
      expect(c.forms).toBe(tr.trainTasks().filter((t) => t.to === lang).length);
      expect(c.copies).toBe(tr.copyTasks().filter((t) => t.to === lang).length);
    }
  });
});
