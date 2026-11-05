import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { FormTrainer } from '../../src/train/trainer';
import { ON, TOP, profileUnits } from '../../src/train/units';

describe('what the thinking units have learned', () => {
  const ds = new Dataset(baseBundle());
  const tr = new FormTrainer(ds, { seed: 2 });
  for (let i = 0; i < 30; i++) tr.trainStep();
  const tasks = ds.formTasks().slice(0, 120);
  const { units, moments } = profileUnits(tr.model, ds, tasks, tr.ids);

  it('reads every letter of every answer, plus the end of each word', () => {
    expect(moments).toBe(tasks.reduce((a, t) => a + [...t.answer].length + 1, 0));
    expect(units).toHaveLength(tr.model.config.ffn);
  });

  it('keeps each unit\'s strongest moments, strongest first, each once', () => {
    for (const u of units) {
      expect(u.top.length).toBeLessThanOrEqual(TOP);
      expect(u.on).toBeGreaterThanOrEqual(0);
      expect(u.on).toBeLessThanOrEqual(1);
      for (let i = 0; i < u.top.length; i++) {
        const m = u.top[i];
        expect(m.value).toBeGreaterThan(ON);
        if (i) expect(m.value).toBeLessThanOrEqual(u.top[i - 1].value);
        expect(m.at).toBeLessThanOrEqual([...m.answer].length);
        expect(m.next).toBe([...m.answer][m.at] ?? '');
      }
      const keys = u.top.map((m) => `${m.answer}|${m.at}|${m.lang}|${m.form}`);
      expect(new Set(keys).size).toBe(keys.length);
      expect(u.pushes).toHaveLength(3);
    }
  });

  it('switching a unit off at its strongest moments is measured (one writing layer: exact)', () => {
    for (const u of units) {
      if (!u.top.length) continue;
      expect(u.off).not.toBeNull();
      const o = u.off!;
      for (const p of [o.withP, o.withoutP]) { expect(p).toBeGreaterThanOrEqual(0); expect(p).toBeLessThanOrEqual(1); }
      for (let i = 1; i < o.helps.length; i++) expect(o.helps[i].drop).toBeLessThanOrEqual(o.helps[i - 1].drop);
      for (const x of o.helps) expect(x.drop).toBeGreaterThan(0);
    }
  });

  it('a theme is shared by as many of the strongest moments as it says', () => {
    for (const u of units) {
      if (!u.theme) continue;
      const { kind, value, count } = u.theme;
      const of = (m: (typeof u.top)[number]) => kind === 'next' ? m.next
        : kind === 'prev' ? (m.at === 0 ? '<start>' : [...m.answer][m.at - 1])
          : kind === 'task' ? `${m.lang}:${m.form}` : kind === 'form' ? m.form : m.lang;
      expect(u.top.filter((m) => of(m) === value)).toHaveLength(count);
    }
  });
});
