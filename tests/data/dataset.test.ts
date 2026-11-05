import { describe, expect, it } from 'vitest';
import { allPacks, baseBundle } from '../../src/data/bundle';
import { Dataset, Pack, classify, editDistance, formKind, mergePacks, validatePacks } from '../../src/data/dataset';
import { buildVocab } from '../../src/nn/vocab';

const bundle = baseBundle();
const vocab = buildVocab(bundle.vocabSpec, bundle.languages, bundle.forms);
const codes = bundle.languages.map((l) => l.code);

describe('data files', () => {
  it('pass validation', () => {
    expect(validatePacks(allPacks(), vocab, codes, bundle.forms)).toEqual([]);
  });

  it('vocabulary: 4 specials + 43 characters + one target token per language × form', () => {
    expect(vocab.size).toBe(4 + 43 + codes.length * bundle.forms.length);
  });

  it('base set: ~150 animals, every language has never-seen forms of every kind', () => {
    const ds = new Dataset(bundle);
    expect(ds.animals.length).toBeGreaterThan(130);
    const tasks = ds.formTasks();
    const unseen = tasks.filter((t) => t.animal.split === 'unseen');
    expect(unseen.length / tasks.length).toBeGreaterThan(0.1);
    expect(unseen.length / tasks.length).toBeLessThan(0.25);
    for (const lang of codes) {
      const kinds = new Set(unseen.filter((t) => t.to === lang).map((t) => formKind(t.source, t.answer)));
      expect([...kinds].sort(), lang).toEqual(['irregular', 'partly', 'pattern']);
    }
  });
});

describe('similarity', () => {
  it('classifies the examples from the design', () => {
    expect(classify('tygr', 'tiger')).toBe('lookalike');
    expect(classify('pes', 'dog')).toBe('different');
    expect(classify('krokodýl', 'crocodile')).toBe('lookalike');
    expect(classify('žirafa', 'giraffe')).toBe('partial');
    expect(classify('veverka', 'squirrel')).toBe('different');
  });

  it('classifies word forms by how much of the name they keep', () => {
    expect(formKind('tygr', 'tygřice')).toBe('pattern');
    expect(formKind('tygr', 'tygr')).toBe('pattern');
    expect(formKind('lev', 'lvice')).toBe('partly');
    expect(formKind('pes', 'štěně')).toBe('irregular');
    expect(formKind('horse', 'mare')).toBe('irregular');
  });

  it('edit distance counts unicode characters', () => {
    expect(editDistance('kočka', 'kocka')).toBe(1);
    expect(editDistance('', 'abc')).toBe(3);
    expect(editDistance('kitten', 'sitting')).toBe(3);
  });
});

describe('validation catches problems', () => {
  const pack = (animals: Pack['animals']): Pack => ({ pack: 't', version: 1, title: {}, animals });
  it('reports bad characters, case, length and duplicates', () => {
    const errs = validatePacks([pack([
      { id: 'a', names: { cs: 'Kočka', en: 'cat' } },
      { id: 'a', names: { cs: 'pes', en: 'dög' } },
      { id: 'b', names: { cs: 'pes', en: 'x'.repeat(21) } },
    ])], vocab, codes, bundle.forms);
    expect(errs.some((e) => e.includes('lower-case'))).toBe(true);
    expect(errs.some((e) => e.includes('duplicate id'))).toBe(true);
    expect(errs.some((e) => e.includes('outside the vocabulary'))).toBe(true);
    expect(errs.some((e) => e.includes('longer than'))).toBe(true);
    expect(errs.some((e) => e.includes('also used by'))).toBe(true);
  });

  it('rejects fields that are no longer used (one right word per answer)', () => {
    const old = { id: 'cat', names: { cs: 'kočka' }, alt: { cs: ['kocour'] }, unsure: ['cs.name'] } as unknown as Pack['animals'][number];
    const errs = validatePacks([pack([old])], vocab, codes, bundle.forms);
    expect(errs).toContain('t/cat: unknown field "alt"');
    expect(errs).toContain('t/cat: unknown field "unsure"');
  });
});

describe('merging', () => {
  it('merges packs by id and keeps unseen sticky', () => {
    const a: Pack = { pack: 'a', version: 1, title: {}, animals: [{ id: 'dog', names: { cs: 'pes', en: 'dog' }, split: 'unseen' }] };
    const b: Pack = { pack: 'b', version: 1, title: {}, animals: [{ id: 'dog', names: { de: 'hund' }, split: 'train' }] };
    const [dog] = mergePacks([a, b]);
    expect(dog.names).toEqual({ cs: 'pes', en: 'dog', de: 'hund' });
    expect(dog.split).toBe('unseen');
  });

  it('builds tasks in both directions', () => {
    const ds = new Dataset(bundle);
    const cat = ds.nameTasks((a) => a.id === 'cat');
    expect(cat.map((t) => `${t.source}->${t.answer}`).sort()).toEqual(['cat->kočka', 'kočka->cat']);
  });

  it('builds word-form tasks within each language', () => {
    const ds = new Dataset(bundle);
    const cat = ds.formTasks((a) => a.id === 'cat');
    expect(cat.map((t) => `${t.to}:${t.form} ${t.source}->${t.answer}`).sort()).toEqual([
      'cs:female kočka->kočka', 'cs:male kočka->kocour', 'cs:young kočka->kotě',
      'en:female cat->queen', 'en:male cat->tom', 'en:young cat->kitten',
    ]);
  });
});
