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

  it('vocabulary: 4 specials + the characters + one target token per language × form', () => {
    expect(bundle.vocabSpec.chars.length).toBe(51);
    expect(vocab.size).toBe(4 + 51 + codes.length * bundle.forms.length);
  });

  it('base set: ~150 animals, every language has never-seen forms of every kind', () => {
    const ds = new Dataset(bundle);
    expect(ds.animals.length).toBeGreaterThan(130);
    const tasks = ds.formTasks();
    const unseen = tasks.filter((t) => t.animal.unseen);
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
      { id: 'a', names: { cs: 'pes', en: 'døg' } },
      { id: 'b', names: { cs: 'pes', en: 'x'.repeat(21) } },
    ])], vocab, codes, bundle.forms);
    expect(errs.some((e) => e.includes('lower-case'))).toBe(true);
    expect(errs.some((e) => e.includes('duplicate id'))).toBe(true);
    expect(errs.some((e) => e.includes('outside the vocabulary'))).toBe(true);
    expect(errs.some((e) => e.includes('longer than'))).toBe(true);
    expect(errs.some((e) => e.includes('also used by'))).toBe(true);
  });

  it('a pack that only adds languages keeps the earlier trained / never-seen split', () => {
    const animals = (p: string, extra: object) => Array.from({ length: 8 }, (_, i) => ({ id: `a${i}`, names: { [p]: `x${'y'.repeat(i)}` }, ...(i === 0 ? extra : {}) }));
    const base: Pack = { pack: 'base', version: 1, title: {}, animals: animals('cs', { unseen: true }) };
    const lang: Pack = { pack: 'lang', version: 1, title: {}, animals: animals('de', {}) };
    expect(validatePacks([base, lang], vocab, codes, bundle.forms)).toEqual([]);
    expect(validatePacks([lang], vocab, codes, bundle.forms)).toContain('pack lang: needs both trained and "unseen": true animals');
  });

  it('rejects fields that are no longer used (one right word per answer)', () => {
    const old = { id: 'cat', names: { cs: 'kočka' }, alt: { cs: ['kocour'] }, unsure: ['cs.name'] } as unknown as Pack['animals'][number];
    const errs = validatePacks([pack([old])], vocab, codes, bundle.forms);
    expect(errs).toContain('t/cat: unknown field "alt"');
    expect(errs).toContain('t/cat: unknown field "unsure"');
  });

  it('marks never-seen animals only with "unseen": true (no split, no kindOverride)', () => {
    const old = { id: 'cat', names: { cs: 'kočka' }, split: 'train', kindOverride: { 'cs-en': 'different' } } as unknown as Pack['animals'][number];
    const bad = { id: 'dog', names: { cs: 'pes' }, unseen: false } as unknown as Pack['animals'][number];
    const errs = validatePacks([pack([old, bad])], vocab, codes, bundle.forms);
    expect(errs).toContain('t/cat: unknown field "split"');
    expect(errs).toContain('t/cat: unknown field "kindOverride"');
    expect(errs).toContain('t/dog: "unseen" must be true or left out');
  });
});

describe('merging', () => {
  it('merges packs by id and keeps unseen sticky', () => {
    const a: Pack = { pack: 'a', version: 1, title: {}, animals: [{ id: 'dog', names: { cs: 'pes', en: 'dog' }, unseen: true }] };
    const b: Pack = { pack: 'b', version: 1, title: {}, animals: [{ id: 'dog', names: { de: 'hund' } }] };
    const [dog] = mergePacks([a, b]);
    expect(dog.names).toEqual({ cs: 'pes', en: 'dog', de: 'hund' });
    expect(dog.unseen).toBe(true);
  });

  it('builds tasks in both directions', () => {
    const ds = new Dataset(bundle);
    const cat = ds.nameTasks((a) => a.id === 'cat');
    const pairs = cat.map((t) => `${t.source}->${t.answer}`);
    expect(pairs).toHaveLength(codes.length * (codes.length - 1));
    expect(pairs).toEqual(expect.arrayContaining(['cat->kočka', 'kočka->cat', 'mačka->katze', 'katze->mačka']));
  });

  it('builds word-form tasks within each language', () => {
    const ds = new Dataset(bundle);
    const cat = ds.formTasks((a) => a.id === 'cat');
    expect(cat.map((t) => `${t.to}:${t.form} ${t.source}->${t.answer}`).sort()).toEqual([
      'cs:female kočka->kočka', 'cs:male kočka->kocour', 'cs:young kočka->kotě',
      'de:female katze->katze', 'de:male katze->kater', 'de:young katze->kätzchen',
      'en:female cat->queen', 'en:male cat->tom', 'en:young cat->kitten',
      'sk:female mačka->mačka', 'sk:male mačka->kocúr', 'sk:young mačka->mačiatko',
    ]);
  });
});
