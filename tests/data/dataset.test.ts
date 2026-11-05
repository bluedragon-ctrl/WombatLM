import { describe, expect, it } from 'vitest';
import { allPacks, baseBundle } from '../../src/data/bundle';
import { Dataset, Pack, classify, editDistance, mergePacks, validatePacks } from '../../src/data/dataset';
import { buildVocab } from '../../src/nn/vocab';

const bundle = baseBundle();
const vocab = buildVocab(bundle.vocabSpec, bundle.languages, bundle.forms);
const codes = bundle.languages.map((l) => l.code);

describe('data files', () => {
  it('pass validation', () => {
    expect(validatePacks(allPacks(), vocab, codes)).toEqual([]);
  });

  it('vocabulary has 49 tokens in v1', () => {
    expect(vocab.size).toBe(49);
  });

  it('base set has ~150 animals, 15–20% unseen, every kind has unseen examples', () => {
    const ds = new Dataset(bundle);
    const n = ds.animals.length;
    expect(n).toBeGreaterThan(130);
    const unseen = ds.animals.filter((a) => a.split === 'unseen');
    expect(unseen.length / n).toBeGreaterThan(0.14);
    expect(unseen.length / n).toBeLessThan(0.21);
    const kinds = new Set(unseen.map((a) => classify(a.names[codes[0]], a.names[codes[1]])));
    expect([...kinds].sort()).toEqual(['different', 'lookalike', 'partial']);
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
    ])], vocab, codes);
    expect(errs.some((e) => e.includes('lower-case'))).toBe(true);
    expect(errs.some((e) => e.includes('duplicate id'))).toBe(true);
    expect(errs.some((e) => e.includes('outside the vocabulary'))).toBe(true);
    expect(errs.some((e) => e.includes('longer than'))).toBe(true);
    expect(errs.some((e) => e.includes('also used by'))).toBe(true);
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

  it('builds tasks in both directions with alternatives', () => {
    const ds = new Dataset(bundle);
    const cat = ds.nameTasks((a) => a.id === 'cat');
    expect(cat.map((t) => `${t.source}->${t.answer}`).sort()).toEqual(['cat->kočka', 'kočka->cat']);
    expect(cat.find((t) => t.to === 'cs')!.accepted).toContain('kocour');
  });
});
