import { describe, expect, it } from 'vitest';
import { Vocab, buildVocab, targetToken } from '../../src/nn/vocab';

const spec = { specials: ['<pad>', '<bos>', '<eos>', '<unk>'], chars: ['a', 'b', 'č', ' '] };
const langs = [{ code: 'cs', name: {} }, { code: 'en', name: {} }];

describe('vocab', () => {
  it('builds specials, characters and target tokens', () => {
    const v = buildVocab(spec, langs, ['name']);
    expect(v.tokens).toEqual(['<pad>', '<bos>', '<eos>', '<unk>', 'a', 'b', 'č', ' ', '<cs>', '<en>']);
    expect(v.targetIds()).toEqual([8, 9]);
  });

  it('adding a form only appends tokens', () => {
    const v1 = buildVocab(spec, langs, ['name']);
    const v2 = buildVocab(spec, langs, ['name', 'young']);
    expect(v2.tokens.slice(0, v1.size)).toEqual(v1.tokens);
    expect(v2.tokens.slice(v1.size)).toEqual([targetToken('cs', 'young'), targetToken('en', 'young')]);
  });

  it('encodes with NFC + lower case and maps unknown characters to <unk>', () => {
    const v = buildVocab(spec, langs, ['name']);
    const decomposed = 'ČAB'; // "Č" written as C + combining caron
    expect(v.encodeWord(decomposed)).toEqual([v.id('č'), v.id('a'), v.id('b')]);
    expect(v.encodeWord('ax')).toEqual([v.id('a'), v.unk]);
    expect(v.unknownChars('axy')).toEqual(['x', 'y']);
    expect(v.decode([v.id('a'), v.id('č'), v.eos, v.id('b')])).toBe('ač');
  });

  it('extend appends new tokens and keeps old indices', () => {
    const v = buildVocab(spec, langs, ['name']);
    const w = v.extend(['ä', 'a', 'ä']);
    expect(w.size).toBe(v.size + 1);
    expect(w.id('ä')).toBe(v.size);
    expect(w.id('č')).toBe(v.id('č'));
    expect(() => new Vocab(['<pad>'])).toThrow();
  });
});
