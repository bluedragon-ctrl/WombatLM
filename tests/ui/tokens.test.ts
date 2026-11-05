import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { targetToken } from '../../src/nn/vocab';
import { learningExample, shortLabel, tokenIndex, tokenKind, tokenName, tokenizeWord } from '../../src/ui/tokens';

const ds = new Dataset(baseBundle());
const vocab = ds.vocab;
const index = tokenIndex(vocab.tokens);
const langName = (c: string) => ({ cs: 'Czech', en: 'English' }[c] ?? c);

describe('tokens on screen', () => {
  it('cuts a word into the same token ids as the model', () => {
    for (const word of ['tygřice', 'Lion Cub', 'mořské prase', 'zebra-x', 'äb5']) {
      expect(tokenizeWord(word, index).map((t) => t.id)).toEqual(vocab.encodeWord(word));
    }
    expect(tokenizeWord('äb', index).map((t) => t.token)).toEqual(['<unk>', 'b']);
  });

  it('shows a learning example as reads / shown so far / guess next, shifted by one', () => {
    const ex = learningExample(index, 'cs', 'female', 'tygr', 'tygřice');
    const task = vocab.id(targetToken('cs', 'female'));
    expect(ex.reads.map((t) => t.id)).toEqual([task, ...vocab.encodeWord('tygr')]);
    expect(ex.shown.map((t) => t.id)).toEqual([task, ...vocab.encodeWord('tygřice')]);
    expect(ex.guess.map((t) => t.id)).toEqual([...vocab.encodeWord('tygřice'), vocab.eos]);
    expect(ex.shown.slice(1)).toEqual(ex.guess.slice(0, -1));
  });

  it('copy practice uses the language token', () => {
    const ex = learningExample(index, 'en', 'name', 'lion', 'lion');
    expect(ex.reads[0].token).toBe('<en>');
    expect(ex.guess.length).toBe(5);
  });

  it('names special tokens plainly for beginners and raw for experts', () => {
    expect(tokenKind('a')).toBe('letter');
    expect(tokenKind('<cs:young>')).toBe('task');
    expect(tokenKind('<en>')).toBe('task');
    expect(tokenKind('<eos>')).toBe('special');
    expect(tokenKind('<unk>')).toBe('unk');
    expect(tokenName('<cs:female>', { langName })).toBe('Czech, female');
    expect(tokenName('<en>', { langName })).toBe('English, name');
    expect(tokenName('<eos>', { langName })).toBe('end');
    expect(tokenName('<eos>', { langName, expert: true })).toBe('<eos>');
    expect(tokenName(' ', { langName })).toBe('␣');
  });

  it('gives every token a one-character label for narrow canvas spots', () => {
    expect(shortLabel('a')).toBe('a');
    expect(shortLabel(' ')).toBe('␣');
    expect(shortLabel('')).toBe('⏎');
    expect(shortLabel('<eos>')).toBe('⏎');
    expect(shortLabel('<cs:female>')).toBe('✱');
    expect(shortLabel('<en>')).toBe('✱');
    expect(shortLabel('<unk>')).toBe('?');
    for (const tok of vocab.tokens) expect([...shortLabel(tok)].length).toBe(1);
  });
});
