import { describe, expect, it } from 'vitest';
import { interestingLetter, pctText } from '../../src/ui/explainView';
import { plainFmt } from '../../src/ui/inside';

const answer = (text: string, out: string[]) => ({ text, steps: out.map((chosen) => ({ chosen })) });

describe('How the model picks one letter: plain helpers', () => {
  it('starts at the letter where the answer stops copying the animal', () => {
    expect(interestingLetter(answer('buvol', [...'buvolice', '']))).toBe(5); // the "i" of -ice
    expect(interestingLetter(answer('tygr', [...'tygřice', '']))).toBe(3); // ř
    expect(interestingLetter(answer('cat', [...'kitten', '']))).toBe(0);
    expect(interestingLetter(answer('lev', [...'lev', '']))).toBe(3); // the end of the word
  });

  it('writes chances as whole percentages without a misleading 0 or 100', () => {
    expect(pctText(0.64)).toBe('64 %');
    expect(pctText(0.001)).toBe('<1 %');
    expect(pctText(0.999)).toBe('>99 %');
    expect(pctText(0)).toBe('0 %');
    expect(pctText(1)).toBe('100 %');
  });

  it('writes small numbers without scientific notation', () => {
    expect(plainFmt(0.0019)).toBe('0.0019');
    expect(plainFmt(-0.00012)).toBe('-0.00012');
    expect(plainFmt(0.5)).toBe('0.500');
    expect(plainFmt(0)).toBe('0');
    expect(plainFmt(3e-8)).not.toMatch(/e/);
  });
});
