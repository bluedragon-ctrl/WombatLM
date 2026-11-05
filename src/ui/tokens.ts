/**
 * Tokens on screen: how a word becomes the numbers the model reads, and how
 * an answer comes out as tokens again.
 *
 * The model reads and writes one character per token. Every token has a
 * number (its id = its place in the token list). Special tokens get a plain
 * name for beginners ("end", "Czech, female"); expert mode shows the raw
 * token (<eos>, <cs:female>). The raw token is always in the tooltip.
 */
import { EOS, UNK, Vocab, targetToken } from '../nn/vocab';
import { h } from './dom';
import { t } from './i18n';

/** One token: its text in the token list and its number. */
export interface Tok {
  token: string;
  id: number;
}

export type TokKind = 'task' | 'special' | 'unk' | 'letter';

const isSpecial = (token: string) => token.length > 1 && token.startsWith('<') && token.endsWith('>');

export function tokenKind(token: string): TokKind {
  if (token === UNK) return 'unk';
  if (!isSpecial(token)) return 'letter';
  return /^<[a-z]+(:[a-z]+)?>$/.test(token) && !['<pad>', '<bos>', '<eos>'].includes(token) ? 'task' : 'special';
}

/** Token text -> its id, for a token list (ready.tokens). */
export function tokenIndex(tokens: readonly string[]): Map<string, number> {
  return new Map(tokens.map((tok, i) => [tok, i]));
}

export function toTok(token: string, index: Map<string, number>): Tok {
  return { token, id: index.get(token) ?? index.get(UNK) ?? -1 };
}

/** A word as the model reads it: one token per character, unknown ones become <unk> (like Vocab.encodeWord). */
export function tokenizeWord(word: string, index: Map<string, number>): Tok[] {
  return [...Vocab.normalize(word)].map((c) => (index.has(c) ? { token: c, id: index.get(c)! } : toTok(UNK, index)));
}

/**
 * One learning example as the model trains on it ("teacher forcing"):
 * - reads: the task token, then the letters of the animal;
 * - shown: what it has been shown of the answer so far before each guess
 *   (it starts with the task token);
 * - guess: what it must guess next at each step (the answer, then "end").
 * `shown[i]` is followed by `guess[i]`: the same answer, shifted by one.
 */
export function learningExample(index: Map<string, number>, lang: string, form: string, source: string, answer: string):
  { reads: Tok[]; shown: Tok[]; guess: Tok[] } {
  const task = toTok(targetToken(lang, form), index);
  const out = tokenizeWord(answer, index);
  return {
    reads: [task, ...tokenizeWord(source, index)],
    shown: [task, ...out],
    guess: [...out, toTok(EOS, index)],
  };
}

export interface ChipOptions {
  /** Raw special tokens (<eos>) instead of plain names. */
  expert?: boolean;
  /** Only the number, as the model sees it. */
  numbers?: boolean;
  /** Language code -> name in the UI language. */
  langName: (code: string) => string;
}

/** The text on a chip: the letter, or a plain name for special tokens. */
export function tokenName(token: string, opts: Pick<ChipOptions, 'expert' | 'langName'>): string {
  if (token === ' ') return '␣';
  if (!isSpecial(token)) return token;
  if (opts.expert) return token;
  if (token === EOS) return t('tok.end');
  if (token === UNK) return '?';
  if (token === '<bos>') return t('tok.start');
  if (token === '<pad>') return '·';
  const m = /^<([a-z]+)(?::([a-z]+))?>$/.exec(token);
  if (!m) return token;
  return t('tok.task', { lang: opts.langName(m[1]), form: t(`tok.form.${m[2] ?? 'name'}`) });
}

/**
 * A one-character label for narrow canvas spots (guess-bar headers, attention
 * row/column labels): letters as they are, the space as ␣, the end ('' or
 * <eos>) as ⏎, task tokens as ✱, <unk> as ?, other special tokens as ·.
 */
export function shortLabel(token: string): string {
  if (token === '' || token === EOS) return '⏎';
  if (token === ' ') return '␣';
  switch (tokenKind(token)) {
    case 'letter': return token;
    case 'task': return '✱';
    case 'unk': return '?';
    default: return '·';
  }
}

/** Tooltip: what the token is, its raw text and its number. */
export function tokenTitle(tok: Tok, opts: Pick<ChipOptions, 'langName'>): string {
  const kind = tokenKind(tok.token);
  const what = kind === 'letter' ? t('tok.tip.letter', { tok: tok.token === ' ' ? '␣' : tok.token })
    : kind === 'unk' ? t('tok.tip.unk')
    : kind === 'task' ? t('tok.tip.task', { name: tokenName(tok.token, { langName: opts.langName }) })
    : tok.token === EOS ? t('tok.tip.end') : tok.token;
  return `${what}\n${t('tok.tip.id', { tok: tok.token === ' ' ? '␣' : tok.token, id: tok.id })}`;
}

/** One token as a chip: the letter (or name) with its number underneath. */
export function tokenChip(tok: Tok, opts: ChipOptions, cls = ''): HTMLElement {
  const kind = tokenKind(tok.token);
  const chip = h('span', { class: `tok tok-${kind}${opts.numbers ? ' num' : ''}${cls ? ` ${cls}` : ''}`, title: tokenTitle(tok, opts) });
  if (opts.numbers) chip.append(h('b', {}, String(tok.id)));
  else chip.append(h('b', {}, tokenName(tok.token, opts)), h('small', {}, String(tok.id)));
  return chip;
}
