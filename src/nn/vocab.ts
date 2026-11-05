/**
 * The model reads and writes one character at a time. The vocabulary is the
 * list of all characters (and a few special tokens) it knows. Each token has
 * a number (its index), and the model only ever sees those numbers.
 *
 * Nothing here is hard-coded: the characters come from data/vocab.json, the
 * target tokens from languages × forms.
 */

export interface VocabSpec {
  specials: string[];
  chars: string[];
}

export interface LanguageSpec {
  code: string;
  name: Record<string, string>;
}

export const PAD = '<pad>';
export const BOS = '<bos>';
export const EOS = '<eos>';
export const UNK = '<unk>';

/** Target token for a (language, form) pair: `<cs>` for names, `<cs:young>` otherwise. */
export function targetToken(lang: string, form: string): string {
  return form === 'name' ? `<${lang}>` : `<${lang}:${form}>`;
}

export class Vocab {
  readonly tokens: readonly string[];
  private readonly index: Map<string, number>;
  readonly pad: number;
  readonly bos: number;
  readonly eos: number;
  readonly unk: number;

  constructor(tokens: readonly string[]) {
    this.tokens = tokens.slice();
    this.index = new Map(this.tokens.map((t, i) => [t, i]));
    if (this.index.size !== this.tokens.length) throw new Error('Vocab has duplicate tokens');
    const need = (t: string) => {
      const i = this.index.get(t);
      if (i === undefined) throw new Error(`Vocab is missing ${t}`);
      return i;
    };
    this.pad = need(PAD);
    this.bos = need(BOS);
    this.eos = need(EOS);
    this.unk = need(UNK);
  }

  get size(): number {
    return this.tokens.length;
  }

  has(token: string): boolean {
    return this.index.has(token);
  }

  id(token: string): number {
    const i = this.index.get(token);
    if (i === undefined) throw new Error(`Unknown token ${token}`);
    return i;
  }

  /** Is this token a single character (as opposed to <pad>, <cs>, …)? */
  isChar(id: number): boolean {
    const t = this.tokens[id];
    return t !== undefined && !(t.length > 1 && t.startsWith('<') && t.endsWith('>'));
  }

  /** Normalise user text the same way as the dataset: NFC, lower case, trimmed. */
  static normalize(text: string): string {
    return text.normalize('NFC').toLowerCase().trim().replace(/\s+/g, ' ');
  }

  /** Characters of `word` that are not in the vocabulary. */
  unknownChars(word: string): string[] {
    return [...new Set([...Vocab.normalize(word)].filter((c) => !this.index.has(c)))];
  }

  /** Word -> token ids (unknown characters become <unk>). No special tokens added. */
  encodeWord(word: string): number[] {
    return [...Vocab.normalize(word)].map((c) => this.index.get(c) ?? this.unk);
  }

  /** Token ids -> text, stopping at <eos> and skipping other special tokens. */
  decode(ids: ArrayLike<number>): string {
    let s = '';
    for (let i = 0; i < ids.length; i++) {
      if (ids[i] === this.eos) break;
      if (this.isChar(ids[i])) s += this.tokens[ids[i]];
    }
    return s;
  }

  /** Token ids of all target tokens, in order. */
  targetIds(): number[] {
    return this.tokens.flatMap((t, i) => (!this.isChar(i) && ![PAD, BOS, EOS, UNK].includes(t) ? [i] : []));
  }

  /**
   * A copy of this vocabulary with `extra` tokens appended at the end
   * (tokens already present are skipped). Existing indices never change.
   */
  extend(extra: readonly string[]): Vocab {
    const add = extra.filter((t, i) => !this.index.has(t) && extra.indexOf(t) === i);
    return new Vocab([...this.tokens, ...add]);
  }
}

/**
 * Build the vocabulary: specials, then characters, then one target token per
 * form × language (form-major, so adding a form later only appends tokens).
 * Models remember their own token list, so a model trained with a different
 * order is remapped by token text when loaded (see serialize.ts).
 */
export function buildVocab(spec: VocabSpec, languages: readonly LanguageSpec[], forms: readonly string[]): Vocab {
  const targets = forms.flatMap((f) => languages.map((l) => targetToken(l.code, f)));
  return new Vocab([...spec.specials, ...spec.chars, ...targets]);
}
