/**
 * Animal data: loading packs, merging them by id, computing how similar two
 * names are ("kind"), building training examples and validating the files.
 *
 * Pure TypeScript (no DOM), so the worker, tests and Node scripts share it.
 */
import { LanguageSpec, Vocab, VocabSpec, buildVocab, targetToken } from '../nn/vocab';

export type Split = 'train' | 'unseen';
export type Kind = 'lookalike' | 'partial' | 'different';
export const KINDS: readonly Kind[] = ['lookalike', 'partial', 'different'];

/** Per-language word lists, e.g. { cs: ["kocour"] }. */
export type LangWords = Record<string, string[]>;

/** Word forms per language, e.g. { cs: { male: "tygr", female: "tygřice", young: "tygře" } }. */
export type LangForms = Record<string, Record<string, string>>;

/** Accepted alternative forms per language, e.g. { cs: { young: ["koťátko"] } }. */
export type LangFormAlts = Record<string, Record<string, string[]>>;

export interface AnimalEntry {
  id: string;
  names: Record<string, string>;
  split?: Split;
  alt?: LangWords;
  /** Male / female / young forms of the name, per language. Missing = no real word. */
  forms?: LangForms;
  /** Alternative forms accepted when scoring (never trained on). */
  formsAlt?: LangFormAlts;
  /** Entries a native speaker still has to confirm, e.g. ["cs.female"]. Ignored by training. */
  unsure?: string[];
  /** Override the computed kind for a language pair, key "cs-en" (sorted codes). */
  kindOverride?: Record<string, Kind>;
}

export interface Pack {
  pack: string;
  version: number;
  title: Record<string, string>;
  animals: AnimalEntry[];
}

/** An animal after merging all packs. */
export interface Animal {
  id: string;
  names: Record<string, string>;
  split: Split;
  alt: LangWords;
  forms: LangForms;
  formsAlt: LangFormAlts;
  unsure: string[];
  kindOverride: Record<string, Kind>;
  /** Pack that introduced the animal. */
  pack: string;
}

// ---------------------------------------------------------------------------
// Similarity
// ---------------------------------------------------------------------------

/** Thresholds on normalised edit distance (0 = identical, 1 = nothing in common). */
export const KIND_THRESHOLDS = { lookalike: 0.34, partial: 0.6 };

export function stripDiacritics(s: string): string {
  return s.normalize('NFD').replace(/\p{M}/gu, '').normalize('NFC');
}

/**
 * Rough "how does it look/sound" spelling used only to classify pairs:
 * strip diacritics, then fold spellings that sound alike across the Latin
 * alphabet (ph→f, ck/c/q→k, y→i, w→v, x→ks). Without this, tygr/tiger and
 * krokodýl/crocodile would count as only "partial" matches.
 */
export function foldSpelling(s: string): string {
  let t = stripDiacritics(s.toLowerCase());
  for (const [a, b] of [['ph', 'f'], ['ck', 'k'], ['c', 'k'], ['q', 'k'], ['y', 'i'], ['w', 'v'], ['x', 'ks']]) {
    t = t.split(a).join(b);
  }
  return t;
}

export function editDistance(a: string, b: string): number {
  const A = [...a], B = [...b];
  let prev = Array.from({ length: B.length + 1 }, (_, j) => j);
  for (let i = 1; i <= A.length; i++) {
    const cur = [i];
    for (let j = 1; j <= B.length; j++) {
      cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (A[i - 1] === B[j - 1] ? 0 : 1));
    }
    prev = cur;
  }
  return prev[B.length];
}

export function similarity(a: string, b: string): number {
  const fa = foldSpelling(a), fb = foldSpelling(b);
  return editDistance(fa, fb) / Math.max(1, fa.length, fb.length);
}

export function classify(a: string, b: string): Kind {
  const d = similarity(a, b);
  return d <= KIND_THRESHOLDS.lookalike ? 'lookalike' : d <= KIND_THRESHOLDS.partial ? 'partial' : 'different';
}

/** How a word form relates to the name it comes from. */
export type FormKind = 'pattern' | 'partly' | 'irregular';
export const FORM_KINDS: readonly FormKind[] = ['pattern', 'partly', 'irregular'];
export const FORM_KIND_THRESHOLDS = { pattern: 0.75, partly: 0.5 };

/** Length of the longest common subsequence (letters in the same order, gaps allowed). */
export function lcsLength(a: string, b: string): number {
  const A = [...a], B = [...b];
  let prev = new Array(B.length + 1).fill(0);
  for (let i = 1; i <= A.length; i++) {
    const cur = [0];
    for (let j = 1; j <= B.length; j++) cur[j] = A[i - 1] === B[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    prev = cur;
  }
  return prev[B.length];
}

/**
 * Does the form keep the name's letters? Share of the name's letters (without
 * diacritics) found in order inside the form: tygr → tygřice keeps all of
 * them ("pattern"), lev → lvice most ("partly"), pes → štěně almost none
 * ("irregular").
 */
export function formKind(name: string, form: string): FormKind {
  const n = stripDiacritics(name), f = stripDiacritics(form);
  const kept = lcsLength(n, f) / Math.max(1, [...n].length);
  return kept >= FORM_KIND_THRESHOLDS.pattern ? 'pattern' : kept >= FORM_KIND_THRESHOLDS.partly ? 'partly' : 'irregular';
}

export function pairKey(a: string, b: string): string {
  return [a, b].sort().join('-');
}

/** Kind of an animal for a language pair (undefined if it lacks one of the names). */
export function animalKind(animal: Animal, langA: string, langB: string): Kind | undefined {
  const a = animal.names[langA], b = animal.names[langB];
  if (!a || !b) return undefined;
  return animal.kindOverride[pairKey(langA, langB)] ?? classify(a, b);
}

// ---------------------------------------------------------------------------
// Merging packs
// ---------------------------------------------------------------------------

function mergeWords(into: LangWords, from?: LangWords) {
  for (const [lang, words] of Object.entries(from ?? {})) {
    into[lang] = [...new Set([...(into[lang] ?? []), ...words])];
  }
}

/** Merge packs in order. Entries with the same id are combined (later packs add languages). */
export function mergePacks(packs: readonly Pack[]): Animal[] {
  const byId = new Map<string, Animal>();
  for (const pack of packs) {
    for (const e of pack.animals) {
      let a = byId.get(e.id);
      if (!a) {
        a = { id: e.id, names: {}, split: e.split ?? 'train', alt: {}, forms: {}, formsAlt: {}, unsure: [], kindOverride: {}, pack: pack.pack };
        byId.set(e.id, a);
      } else if (e.split === 'unseen') {
        a.split = 'unseen'; // unseen anywhere means unseen everywhere (no leakage)
      }
      Object.assign(a.names, normalizeRecord(e.names));
      for (const [lang, forms] of Object.entries(e.forms ?? {})) {
        a.forms[lang] = { ...a.forms[lang], ...normalizeRecord(forms) };
      }
      for (const [lang, alts] of Object.entries(e.formsAlt ?? {})) {
        a.formsAlt[lang] ??= {};
        mergeWords(a.formsAlt[lang], alts);
      }
      mergeWords(a.alt, e.alt);
      a.unsure = [...new Set([...a.unsure, ...(e.unsure ?? [])])];
      Object.assign(a.kindOverride, e.kindOverride ?? {});
    }
  }
  return [...byId.values()];
}

function normalizeRecord(r: Record<string, string>): Record<string, string> {
  return Object.fromEntries(Object.entries(r).map(([k, v]) => [k, Vocab.normalize(v)]));
}

// ---------------------------------------------------------------------------
// The dataset
// ---------------------------------------------------------------------------

/** One translation task: write `animal`'s name (form) in `to`, starting from its name in `from`. */
export interface Task {
  animal: Animal;
  from: string;
  to: string;
  form: string;
  source: string;
  /** Main answer (used for training). */
  answer: string;
  /** All accepted answers (main + alternatives), for evaluation. */
  accepted: string[];
}

export interface DataBundle {
  vocabSpec: VocabSpec;
  languages: LanguageSpec[];
  forms: string[];
  packs: Pack[];
}

export class Dataset {
  readonly vocab: Vocab;
  readonly languages: LanguageSpec[];
  readonly forms: string[];
  animals: Animal[];
  packs: Pack[];

  constructor(bundle: DataBundle, vocab?: Vocab) {
    this.languages = bundle.languages;
    this.forms = bundle.forms;
    this.vocab = vocab ?? buildVocab(bundle.vocabSpec, bundle.languages, bundle.forms);
    this.packs = bundle.packs.slice();
    this.animals = mergePacks(this.packs);
  }

  get languageCodes(): string[] {
    return this.languages.map((l) => l.code);
  }

  /** Add (or replace) a pack and re-merge. */
  addPack(pack: Pack): void {
    this.packs = [...this.packs.filter((p) => p.pack !== pack.pack), pack];
    this.animals = mergePacks(this.packs);
  }

  removePack(packId: string): void {
    this.packs = this.packs.filter((p) => p.pack !== packId);
    this.animals = mergePacks(this.packs);
  }

  /**
   * Name-translation tasks for every ordered language pair an animal has.
   * With `identity`, only the "keep it as it is" tasks (same language in and
   * out, e.g. <cs> tygr -> tygr) are returned instead.
   */
  nameTasks(filter?: (a: Animal) => boolean, identity = false): Task[] {
    const tasks: Task[] = [];
    for (const animal of this.animals) {
      if (filter && !filter(animal)) continue;
      const langs = this.languageCodes.filter((l) => animal.names[l]);
      for (const from of langs) {
        for (const to of langs) {
          if ((from === to) !== identity) continue;
          const answer = animal.names[to];
          tasks.push({
            animal, from, to, form: 'name', source: animal.names[from], answer,
            accepted: [answer, ...(animal.alt[to] ?? []).map(Vocab.normalize)],
          });
        }
      }
    }
    return tasks;
  }

  /** Word-form tasks (name -> male / female / young) within each language. */
  formTasks(filter?: (a: Animal) => boolean): Task[] {
    const tasks: Task[] = [];
    for (const animal of this.animals) {
      if (filter && !filter(animal)) continue;
      for (const lang of this.languageCodes) {
        const name = animal.names[lang];
        if (!name) continue;
        for (const form of this.forms) {
          const answer = animal.forms[lang]?.[form];
          if (form === 'name' || !answer) continue;
          tasks.push({
            animal, from: lang, to: lang, form, source: name, answer,
            accepted: [answer, ...(animal.formsAlt[lang]?.[form] ?? []).map(Vocab.normalize)],
          });
        }
      }
    }
    return tasks;
  }

  /** Token id of the target token for a language and form. */
  targetId(lang: string, form = 'name'): number {
    return this.vocab.id(targetToken(lang, form));
  }

  /**
   * Guess which language a word is in by looking it up in the data
   * (any form, main or alternative). Returns undefined if unknown.
   */
  detectLanguage(word: string): string | undefined {
    const w = Vocab.normalize(word);
    for (const a of this.animals) {
      for (const lang of this.languageCodes) {
        if (a.names[lang] === w || (a.alt[lang] ?? []).includes(w)) return lang;
      }
    }
    return undefined;
  }

  /** Find the animal (and language) whose name is `word`. */
  findByName(word: string): { animal: Animal; lang: string } | undefined {
    const w = Vocab.normalize(word);
    for (const a of this.animals) {
      for (const lang of this.languageCodes) {
        if (a.names[lang] === w || (a.alt[lang] ?? []).includes(w)) return { animal: a, lang };
      }
    }
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export const MAX_WORD_LENGTH = 20;

export interface ValidationOptions {
  /** Packs smaller than this need not have both splits (user packs). */
  minPackSizeForSplits?: number;
}

/** Check packs against the rules in the design (section 4.5). Returns a list of problems. */
export function validatePacks(
  packs: readonly Pack[], vocab: Vocab, languages: readonly string[], forms: readonly string[], opts: ValidationOptions = {},
): string[] {
  const wordForms = forms.filter((f) => f !== 'name');
  const errors: string[] = [];
  const minSplit = opts.minPackSizeForSplits ?? 8;
  const ids = new Map<string, string>();
  const namesSeen = new Map<string, string>(); // "lang:word" -> id
  const checkWord = (where: string, lang: string, word: string) => {
    if (!languages.includes(lang)) errors.push(`${where}: unknown language "${lang}"`);
    if (typeof word !== 'string' || word.length === 0) { errors.push(`${where}: empty word`); return; }
    if (word !== word.normalize('NFC')) errors.push(`${where}: "${word}" is not NFC-normalised`);
    if (word !== word.toLowerCase()) errors.push(`${where}: "${word}" is not lower-case`);
    if (word !== word.trim() || /\s{2,}/.test(word)) errors.push(`${where}: "${word}" has extra spaces`);
    if ([...word].length > MAX_WORD_LENGTH) errors.push(`${where}: "${word}" is longer than ${MAX_WORD_LENGTH}`);
    const bad = vocab.unknownChars(word);
    if (bad.length) errors.push(`${where}: "${word}" has characters outside the vocabulary: ${bad.join(' ')}`);
  };
  for (const pack of packs) {
    if (!pack.pack || !Array.isArray(pack.animals)) { errors.push(`pack ${pack.pack}: malformed`); continue; }
    const splits = new Set<string>();
    for (const a of pack.animals) {
      const where = `${pack.pack}/${a.id}`;
      if (!a.id) errors.push(`${pack.pack}: animal without id`);
      if (a.split !== undefined && a.split !== 'train' && a.split !== 'unseen') errors.push(`${where}: bad split "${a.split}"`);
      splits.add(a.split ?? 'train');
      const prevPack = ids.get(a.id);
      // The same id in another pack is allowed only to add new languages.
      if (prevPack === pack.pack) errors.push(`${where}: duplicate id`);
      ids.set(a.id, pack.pack);
      for (const [lang, word] of Object.entries(a.names ?? {})) {
        checkWord(`${where} names.${lang}`, lang, word);
        const key = `${lang}:${word}`;
        const other = namesSeen.get(key);
        if (other && other !== a.id) errors.push(`${where}: name "${word}" (${lang}) is also used by ${other}`);
        namesSeen.set(key, a.id);
      }
      if (Object.keys(a.names ?? {}).length === 0) errors.push(`${where}: no names`);
      for (const [lang, words] of Object.entries(a.alt ?? {})) {
        if (!Array.isArray(words)) { errors.push(`${where} alt.${lang}: must be a list`); continue; }
        words.forEach((w) => checkWord(`${where} alt.${lang}`, lang, w));
      }
      for (const [lang, fs] of Object.entries(a.forms ?? {})) {
        if (!a.names?.[lang]) errors.push(`${where}: forms.${lang} but no ${lang} name`);
        for (const [form, word] of Object.entries(fs)) {
          if (!wordForms.includes(form)) errors.push(`${where}: unknown form "${form}"`);
          checkWord(`${where} forms.${lang}.${form}`, lang, word);
        }
      }
      for (const [lang, fs] of Object.entries(a.formsAlt ?? {})) {
        for (const [form, words] of Object.entries(fs)) {
          if (!a.forms?.[lang]?.[form]) errors.push(`${where}: formsAlt.${lang}.${form} without a main form`);
          if (!Array.isArray(words)) { errors.push(`${where} formsAlt.${lang}.${form}: must be a list`); continue; }
          words.forEach((w) => checkWord(`${where} formsAlt.${lang}.${form}`, lang, w));
        }
      }
      for (const u of a.unsure ?? []) {
        const [lang, field] = u.split('.');
        if (!(field === 'name' ? a.names?.[lang] : a.forms?.[lang]?.[field])) errors.push(`${where}: unsure "${u}" points at nothing`);
      }
      for (const [k, v] of Object.entries(a.kindOverride ?? {})) {
        if (!KINDS.includes(v)) errors.push(`${where}: bad kindOverride ${k}=${v}`);
      }
    }
    if (pack.animals.length >= minSplit && (!splits.has('train') || !splits.has('unseen'))) {
      errors.push(`pack ${pack.pack}: needs both "train" and "unseen" animals`);
    }
  }
  return errors;
}
