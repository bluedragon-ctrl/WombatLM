/**
 * Animal data: loading packs, merging them by id, computing how similar two
 * names are ("kind"), building training examples and validating the files.
 *
 * Pure TypeScript (no DOM), so the worker, tests and Node scripts share it.
 */
import { LanguageSpec, Vocab, VocabSpec, buildVocab, targetToken } from '../nn/vocab';

export type Kind = 'lookalike' | 'partial' | 'different';

/** Word forms per language, e.g. { cs: { male: "tygr", female: "tygřice", young: "tygře" } }. */
export type LangForms = Record<string, Record<string, string>>;

export interface AnimalEntry {
  id: string;
  names: Record<string, string>;
  /** Male / female / young forms of the name, per language. Missing = no real word. */
  forms?: LangForms;
  /** Never trained on: kept aside to test whether the model learned the patterns. */
  unseen?: true;
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
  forms: LangForms;
  /** Never trained on (unseen in any pack means unseen everywhere). */
  unseen: boolean;
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

// ---------------------------------------------------------------------------
// Merging packs
// ---------------------------------------------------------------------------

/** Merge packs in order. Entries with the same id are combined (later packs add languages). */
export function mergePacks(packs: readonly Pack[]): Animal[] {
  const byId = new Map<string, Animal>();
  for (const pack of packs) {
    for (const e of pack.animals) {
      let a = byId.get(e.id);
      if (!a) {
        a = { id: e.id, names: {}, forms: {}, unseen: false, pack: pack.pack };
        byId.set(e.id, a);
      }
      if (e.unseen) a.unseen = true; // unseen anywhere means unseen everywhere (no leakage)
      Object.assign(a.names, normalizeRecord(e.names));
      for (const [lang, forms] of Object.entries(e.forms ?? {})) {
        a.forms[lang] = { ...a.forms[lang], ...normalizeRecord(forms) };
      }
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
  /** The right answer ('' when the language has no real word). */
  answer: string;
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
animal, from, to, form: 'name', source: animal.names[from], answer });
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
          tasks.push({ animal, from: lang, to: lang, form, source: name, answer });
        }
      }
    }
    return tasks;
  }

  /**
   * Word forms the data does not have (žirafa has no Czech male/female/young
   * word). The answer is empty: whatever the model writes is its
   * own invention.
   */
  missingFormTasks(filter?: (a: Animal) => boolean): Task[] {
    const tasks: Task[] = [];
    for (const animal of this.animals) {
      if (filter && !filter(animal)) continue;
      for (const lang of this.languageCodes) {
        const name = animal.names[lang];
        if (!name) continue;
        for (const form of this.forms) {
          if (form === 'name' || animal.forms[lang]?.[form]) continue;
          tasks.push({ animal, from: lang, to: lang, form, source: name, answer: '' });
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
   * (by its name). Returns undefined if unknown.
   */
  detectLanguage(word: string): string | undefined {
    const w = Vocab.normalize(word);
    for (const a of this.animals) {
      for (const lang of this.languageCodes) {
        if (a.names[lang] === w) return lang;
      }
    }
    return undefined;
  }

  /** Find the animal (and language) whose name is `word`. */
  findByName(word: string): { animal: Animal; lang: string } | undefined {
    const w = Vocab.normalize(word);
    for (const a of this.animals) {
      for (const lang of this.languageCodes) {
        if (a.names[lang] === w) return { animal: a, lang };
      }
    }
    return undefined;
  }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export const MAX_WORD_LENGTH = 20;

/** The fields an animal entry may have. One right word per answer: no alternatives. */
const ENTRY_FIELDS = ['id', 'names', 'forms', 'unseen'];

export interface ValidationOptions {
  /** Packs smaller than this need not have both trained and unseen animals (user packs). */
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
    let trained = 0, unseen = 0, added = 0;
    for (const a of pack.animals) {
      const where = `${pack.pack}/${a.id}`;
      if (!a.id) errors.push(`${pack.pack}: animal without id`);
      if (a.unseen !== undefined && a.unseen !== true) errors.push(`${where}: "unseen" must be true or left out`);
      const prevPack = ids.get(a.id);
      // Animals an earlier pack introduced keep its trained / never-seen split.
      if (prevPack === undefined) {
        added++;
        if (a.unseen) unseen++; else trained++;
      }
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
      for (const key of Object.keys(a)) {
        if (!ENTRY_FIELDS.includes(key)) errors.push(`${where}: unknown field "${key}"`);
      }
      for (const [lang, fs] of Object.entries(a.forms ?? {})) {
        if (!a.names?.[lang]) errors.push(`${where}: forms.${lang} but no ${lang} name`);
        for (const [form, word] of Object.entries(fs)) {
          if (!wordForms.includes(form)) errors.push(`${where}: unknown form "${form}"`);
          checkWord(`${where} forms.${lang}.${form}`, lang, word);
        }
      }
    }
    if (added >= minSplit && (!trained || !unseen)) {
      errors.push(`pack ${pack.pack}: needs both trained and "unseen": true animals`);
    }
  }
  return errors;
}
