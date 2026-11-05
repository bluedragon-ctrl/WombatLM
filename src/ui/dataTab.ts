/**
 * "Learning data" tab: everything the model learns from and is tested on.
 * Three views of the same data:
 * - table: one row per animal and language (an overview);
 * - questions: every question exactly as the model gets it (task token +
 *   name), its right answer, and whether it is learned from, tested on, or
 *   both;
 * - records: the data records themselves, every field, and which of them
 *   ever reach the model.
 *
 * The table: one table per
 * language. For each animal: its name (copy practice) and the right male /
 * female / young word where the data has one. Never-seen animals are listed
 * too, greyed out: they are kept aside to test it, and their answers stay
 * hidden until revealed (like on the Ask tab).
 *
 * Clicking a word shows that learning example as tokens, the way the model
 * trains on it: what it reads, what it is shown of the answer so far, and
 * the next token it must guess at each step.
 *
 * This mirrors the trainer (`FormTrainer.defaultMix`): word-form tasks and
 * "copy the name" tasks of train-split animals, main answers only.
 */
import type { AnimalInfo, ReadyMsg } from '../train/protocol';
import { h } from './dom';
import { t, uiLang } from './i18n';
import { learningExample, Tok, tokenChip, tokenIndex, tokenizeWord, tokenName, toTok } from './tokens';
import { targetToken } from '../nn/vocab';

type DataFilter = 'all' | 'train' | 'unseen';
const FILTERS: DataFilter[] = ['all', 'train', 'unseen'];
type DataView = 'table' | 'questions' | 'records';
const VIEWS: DataView[] = ['table', 'questions', 'records'];

/**
 * One question the data can ask: write `form` of the animal in `lang`,
 * starting from its name. This is all the model is given (a task token for
 * language + form, and the letters of `source`); the rest says how the
 * question is used.
 */
export interface DataQuestion {
  id: string;
  lang: string;
  form: string;
  source: string;
  /** The right answer; '' when the language has no real word (the model can only invent one). */
  answer: string;
  split: 'train' | 'unseen';
  /** In the learning data: the trainer picks it for training steps. */
  learn: boolean;
  /** Asked when testing (Test tab). With no answer, whatever it writes counts as invented. */
  test: boolean;
}

/**
 * Every question for the animals (in one language, or all), the way the
 * trainer and the tests use them (`FormTrainer.defaultMix`, `evaluate`):
 * - names: "copy the name" is learned for animals it learns from; never tested;
 * - forms with a real word: learned (animals it learns from) and tested (all animals);
 * - forms without a real word: never learned; tested as "what does it invent".
 */
export function dataQuestions(ready: Pick<ReadyMsg, 'animals' | 'forms' | 'languages'>, lang?: string): DataQuestion[] {
  const out: DataQuestion[] = [];
  const langs = lang ? [lang] : ready.languages.map((l) => l.code);
  for (const a of ready.animals) {
    for (const l of langs) {
      const source = a.names[l];
      if (!source) continue;
      const train = a.split === 'train';
      for (const form of ready.forms) {
        const isName = form === 'name';
        const answer = isName ? source : a.forms[l]?.[form] ?? '';
        out.push({
          id: a.id, lang: l, form, source, answer, split: a.split,
          learn: train && !!answer,
          test: !isName,
        });
      }
    }
  }
  return out;
}

/** Counts of what the model learns from, per language and in total. */
export interface DataCounts {
  animals: number;
  train: number;
  unseen: number;
  /** Learning examples: word forms and "copy the name" of train-split animals. */
  forms: number;
  copies: number;
  /** Test questions with a right answer (all animals), and those of never-seen animals. */
  tests: number;
  unseenTests: number;
  /** Test questions without a real word: whatever it writes is its own invention. */
  inventions: number;
}

export function countData(ready: Pick<ReadyMsg, 'animals' | 'forms'>, lang?: string): DataCounts {
  const c: DataCounts = { animals: 0, train: 0, unseen: 0, forms: 0, copies: 0, tests: 0, unseenTests: 0, inventions: 0 };
  for (const a of ready.animals) {
    const langs = lang ? [lang] : Object.keys(a.names);
    const has = langs.filter((l) => a.names[l]);
    if (has.length === 0) continue;
    c.animals++;
    const train = a.split === 'train';
    if (train) c.train++; else c.unseen++;
    for (const l of has) {
      if (train) c.copies++;
      for (const f of ready.forms) {
        if (f === 'name') continue;
        if (!a.forms[l]?.[f]) { c.inventions++; continue; }
        c.tests++;
        if (train) c.forms++; else c.unseenTests++;
      }
    }
  }
  return c;
}

export class DataTab {
  readonly el: HTMLElement;
  private ready?: ReadyMsg;
  private lang = '';
  private filter: DataFilter = 'all';
  private view: DataView = 'table';
  private viewPick: HTMLElement;
  private viewNote: HTMLElement;
  private legend: HTMLElement;
  private search = '';
  private reveal = false;
  private summary: HTMLElement;
  private langPick: HTMLElement;
  private filterPick: HTMLElement;
  private table: HTMLElement;
  private example: HTMLElement;
  private expert = false;
  /** The learning example shown as tokens. */
  private sel?: { id: string; lang: string; form: string };

  /** `onAsk`: "ask the model this question" was pressed; ask it on the Ask tab. */
  constructor(private onAsk: (text: string, lang: string, form: string) => void) {
    this.summary = h('div', { class: 'data-summary', 'data-test': 'data-summary' });
    this.langPick = h('div', { class: 'picker', role: 'group', 'aria-label': t('language') });
    this.filterPick = h('div', { class: 'picker', role: 'group' });
    this.viewPick = h('div', { class: 'picker data-views', role: 'group', 'data-test': 'data-views' });
    for (const v of VIEWS) {
      const b = h('button', { 'data-view': v, title: t(`data.view.${v}.tip`) }, t(`data.view.${v}`));
      b.onclick = () => { this.view = v; this.render(); };
      this.viewPick.append(b);
    }
    this.viewNote = h('p', { class: 'small data-view-note', 'data-test': 'data-view-note' });
    for (const f of FILTERS) {
      const b = h('button', { 'data-filter': f }, t(`data.filter.${f}`));
      b.onclick = () => { this.filter = f; this.render(); };
      this.filterPick.append(b);
    }
    const search = h('input', { type: 'search', class: 'data-search', placeholder: t('data.search'), spellcheck: 'false', 'aria-label': t('data.search') });
    search.oninput = () => { this.search = search.value.trim().toLowerCase(); this.render(); };
    const revealBox = h('input', { type: 'checkbox', 'data-test': 'data-reveal' });
    revealBox.onchange = () => { this.reveal = revealBox.checked; this.render(); };
    this.table = h('div', { class: 'data-table', 'data-test': 'data-table' });
    this.example = h('section', { class: 'panel example compact-tokens', 'data-test': 'example' });
    const tablePanel = h('section', { class: 'panel' },
      h('div', { class: 'phase-head' }, h('h2', {}, t('data.title')),
        h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, t('data.help')))),
      h('p', { class: 'practice-explain' }, t('data.explain')),
      h('div', { class: 'data-given', 'data-test': 'data-given' },
        h('b', {}, t('data.given.title')), ' ', t('data.given.body')),
      this.summary,
      this.viewPick,
      this.viewNote,
      h('div', { class: 'row data-controls' }, this.langPick, this.filterPick, search,
        h('label', { class: 'small' }, revealBox, ' ', t('data.reveal'))),
      this.legend = h('div', { class: 'data-legend small' }),
      this.table);
    this.el = h('div', { class: 'data-layout' }, tablePanel, this.example);
  }

  setExpert(on: boolean): void {
    this.expert = on;
    this.render();
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    if (!ready.languages.some((l) => l.code === this.lang)) this.lang = ready.languages[0].code;
    if (!this.sel || !ready.animals.some((a) => a.id === this.sel!.id)) this.sel = this.firstExample(ready);
    this.langPick.replaceChildren(...ready.languages.map((l) => {
      const b = h('button', { 'data-lang': l.code }, l.name[uiLang()] ?? l.code);
      b.onclick = () => { this.lang = l.code; this.render(); };
      return b;
    }));
    this.render();
  }

  render(): void {
    const ready = this.ready;
    if (!ready) return;
    for (const b of this.langPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.lang === this.lang);
    for (const b of this.filterPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.filter === this.filter);

    const all = countData(ready);
    const here = countData(ready, this.lang);
    const langName = ready.languages.find((l) => l.code === this.lang)?.name[uiLang()] ?? this.lang;
    const n = (v: number) => v.toLocaleString(uiLang());
    this.summary.replaceChildren(
      h('div', { class: 'stat' }, h('b', {}, n(all.forms + all.copies)), ' ', t('data.sum.examples')),
      h('div', { class: 'stat' }, h('b', {}, n(all.forms)), ' ', t('data.sum.forms')),
      h('div', { class: 'stat' }, h('b', {}, n(all.copies)), ' ', t('data.sum.copies')),
      h('div', { class: 'stat' }, h('b', {}, n(all.train)), ' ', t('data.sum.train')),
      h('div', { class: 'stat kept' }, h('b', {}, n(all.unseen)), ' ', t('data.sum.unseen')),
      h('div', { class: 'stat' }, h('b', {}, n(all.tests)), ' ', t('data.sum.tests', { unseen: n(all.unseenTests) })),
      h('div', { class: 'stat kept' }, h('b', {}, n(all.inventions)), ' ', t('data.sum.inventions')),
      h('p', { class: 'small' }, t('data.sum.lang', { lang: langName, examples: n(here.forms + here.copies), forms: n(here.forms), copies: n(here.copies) })));

    this.langPick.hidden = this.view === 'records';
    for (const b of this.viewPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.view === this.view);
    this.viewNote.textContent = t(`data.view.${this.view}.note`);
    if (this.view === 'table') this.renderTable(ready);
    else if (this.view === 'questions') this.renderQuestions(ready);
    else this.renderRecords(ready);
    this.renderExample(ready);
  }

  private renderTable(ready: ReadyMsg): void {
    this.legend.hidden = false;
    this.legend.replaceChildren(
      h('span', { class: 'learned' }, t('data.legend.learned')),
      h('span', { class: 'missing' }, `— ${t('data.legend.missing')}`),
      h('span', { class: 'kept' }, t('data.legend.kept')));
    const lang = this.lang;
    const forms = ready.forms.filter((f) => f !== 'name');
    const list = this.animalsShown(ready, lang);
    const rows = list.map((a) => {
      const kept = a.split === 'unseen';
      const word = (text: string, form: string, cls: string, title: string) => {
        const sel = this.sel?.id === a.id && this.sel.lang === lang && this.sel.form === form;
        const cell = h('td', { class: `cell ${cls}${sel ? ' sel' : ''}`, title }, text);
        cell.onclick = () => { this.sel = { id: a.id, lang, form }; this.render(); };
        return cell;
      };
      const cells = forms.map((f) => {
        const w = a.forms[lang]?.[f];
        if (!w) return h('td', { class: 'cell missing', title: t('data.tip.missing') }, '—');
        if (kept && !this.reveal) return h('td', { class: 'cell kept hidden', title: t('data.tip.hidden') }, '?');
        return word(w, f, kept ? 'kept' : 'learned', kept ? t('data.tip.kept') : t('data.tip.learned', { from: a.names[lang], to: w }));
      });
      const name = word(a.names[lang], 'name', kept ? 'kept name' : 'learned name',
        kept ? t('data.tip.kept') : t('data.tip.copy', { word: a.names[lang] }));
      if (kept) name.append(h('span', { class: 'never' }, t('neverSeen')));
      return h('tr', { class: kept ? 'kept' : '' }, name, ...cells);
    });
    this.table.replaceChildren(
      h('p', { class: 'small' }, t('data.shown', { n: list.length })),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, t('data.col.name')),
          ...forms.map((f) => h('th', {}, t(`form.${f}`))))),
        h('tbody', {}, ...rows)));
  }

  /** Animals that pass the split filter and the search, sorted by name. */
  private animalsShown(ready: ReadyMsg, lang: string): AnimalInfo[] {
    return ready.animals
      .filter((a) => a.names[lang])
      .filter((a) => this.filter === 'all' || a.split === this.filter)
      .filter((a) => !this.search || this.matches(a, lang))
      .sort((a, b) => a.names[lang].localeCompare(b.names[lang], lang));
  }

  /** Every question, as the model gets it, with its answers and how it is used. */
  private renderQuestions(ready: ReadyMsg): void {
    this.legend.hidden = true;
    const lang = this.lang;
    const index = tokenIndex(ready.tokens);
    const langName = (code: string) => ready.languages.find((l) => l.code === code)?.name[uiLang()] ?? code;
    const opts = { expert: this.expert, langName };
    const animals = this.animalsShown(ready, lang);
    const order = new Map(animals.map((a, i) => [a.id, i]));
    const qs = dataQuestions(ready, lang)
      .filter((q) => order.has(q.id))
      .sort((a, b) => order.get(a.id)! - order.get(b.id)! || ready.forms.indexOf(a.form) - ready.forms.indexOf(b.form));
    const n = (v: number) => v.toLocaleString(uiLang());
    const learn = qs.filter((q) => q.learn).length;
    const test = qs.filter((q) => q.test && q.answer).length;
    const invent = qs.filter((q) => q.test && !q.answer).length;
    const unused = qs.filter((q) => !q.learn && !q.test).length;
    const yes = (on: boolean, text = '✓') => h('td', { class: on ? 'use yes' : 'use no' }, on ? text : '—');
    const rows = qs.map((q) => {
      const hidden = q.split === 'unseen' && !this.reveal && q.form !== 'name';
      const sel = this.sel?.id === q.id && this.sel.lang === q.lang && this.sel.form === q.form;
      const answer = !q.answer ? h('span', { class: 'missing' }, t('data.q.noWord'))
        : hidden ? h('span', { class: 'hidden', title: t('data.tip.hidden') }, '?')
        : h('span', {}, q.answer);
      const tr = h('tr', { class: `${q.split === 'unseen' ? 'kept' : ''}${sel ? ' sel' : ''}` },
        h('td', { class: 'q' }, h('span', { class: 'chips' }, tokenChip(toTok(targetToken(q.lang, q.form), index), opts)), ' ', h('b', {}, q.source)),
        h('td', { class: 'arrow' }, '→'),
        h('td', { class: 'ans' }, answer),
        yes(q.learn),
        yes(q.test, q.answer ? '✓' : t('data.q.invents')),
        h('td', { class: 'small' }, q.split === 'unseen' ? t('neverSeen') : ''));
      tr.onclick = () => { this.sel = { id: q.id, lang: q.lang, form: q.form }; this.render(); };
      return tr;
    });
    this.table.replaceChildren(
      h('p', { class: 'small', 'data-test': 'data-q-count' }, t('data.q.shown', { n: n(qs.length), learn: n(learn), test: n(test), invent: n(invent), unused: n(unused) })),
      h('table', { class: 'questions', 'data-test': 'data-questions' },
        h('thead', {}, h('tr', {},
          h('th', {}, t('data.q.col.question')), h('th', {}),
          h('th', {}, t('data.q.col.answer')),
          h('th', {}, t('data.q.col.learn')),
          h('th', {}, t('data.q.col.test')),
          h('th', {}))),
        h('tbody', {}, ...rows)));
  }

  /** The data records themselves: every field, with what reaches the model. */
  private renderRecords(ready: ReadyMsg): void {
    this.legend.hidden = false;
    this.legend.replaceChildren(h('dl', { class: 'record-fields' },
      ...(['id', 'pack', 'split', 'names', 'forms'] as const).flatMap((f) => [
        h('dt', {}, h('code', {}, f)), h('dd', {}, t(`data.rec.${f}`))])));
    const hiddenMark = t('data.rec.hidden');
    const animals = ready.animals
      .filter((a) => this.filter === 'all' || a.split === this.filter)
      .filter((a) => !this.search || Object.keys(a.names).some((l) => this.matches(a, l)))
      .sort((a, b) => a.id.localeCompare(b.id));
    const mask = <T>(a: AnimalInfo, v: T): T | string => (a.split === 'unseen' && !this.reveal ? hiddenMark : v);
    const records = animals.map((a) => {
      const rec: Record<string, unknown> = { id: a.id, pack: a.pack, split: a.split, names: a.names };
      if (Object.keys(a.forms).length) rec.forms = mask(a, a.forms);
      return h('pre', { class: `record${a.split === 'unseen' ? ' kept' : ''}` }, recordJson(rec));
    });
    this.table.replaceChildren(
      h('p', { class: 'small' }, t('data.rec.shown', { n: animals.length })),
      h('div', { class: 'records', 'data-test': 'data-records' }, ...records));
  }

  /** Start with a word-form example of the first animal it learns from, so the panel is never empty. */
  private firstExample(ready: ReadyMsg): DataTab['sel'] {
    const lang = this.lang;
    const train = ready.animals.filter((a) => a.split === 'train' && a.names[lang])
      .sort((a, b) => a.names[lang].localeCompare(b.names[lang], lang));
    for (const form of ['female', ...ready.forms.filter((f) => f !== 'name')]) {
      const a = train.find((x) => x.forms[lang]?.[form]);
      if (a) return { id: a.id, lang, form };
    }
    return train[0] && { id: train[0].id, lang, form: 'name' };
  }

  /** The selected learning example as tokens: reads, shown so far, must guess next. */
  private renderExample(ready: ReadyMsg): void {
    const sel = this.sel;
    const a = sel && ready.animals.find((x) => x.id === sel.id);
    if (!sel || !a?.names[sel.lang]) {
      this.example.replaceChildren(h('h2', {}, t('example.title')), h('p', { class: 'small' }, t('example.pick')));
      return;
    }
    const { lang, form } = sel;
    const index = tokenIndex(ready.tokens);
    const langName = (code: string) => ready.languages.find((l) => l.code === code)?.name[uiLang()] ?? code;
    const opts = { expert: this.expert, langName };
    const source = a.names[lang];
    const answer = form === 'name' ? source : a.forms[lang]?.[form] ?? '';
    const kept = a.split === 'unseen';
    const chips = (toks: Tok[]) => h('span', { class: 'chips' }, ...toks.map((tok) => tokenChip(tok, opts)));
    const row = (label: string, el: HTMLElement) => h('div', { class: 'token-row stacked' }, h('span', { class: 'token-label small' }, label), el);
    const ask = h('button', { 'data-test': 'example-ask' }, t('example.ask'));
    ask.onclick = () => this.onAsk(source, lang, form);
    const shownAnswer = kept && !this.reveal && form !== 'name' ? '?' : answer;
    const parts: (Node | string)[] = [
      h('div', { class: 'phase-head' }, h('h2', {}, t('example.title'))),
      h('p', { class: 'example-question' }, t('example.question', { lang: langName(lang), form: t(`form.${form}`), from: source, to: shownAnswer })),
    ];
    if (kept || !answer) {
      parts.push(row(t('example.reads'), chips([toTok(targetToken(lang, form), index), ...tokenizeWord(source, index)])),
        h('p', { class: 'small kept-note' }, t('example.kept')));
    } else {
      const ex = learningExample(index, lang, form, source, answer);
      const name = (tok: Tok) => tokenName(tok.token, opts);
      const caption = h('p', { class: 'small example-caption', 'data-test': 'example-caption' });
      const cells: HTMLElement[][] = [];
      const highlight = (i: number | null) => {
        cells.forEach((col, j) => col.forEach((c) => c.classList.toggle('hl', j === i)));
        caption.textContent = i === null ? '' : t('example.guessTip', { sofar: ex.shown.slice(0, i + 1).map(name).join(' '), next: name(ex.guess[i]) });
      };
      const col = (i: number, content: Node | string, cls: string) => {
        const td = h('td', { class: cls }, content);
        td.onmouseenter = () => highlight(i);
        td.onmouseleave = () => highlight(null);
        (cells[i] ??= []).push(td);
        return td;
      };
      const n = ex.shown.length;
      const label = (text: string) => h('tr', {}, h('th', { class: 'small', colspan: n }, text));
      const grid = h('div', { class: 'example-scroll' }, h('table', { class: 'example-grid', 'data-test': 'example-grid' },
        label(t('example.shown')),
        h('tr', {}, ...ex.shown.map((tok, i) => col(i, tokenChip(tok, opts), 'shown'))),
        h('tr', {}, ...ex.shown.map((_, i) => col(i, '↓', 'arrow'))),
        h('tr', {}, ...ex.guess.map((tok, i) => col(i, tokenChip(tok, opts), 'guess'))),
        label(t('example.guess'))));
      parts.push(row(t('example.reads'), chips(ex.reads)), grid, caption,
        h('p', { class: 'small' }, t('example.explain', { n: ex.guess.length })));
    }
    parts.push(h('div', { class: 'row' }, ask));
    this.example.replaceChildren(...parts);
  }

  private matches(a: AnimalInfo, lang: string): boolean {
    // Hidden answers of never-seen animals must not be findable by searching.
    const hidden = a.split === 'unseen' && !this.reveal;
    if (!a.names[lang]) return false;
    const words = [a.names[lang], a.id, ...(hidden ? [] : Object.values(a.forms[lang] ?? {}))];
    return words.some((w) => w.toLowerCase().includes(this.search));
  }
}

/** Readable JSON: one line per field; objects by language get one line per language. */
function recordJson(rec: Record<string, unknown>): string {
  const one = (v: unknown) => JSON.stringify(v).replace(/,"/g, ', "').replace(/":/g, '": ');
  const lines = Object.entries(rec).map(([k, v]) => {
    const nested = v && typeof v === 'object' && !Array.isArray(v) && Object.values(v).some((x) => x && typeof x === 'object');
    if (!nested) return `  "${k}": ${one(v)}`;
    const inner = Object.entries(v as Record<string, unknown>).map(([l, x]) => `    "${l}": ${one(x)}`);
    return `  "${k}": {\n${inner.join(',\n')}\n  }`;
  });
  return `{\n${lines.join(',\n')}\n}`;
}
