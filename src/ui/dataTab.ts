/**
 * "Learning data" tab: everything the model learns from, as one table per
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

/** Counts of what the model learns from, per language and in total. */
export interface DataCounts {
  animals: number;
  train: number;
  unseen: number;
  /** Learning examples: word forms and "copy the name" of train-split animals. */
  forms: number;
  copies: number;
}

export function countData(ready: Pick<ReadyMsg, 'animals' | 'forms'>, lang?: string): DataCounts {
  const c: DataCounts = { animals: 0, train: 0, unseen: 0, forms: 0, copies: 0 };
  for (const a of ready.animals) {
    const langs = lang ? [lang] : Object.keys(a.names);
    const has = langs.filter((l) => a.names[l]);
    if (has.length === 0) continue;
    c.animals++;
    if (a.split === 'unseen') { c.unseen++; continue; }
    c.train++;
    for (const l of has) {
      c.copies++;
      for (const f of ready.forms) if (f !== 'name' && a.forms[l]?.[f]) c.forms++;
    }
  }
  return c;
}

export class DataTab {
  readonly el: HTMLElement;
  private ready?: ReadyMsg;
  private lang = '';
  private filter: DataFilter = 'all';
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
      this.summary,
      h('div', { class: 'row data-controls' }, this.langPick, this.filterPick, search,
        h('label', { class: 'small' }, revealBox, ' ', t('data.reveal'))),
      h('div', { class: 'data-legend small' },
        h('span', { class: 'learned' }, t('data.legend.learned')),
        h('span', { class: 'missing' }, `— ${t('data.legend.missing')}`),
        h('span', { class: 'kept' }, t('data.legend.kept'))),
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
      h('p', { class: 'small' }, t('data.sum.lang', { lang: langName, examples: n(here.forms + here.copies), forms: n(here.forms), copies: n(here.copies) })));

    const lang = this.lang;
    const forms = ready.forms.filter((f) => f !== 'name');
    const list = ready.animals
      .filter((a) => a.names[lang])
      .filter((a) => this.filter === 'all' || a.split === this.filter)
      .filter((a) => !this.search || this.matches(a, lang))
      .sort((a, b) => a.names[lang].localeCompare(b.names[lang], lang));
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
    this.renderExample(ready);
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
    const words = [a.names[lang], ...(hidden ? [] : Object.values(a.forms[lang] ?? {}))];
    return words.some((w) => w.toLowerCase().includes(this.search));
  }
}
