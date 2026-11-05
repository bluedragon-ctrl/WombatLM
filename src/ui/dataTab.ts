/**
 * "Learning data" tab: everything the model learns from, as one table per
 * language. For each animal: its name (copy practice) and the right male /
 * female / young word where the data has one. Never-seen animals are listed
 * too, greyed out: they are kept aside to test it, and their answers stay
 * hidden until revealed (like on the Ask tab).
 *
 * This mirrors the trainer (`FormTrainer.defaultMix`): word-form tasks and
 * "copy the name" tasks of train-split animals, main answers only.
 */
import type { AnimalInfo, ReadyMsg } from '../train/protocol';
import { h } from './dom';
import { t, uiLang } from './i18n';

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

  /** `onAsk`: a word was clicked; ask the model that question on the Ask tab. */
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
    this.el = h('section', { class: 'panel' },
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
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    if (!ready.languages.some((l) => l.code === this.lang)) this.lang = ready.languages[0].code;
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
        const cell = h('td', { class: `cell ${cls}`, title }, text);
        cell.onclick = () => this.onAsk(a.names[lang], lang, form);
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

  private matches(a: AnimalInfo, lang: string): boolean {
    // Hidden answers of never-seen animals must not be findable by searching.
    const hidden = a.split === 'unseen' && !this.reveal;
    const words = [a.names[lang], ...(hidden ? [] : Object.values(a.forms[lang] ?? {}))];
    return words.some((w) => w.toLowerCase().includes(this.search));
  }
}
