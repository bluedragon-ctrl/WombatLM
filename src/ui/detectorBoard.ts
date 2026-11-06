/**
 * The detector board: all thinking units ("detectors") of the writer at a
 * glance, one square each, so a learner sees there are many of them and how
 * they divide the work.
 *
 * - "job" mode (What have the detectors learned?): each square is coloured by
 *   the kind of job the detector found (about to write a letter / just wrote
 *   a letter / a language or kind of word / no clear job) and labelled with
 *   its letter; a dot marks the ones that make a real difference on their own.
 * - "now" mode (Think stage of "How the model picks one letter"): each square
 *   glows orange as strongly as the detector is switched on for this letter;
 *   the ones the letter needs most are outlined.
 *
 * Clicking a square picks that detector (its card is shown by the caller).
 */
import type { UnitInfo } from '../train/protocol';
import { h } from './dom';
import { t } from './i18n';
import { shortLabel } from './tokens';
import { jobKind, matters, strengthWord, themeText, type JobKind } from './unitCard';

/** A very short label for a square: the letter, or the language code, or nothing. */
function cellLabel(u: UnitInfo): string {
  const th = u.theme;
  if (!th) return '';
  switch (th.kind) {
    case 'next': return th.value === '' ? '⏎' : shortLabel(th.value);
    case 'prev': return th.value === '<start>' ? '▸' : shortLabel(th.value);
    case 'task': { const [l, f] = th.value.split(':'); return `${l}${FORM_MARK[f] ?? ''}`; }
    case 'lang': return th.value;
    case 'form': return FORM_MARK[th.value] ?? th.value.slice(0, 1);
  }
}

/** Short marks for the kinds of word, so they do not look like language codes. */
const FORM_MARK: Record<string, string> = { male: '♂', female: '♀', young: '🐣', name: 'N' };

export interface BoardOptions {
  /** "now" mode: how strongly each unit is on for the letter looked at. */
  now?: ArrayLike<number>;
  /** "now" mode: the units the letter needs most (outlined). */
  needed?: ReadonlySet<number>;
  selected?: number | null;
  onPick?: (unit: number) => void;
  langName: (code: string) => string;
}

export function detectorBoard(units: readonly UnitInfo[], opts: BoardOptions): HTMLElement {
  const now = opts.now;
  const max = now ? Math.max(1e-6, ...Array.from(now)) : 1;
  const cells = units.map((u) => {
    const kind = jobKind(u);
    const job = u.theme ? themeText(u.theme, opts.langName) : t('board.noJob');
    const cls = ['dcell', `k-${kind}`];
    let style = '';
    let tip = `${t('unit.title', { n: u.unit + 1 })}: ${job}`;
    if (now) {
      const v = Math.max(0, now[u.unit]);
      const a = v / max;
      cls.push('now');
      if (v <= 0.05) cls.push('off');
      else style = `background:rgba(249,115,22,${(0.15 + 0.85 * a).toFixed(2)});${a > 0.55 ? 'color:#fff;' : ''}`;
      if (opts.needed?.has(u.unit)) cls.push('needed');
      tip += `\n${v > 0.05 ? t('board.tipOn', { strength: strengthWord(a) }) : t('board.tipOff')}`;
    } else if (matters(u)) {
      cls.push('alone');
      tip += `\n${t('board.tipAlone')}`;
    }
    if (opts.selected === u.unit) cls.push('sel');
    const label = cellLabel(u);
    if ([...label].length > 2) cls.push('long');
    else if (u.theme?.kind === 'form') cls.push('sym');
    const b = h('button', { class: cls.join(' '), style, title: tip, 'data-unit': String(u.unit), 'aria-label': tip }, label);
    if (opts.onPick) b.onclick = () => opts.onPick!(u.unit);
    return b;
  });
  return h('div', { class: `dboard${now ? ' now' : ''}`, 'data-test': now ? 'board-now' : 'board' }, ...cells);
}

/** The legend for "job" mode, with how many detectors are in each group. */
export function boardLegend(units: readonly UnitInfo[]): HTMLElement {
  const count = (k: JobKind) => units.filter((u) => jobKind(u) === k).length;
  const item = (k: JobKind) => h('span', { class: 'dlegend-item' }, h('span', { class: `dcell mini k-${k}` }), t(`board.kind.${k}`, { n: count(k) }));
  const alone = units.filter(matters).length;
  return h('div', { class: 'dlegend small' },
    item('next'), item('prev'), item('word'), item('none'),
    units.some((u) => u.off) ? h('span', { class: 'dlegend-item' }, h('span', { class: 'dcell mini alone' }), t('board.alone', { n: alone })) : null);
}

/** A small "how a detector works" picture: numbers in → does it match my pattern? → on: adds its nudge. */
export function detectorFlow(): HTMLElement {
  const box = (title: string, sub: string, cls = '') => h('div', { class: `dflow-box ${cls}` }, h('b', {}, title), h('span', { class: 'small' }, sub));
  const arrow = () => h('span', { class: 'dflow-arrow', 'aria-hidden': 'true' }, '→');
  return h('div', { class: 'dflow', 'data-test': 'detector-flow' },
    box(t('flowd.in'), t('flowd.inSub')),
    arrow(),
    box(t('flowd.check'), t('flowd.checkSub'), 'det'),
    arrow(),
    box(t('flowd.on'), t('flowd.onSub'), 'on'));
}
