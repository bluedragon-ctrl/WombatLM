/**
 * A card for one thinking unit ("detector"): what switches it on most in the
 * learning data (with a plain-language theme when there is a clear one), and
 * what really happens when it is switched off at those moments (measured).
 * Used by "What have the detectors learned?" and by the Think stage of "How
 * the model picks one letter" (which adds the same for the letter shown).
 */
import type { UnitInfo, UnitMoment, UnitTheme } from '../train/protocol';
import { h } from './dom';
import { t } from './i18n';
import { shortLabel } from './tokens';
import { pctText } from './format';

/** How many of its strongest moments a card shows. */
const SHOW = 5;
/** …and a compact card (Think stage). */
const COMPACT_SHOW = 3;
/** A change of the right letter's chance smaller than this counts as "almost nothing". */
const SMALL = 0.02;

/** One sentence: what switching it off does to the right letter's chance (falls / rises / almost nothing). */
export function offText(withP: number, withoutP: number, keys: { fall: string; rise: string; none: string }, vars: Record<string, string> = {}): string {
  const v = { ...vars, with: pctText(withP), without: pctText(withoutP) };
  if (withoutP < withP - SMALL) return t(keys.fall, v);
  if (withoutP > withP + SMALL) return t(keys.rise, v);
  return t(keys.none, v);
}

export function themeText(theme: UnitTheme, langName: (code: string) => string): string {
  const form = (f: string) => t(`tok.form.${f}`);
  switch (theme.kind) {
    case 'next': return theme.value === '' ? t('unit.theme.end') : t('unit.theme.next', { v: theme.value });
    case 'prev': return theme.value === '<start>' ? t('unit.theme.start') : t('unit.theme.prev', { v: theme.value });
    case 'task': { const [l, f] = theme.value.split(':'); return t('unit.theme.task', { lang: langName(l), form: form(f) }); }
    case 'form': return t('unit.theme.form', { form: form(theme.value) });
    case 'lang': return t('unit.theme.lang', { lang: langName(theme.value) });
  }
}

/** One moment: the answer with the letter being written highlighted (⏎ = the end); `strength` (0..1) shades the bar under it. */
export function momentChip(m: UnitMoment, langName: (code: string) => string, strength = 1): HTMLElement {
  return h('span', {
    class: 'umoment',
    style: `--s:${Math.max(0.15, Math.min(1, strength)).toFixed(2)}`,
    title: t('unit.momentTip', { source: m.source, lang: langName(m.lang), form: t(`tok.form.${m.form}`), k: m.at + 1, v: m.value.toFixed(2) }),
  },
  h('span', { class: 'done' }, m.answer.slice(0, m.at)),
  h('b', {}, m.next === '' ? '⏎' : m.next),
  h('span', { class: 'rest' }, m.answer.slice(m.at + 1)));
}

/**
 * The card, in three parts: when it switches on (its job and its strongest
 * moments), what happens without it (measured: the right letter's chance with
 * and without it, as two bars) and how often it is on at all. With `now` (its
 * value for the letter looked at in "picks one letter"), a bar shows how
 * strongly it is on now, compared with its strongest moment in the learning data.
 * `compact` (the Think stage in plain mode): only its job, 3 moments without
 * a label, and with/without bars for the letter looked at now.
 */
export function unitCard(u: UnitInfo, langName: (code: string) => string,
  opts: { now?: number; nowOff?: { letter: string; withP: number; withoutP: number }; expert?: boolean; picked?: boolean; compact?: boolean } = {}): HTMLElement {
  const head = h('div', { class: 'uhead' }, h('b', {}, t('unit.title', { n: u.unit + 1 })));
  const strongest = Math.max(1e-6, u.top[0]?.value ?? opts.now ?? 1);
  if (opts.now !== undefined) {
    const a = Math.min(1, opts.now / strongest);
    head.append(h('span', { class: 'unow small', title: t('unit.nowTip') },
      t('unit.now', { strength: strengthWord(a) }), ' ',
      h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.max(3, Math.round(a * 100))}%` }))));
  }
  const theme = u.theme
    ? h('p', { class: 'utheme' }, t('unit.themeLine', { theme: themeText(u.theme, langName) }), ' ', sureDots(u.theme.count, u.theme.of))
    : h('p', { class: 'utheme none' }, t('unit.noTheme'));
  const moments = u.top.slice(0, opts.compact ? COMPACT_SHOW : SHOW);
  const chips = h('div', { class: 'umoments', title: t('unit.moments') }, ...moments.map((m) => momentChip(m, langName, m.value / strongest)));
  const cls = `ucard k-${jobKind(u)}${u.theme ? '' : ' unclear'}${opts.picked ? ' picked' : ''}`;
  if (opts.compact) {
    const n = opts.nowOff;
    const verdict = n ? (n.withoutP < n.withP - SMALL ? 'fall' : n.withoutP > n.withP + SMALL ? 'rise' : 'none') : '';
    return h('div', { class: `${cls} compact`, 'data-unit': String(u.unit) },
      head, theme, chips,
      n ? h('div', { class: 'uoffbox', 'data-test': 'unit-off-now' },
        h('p', { class: 'ulabel small' }, t('unit.offNowLabel', { tok: shortLabel(n.letter) })),
        h('div', { class: 'ubars small' },
          bar(t('unit.with'), n.withP, 'with'),
          bar(t('unit.without'), n.withoutP, `without ${verdict}`))) : null);
  }
  const parts: (HTMLElement | null)[] = [
    head,
    theme,
    h('p', { class: 'ulabel small' }, t('unit.moments')),
    chips,
  ];
  // What it really does: switched off at these moments (measured), and for the letter shown now.
  if (opts.nowOff) {
    const n = opts.nowOff;
    parts.push(h('p', { class: 'uoff now', 'data-test': 'unit-off-now' },
      offText(n.withP, n.withoutP, { fall: 'unit.offNow.fall', rise: 'unit.offNow.rise', none: 'unit.offNow.none' }, { tok: shortLabel(n.letter) })));
  }
  if (u.off) {
    const o = u.off;
    const verdict = o.withoutP < o.withP - SMALL ? 'fall' : o.withoutP > o.withP + SMALL ? 'rise' : 'none';
    parts.push(h('div', { class: 'uoffbox', 'data-test': 'unit-off' },
      h('p', { class: 'ulabel small' }, t('unit.offLabel')),
      h('div', { class: 'ubars small' },
        bar(t('unit.with'), o.withP, 'with'),
        bar(t('unit.without'), o.withoutP, `without ${verdict}`)),
      h('p', { class: `uverdict small ${verdict}` }, t(`unit.verdict.${verdict}`)),
      o.helps.length ? h('div', { class: 'uhelps small', title: t('unit.helpsTip') }, t('unit.helps'), ' ',
        ...o.helps.map((x) => h('span', { class: 'uhelp' }, h('b', {}, shortLabel(x.char)), ` −${Math.max(1, Math.round(x.drop * 100))}`))) : null));
  }
  // Expert: the straight-line hint from its pattern alone.
  if (opts.expert || !u.off) {
    parts.push(h('div', { class: 'uvotes small', title: t('unit.votesTip') }, t('unit.votes'), ' ',
      ...u.pushes.map((p) => h('b', { class: 'uvote' }, shortLabel(p.char))),
      opts.expert ? h('span', { class: 'muted' }, ` (${u.pushes.map((p) => p.score.toFixed(2)).join(', ')})`) : null));
  }
  parts.push(h('p', { class: 'ufoot small' }, t('unit.onShare', { pct: Math.round(u.on * 100) })));
  return h('div', { class: cls, 'data-unit': String(u.unit) }, ...parts);
}

/** One labelled bar: the right letter's chance (0..1). */
function bar(label: string, p: number, cls: string): HTMLElement {
  return h('div', { class: `ubar ${cls}` },
    h('span', {}, label),
    h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.max(1, Math.round(p * 100))}%` })),
    h('b', {}, pctText(p)));
}

/** How sure the job is: one dot per strongest moment, filled if it shares the job. */
function sureDots(count: number, of: number): HTMLElement {
  return h('span', { class: 'udots', title: t('unit.dotsTip', { count, of }), 'aria-label': t('unit.dotsTip', { count, of }) },
    ...Array.from({ length: of }, (_, i) => h('span', { class: i < count ? 'on' : '' })));
}

export type JobKind = 'next' | 'prev' | 'word' | 'none';

/** The kind of job, as a colour group (the same on the cards and the board). */
export function jobKind(u: UnitInfo): JobKind {
  const k = u.theme?.kind;
  if (k === 'next') return 'next';
  if (k === 'prev') return 'prev';
  if (k) return 'word';
  return 'none';
}

/** "strongly" / "clearly" / "a little", for how strongly a unit is on (0..1 of its strongest). */
export function strengthWord(a: number): string {
  return t(a >= 0.66 ? 'unit.strong' : a >= 0.33 ? 'unit.medium' : 'unit.weak');
}

/** How much switching it off at its strongest moments lowers the right letter's chance (0 if unknown). */
export function offDrop(u: UnitInfo): number {
  return u.off ? u.off.withP - u.off.withoutP : 0;
}

/** Does it make a real difference on its own (switched off at its strongest moments)? */
export function matters(u: UnitInfo): boolean {
  return offDrop(u) > SMALL;
}

/**
 * Units with a clear theme first (the clearer, the earlier); among equally
 * clear ones, those the model needs most (switched off, the right letter
 * loses most); then the rest the same way.
 */
export function clearestFirst(units: readonly UnitInfo[]): UnitInfo[] {
  return units.slice().sort((a, b) => (b.theme?.count ?? 0) - (a.theme?.count ?? 0) || offDrop(b) - offDrop(a) || b.on - a.on);
}
