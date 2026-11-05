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

/** One moment: the answer with the letter being written highlighted (⏎ = the end). */
export function momentChip(m: UnitMoment, langName: (code: string) => string): HTMLElement {
  return h('span', {
    class: 'umoment',
    title: t('unit.momentTip', { source: m.source, lang: langName(m.lang), form: t(`tok.form.${m.form}`), k: m.at + 1, v: m.value.toFixed(2) }),
  },
  h('span', { class: 'done' }, m.answer.slice(0, m.at)),
  h('b', {}, m.next === '' ? '⏎' : m.next),
  h('span', { class: 'rest' }, m.answer.slice(m.at + 1)));
}

/**
 * The card. With `now` (its value for the letter looked at in "picks one
 * letter"), a bar shows how strongly it is on now, compared with its
 * strongest moment in the learning data.
 */
export function unitCard(u: UnitInfo, langName: (code: string) => string,
  opts: { now?: number; nowOff?: { letter: string; withP: number; withoutP: number }; expert?: boolean } = {}): HTMLElement {
  const head = h('div', { class: 'uhead' },
    h('b', {}, t('unit.title', { n: u.unit + 1 })),
    h('span', { class: 'small' }, t('unit.onShare', { pct: Math.round(u.on * 100) })));
  if (opts.now !== undefined) {
    const max = Math.max(1e-6, u.top[0]?.value ?? opts.now);
    head.append(h('span', { class: 'unow small', title: t('unit.nowTip') },
      t('unit.now'), ' ',
      h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.max(3, Math.min(100, Math.round((opts.now / max) * 100)))}%` }))));
  }
  const theme = u.theme
    ? h('p', { class: 'utheme' }, t('unit.themeLine', { theme: themeText(u.theme, langName), count: u.theme.count, of: u.theme.of }))
    : h('p', { class: 'utheme none' }, t('unit.noTheme'));
  // What it really does: switched off at these moments (measured), and for the letter shown now.
  const effect: HTMLElement[] = [];
  if (opts.nowOff) {
    const n = opts.nowOff;
    effect.push(h('p', { class: 'uoff now', 'data-test': 'unit-off-now' },
      offText(n.withP, n.withoutP, { fall: 'unit.offNow.fall', rise: 'unit.offNow.rise', none: 'unit.offNow.none' }, { tok: shortLabel(n.letter) })));
  }
  if (u.off) {
    effect.push(h('p', { class: 'uoff small', 'data-test': 'unit-off' },
      offText(u.off.withP, u.off.withoutP, { fall: 'unit.off.fall', rise: 'unit.off.rise', none: 'unit.off.none' })));
    if (u.off.helps.length) {
      effect.push(h('div', { class: 'uhelps small', title: t('unit.helpsTip') }, t('unit.helps'), ' ',
        ...u.off.helps.map((x) => h('span', { class: 'uhelp' }, h('b', {}, shortLabel(x.char)), ` −${Math.max(1, Math.round(x.drop * 100))}`))));
    }
  }
  // Expert: the straight-line hint from its pattern alone.
  if (opts.expert || !u.off) {
    effect.push(h('div', { class: 'uvotes small', title: t('unit.votesTip') }, t('unit.votes'), ' ',
      ...u.pushes.map((p) => h('b', { class: 'uvote' }, shortLabel(p.char))),
      opts.expert ? h('span', { class: 'muted' }, ` (${u.pushes.map((p) => p.score.toFixed(2)).join(', ')})`) : null));
  }
  return h('div', { class: `ucard${u.theme ? '' : ' unclear'}`, 'data-unit': String(u.unit) },
    head,
    theme,
    h('div', { class: 'umoments' }, ...u.top.slice(0, SHOW).map((m) => momentChip(m, langName))),
    ...effect);
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
