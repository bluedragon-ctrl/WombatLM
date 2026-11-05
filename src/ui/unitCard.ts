/**
 * A card for one thinking unit ("detector"): what switches it on most in the
 * learning data (with a plain-language theme when there is a clear one) and
 * which letters it votes for. Used by "What have the detectors learned?" and
 * by the Think stage of "How the model picks one letter".
 */
import type { UnitInfo, UnitMoment, UnitTheme } from '../train/protocol';
import { h } from './dom';
import { t } from './i18n';
import { shortLabel } from './tokens';

/** How many of its strongest moments a card shows. */
const SHOW = 5;

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
export function unitCard(u: UnitInfo, langName: (code: string) => string, opts: { now?: number; expert?: boolean } = {}): HTMLElement {
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
  return h('div', { class: `ucard${u.theme ? '' : ' unclear'}`, 'data-unit': String(u.unit) },
    head,
    theme,
    h('div', { class: 'umoments' }, ...u.top.slice(0, SHOW).map((m) => momentChip(m, langName))),
    h('div', { class: 'uvotes small', title: t('unit.votesTip') }, t('unit.votes'), ' ',
      ...u.pushes.map((p) => h('b', { class: 'uvote' }, shortLabel(p.char))),
      opts.expert ? h('span', { class: 'muted' }, ` (${u.pushes.map((p) => p.score.toFixed(2)).join(', ')})`) : null));
}

/** Units with a clear theme first (the clearer, the earlier), then the rest by how often they are on. */
export function clearestFirst(units: readonly UnitInfo[]): UnitInfo[] {
  return units.slice().sort((a, b) => (b.theme?.count ?? 0) - (a.theme?.count ?? 0) || b.on - a.on);
}
