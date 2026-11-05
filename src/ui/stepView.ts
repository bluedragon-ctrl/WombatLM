/**
 * "What this step changed" (Model learning tab): after Step ×1 while paused,
 * the whole learning step in numbers:
 *   measure → push back (gradient) → limit (clipping) → nudge (Adam) → check;
 * which numbers moved the most / were pushed the hardest, and for any one
 * number the exact calculation of its change, with each example's share of
 * its push.
 *
 * Recorded by the worker only for single steps (see train/stepDetail.ts):
 * one recorded step costs about as much as 20 ordinary ones.
 */
import type { MatrixInfo, NumberRef, ReadyMsg, StepDetailMsg, WhyMsg } from '../train/protocol';
import type { PracticeBatch } from '../train/trainer';
import { h } from './dom';
import { t, uiLang } from './i18n';
import { axisText, roleLabel } from './inside';

type List = 'change' | 'push';
/** Examples listed one by one in a push's shares (the rest are summed up). */
const SHARES = 8;

export class StepView {
  readonly el: HTMLElement;
  private ready?: ReadyMsg;
  private detail?: StepDetailMsg;
  private why?: WhyMsg;
  private practice?: PracticeBatch;
  private list: List = 'change';
  private selected: { name: string; index: number } | null = null;
  private body: HTMLElement;
  /** The model's step now, and whether it is learning. */
  private step = 0;
  private running = false;

  constructor(private requestWhy: (name: string, index: number) => void, private langName: (code: string) => string) {
    this.body = h('div', { class: 'sd-body', 'aria-live': 'polite' });
    this.el = h('section', { class: 'panel phase build step-detail', 'data-test': 'step-detail' },
      h('div', { class: 'phase-head' }, h('h2', {}, t('sd.title')),
        h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, t('sd.help')))),
      this.body);
    this.render();
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    this.detail = undefined;
    this.why = undefined;
    this.selected = null;
    this.render();
  }

  /** The model's current step: the recorded step is shown only while it is still the latest. */
  setStep(step: number, running: boolean, practice: PracticeBatch | undefined): void {
    const changed = step !== this.step || running !== this.running || practice !== this.practice;
    this.step = step;
    this.running = running;
    this.practice = practice;
    if (changed) this.render();
  }

  /** Is there a recorded step for the model as it is now? */
  current(): boolean {
    return !!this.detail && this.detail.step === this.step && !this.running;
  }

  onDetail(msg: StepDetailMsg): void {
    this.detail = msg;
    this.why = undefined;
    // Explain the number that moved the most right away, so there is always one example.
    const first = msg.top.change[0];
    this.selected = first ? { name: first.name, index: first.index } : null;
    if (first) this.requestWhy(first.name, first.index);
    this.render();
  }

  onWhy(msg: WhyMsg): void {
    if (!this.selected || !msg.ref || msg.ref.name !== this.selected.name || msg.ref.index !== this.selected.index) {
      if (!msg.missing) return;
    }
    this.why = msg;
    this.render();
  }

  /** Explain one number (from the lists here, or a click on a heatmap). */
  pick(name: string, index: number): void {
    this.selected = { name, index };
    this.why = undefined;
    this.requestWhy(name, index);
    this.render();
  }

  // -------------------------------------------------------------------------

  private info(name: string): MatrixInfo | undefined {
    return this.ready?.matrices.find((m) => m.name === name);
  }

  /** "letter table · letter 'ř' · number 5" */
  numberName(ref: { name: string; row: number; col: number }): string {
    const m = this.info(ref.name);
    if (!m || !this.ready) return `${ref.name}[${ref.row}, ${ref.col}]`;
    const tokens = this.ready.tokens;
    const parts = [roleLabel(m)];
    const r = m.rows > 1 ? axisText(m.rowKind, ref.row, tokens) : '';
    const c = axisText(m.colKind, ref.col, tokens);
    const layer = /^(dim|hidden)$/.test(m.rowKind) && /^(dim|hidden)$/.test(m.colKind);
    if (layer && r && c) parts.push(`${r} → ${c}`);
    else parts.push(...[r, c].filter(Boolean));
    const part = partName(ref.name);
    return (part ? `${part}: ` : '') + parts.join(' · ');
  }

  private render(): void {
    const d = this.detail;
    if (!this.ready) { this.body.replaceChildren(); return; }
    if (!d || !this.current()) {
      this.body.replaceChildren(h('p', { class: 'sd-empty' }, t(this.running ? 'sd.running' : 'sd.empty')));
      return;
    }
    const p = this.practice?.step === d.step ? this.practice : undefined;
    const n = (v: number) => v.toLocaleString(uiLang());
    const flow = h('ol', { class: 'sd-flow' },
      stage(1, t('sd.f.measure'), p ? t('sd.f.measureVal', { loss: p.loss.toFixed(2) }) : ''),
      stage(2, t('sd.f.push'), t('sd.f.pushVal', { len: num(d.gradNorm) })),
      stage(3, t('sd.f.limit'), d.clip < 1 ? t('sd.f.limitOn', { limit: d.clipNorm, f: d.clip.toFixed(2) }) : t('sd.f.limitOff', { limit: d.clipNorm })),
      stage(4, t('sd.f.nudge'), d.t < d.warmupSteps
        ? t('sd.f.nudgeWarm', { lr: num(d.lr), t: d.t, w: d.warmupSteps, full: num(d.baseLr) })
        : t('sd.f.nudgeVal', { lr: num(d.lr) })),
      stage(5, t('sd.f.check'), p?.check ? t('sd.f.checkVal', { before: p.check.lossBefore.toFixed(3), after: p.check.lossAfter.toFixed(3) }) : ''));
    const check = p?.check ? h('p', { class: 'sd-check', 'data-test': 'sd-check' },
      t(p.check.lossAfter < p.check.lossBefore ? 'sd.checkBetter' : 'sd.checkWorse', {
        before: p.check.lossBefore.toFixed(3), after: p.check.lossAfter.toFixed(3),
        up: p.check.lettersUp, down: p.check.lettersDown, b: p.batchSize,
      }), ' ', h('span', { class: 'small' }, t('sd.checkNote'))) : '';
    const moved = h('p', {}, t('sd.moved', { moved: n(d.numbers.moved), total: n(d.numbers.total), max: num(d.numbers.max), median: num(d.numbers.median) }));
    const tabs = h('div', { class: 'picker', role: 'group' }, ...(['change', 'push'] as List[]).map((k) => {
      const b = h('button', { class: k === this.list ? 'on' : '', 'data-list': k, title: t(`sd.list.${k}Tip`) }, t(`sd.list.${k}`));
      b.onclick = () => { this.list = k; this.render(); };
      return b;
    }));
    const refs = d.top[this.list];
    const table = h('table', { class: 'sd-numbers', 'data-test': 'sd-numbers' },
      h('thead', {}, h('tr', {},
        h('th', {}, t('sd.col.number')), h('th', { class: 'num' }, t('sd.col.before')), h('th', { class: 'num' }, t('sd.col.after')),
        h('th', { class: 'num' }, t('sd.col.change')), h('th', { class: 'num' }, t('sd.col.push')))),
      h('tbody', {}, ...refs.map((r) => this.numberRow(r))));
    this.body.replaceChildren(flow, check, moved,
      h('div', { class: 'row sd-list-head' }, h('h3', {}, t('sd.numbersTitle')), tabs),
      h('p', { class: 'small' }, t(`sd.list.${this.list}Help`)),
      table,
      h('p', { class: 'small' }, t('sd.pickHint')),
      this.whyPanel());
  }

  private numberRow(r: NumberRef): HTMLElement {
    const on = this.selected?.name === r.name && this.selected.index === r.index;
    const tr = h('tr', { class: on ? 'on' : '', tabindex: 0, 'data-number': `${r.name}:${r.index}` },
      h('td', {}, this.numberName(r)),
      h('td', { class: 'num' }, num(r.before)),
      h('td', { class: 'num' }, num(r.after)),
      h('td', { class: `num ${r.after > r.before ? 'up' : 'down'}` }, signed(r.after - r.before)),
      h('td', { class: 'num' }, signed(r.grad)));
    tr.onclick = () => this.pick(r.name, r.index);
    tr.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); this.pick(r.name, r.index); } };
    return tr;
  }

  /** The whole calculation of the selected number's change. */
  private whyPanel(): HTMLElement {
    const box = h('div', { class: 'sd-why', 'data-test': 'sd-why' });
    if (!this.selected) return box;
    const w = this.why;
    if (!w) { box.append(h('p', { class: 'small' }, t('sd.why.wait'))); return box; }
    if (w.missing || !w.ref || !w.calc) { box.append(h('p', { class: 'small' }, t('sd.why.missing'))); return box; }
    const c = w.calc, r = w.ref;
    const way = c.grad > 0 ? t('sd.why.down') : c.grad < 0 ? t('sd.why.up') : t('sd.why.none');
    const steps: HTMLElement[] = [];
    // 1. The push and where it comes from.
    steps.push(whyStep(1, t('sd.why.pushTitle'),
      h('p', {}, t('sd.why.push', { g: signed(c.grad), small: signed(c.grad * 0.01), way })),
      w.shares ? this.shares(w) : h('p', { class: 'small' }, t('sd.why.noSplit'))));
    // 2. Clipping.
    steps.push(whyStep(2, t('sd.why.limitTitle'), h('p', {}, c.clip < 1
      ? t('sd.why.limitOn', { g: signed(c.grad), f: c.clip.toFixed(3), gu: signed(c.gradUsed), len: num(this.detail!.gradNorm), limit: this.detail!.clipNorm })
      : t('sd.why.limitOff', { g: signed(c.grad), len: num(this.detail!.gradNorm), limit: this.detail!.clipNorm }))));
    // 3. Adam's memory.
    steps.push(whyStep(3, t('sd.why.memTitle'),
      h('p', {}, t('sd.why.mem')),
      h('p', { class: 'sd-eq' }, `m = ${c.beta1} × ${signed(c.mPrev)} + ${round(1 - c.beta1)} × ${signed(c.gradUsed)} = ${signed(c.m)}`),
      h('p', { class: 'sd-eq' }, `v = ${c.beta2} × ${num(c.vPrev)} + ${round(1 - c.beta2)} × (${signed(c.gradUsed)})² = ${num(c.v)}`),
      h('p', {}, t('sd.why.start', { t: c.t, bc1: c.bc1.toFixed(3), bc2: c.bc2.toFixed(3) })),
      h('p', { class: 'sd-eq' }, `m̂ = ${signed(c.m)} / ${c.bc1.toFixed(3)} = ${signed(c.mHat)}     √v̂ = √(${num(c.v)} / ${c.bc2.toFixed(3)}) = ${num(Math.sqrt(c.vHat))}`)));
    // 4. The nudge.
    steps.push(whyStep(4, t('sd.why.nudgeTitle'),
      h('p', { class: 'sd-eq' }, `${t('sd.why.move')} = ${num(c.lr)} × ${signed(c.mHat)} / ${num(Math.sqrt(c.vHat))} = ${signed(c.move)}`),
      h('p', { class: 'small' }, c.lr < c.baseLr ? t('sd.why.lrWarm', { lr: num(c.lr), full: num(c.baseLr), t: c.t, w: c.warmupSteps }) : t('sd.why.lr', { lr: num(c.lr) })),
      h('p', { class: 'small' }, t('sd.why.ratio', { r: (c.mHat / (Math.sqrt(c.vHat) + c.eps)).toFixed(2) }))));
    // 5. The result.
    const decay = c.decay !== 1 ? ` × ${c.decay.toFixed(6)}` : '';
    const ok = Math.abs((w.now ?? c.after) - c.after) < 1e-5;
    steps.push(whyStep(5, t('sd.why.resultTitle'),
      h('p', { class: 'sd-eq' }, `${num(r.before)}${decay} − (${signed(c.move)}) = ${num(c.after)}`),
      h('p', { class: 'small' }, ok ? t('sd.why.matches', { now: num(w.now ?? c.after) }) : t('sd.why.differs', { now: num(w.now ?? c.after) }))));
    box.append(
      h('h3', {}, t('sd.why.title', { name: this.numberName(r) })),
      h('p', { class: 'small' }, t('sd.why.summary', { before: num(r.before), after: num(r.after), d: signed(r.after - r.before) })),
      h('ol', { class: 'sd-why-steps' }, ...steps));
    return box;
  }

  /** Each example's share of the push: biggest first, the rest summed up; they add up to the push. */
  private shares(w: WhyMsg): HTMLElement {
    const items = w.shares!.map((s) => ({
      label: `${s.source} → ${s.target}`,
      task: s.form === 'name' ? t('practice.copy') : `${t(`form.${s.form}`)} · ${this.langName(s.lang)}`,
      v: s.share,
    }));
    const sorted = items.slice().sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const shown = sorted.slice(0, SHARES);
    const rest = sorted.slice(SHARES);
    const lines = shown.map((x) => ({ label: x.label, task: x.task, v: x.v }));
    if (rest.length) lines.push({ label: t('sd.why.rest', { n: rest.length }), task: '', v: rest.reduce((a, b) => a + b.v, 0) });
    const max = Math.max(1e-12, ...lines.map((l) => Math.abs(l.v)));
    const total = items.reduce((a, b) => a + b.v, 0);
    return h('div', {},
      h('p', { class: 'small' }, t('sd.why.sharesHelp')),
      h('table', { class: 'xcontrib sd-shares', 'data-test': 'sd-shares' }, h('tbody', {},
        ...lines.map((l) => h('tr', {},
          h('td', {}, l.label, l.task ? h('span', { class: 'task' }, ` ${l.task}`) : ''),
          h('td', { class: 'bar' }, h('span', { class: `sbar ${l.v < 0 ? 'neg' : 'pos'}`, style: `width:${Math.round((Math.abs(l.v) / max) * 100)}%` })),
          h('td', { class: 'val' }, signed(l.v)))),
        h('tr', { class: 'total' }, h('td', {}, t('sd.why.total')), h('td', {}), h('td', { class: 'val' }, signed(total))))));
  }
}

/** The part of the model a parameter belongs to, in plain words ("" for letters and positions). */
function partName(name: string): string {
  const m = name.match(/^(enc|dec)\d+\.(self|cross|ffn|ln1|ln2|lnX)/);
  if (m) {
    const where = t(m[1] === 'enc' ? 'inside.group.encoder' : 'inside.group.decoder');
    const what = m[2].startsWith('ln') ? t('inside.part.norm') : t(`flow.${m[1]}.${m[2]}`);
    return `${where}, ${what}`;
  }
  if (/^(enc|dec)Norm/.test(name)) return t(name.startsWith('enc') ? 'inside.group.encoder' : 'inside.group.decoder');
  return '';
}

function stage(i: number, what: string, val: string): HTMLElement {
  return h('li', {}, h('span', { class: 'num' }, String(i)), h('span', { class: 'what' }, what), h('span', { class: 'val' }, val));
}

function whyStep(i: number, title: string, ...body: (HTMLElement | string)[]): HTMLElement {
  return h('li', {}, h('span', { class: 'num' }, String(i)), h('div', {}, h('h4', {}, title), ...body));
}

/** A small number readable without exponents where possible. */
export function num(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 0.1) return v.toFixed(4);
  if (a >= 0.0001) return v.toFixed(6);
  return v.toExponential(2);
}

function signed(v: number): string {
  return v > 0 ? `+${num(v)}` : num(v);
}

function round(v: number): string {
  return String(Math.round(v * 1000) / 1000);
}
