/**
 * "How the model picks one letter" ("How it picks a letter" tab), for one letter of
 * the answer, stage by stage.
 *
 * Plain mode tells it without rows of numbers: after each stage, which letter
 * the model would write if it stopped there and how sure it is ("what would it
 * write now", the logit lens), where it looks (both heads together) and how
 * many detectors switch on.
 *
 * Expert mode adds every number the model computes on the way, with what each
 * stage adds highlighted. Clicking a square shows where its value comes from
 * (the exact sum), and the weights it used are outlined in the heatmaps.
 *
 * Computed by the worker only on request (`explain`), for one letter at a time.
 */
import { EOS } from '../nn/vocab';
import type { AnswerMsg, ExplainLens, ExplainMsg, ExplainStage, UnitsMsg } from '../train/protocol';
import { unitCard } from './unitCard';
import { pctText } from './format';
import { sequential, symmetricScale } from '../viz/colors';
import { drawStrip, stripCellAt, stripLayout, topIndices } from '../viz/vectorStrip';
import type { HeatmapMarks } from '../viz/weightHeatmap';
import { h, showTip } from './dom';
import { t } from './i18n';
import { shortLabel, tokenName } from './tokens';

export interface Question {
  text: string;
  lang: string;
  form: string;
}

/** Which square was clicked: strip `strip` of stage row `row`, square `index`. */
interface Pick {
  row: string;
  strip: string;
  index: number;
}

/** Thinking units shown in the Think stage (the most strongly on). */
const STRONG_UNITS = 3;
/** A unit is "needed" for a letter if switching it off lowers the letter's chance by at least this. */
const NEEDED = 0.02;
/** How many squares with the biggest change get an outline. */
const TOP = 3;
/** Contributions listed one by one (the rest are summed up). */
const LIST = 8;

export class ExplainView {
  readonly el: HTMLElement;
  /** Selected guess-bar column (null = none yet). */
  position: number | null = null;
  private answer?: AnswerMsg;
  private msg?: ExplainMsg;
  private expert = false;
  private numbers = false;
  private pick: Pick | null = null;
  private lastKey = '';
  private picker: HTMLElement;
  private body: HTMLElement;
  private numbersRow: HTMLElement;
  private helpShort: HTMLElement;
  private helpMore: HTMLElement;
  /** The question of the latest answer, and whether the user picked its letter (otherwise the most interesting one is shown). */
  private question = '';
  private picked = false;
  /** What each thinking unit has learned (for the Think stage). */
  private units?: UnitsMsg;

  constructor(
    private request: (q: Question, position: number) => void,
    private onMarks: (marks: Map<string, HeatmapMarks>) => void,
    private langName: (code: string) => string,
  ) {
    this.picker = h('div', { class: 'xpick', 'data-test': 'explain-pick', role: 'group', 'aria-label': t('explain.pick') });
    const numbers = h('input', { type: 'checkbox', 'data-test': 'explain-numbers' });
    numbers.onchange = () => { this.numbers = numbers.checked; this.render(); };
    this.body = h('div', { class: 'xbody', 'aria-live': 'polite' });
    this.numbersRow = h('div', { class: 'row small', hidden: true }, h('label', {}, numbers, ' ', t('explain.numbers')));
    this.helpShort = h('span', {});
    this.helpMore = h('p', {});
    this.el = h('section', { class: 'panel explain', 'data-test': 'explain' },
      h('h2', {}, t('explain.title')),
      h('p', { class: 'small xhelp' }, this.helpShort, ' ', h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), this.helpMore)),
      this.picker,
      this.numbersRow,
      this.body);
    this.renderHelp();
    this.render();
  }

  setExpert(on: boolean): void {
    this.expert = on;
    this.numbersRow.hidden = !on;
    this.renderHelp();
    this.renderPicker();
    this.render();
  }

  private renderHelp(): void {
    this.helpShort.textContent = t('explain.helpShort');
    this.helpMore.textContent = this.expert ? t('explain.help') : t('explain.helpMore');
  }

  /** The answer shown on the Ask tab: its letters are the ones to pick from. */
  setAnswer(a: AnswerMsg | undefined): void {
    this.answer = a;
    if (a && a.steps.length) {
      const q = `${a.text}|${a.lang}|${a.form}`;
      if (q !== this.question) { this.question = q; this.picked = false; }
      // Until the user picks a letter: the one where the answer stops copying the animal
      // (it can move while the model learns).
      if (!this.picked) this.position = interestingLetter(a);
      this.position = Math.min(this.position ?? 0, a.steps.length - 1);
    }
    this.renderPicker();
    if (!a) this.render();
  }

  setUnits(u: UnitsMsg | undefined): void {
    this.units = u;
    if (this.msg) this.render();
  }

  /** Choose a letter (guess-bar column) and look inside it. */
  select(position: number): void {
    if (this.answer) this.question = `${this.answer.text}|${this.answer.lang}|${this.answer.form}`;
    this.picked = true;
    this.position = position;
    if (this.answer && this.answer.steps.length) this.position = Math.min(position, this.answer.steps.length - 1);
    this.renderPicker();
    this.refresh();
  }

  /** Ask the worker again if the question, the letter or the model changed (call while visible). */
  refresh(): void {
    const a = this.answer;
    if (!a || this.position === null || !a.steps.length) return;
    const key = `${a.step}|${a.text}|${a.lang}|${a.form}|${this.position}`;
    if (key === this.lastKey) return;
    this.lastKey = key;
    this.request({ text: a.text, lang: a.lang, form: a.form }, this.position);
  }

  onExplain(msg: ExplainMsg): void {
    const same = this.msg && this.msg.position === msg.position && this.msg.text === msg.text && this.msg.form === msg.form && this.msg.lang === msg.lang;
    this.msg = msg;
    if (!same) this.pick = null;
    this.render();
  }

  // -------------------------------------------------------------------------

  private renderPicker(): void {
    const a = this.answer;
    if (!a || !a.steps.length) { this.picker.replaceChildren(); return; }
    const buttons = a.steps.map((s, i) => {
      const b = h('button', { class: `xletter${i === this.position ? ' on' : ''}`, 'data-pos': String(i), title: this.full(s.chosen) }, shortLabel(s.chosen));
      b.onclick = () => this.select(i);
      return b;
    });
    this.picker.replaceChildren(h('span', { class: 'from' }, `${a.text} →`), ...buttons);
  }

  private full(token: string): string {
    return tokenName(token || EOS, { expert: this.expert, langName: this.langName });
  }

  private render(): void {
    const m = this.msg;
    if (!this.answer) {
      this.body.replaceChildren(h('p', { class: 'small' }, t('explain.noAnswer')));
      this.onMarks(new Map());
      return;
    }
    if (!m) { this.body.replaceChildren(h('p', { class: 'small' }, t('explain.wait'))); return; }
    const ex = this.expert;
    const p = m.position;
    const written = m.decIn.slice(1).join('');
    // One colour scale for all the letter's numbers, so sizes can be compared from step to step.
    const all = new Float32Array([...m.start.x, ...m.start.token, ...m.start.pos, ...m.stages.flatMap((s) => [...s.add, ...s.x]), ...m.final]);
    const scale = symmetricScale(all);
    const rows: HTMLElement[] = [];
    let num = 1;
    const tokIn = m.decIn[p];
    const lens = m.lens ?? [];
    const startStrips = ex ? [
      this.strip('start', 'token', t('explain.strip.token', { tok: shortLabel(tokIn) }), m.start.token, scale),
      this.strip('start', 'pos', t('explain.strip.pos', { n: p + 1 }), m.start.pos, scale),
      this.strip('start', 'x', t('explain.strip.in'), m.start.x, scale),
    ] : [];
    if (lens[0]) startStrips.push(this.lens(lens[0], m.chosen));
    rows.push(this.stageRow('start', num++, t(p === 0 ? 'explain.start.titleTask' : 'explain.start.title', { tok: this.full(tokIn) }),
      'start', t('explain.start.help', { n: p + 1, d: m.start.x.length }), startStrips, null, lens[0] ? { from: null, to: lens[0].chosen } : null));
    let before = m.start.x;
    const layers = new Set(m.stages.map((s) => s.name.split('.')[0])).size;
    m.stages.forEach((s, i) => {
      const kind = s.name.split('.')[1]; // self, cross, ffn
      const layer = layers > 1 ? ` ${t('explain.layer', { n: Number(s.name.match(/\d+/)?.[0] ?? 0) + 1 })}` : '';
      const marks = new Set(topIndices(s.add, TOP));
      const strips: HTMLElement[] = [];
      if (s.kind === 'attend') strips.push(ex ? this.attention(s, kind === 'self' ? m.decIn : m.input) : this.attentionPlain(s, kind === 'self' ? m.decIn : m.input));
      else if (ex) strips.push(this.strip(s.name, 'hidden', t('explain.strip.units', { on: unitsOn(s), n: s.sources }), s.hidden, symmetricScale(s.hidden)));
      else strips.push(h('p', { class: 'small xunits' }, t('explain.unitsOn', { on: unitsOn(s), n: s.sources })));
      if (s.kind === 'think') strips.push(this.strongUnits(s));
      if (ex) {
        strips.push(
          this.strip(s.name, 'add', t('explain.strip.add'), s.add, scale, marks),
          this.strip(s.name, 'x', t('explain.strip.now'), s.x, scale, marks));
      }
      const l = lens[i + 1];
      if (l) strips.push(this.lens(l, m.chosen));
      rows.push(this.stageRow(s.name, num++, t(`explain.${kind}.title`) + layer, kind, t(`explain.${kind}.help`), strips,
        ex ? norm(s.add) / Math.max(1e-9, norm(before)) : null, l ? { from: lens[i].chosen, to: l.chosen } : null));
      before = s.x;
    });
    rows.push(this.stageRow('guess', num, t('explain.guess.title'), 'guess', t('explain.guess.help'), [
      ...(ex ? [this.strip('guess', 'final', t('explain.strip.final'), m.final, symmetricScale(m.final))] : []),
      this.guesses(m),
    ], null, null));
    const caption = p === 0 ? t('explain.caption0', { n: m.columns, chosen: this.full(m.chosen) })
      : t('explain.caption', { k: p + 1, n: m.columns, written, chosen: this.full(m.chosen) });
    this.body.replaceChildren(
      h('p', { class: 'xcaption', 'data-test': 'explain-caption' }, caption, ' ', h('span', { class: 'small' }, t('explain.step', { step: m.step }))),
      ...(lens.length ? [this.ladder(m)] : []),
      ...(ex ? [h('p', { class: 'small' }, t('explain.marks'))] : []),
      h('ol', { class: `xstages${ex ? '' : ' plain'}` }, ...rows));
    this.onMarks(this.marks());
  }

  /**
   * "How sure it is about the letter, stage by stage": the chance of the
   * written letter after each stage, and which stage raised it the most.
   */
  private ladder(m: ExplainMsg): HTMLElement {
    const tok = this.full(m.chosen);
    const names = ['start', ...m.stages.map((s) => s.name.split('.')[1])];
    const chips: HTMLElement[] = [];
    m.lens.forEach((l, i) => {
      if (i) chips.push(h('span', { class: 'xladder-arrow', 'aria-hidden': 'true' }, '→'));
      chips.push(h('span', { class: 'xladder-chip', title: t(`explain.name.${names[i]}`) },
        h('span', { class: 'num' }, String(i + 1)),
        h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.max(2, Math.round(l.chosen * 100))}%` })),
        h('b', {}, pctText(l.chosen))));
    });
    let best = -1, gain = 0.05;
    for (let i = 1; i < m.lens.length; i++) {
      const g = m.lens[i].chosen - m.lens[i - 1].chosen;
      if (g > gain) { gain = g; best = i; }
    }
    const verdict = best > 0 ? t('explain.jump', { tok, n: best + 1, stage: t(`explain.name.${names[best]}`) })
      : m.lens[0].chosen > 0.5 ? t('explain.jumpNone', { tok }) : '';
    return h('div', { class: 'xladder', 'data-test': 'explain-ladder' },
      h('span', { class: 'small' }, t('explain.ladder', { tok })),
      h('div', { class: 'xladder-chips' }, ...chips),
      verdict ? h('p', { class: 'small xverdict' }, verdict) : null);
  }

  /**
   * The thinking units this letter needs most: switched off one at a time,
   * the written letter loses the most chance (measured). If none matters much
   * on its own, the ones most strongly on, with a sentence that they work as a team.
   */
  private strongUnits(s: ExplainStage): HTMLElement {
    const u = this.units;
    if (!u || u.layer !== s.name.split('.')[0]) return h('p', { class: 'small xunits' }, t('explain.ffn.unitsWait'));
    const m = this.msg!;
    const withP = m.lens.at(-1)?.chosen ?? 0;
    const on = Array.from(s.hidden, (v, k) => ({ v, k, drop: m.unitsOff ? withP - m.unitsOff[k] : 0 })).filter((x) => x.v > 0.05);
    const needed = on.filter((x) => x.drop >= NEEDED).sort((a, b) => b.drop - a.drop);
    const team = needed.length === 0;
    const strongest = (team ? on.sort((a, b) => b.v - a.v) : needed).slice(0, STRONG_UNITS);
    const tok = shortLabel(m.chosen);
    return h('div', { class: 'xunitcards', 'data-test': 'explain-units' },
      h('p', { class: 'small' }, team ? t('explain.ffn.unitsTeam', { tok }) : t('explain.ffn.unitsNeeded', { tok })),
      h('div', { class: 'ucards' }, ...strongest.map((x) => unitCard(u.units[x.k], this.langName, {
        now: x.v, expert: this.expert,
        nowOff: this.msg?.unitsOff ? { letter: this.msg.chosen, withP: this.msg.lens.at(-1)?.chosen ?? 0, withoutP: this.msg.unitsOff[x.k] } : undefined,
      }))));
  }

  /** "Would write now": the 3 most likely letters at this point, the finally written one outlined. */
  private lens(l: ExplainLens, chosen: string): HTMLElement {
    return h('div', { class: 'xlens', 'data-test': 'explain-lens' },
      h('span', { class: 'xlabel small' }, t('explain.wouldWrite')),
      ...l.top.map((g) => h('span', { class: `xlens-g${g.char === chosen ? ' chosen' : ''}`, title: this.full(g.char) },
        h('b', {}, shortLabel(g.char)),
        h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.max(2, Math.round(g.p * 100))}%` })),
        h('span', { class: 'small' }, pctText(g.p)))));
  }

  /** Attention in plain mode: both heads together (their average), and where it looks most. */
  private attentionPlain(s: ExplainStage, keys: string[]): HTMLElement {
    const w = Array.from({ length: s.sources }, (_, i) => {
      let a = 0;
      for (let hd = 0; hd < s.heads; hd++) a += s.weights[hd * s.sources + i];
      return a / s.heads;
    });
    const top = w.indexOf(Math.max(...w));
    return h('div', { class: 'xatt-wrap' },
      h('div', { class: 'xatt-head' },
        h('span', { class: 'xlabel small' }, t('explain.looksAt')),
        ...keys.slice(0, s.sources).map((k, i) => h('span', { class: 'xatt', style: `background:${sequential(w[i])};color:${w[i] > 0.6 ? '#fff' : 'inherit'}`, title: `${this.full(k)}: ${pctText(w[i])}` },
          h('b', {}, shortLabel(k)), h('small', {}, `${Math.round(w[i] * 100)}`)))),
      h('p', { class: 'small xmostly' }, t('explain.mostly', { tok: this.full(keys[top]), pct: pctText(w[top]) })));
  }

  private stageRow(id: string, num: number, title: string, kind: string, help: string, strips: HTMLElement[], change: number | null,
    effect: { from: number | null; to: number } | null): HTMLElement {
    const side = h('div', { class: 'xchange' });
    if (change !== null) {
      side.title = t('explain.changeHelp');
      side.append(
        h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.min(100, Math.round(change * 100))}%` })),
        h('span', { class: 'small' }, t('explain.change', { pct: Math.round(change * 100) })));
    }
    if (effect && this.msg) {
      const up = effect.from === null ? '' : effect.to > effect.from + 0.005 ? ' up' : effect.to < effect.from - 0.005 ? ' down' : '';
      const arrow = up === ' up' ? '▲' : up === ' down' ? '▼' : '';
      side.append(h('span', { class: `xeffect small${up}`, 'data-test': 'explain-effect' },
        t('explain.effect', { tok: shortLabel(this.msg.chosen) }), ': ',
        effect.from === null ? '' : `${pctText(effect.from)} → `,
        h('b', {}, arrow ? `${pctText(effect.to)} ${arrow}` : pctText(effect.to))));
    }
    // Plain mode: one short sentence, the full explanation behind "?". Expert: the full explanation.
    const text = this.expert ? h('p', { class: 'small' }, help)
      : h('p', { class: 'small' }, t(`explain.${kind}.short`), ' ', h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, help)));
    const li = h('li', { class: 'xstage', 'data-stage': id },
      h('span', { class: 'num' }, String(num)),
      h('div', { class: 'xtext' }, h('h3', {}, title), text),
      h('div', { class: 'xstrips' }, ...strips),
      side);
    if (this.pick?.row === id) li.append(h('div', { class: 'xdetail', 'data-test': 'explain-detail' }, ...this.detail(this.pick)));
    return li;
  }

  /** A labelled strip of squares; clicking a square explains it. */
  private strip(row: string, name: string, label: string, values: Float32Array, scale: number, marks?: ReadonlySet<number>): HTMLElement {
    const layout = stripLayout(values.length, this.numbers);
    const canvas = h('canvas', { class: 'xstrip', 'data-strip': `${row}:${name}` });
    const selected = this.pick?.row === row && this.pick.strip === name ? this.pick.index : null;
    const draw = (hover: number | null = null) => drawStrip(canvas, values, layout, scale, { marks, selected, hover });
    draw();
    canvas.addEventListener('mousemove', (e) => {
      const i = stripCellAt(layout, e.offsetX, e.offsetY);
      showTip(e, i === null ? null : `${name === 'hidden' ? t('axis.hidden', { n: i + 1 }) : t('axis.dim', { n: i + 1 })}: ${fmt(values[i])}\n${t('explain.click')}`);
      draw(i);
    });
    canvas.addEventListener('mouseleave', (e) => { showTip(e, null); draw(); });
    canvas.addEventListener('click', (e) => {
      const i = stripCellAt(layout, e.offsetX, e.offsetY);
      if (i === null) return;
      showTip(e, null);
      this.pick = selected === i ? null : { row, strip: name, index: i };
      this.render();
    });
    return h('div', { class: 'xstrip-row' }, h('span', { class: 'xlabel small' }, label), canvas);
  }

  /** Attention: per head, how much it looked at each position. */
  private attention(s: ExplainStage, keys: string[]): HTMLElement {
    const heads = Array.from({ length: s.heads }, (_, hd) => h('div', { class: 'xatt-head' },
      h('span', { class: 'xlabel small' }, t('explain.head', { n: hd + 1 })),
      ...keys.slice(0, s.sources).map((k, i) => {
        const w = s.weights[hd * s.sources + i];
        return h('span', { class: 'xatt', style: `background:${sequential(w)};color:${w > 0.6 ? '#fff' : 'inherit'}`, title: `${this.full(k)}: ${Math.round(w * 100)} %` },
          h('b', {}, shortLabel(k)), h('small', {}, `${Math.round(w * 100)}`));
      })));
    return h('div', { class: 'xatt-wrap' }, ...heads);
  }

  /** The top guesses; clicking one shows how its score is made. */
  private guesses(m: ExplainMsg): HTMLElement {
    const list = m.guesses.map((g, i) => {
      const on = this.pick?.row === 'guess' && this.pick.strip === 'guess' && this.pick.index === i;
      const b = h('button', { class: `xguess${g.char === m.chosen ? ' chosen' : ''}${on ? ' on' : ''}`, title: this.full(g.char) },
        h('b', {}, shortLabel(g.char)),
        h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.max(1, Math.round(g.p * 100))}%` })),
        h('span', { class: 'small' }, pctText(g.p)),
        this.expert ? h('span', { class: 'small score' }, t('explain.score', { s: g.score.toFixed(2) })) : null);
      b.onclick = () => { this.pick = on ? null : { row: 'guess', strip: 'guess', index: i }; this.render(); };
      return b;
    });
    return h('div', { class: 'xguesses', 'data-test': 'explain-guesses' }, ...list);
  }

  // -------------------------------------------------------------------------
  // Where does a number come from?

  private detail(pk: Pick): (HTMLElement | string)[] {
    const m = this.msg!;
    const j = pk.index;
    if (pk.row === 'start') {
      return [h('p', {}, t('explain.d.start', { n: j + 1, tok: fmt(m.start.token[j]), pos: fmt(m.start.pos[j]), sum: fmt(m.start.x[j]) }))];
    }
    if (pk.row === 'guess') {
      if (pk.strip === 'final') return [h('p', {}, t('explain.d.final', { n: j + 1, v: fmt(m.final[j]) }))];
      const g = m.guesses[j];
      const items = Array.from(g.contrib, (v, i) => ({ label: t('explain.d.product', { n: i + 1, a: fmt(m.final[i]), b: fmt(g.row[i]) }), v }));
      return [
        h('p', {}, t('explain.d.score', { tok: shortLabel(g.char), s: fmt(g.score), p: Math.round(g.p * 100) })),
        this.contribList(items, { label: t('explain.d.favourite'), v: g.bias }, g.score),
      ];
    }
    const idx = m.stages.findIndex((s) => s.name === pk.row);
    const s = m.stages[idx];
    const prev = idx === 0 ? m.start.x : m.stages[idx - 1].x;
    if (pk.strip === 'x') return [h('p', {}, t('explain.d.now', { n: j + 1, before: fmt(prev[j]), add: fmt(s.add[j]), now: fmt(s.x[j]) }))];
    if (pk.strip === 'hidden') {
      return [h('p', {}, t(s.hidden[j] > 0.05 ? 'explain.d.unitOn' : 'explain.d.unitOff', { n: j + 1, v: fmt(s.hidden[j]) }))];
    }
    // What this step adds to number j: split by source.
    const d = s.add.length;
    const keys = s.name.endsWith('.self') ? m.decIn : m.input;
    const items = Array.from({ length: s.sources }, (_, k) => ({
      label: s.kind === 'attend' ? t('explain.d.from', { tok: shortLabel(keys[k]), n: k + 1 }) : t('explain.d.unit', { n: k + 1, v: fmt(s.hidden[k]) }),
      v: s.contrib[k * d + j],
    }));
    return [
      h('p', {}, t(s.kind === 'attend' ? 'explain.d.addAttend' : 'explain.d.addThink', { n: j + 1, v: fmt(s.add[j]) })),
      this.contribList(items, { label: t('explain.d.bias'), v: s.bias[j] }, s.add[j]),
    ];
  }

  /** The biggest parts of a sum, one per line with a bar, then the rest, the bias and the total. */
  private contribList(items: { label: string; v: number }[], bias: { label: string; v: number }, total: number): HTMLElement {
    const sorted = items.slice().sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
    const shown = sorted.slice(0, LIST);
    const rest = sorted.slice(LIST);
    const lines = [...shown];
    if (rest.length) lines.push({ label: t('explain.d.rest', { n: rest.length }), v: rest.reduce((a, b) => a + b.v, 0) });
    lines.push(bias);
    const max = Math.max(1e-9, ...lines.map((l) => Math.abs(l.v)));
    return h('table', { class: 'xcontrib' },
      h('tbody', {}, ...lines.map((l) => h('tr', {},
        h('td', {}, l.label),
        h('td', { class: 'bar' }, h('span', { class: `sbar ${l.v < 0 ? 'neg' : 'pos'}`, style: `width:${Math.round((Math.abs(l.v) / max) * 100)}%` })),
        h('td', { class: 'val' }, signed(l.v)))),
      h('tr', { class: 'total' }, h('td', {}, t('explain.d.total')), h('td', {}), h('td', { class: 'val' }, fmt(total)))));
  }

  /** Weights used for this letter (and for the clicked square), to outline in the heatmaps. */
  private marks(): Map<string, HeatmapMarks> {
    const m = this.msg;
    const out = new Map<string, HeatmapMarks>();
    if (!m) return out;
    const add = (name: string, mk: HeatmapMarks) => {
      const o = out.get(name) ?? {};
      out.set(name, { rows: [...(o.rows ?? []), ...(mk.rows ?? [])], cols: [...(o.cols ?? []), ...(mk.cols ?? [])] });
    };
    const p = m.position;
    const chosen = m.guesses.find((g) => g.char === m.chosen)?.token;
    add('tokEmb', { rows: [m.decInIds[p], ...(chosen !== undefined ? [chosen] : [])] });
    add('decPos', { rows: [p] });
    if (chosen !== undefined) add('outBias', { cols: [chosen] });
    const pk = this.pick;
    if (!pk) return out;
    if (pk.row === 'guess' && pk.strip === 'guess') {
      const g = m.guesses[pk.index];
      add('tokEmb', { rows: [g.token] });
      add('outBias', { cols: [g.token] });
    } else if (pk.row === 'start') {
      add('tokEmb', { cols: [pk.index] });
      add('decPos', { cols: [pk.index] });
    } else {
      const s = m.stages.find((st) => st.name === pk.row);
      if (!s) return out;
      const block = s.name.split('.')[0];
      if (s.kind === 'attend' && pk.strip === 'add') add(`${s.name}.o.W`, { cols: [pk.index] });
      if (s.kind === 'think' && pk.strip === 'add') add(`${block}.ffn.down.W`, { cols: [pk.index] });
      if (s.kind === 'think' && pk.strip === 'hidden') {
        add(`${block}.ffn.up.W`, { cols: [pk.index] });
        add(`${block}.ffn.down.W`, { rows: [pk.index] });
      }
    }
    return out;
  }
}

/** Thinking units that are (clearly) on. */
function unitsOn(s: ExplainStage): number {
  return s.hidden.filter((v) => v > 0.05).length;
}

export { pctText };

/**
 * The most interesting letter of an answer to start with: the first one where
 * it stops copying the animal's name (buvol → buvol|i|ce), or the first letter
 * if it differs straight away.
 */
export function interestingLetter(a: { text: string; steps: readonly { chosen: string }[] }): number {
  const src = [...a.text];
  for (let i = 0; i < a.steps.length; i++) if (a.steps[i].chosen !== src[i]) return i;
  return 0;
}

function norm(a: Float32Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * a[i];
  return Math.sqrt(s);
}

function fmt(v: number): string {
  const a = Math.abs(v);
  return a === 0 ? '0' : a >= 0.001 ? v.toFixed(3) : v.toExponential(1);
}

function signed(v: number): string {
  return v >= 0 ? `+${fmt(v)}` : fmt(v);
}
