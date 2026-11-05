/**
 * "Inside the model" tab:
 * - "Where is it learning?": a diagram of the model's parts, each with how
 *   much of its numbers changed per step lately (one scale for all) and a
 *   line of that over time; clicking a part scrolls to its heatmaps;
 * - "Look inside one letter" (see explainView.ts);
 * - a heatmap per weight matrix (weights, their change in the last steps, or
 *   the gradient), with letters / positions on the axes where they mean
 *   something and plain-language tooltips;
 * - in expert mode, "follow the signal" for the current word.
 */
import type { AxisKind } from '../nn/transformer';
import type { AnswerMsg, ExplainMsg, InspectMode, MatrixChange, MatrixInfo, ReadyMsg, WeightsMsg } from '../train/protocol';
import { DIVERGING_ENDS, INK, symmetricScale } from '../viz/colors';
import { prepare } from '../viz/canvas';
import { drawHeatmap, heatmapCellAt, heatmapLayout, HeatmapLayout, HeatmapMarks } from '../viz/weightHeatmap';
import { h, showTip } from './dom';
import { ExplainView, Question } from './explainView';
import { t, uiLang } from './i18n';
import { shortLabel } from './tokens';

const MODES: InspectMode[] = ['weights', 'delta', 'grad'];

/** A "?" that opens the full explanation (the panel itself keeps one short sentence). */
const help = (text: string) => h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, text));

/** Which part of the model a matrix belongs to, for grouping. */
function group(name: string): string {
  if (/^enc(\d|Norm)/.test(name)) return 'encoder';
  if (/^dec(\d|Norm)/.test(name)) return 'decoder';
  return 'letters';
}

/** Sub-group inside a block: self-attention, cross-attention, thinking (FFN), norms. */
function part(name: string): string {
  const m = name.match(/^(?:enc|dec)\d+\.(self|cross|ffn|ln1|ln2|lnX)/);
  if (!m) return '';
  return m[1].startsWith('ln') ? 'norm' : m[1];
}

/**
 * Boxes of the "where is it learning" diagram, in the order the signal flows.
 * Each volume knob counts with the step it prepares.
 */
const BOXES: { id: string; column: string; match: RegExp; target: string }[] = [
  { id: 'letters', column: 'letters', match: /^(tokEmb|outBias)$/, target: '[data-matrix-card=tokEmb]' },
  { id: 'positions', column: 'letters', match: /^(encPos|decPos)$/, target: '[data-matrix-card=encPos]' },
  { id: 'enc.self', column: 'encoder', match: /^enc\d+\.(self|ln1)\./, target: '[data-part="encoder.self"]' },
  { id: 'enc.ffn', column: 'encoder', match: /^(enc\d+\.(ffn|ln2)\.|encNorm)/, target: '[data-part="encoder.ffn"]' },
  { id: 'dec.self', column: 'decoder', match: /^dec\d+\.(self|ln1)\./, target: '[data-part="decoder.self"]' },
  { id: 'dec.cross', column: 'decoder', match: /^dec\d+\.(cross|lnX)\./, target: '[data-part="decoder.cross"]' },
  { id: 'dec.ffn', column: 'decoder', match: /^(dec\d+\.(ffn|ln2)\.|decNorm)/, target: '[data-part="decoder.ffn"]' },
];

/** Points kept per box for the "over time" line. */
const HISTORY = 240;

interface Card {
  info: MatrixInfo;
  canvas: HTMLCanvasElement;
  scaleLabel: HTMLElement;
  rateLabel: HTMLElement;
  layout: HeatmapLayout;
  data: Float32Array | null;
  /** Own colour scale (99th percentile of |values|) and the one in use (own or shared). */
  ownScale: number;
  scale: number;
}

interface Box {
  el: HTMLElement;
  fill: HTMLElement;
  value: HTMLElement;
  spark: HTMLCanvasElement;
  history: { step: number; rate: number }[];
  rate: number | null;
}

export class InsideTab {
  readonly el: HTMLElement;
  mode: InspectMode = 'delta';
  private ready?: ReadyMsg;
  private expert = false;
  private cards: Card[] = [];
  private modeButtons: HTMLButtonElement[] = [];
  private modeNote: HTMLElement;
  private body: HTMLElement;
  private signal: HTMLElement;
  private answer?: AnswerMsg;
  /** One colour scale for every heatmap (honest sizes), or one per matrix. */
  private shared = true;
  private overview: HTMLElement;
  private flow: HTMLElement;
  private boxes = new Map<string, Box>();
  private lastSummaryStep = -1;
  /** Change per step of each matrix (share of its size), from the latest summary. */
  private rates = new Map<string, number>();
  private explain: ExplainView;
  /** Weights used by the letter in "look inside one letter", outlined in the heatmaps. */
  private marks = new Map<string, HeatmapMarks>();
  /** A click on a number asks why it changed (set by the app). */
  onPickNumber?: (name: string, index: number) => void;
  /** The latest step was recorded in detail, so a click can explain a number (set by the app). */
  canWhy = false;

  constructor(private onModeChange: () => void, requestExplain: (q: Question, position: number) => void) {
    this.modeNote = h('span', { class: 'small', 'data-test': 'mode-note' });
    const modes = h('div', { class: 'picker', role: 'group', 'aria-label': t('inside.mode') });
    for (const m of MODES) {
      const b = h('button', { 'data-mode': m }, t(`inside.mode.${m}`));
      b.onclick = () => { this.mode = m; this.renderModeButtons(); this.onModeChange(); };
      modes.append(b);
      this.modeButtons.push(b);
    }
    const sharedBox = h('input', { type: 'checkbox', checked: true, 'data-test': 'shared-scale' });
    sharedBox.onchange = () => { this.shared = sharedBox.checked; this.rescale(); };
    const legend = h('div', { class: 'legend' },
      h('span', { class: 'swatch-box', style: `background:${DIVERGING_ENDS.neg}` }), t('inside.negative'),
      h('span', { class: 'swatch-box', style: 'background:#fff' }), t('inside.zero'),
      h('span', { class: 'swatch-box', style: `background:${DIVERGING_ENDS.pos}` }), t('inside.positive'),
      h('label', { class: 'shared', title: t('inside.sharedHelp') }, sharedBox, ' ', t('inside.shared')));
    this.body = h('div', { class: 'inside-body', 'data-test': 'heatmaps' });
    this.signal = h('section', { class: 'panel signal', hidden: true, 'data-test': 'signal' });
    this.flow = h('div', { class: 'flow' });
    this.overview = h('section', { class: 'panel overview', 'data-test': 'overview' },
      h('h2', {}, t('flow.title')),
      h('p', { class: 'small' }, t('flow.helpShort'), ' ', help(t('flow.help'))),
      this.flow);
    this.explain = new ExplainView(requestExplain, (marks) => { this.marks = marks; for (const c of this.cards) this.draw(c); }, (code) => this.langName(code));
    this.el = h('div', { class: 'inside plain' },
      h('section', { class: 'panel inside-head' },
        h('p', { class: 'intro', 'data-test': 'inside-intro' }),
        h('p', { class: 'small' }, t('inside.guide'))),
      this.overview,
      this.explain.el,
      h('section', { class: 'panel inside-head' },
        h('h2', {}, t('inside.heatTitle')),
        h('div', { class: 'row' }, modes, this.modeNote),
        h('p', { class: 'small' }, t('inside.helpShort'), ' ', help(t('inside.help'))),
        legend),
      this.body,
      this.signal);
    this.renderModeButtons();
  }

  /** Switch what the heatmaps show (e.g. "Change" from the learning-cycle link on the Model learning tab). */
  setMode(mode: InspectMode): void {
    this.mode = mode;
    this.renderModeButtons();
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    this.el.querySelector('.intro')!.textContent = t('inside.intro', { n: ready.params.toLocaleString(uiLang()) });
    this.lastSummaryStep = -1;
    this.rates.clear();
    this.buildFlow();
    this.build();
  }

  setExpert(on: boolean): void {
    this.expert = on;
    // Plain mode hides the engineering details (rates, shapes, colour scales): see style.css.
    this.el.classList.toggle('plain', !on);
    this.signal.hidden = !on;
    this.explain.setExpert(on);
    this.build();
    this.renderSignal();
  }

  /** Matrix names to request (only visible ones). */
  wanted(): string[] {
    return this.cards.map((c) => c.info.name);
  }

  /** The tab became visible (or its answer changed while visible): bring "look inside one letter" up to date. */
  shown(): void {
    this.explain.refresh();
  }

  /** Look inside letter `position` of the current answer (from a click on the guess bars). */
  explainAt(position: number): void {
    this.explain.select(position);
  }

  scrollToExplain(): void {
    this.explain.el.scrollIntoView?.({ block: 'start' });
  }

  onExplain(msg: ExplainMsg): void {
    this.explain.onExplain(msg);
  }

  private langName(code: string): string {
    return this.ready?.languages.find((l) => l.code === code)?.name[uiLang()] ?? code;
  }

  private visible(m: MatrixInfo): boolean {
    return this.expert || m.rows > 1;
  }

  // -------------------------------------------------------------------------
  // "Where is it learning?"

  private buildFlow(): void {
    if (!this.ready) return;
    const names = this.ready.matrices.map((m) => m.name);
    this.boxes.clear();
    const columns = new Map<string, HTMLElement[]>();
    for (const b of BOXES) {
      if (!names.some((n) => b.match.test(n))) continue; // e.g. a model without an encoder
      const fill = h('span', { class: 'fill' });
      const value = h('span', { class: 'val' }, '—');
      const spark = h('canvas', { class: 'spark' });
      const el = h('button', { class: 'flow-box', 'data-box': b.id, title: t('flow.click') },
        h('span', { class: 'name' }, t(`flow.${b.id}`)),
        h('span', { class: 'track' }, fill),
        value,
        spark);
      el.onclick = () => this.el.querySelector(b.target)?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
      this.boxes.set(b.id, { el, fill, value, spark, history: [], rate: null });
      if (!columns.has(b.column)) columns.set(b.column, []);
      columns.get(b.column)!.push(el);
    }
    const cols = ['letters', 'encoder', 'decoder'].filter((c) => columns.has(c)).map((c) => h('div', { class: `flow-col ${c}` },
      h('h3', {}, t(`inside.group.${c}`)),
      ...columns.get(c)!));
    const arrow = () => h('span', { class: 'flow-arrow', 'aria-hidden': 'true' }, '→');
    const parts: HTMLElement[] = [];
    cols.forEach((c, i) => { if (i) parts.push(arrow()); parts.push(c); });
    parts.push(arrow(), h('div', { class: 'flow-end' }, t('flow.end')));
    this.flow.replaceChildren(...parts);
    this.drawFlow();
  }

  /** New change summary: one rate per box (share of its numbers changed per step), and per matrix for the captions. */
  private onSummary(step: number, summary: MatrixChange[]): void {
    if (step < this.lastSummaryStep) { for (const b of this.boxes.values()) b.history = []; }
    this.rates.clear();
    for (const m of summary) if (m.steps > 0 && m.norm > 0) this.rates.set(m.name, m.change / m.norm / m.steps);
    const fresh = step !== this.lastSummaryStep;
    for (const def of BOXES) {
      const box = this.boxes.get(def.id);
      if (!box) continue;
      let c2 = 0, n2 = 0, steps = 0;
      for (const m of summary) if (def.match.test(m.name)) { c2 += m.change * m.change; n2 += m.norm * m.norm; steps = Math.max(steps, m.steps); }
      box.rate = steps > 0 && n2 > 0 ? Math.sqrt(c2) / Math.sqrt(n2) / steps : null;
      if (fresh && box.rate !== null) {
        box.history.push({ step, rate: box.rate });
        if (box.history.length > HISTORY) box.history = box.history.filter((_, i) => i % 2 === 0);
      }
    }
    this.lastSummaryStep = step;
    this.drawFlow();
    for (const c of this.cards) {
      const r = this.rates.get(c.info.name);
      c.rateLabel.textContent = r === undefined ? '' : ` · ${t('inside.rate', { pct: pct(r) })}`;
    }
  }

  private drawFlow(): void {
    const boxes = [...this.boxes.values()];
    const max = Math.max(1e-12, ...boxes.map((b) => b.rate ?? 0));
    const histMax = Math.max(1e-12, ...boxes.flatMap((b) => b.history.map((p) => p.rate)));
    const first = Math.min(...boxes.flatMap((b) => b.history.map((p) => p.step)));
    const last = Math.max(...boxes.flatMap((b) => b.history.map((p) => p.step)));
    for (const b of boxes) {
      b.fill.style.width = b.rate === null ? '0' : `${Math.max(2, Math.round((b.rate / max) * 100))}%`;
      b.value.textContent = b.rate === null ? t('flow.none') : t('inside.rate', { pct: pct(b.rate) });
      b.el.title = b.rate === null ? t('flow.click') : `${t('inside.rate', { pct: pct(b.rate) })}\n${t('flow.click')}`;
      const W = 120, H = 26;
      const ctx = prepare(b.spark, W, H);
      ctx.strokeStyle = '#e5e7eb';
      ctx.beginPath(); ctx.moveTo(0, H - 1.5); ctx.lineTo(W, H - 1.5); ctx.stroke();
      if (b.history.length < 2 || last <= first) continue;
      // One scale for every box, so the lines can be compared too.
      ctx.strokeStyle = INK;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      b.history.forEach((p, i) => {
        const x = ((p.step - first) / (last - first)) * (W - 2) + 1;
        const y = H - 2 - (p.rate / histMax) * (H - 4);
        if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      });
      ctx.stroke();
    }
  }

  // -------------------------------------------------------------------------
  // Heatmaps

  private build(): void {
    if (!this.ready) return;
    const old = new Map(this.cards.map((c) => [c.info.name, c]));
    this.cards = [];
    const groups = new Map<string, Map<string, HTMLElement[]>>();
    for (const info of this.ready.matrices) {
      if (!this.visible(info)) continue;
      const card = this.card(info, old.get(info.name));
      const g = group(info.name), p = part(info.name);
      if (!groups.has(g)) groups.set(g, new Map());
      const parts = groups.get(g)!;
      if (!parts.has(p)) parts.set(p, []);
      parts.get(p)!.push(card);
    }
    const order = ['letters', 'encoder', 'decoder'];
    const sorted = [...groups].sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
    this.body.replaceChildren(...sorted.map(([g, parts]) => h('section', { class: `panel group ${g}`, 'data-group': g },
      h('h2', {}, t(`inside.group.${g}`)),
      h('p', { class: 'small' }, t(`inside.groupHelp.${g}`)),
      h('div', { class: 'parts' }, ...[...parts].map(([p, cards]) => h('div', { class: 'part', 'data-part': `${g}.${p}` },
        p ? h('h3', {}, t(`inside.part.${p}`)) : null,
        h('div', { class: 'cards' }, ...cards)))))));
    this.rescale();
  }

  /** Labels for an axis that means something (letters, positions); none for plain "number n" axes. */
  private axisLabels(kind: AxisKind, n: number): string[] | undefined {
    if (kind === 'token') return this.ready!.tokens.slice(0, n).map(shortLabel);
    if (kind === 'position') return Array.from({ length: n }, (_, i) => String(i + 1));
    return undefined;
  }

  private card(info: MatrixInfo, prev?: Card): HTMLElement {
    const canvas = h('canvas', { class: 'heat', 'data-matrix': info.name });
    const scaleLabel = h('span', { class: 'scale' });
    const r = this.rates.get(info.name);
    const rateLabel = h('span', { class: 'rate' }, r === undefined ? '' : ` · ${t('inside.rate', { pct: pct(r) })}`);
    // Plain mode: the number × number matrices have no readable axes, so they are small thumbnails.
    const plainAxes = (k: AxisKind) => k === 'dim' || k === 'hidden';
    const thumb = !this.expert && plainAxes(info.rowKind) && plainAxes(info.colKind);
    const layout = heatmapLayout(info.rows, info.cols, thumb ? 3 : 5, {
      rows: info.rows > 1 ? this.axisLabels(info.rowKind, info.rows) : undefined,
      cols: this.axisLabels(info.colKind, info.cols),
    });
    const c: Card = { info, canvas, scaleLabel, rateLabel, layout, data: prev?.data ?? null, ownScale: prev?.ownScale ?? 1, scale: prev?.scale ?? 1 };
    this.cards.push(c);
    canvas.addEventListener('mousemove', (e) => {
      const cell = heatmapCellAt(c.layout, e.offsetX, e.offsetY);
      showTip(e, cell && c.data ? this.describe(c, cell.row, cell.col) : null);
      this.draw(c, cell);
    });
    canvas.addEventListener('mouseleave', (e) => { showTip(e, null); this.draw(c); });
    canvas.addEventListener('click', (e) => {
      const cell = heatmapCellAt(c.layout, e.offsetX, e.offsetY);
      if (!cell || !this.canWhy) return;
      showTip(e, null);
      this.onPickNumber?.(c.info.name, cell.row * c.info.cols + cell.col);
    });
    const label = h('div', { class: 'label' }, roleLabel(info));
    if (this.expert) label.append(h('code', {}, ` ${info.name}`));
    return h('figure', { class: `card${thumb ? ' thumb' : ''}`, 'data-matrix-card': info.name }, label, canvas,
      h('figcaption', { class: 'small' }, `${info.rows} × ${info.cols}`, scaleLabel, rateLabel));
  }

  private describe(c: Card, row: number, col: number): string {
    const v = c.data![row * c.info.cols + col];
    const tokens = this.ready!.tokens;
    const r = axisText(c.info.rowKind, row, tokens), k = axisText(c.info.colKind, col, tokens);
    // A layer's matrix connects an incoming number (row) to an outgoing one (column).
    const layer = /^(dim|hidden)$/.test(c.info.rowKind) && /^(dim|hidden)$/.test(c.info.colKind);
    const where = layer ? `${r} → ${k}` : [r, k].filter(Boolean).join(', ');
    const lines = [`${roleLabel(c.info)} — ${where}`, tipText(c.info, row, col, tokens), t(`inside.val.${this.mode}`, { v: this.expert ? fmt(v) : plainFmt(v) })];
    if (this.expert) lines.push(`${c.info.name}[${row}, ${col}]`);
    if (this.canWhy) lines.push(t('inside.whyClick'));
    return lines.filter(Boolean).join('\n');
  }

  private draw(c: Card, highlight: { row: number; col: number } | null = null): void {
    drawHeatmap(c.canvas, c.data ?? new Float32Array(c.info.rows * c.info.cols), c.layout, c.scale, highlight, this.marks.get(c.info.name));
  }

  /** Colour scale per matrix, or one shared by all (the largest of the matrices' own scales). */
  private rescale(): void {
    const big = this.cards.filter((c) => c.info.rows > 1 && c.data?.some((v) => v !== 0));
    const shared = Math.max(1e-12, ...big.map((c) => c.ownScale));
    for (const c of this.cards) {
      c.scale = this.shared && big.length ? shared : c.ownScale;
      c.scaleLabel.textContent = c.data?.some((v) => v !== 0) ? ` · ±${fmt(c.scale)}` : '';
      this.draw(c);
    }
  }

  onWeights(msg: WeightsMsg): void {
    if (msg.summary) this.onSummary(msg.step, msg.summary);
    if (msg.mode !== this.mode) return;
    const byName = new Map(msg.matrices.map((m) => [m.name, m.data]));
    for (const c of this.cards) {
      const d = byName.get(c.info.name);
      if (!d) continue;
      c.data = d;
      c.ownScale = symmetricScale(d);
    }
    this.rescale();
    this.modeNote.textContent = msg.mode === 'weights' ? t('inside.note.weights', { step: msg.step })
      : msg.steps === 0 ? t(`inside.note.${msg.mode}None`)
        : msg.mode === 'delta' ? t(msg.steps === 1 ? 'inside.note.delta1' : 'inside.note.delta', { n: msg.steps }) : t('inside.note.grad', { step: msg.step });
  }

  setAnswer(a: AnswerMsg | undefined): void {
    this.answer = a;
    this.explain.setAnswer(a);
    if (this.expert) this.renderSignal();
  }

  private renderModeButtons(): void {
    for (const b of this.modeButtons) b.classList.toggle('on', b.dataset.mode === this.mode);
  }

  /** "Follow the signal": the hidden numbers of every position, layer by layer. */
  private renderSignal(): void {
    const a = this.answer;
    if (!this.expert || !a) return;
    const input = a.input.map((tok, i) => (i === 0 ? '✱' : tok === ' ' ? '␣' : tok));
    // Decoder row i: the letter going in at position i, and the letter it is used to guess.
    const decIn = ['✱', ...a.steps.map((s) => shortLabel(s.chosen))];
    const decRows = decIn.map((tok, i) => (i < a.steps.length ? `${tok}→${shortLabel(a.steps[i].chosen)}` : tok));
    const strips = a.activations.map((act) => {
      const enc = act.name.startsWith('enc');
      const labels = (enc ? input : decRows).slice(0, act.T);
      const layout = heatmapLayout(act.T, act.d, 12);
      const canvas = h('canvas', { class: 'heat' });
      const scale = symmetricScale(act.data);
      drawHeatmap(canvas, act.data, layout, scale);
      canvas.addEventListener('mousemove', (e) => {
        const cell = heatmapCellAt(layout, e.offsetX, e.offsetY);
        const who = enc ? `'${labels[cell?.row ?? 0]}'` : t('signal.decRow', { tok: decIn[cell?.row ?? 0], next: shortLabel(a.steps[cell?.row ?? 0]?.chosen ?? '') });
        showTip(e, cell ? `${who}, ${t('axis.dim', { n: cell.col + 1 })}: ${fmt(act.data[cell.row * act.d + cell.col])}` : null);
      });
      canvas.addEventListener('mouseleave', (e) => showTip(e, null));
      return h('figure', { class: 'card strip' },
        h('div', { class: 'label' }, t(`signal.${act.name.replace(/\d+/, 'N')}`, { n: Number(act.name.match(/\d+/)?.[0] ?? 0) + 1 }), h('code', {}, ` ${act.name}`)),
        h('div', { class: 'strip-body' },
          h('div', { class: 'strip-labels' }, ...labels.map((l) => h('div', {}, l))),
          canvas));
    });
    this.signal.replaceChildren(
      h('h2', {}, t('signal.title')),
      h('p', { class: 'small' }, t('signal.help', { word: `${a.text} → ${a.output}` })),
      h('div', { class: 'cards' }, ...strips));
  }
}

function fmt(v: number): string {
  const a = Math.abs(v);
  return a === 0 ? '0' : a >= 0.01 ? v.toFixed(3) : v.toExponential(1);
}

/** A number without scientific notation: 0.0019 rather than 1.9e-3, "almost 0" when tinier than 0.000001. */
export function plainFmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  if (a >= 0.01) return v.toFixed(3);
  if (a < 1e-6) return t('inside.almostZero');
  return v.toFixed(Math.min(8, 1 - Math.floor(Math.log10(a))));
}

/** A share (0..1) as a percentage with enough digits for small changes. */
function pct(r: number): string {
  const p = r * 100;
  return p >= 1 ? p.toFixed(1) : p >= 0.01 ? p.toFixed(2) : p.toExponential(1);
}

/** Plain-language name of a matrix (technical name in expert mode is added by the caller). */
export function roleLabel(m: MatrixInfo): string {
  if (m.name === 'encPos') return t('role.encPos');
  if (m.name === 'decPos') return t('role.decPos');
  const key = m.role.endsWith('.bias') ? 'role.bias' : `role.${m.role}`;
  const s = t(key);
  return s === key ? m.role : s;
}

/** What one number of a matrix does, in plain words. */
export function tipText(m: MatrixInfo, row: number, col: number, tokens: readonly string[]): string {
  const vars = { r: row + 1, c: col + 1, tok: axisText('token', m.rowKind === 'token' ? row : col, tokens), d: m.cols };
  if (m.role.endsWith('.bias')) return t('tip.bias', { what: axisText(m.colKind, col, tokens) });
  const key = `tip.${m.role}`;
  const s = t(key, vars);
  return s === key ? '' : s;
}

/** What row/column number `i` of an axis means, e.g. "letter 'ž'". */
export function axisText(kind: AxisKind, i: number, tokens: readonly string[]): string {
  switch (kind) {
    case 'token': return t('axis.token', { tok: tokens[i] === ' ' ? '␣' : tokens[i] });
    case 'position': return t('axis.position', { n: i + 1 });
    case 'dim': return t('axis.dim', { n: i + 1 });
    case 'hidden': return t('axis.hidden', { n: i + 1 });
    case 'none': return '';
  }
}
