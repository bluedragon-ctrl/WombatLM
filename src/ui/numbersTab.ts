/**
 * "The numbers" tab (expert only): the real numbers behind the other tabs,
 * for readers who know how neural networks work. The other tabs explain the
 * ideas in plain words; this one shows the arithmetic.
 * - "The model in numbers": its sizes and learning settings, live values of
 *   the current step, and a table of every weight matrix (shape, how many
 *   numbers, share of the model, change per step);
 * - "How fast each part changes": a diagram of the model's parts, each with
 *   how much of its numbers changed per step lately and a line over time;
 * - "All the numbers": one heatmap per weight matrix (values, recent change or
 *   gradient), with technical names, shapes and colour scales;
 * - "Follow the signal": the hidden numbers of every position of the current
 *   answer, layer by layer;
 * - "What this step changed" (see stepView.ts): after Step ×1, the gradient,
 *   clipping and Adam calculation of any number. A click on a heatmap square
 *   explains that number there.
 */
import type { AxisKind } from '../nn/transformer';
import type { AnswerMsg, InspectMode, MatrixChange, MatrixInfo, ReadyMsg, TickMsg, WeightsMsg } from '../train/protocol';
import { DIVERGING_ENDS, INK, symmetricScale } from '../viz/colors';
import { prepare } from '../viz/canvas';
import { drawHeatmap, heatmapCellAt, heatmapLayout, HeatmapLayout, HeatmapMarks } from '../viz/weightHeatmap';
import { h, showTip } from './dom';
import { t, uiLang } from './i18n';
import { axisText, BOXES, fmt, modeNote, roleLabel, tipText } from './inside';
import type { StepView } from './stepView';
import { shortLabel } from './tokens';

const MODES: InspectMode[] = ['weights', 'delta', 'grad'];
/** Points kept per box for the "over time" line. */
const HISTORY = 240;

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

export class NumbersTab {
  readonly el: HTMLElement;
  mode: InspectMode = 'weights';
  private ready?: ReadyMsg;
  private cards: Card[] = [];
  private modeButtons: HTMLButtonElement[] = [];
  private modeNote: HTMLElement;
  private body: HTMLElement;
  private signal: HTMLElement;
  private answer?: AnswerMsg;
  /** One colour scale for every heatmap (honest sizes), or one per matrix. */
  private shared = true;
  private flow: HTMLElement;
  private boxes = new Map<string, Box>();
  private lastSummaryStep = -1;
  /** Change per step of each matrix (share of its size), from the latest summary. */
  private rates = new Map<string, number>();
  /** Weights used by the letter in "look inside one letter" (Inside tab), outlined in the heatmaps. */
  private marks = new Map<string, HeatmapMarks>();
  private sizes: HTMLElement;
  private live: HTMLElement;
  private matrixRows: HTMLElement;
  /** The latest step was recorded in detail, so a click can explain a number (set by the app). */
  canWhy = false;
  /** A click on a number asks why it changed (set by the app). */
  onPickNumber?: (name: string, index: number) => void;

  constructor(private onModeChange: () => void, private stepView: StepView) {
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
    this.signal = h('section', { class: 'panel signal', 'data-test': 'signal' }, h('h2', {}, t('signal.title')), h('p', { class: 'small' }, t('signal.wait')));
    this.flow = h('div', { class: 'flow' });
    this.sizes = h('dl', { class: 'nums-sizes', 'data-test': 'nums-sizes' });
    this.live = h('dl', { class: 'nums-sizes nums-live', 'data-test': 'nums-live' });
    this.matrixRows = h('tbody');
    const overview = h('section', { class: 'panel overview', 'data-test': 'overview' },
      h('h2', {}, t('flow.title')),
      h('p', { class: 'small' }, t('flow.helpShort'), ' ', help(t('flow.help'))),
      this.flow);
    this.el = h('div', { class: 'inside numbers' },
      h('section', { class: 'panel inside-head nums-intro' },
        h('h2', {}, t('nums.title')),
        h('p', { class: 'small' }, t('nums.intro')),
        h('ul', { class: 'small nums-toc' }, ...['sizes', 'flow', 'heat', 'signal', 'step'].map((k) => h('li', {}, t(`nums.toc.${k}`))))),
      h('section', { class: 'panel nums-model', 'data-test': 'nums-model' },
        h('h2', {}, t('nums.model.title')),
        h('p', { class: 'small' }, t('nums.model.help')),
        h('div', { class: 'nums-two' },
          h('div', {}, h('h3', {}, t('nums.model.sizes')), this.sizes),
          h('div', {}, h('h3', {}, t('nums.model.live')), this.live)),
        h('details', { class: 'nums-matrices', 'data-test': 'nums-matrices' },
          h('summary', {}, t('nums.model.matrices')),
          h('p', { class: 'small' }, t('nums.model.matricesHelp')),
          h('table', { class: 'sd-numbers nums-table' },
            h('thead', {}, h('tr', {},
              h('th', {}, t('nums.col.part')), h('th', {}, t('nums.col.what')), h('th', {}, t('nums.col.name')),
              h('th', { class: 'num' }, t('nums.col.shape')), h('th', { class: 'num' }, t('nums.col.count')),
              h('th', { class: 'num' }, t('nums.col.share')), h('th', { class: 'num' }, t('nums.col.rate')))),
            this.matrixRows))),
      overview,
      h('section', { class: 'panel inside-head heat-head' },
        h('h2', {}, t('inside.heatTitle')),
        h('div', { class: 'row' }, modes, this.modeNote),
        h('p', { class: 'small' }, t('inside.helpShort'), ' ', help(t('inside.help'))),
        legend),
      this.body,
      this.signal,
      stepView.el);
    this.renderModeButtons();
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    this.lastSummaryStep = -1;
    this.rates.clear();
    this.renderSizes();
    this.renderMatrixTable();
    this.buildFlow();
    this.build();
  }

  /** Matrix names to request: all of them. */
  wanted(): string[] {
    return this.ready ? this.ready.matrices.map((m) => m.name) : [];
  }

  setMode(mode: InspectMode): void {
    this.mode = mode;
    this.renderModeButtons();
  }

  /** Scroll to a part of the heatmaps (from a click on the Inside tab's scan). */
  scrollTo(selector: string): void {
    this.el.querySelector(selector)?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
  }

  scrollToStep(): void {
    this.stepView.el.scrollIntoView?.({ block: 'start' });
  }

  /** Outline the numbers used by the letter chosen in "look inside one letter". */
  setMarks(marks: Map<string, HeatmapMarks>): void {
    this.marks = marks;
    for (const c of this.cards) this.draw(c);
  }

  // -------------------------------------------------------------------------
  // The model in numbers

  private renderSizes(): void {
    const r = this.ready;
    if (!r) return;
    const c = r.config;
    const n = (v: number) => v.toLocaleString(uiLang());
    const item = (key: string, value: string, tech?: string) => [
      h('dt', {}, t(`nums.size.${key}`), tech ? h('code', {}, ` ${tech}`) : ''),
      h('dd', {}, value)];
    this.sizes.replaceChildren(
      ...item('params', n(r.params)),
      ...item('vocab', n(c.vocabSize), 'V'),
      ...item('dModel', n(c.dModel), 'd_model'),
      ...item('heads', t('nums.size.headsVal', { n: c.heads, d: c.dModel / c.heads }), 'h'),
      ...item('ffn', n(c.ffn), 'd_ff'),
      ...item('layers', t('nums.size.layersVal', { enc: c.encLayers, dec: c.decLayers })),
      ...item('maxLen', n(c.maxLen)),
      ...item('dropout', String(c.dropout), 'p'),
      ...item('adam', `β₁ ${c.beta1} · β₂ ${c.beta2} · ε ${c.eps}`),
      ...item('clip', String(c.clipNorm), '‖g‖ ≤'),
      ...item('warmup', t('nums.size.warmupVal', { n: c.warmupSteps })),
      ...item('decay', String(c.weightDecay)),
      ...item('copy', `1 : ${c.copyWeight}`));
  }

  /** Live values of the latest step (cheap: called on every tick). */
  setTick(tk: TickMsg): void {
    const item = (key: string, value: string, tech?: string) => [
      h('dt', {}, t(`nums.live.${key}`), tech ? h('code', {}, ` ${tech}`) : ''),
      h('dd', {}, value)];
    this.live.replaceChildren(
      ...item('step', tk.step.toLocaleString(uiLang())),
      ...item('loss', tk.lossEma === null ? '–' : tk.lossEma.toFixed(4), 'EMA'),
      ...item('lr', tk.step === 0 ? '–' : tk.lr.toPrecision(3), 'η'),
      ...item('batch', String(tk.batchSize), 'B'),
      ...item('grad', tk.step === 0 ? '–' : tk.gradNorm.toFixed(4), '‖g‖'),
      ...item('speed', tk.running ? t('nums.live.speedVal', { n: Math.round(tk.stepsPerSec) }) : t('nums.live.paused')));
  }

  private renderMatrixTable(): void {
    const r = this.ready;
    if (!r) return;
    const partOf = (name: string) => BOXES.find((b) => b.match.test(name));
    this.matrixRows.replaceChildren(...r.matrices.map((m) => {
      const count = m.rows * m.cols;
      const box = partOf(m.name);
      const rate = this.rates.get(m.name);
      const tr = h('tr', { 'data-row-matrix': m.name, tabindex: 0, title: t('nums.col.click') },
        h('td', {}, box ? t(`flow.${box.id}`) : ''),
        h('td', {}, roleLabel(m)),
        h('td', {}, h('code', {}, m.name)),
        h('td', { class: 'num' }, m.rows > 1 ? `${m.rows} × ${m.cols}` : String(m.cols)),
        h('td', { class: 'num' }, count.toLocaleString(uiLang())),
        h('td', { class: 'num' }, `${((count / r.params) * 100).toFixed(1)} %`),
        h('td', { class: 'num' }, rate === undefined ? '–' : `${pct(rate)} %`));
      const go = () => this.scrollTo(`[data-matrix-card="${m.name}"]`);
      tr.onclick = go;
      tr.onkeydown = (e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); go(); } };
      return tr;
    }));
  }

  // -------------------------------------------------------------------------
  // "How fast each part changes"

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
      el.onclick = () => this.scrollTo(b.target);
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
    if (fresh) this.renderMatrixTable();
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
    const layout = heatmapLayout(info.rows, info.cols, 5, {
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
    const label = h('div', { class: 'label' }, roleLabel(info), h('code', {}, ` ${info.name}`));
    return h('figure', { class: 'card', 'data-matrix-card': info.name }, label, canvas,
      h('figcaption', { class: 'small' }, `${info.rows} × ${info.cols}`, scaleLabel, rateLabel));
  }

  private describe(c: Card, row: number, col: number): string {
    const v = c.data![row * c.info.cols + col];
    const tokens = this.ready!.tokens;
    const r = axisText(c.info.rowKind, row, tokens), k = axisText(c.info.colKind, col, tokens);
    // A layer's matrix connects an incoming number (row) to an outgoing one (column).
    const layer = /^(dim|hidden)$/.test(c.info.rowKind) && /^(dim|hidden)$/.test(c.info.colKind);
    const where = layer ? `${r} → ${k}` : [r, k].filter(Boolean).join(', ');
    const lines = [`${roleLabel(c.info)} — ${where}`, tipText(c.info, row, col, tokens), t(`inside.val.${this.mode}`, { v: fmt(v) }), `${c.info.name}[${row}, ${col}]`];
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
    this.modeNote.textContent = modeNote(msg);
  }

  setAnswer(a: AnswerMsg | undefined): void {
    this.answer = a;
    this.renderSignal();
  }

  private renderModeButtons(): void {
    for (const b of this.modeButtons) b.classList.toggle('on', b.dataset.mode === this.mode);
  }

  // -------------------------------------------------------------------------
  // "Follow the signal": the hidden numbers of every position, layer by layer.

  private renderSignal(): void {
    const a = this.answer;
    if (!a) return;
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

/** A share (0..1) as a percentage with enough digits for small changes. */
function pct(r: number): string {
  const p = r * 100;
  return p >= 1 ? p.toFixed(1) : p >= 0.01 ? p.toFixed(2) : p.toExponential(1);
}
