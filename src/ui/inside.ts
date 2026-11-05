/**
 * "Inside the model" tab:
 * - "Where is it learning?": a "brain scan" of all the model's numbers, one
 *   dot each, grouped by part in the order the signal flows, glowing where
 *   they change (or are pushed, or are big); clicking a part opens its
 *   heatmaps (which plain mode keeps folded away until then);
 * - expert: "How fast each part changes": a diagram of the model's parts, each with how
 *   much of its numbers changed per step lately (one scale for all) and a
 *   line of that over time; clicking a part scrolls to its heatmaps;
 * - "Look inside one letter" (see explainView.ts);
 * - "What have the detectors learned?": the writer's thinking units, each
 *   with the moments of the learning data that switch it on most (see
 *   train/units.ts and unitCard.ts);
 * - a heatmap per weight matrix (weights, their change in the last steps, or
 *   the gradient), with letters / positions on the axes where they mean
 *   something and plain-language tooltips;
 * - in expert mode, "follow the signal" for the current word.
 */
import type { AxisKind } from '../nn/transformer';
import type { AnswerMsg, ExplainMsg, InspectMode, MatrixChange, MatrixInfo, ReadyMsg, UnitsMsg, WeightsMsg } from '../train/protocol';
import { clearestFirst, matters, unitCard } from './unitCard';
import { DIVERGING_ENDS, INK, symmetricScale } from '../viz/colors';
import { prepare } from '../viz/canvas';
import { drawHeatmap, heatmapCellAt, heatmapLayout, HeatmapLayout, HeatmapMarks } from '../viz/weightHeatmap';
import { drawScan, GLOW_GRADIENT, scanCellAt, scanLayout, ScanLayout, ScanPiece } from '../viz/brainScan';
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
export const BOXES: { id: string; column: string; match: RegExp; target: string }[] = [
  { id: 'letters', column: 'letters', match: /^(tokEmb|outBias)$/, target: '[data-matrix-card=tokEmb]' },
  { id: 'positions', column: 'letters', match: /^(encPos|decPos)$/, target: '[data-matrix-card=encPos]' },
  { id: 'enc.self', column: 'encoder', match: /^enc\d+\.(self|ln1)\./, target: '[data-part="encoder.self"]' },
  { id: 'enc.ffn', column: 'encoder', match: /^(enc\d+\.(ffn|ln2)\.|encNorm)/, target: '[data-part="encoder.ffn"]' },
  { id: 'dec.self', column: 'decoder', match: /^dec\d+\.(self|ln1)\./, target: '[data-part="decoder.self"]' },
  { id: 'dec.cross', column: 'decoder', match: /^dec\d+\.(cross|lnX)\./, target: '[data-part="decoder.cross"]' },
  { id: 'dec.ffn', column: 'decoder', match: /^(dec\d+\.(ffn|ln2)\.|decNorm)/, target: '[data-part="decoder.ffn"]' },
];

/** The scan calls a part busiest / quietest only if its numbers move at least this much more / less than average. */
const BUSY = 1.3;
const QUIET = 0.7;
/** "What have the detectors learned?": at most one request per this many ms while learning; clear ones shown in plain mode. */
const UNITS_EVERY = 15000;
const UNITS_CLEAR = 12;

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

interface ScanTile {
  def: (typeof BOXES)[number];
  canvas: HTMLCanvasElement;
  layout: ScanLayout;
  share: HTMLElement;
  /** Share of all the change (or push) in this part, 0..1, or null in "Values" mode. */
  part: number | null;
  hover: { piece: ScanPiece } | null;
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
  private scanNote: HTMLElement;
  private scanEl: HTMLElement;
  private scanFlow: HTMLElement;
  private scanLegend: HTMLElement;
  private scanVerdict: HTMLElement;
  private tiles: ScanTile[] = [];
  /** The latest numbers of every matrix (in the current mode), for the scan. */
  private all = new Map<string, Float32Array>();
  private scanScale = 1;
  /** Plain mode: the heatmaps are folded away until asked for (button, or a click on the scan). */
  private numbersOpen = false;
  private numbersButton: HTMLButtonElement;
  private heatHead: HTMLElement;
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
  /** "What have the detectors learned?" */
  private unitsEl: HTMLElement;
  private unitsBody: HTMLElement;
  private unitsNote: HTMLElement;
  private unitsTeam: HTMLElement;
  private unitsButton: HTMLButtonElement;
  private units?: UnitsMsg;
  private unitsAll = false;
  private unitsAsked = 0;
  private unitsInFlight = false;
  /** Ask the worker what the thinking units have learned (set by the app). */
  requestUnits?: () => void;
  /** A click on a number asks why it changed (set by the app). */
  onPickNumber?: (name: string, index: number) => void;
  /** The latest step was recorded in detail, so a click can explain a number (set by the app). */
  canWhy = false;

  constructor(private onModeChange: () => void, requestExplain: (q: Question, position: number) => void) {
    this.modeNote = h('span', { class: 'small', 'data-test': 'mode-note' });
    this.scanNote = h('span', { class: 'small', 'data-test': 'scan-note' });
    // The scan and the heatmaps show the same numbers, so both have the same mode buttons.
    const modePicker = () => {
      const modes = h('div', { class: 'picker', role: 'group', 'aria-label': t('inside.mode') });
      for (const m of MODES) {
        const b = h('button', { 'data-mode': m }, t(`inside.mode.${m}`));
        b.onclick = () => { this.mode = m; this.renderModeButtons(); this.onModeChange(); };
        modes.append(b);
        this.modeButtons.push(b);
      }
      return modes;
    };
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
    this.scanFlow = h('div', { class: 'flow scan-flow' });
    this.scanLegend = h('div', { class: 'legend scan-legend' });
    this.scanVerdict = h('p', { class: 'scan-verdict', 'data-test': 'scan-verdict' });
    this.numbersButton = h('button', { class: 'numbers-toggle', 'data-test': 'numbers-toggle' });
    this.numbersButton.onclick = () => { this.numbersOpen = !this.numbersOpen; this.renderNumbersOpen(); };
    this.scanEl = h('section', { class: 'panel scan', 'data-test': 'scan' },
      h('h2', {}, t('scan.title')),
      h('p', { class: 'small' }, h('span', { class: 'scan-help' }), ' ', help(t('scan.help'))),
      h('div', { class: 'row' }, modePicker(), this.scanNote),
      this.scanFlow,
      this.scanLegend,
      this.scanVerdict,
      this.numbersButton);
    this.overview = h('section', { class: 'panel overview', hidden: true, 'data-test': 'overview' },
      h('h2', {}, t('flow.title')),
      h('p', { class: 'small' }, t('flow.helpShort'), ' ', help(t('flow.help'))),
      this.flow);
    this.explain = new ExplainView(requestExplain, (marks) => { this.marks = marks; for (const c of this.cards) this.draw(c); }, (code) => this.langName(code));
    this.unitsBody = h('div', { class: 'ucards', 'data-test': 'unit-cards' });
    this.unitsNote = h('p', { class: 'small', 'data-test': 'units-note' }, t('units.wait'));
    this.unitsTeam = h('p', { class: 'units-team', 'data-test': 'units-team' });
    this.unitsButton = h('button', { class: 'units-toggle', 'data-test': 'units-toggle' });
    this.unitsButton.onclick = () => { this.unitsAll = !this.unitsAll; this.renderUnits(); };
    this.unitsEl = h('section', { class: 'panel units', 'data-test': 'units' },
      h('h2', {}, t('units.title')),
      h('p', { class: 'small' }, h('span', { class: 'units-help' }), ' ', help(t('units.help'))),
      this.unitsNote,
      this.unitsTeam,
      this.unitsBody,
      this.unitsButton);
    this.heatHead = h('section', { class: 'panel inside-head heat-head' },
      h('h2', {}, t('inside.heatTitle')),
      h('div', { class: 'row' }, modePicker(), this.modeNote),
      h('p', { class: 'small' }, t('inside.helpShort'), ' ', help(t('inside.help'))),
      legend);
    this.el = h('div', { class: 'inside plain' },
      h('section', { class: 'panel inside-head' },
        h('p', { class: 'intro', 'data-test': 'inside-intro' }),
        h('p', { class: 'small' }, t('inside.guide'))),
      this.scanEl,
      this.overview,
      this.explain.el,
      this.unitsEl,
      this.heatHead,
      this.body,
      this.signal);
    this.renderModeButtons();
    this.renderNumbersOpen();
  }

  /** Plain mode: the heatmaps only when asked for. Expert: always. */
  private renderNumbersOpen(): void {
    const open = this.expert || this.numbersOpen;
    this.heatHead.hidden = !open;
    this.body.hidden = !open;
    this.numbersButton.hidden = this.expert;
    this.numbersButton.textContent = t(this.numbersOpen ? 'scan.hideNumbers' : 'scan.showNumbers');
  }

  /** Switch what the heatmaps show (e.g. "Change" from the learning-cycle link on the Model learning tab). */
  setMode(mode: InspectMode): void {
    this.mode = mode;
    this.renderModeButtons();
  }

  /** Open the heatmaps (plain mode) and scroll to `selector`. */
  private openNumbers(selector: string): void {
    this.numbersOpen = true;
    this.renderNumbersOpen();
    this.el.querySelector(selector)?.scrollIntoView?.({ block: 'start', behavior: 'smooth' });
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    this.el.querySelector('.intro')!.textContent = t('inside.intro', { n: ready.params.toLocaleString(uiLang()) });
    this.lastSummaryStep = -1;
    this.rates.clear();
    this.all.clear();
    this.units = undefined;
    this.explain.setUnits(undefined);
    this.renderUnits();
    this.buildFlow();
    this.buildScan();
    this.build();
  }

  setExpert(on: boolean): void {
    this.expert = on;
    // Plain mode hides the engineering details (rates, shapes, colour scales): see style.css.
    this.el.classList.toggle('plain', !on);
    this.signal.hidden = !on;
    this.overview.hidden = !on;
    this.renderNumbersOpen();
    this.explain.setExpert(on);
    this.renderUnits();
    this.build();
    this.renderSignal();
  }

  /** Matrix names to request: all of them (the scan shows every number). */
  wanted(): string[] {
    return this.ready ? this.ready.matrices.map((m) => m.name) : [];
  }

  /** The tab became visible (or its answer changed while visible): bring "look inside one letter" up to date. */
  shown(): void {
    this.explain.refresh();
    this.askUnits(true);
  }

  /** Ask for the units again: when the tab is shown, or (while learning) every few seconds at most. */
  private askUnits(now = false): void {
    if (!this.requestUnits || this.unitsInFlight) return;
    if (!now && Date.now() - this.unitsAsked < UNITS_EVERY) return;
    this.unitsInFlight = true;
    this.unitsAsked = Date.now();
    this.requestUnits();
  }

  onUnits(msg: UnitsMsg): void {
    this.unitsInFlight = false;
    this.units = msg;
    this.explain.setUnits(msg);
    this.renderUnits();
  }

  private renderUnits(): void {
    const u = this.units;
    this.el.querySelector('.units-help')!.textContent = t('units.helpShort', { n: u?.units.length ?? 64 });
    if (!u) { this.unitsNote.textContent = t('units.wait'); this.unitsBody.replaceChildren(); this.unitsButton.hidden = true; return; }
    const sorted = clearestFirst(u.units);
    const clear = sorted.filter((x) => x.theme);
    const all = this.expert || this.unitsAll || clear.length === 0;
    const shown = all ? sorted : clear.slice(0, UNITS_CLEAR);
    const alone = u.units.filter(matters).length;
    this.unitsNote.textContent = `${t(all ? 'units.countAll' : 'units.count', { k: shown.length, clear: clear.length, n: u.units.length, m: u.moments.toLocaleString(uiLang()) })} ${t('units.step', { step: u.step })}`;
    this.unitsTeam.textContent = u.units.some((x) => x.off) ? t('units.team', { k: alone, n: u.units.length }) : '';
    this.unitsBody.replaceChildren(...shown.map((x) => unitCard(x, (c) => this.langName(c), { expert: this.expert })));
    this.unitsButton.hidden = this.expert || clear.length === 0;
    this.unitsButton.textContent = this.unitsAll ? t('units.showClear') : t('units.showAll', { n: u.units.length });
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
  // "Where is it learning?": the brain scan

  private buildScan(): void {
    if (!this.ready) return;
    this.tiles = [];
    const columns = new Map<string, HTMLElement[]>();
    for (const def of BOXES) {
      const ms = this.ready.matrices.filter((m) => def.match.test(m.name));
      if (!ms.length) continue;
      const layout = scanLayout(ms.map((m) => ({ name: m.name, size: m.rows * m.cols })));
      const canvas = h('canvas', { class: 'scan-canvas' });
      const share = h('span', { class: 'share small' });
      const tile: ScanTile = { def, canvas, layout, share, part: null, hover: null };
      const el = h('button', { class: 'scan-tile', 'data-box': def.id, title: t('scan.click') },
        h('span', { class: 'name' }, t(`flow.${def.id}`)),
        canvas,
        share);
      el.onclick = () => this.openNumbers(def.target);
      canvas.addEventListener('mousemove', (e) => {
        const cell = scanCellAt(layout, e.offsetX, e.offsetY);
        tile.hover = cell ? { piece: cell.piece } : null;
        showTip(e, cell ? this.describeDot(def.id, cell.piece.name, cell.index) : null);
        this.drawTile(tile);
      });
      canvas.addEventListener('mouseleave', (e) => { showTip(e, null); tile.hover = null; this.drawTile(tile); });
      this.tiles.push(tile);
      if (!columns.has(def.column)) columns.set(def.column, []);
      columns.get(def.column)!.push(el);
    }
    const cols = ['letters', 'encoder', 'decoder'].filter((c) => columns.has(c)).map((c) => h('div', { class: `flow-col ${c}` },
      h('h3', {}, t(`inside.group.${c}`)),
      ...columns.get(c)!));
    const parts: HTMLElement[] = [];
    cols.forEach((c, i) => { if (i) parts.push(h('span', { class: 'flow-arrow', 'aria-hidden': 'true' }, '→')); parts.push(c); });
    parts.push(h('span', { class: 'flow-arrow', 'aria-hidden': 'true' }, '→'), h('div', { class: 'flow-end' }, t('flow.end')));
    this.scanFlow.replaceChildren(...parts);
    this.el.querySelector('.scan-help')!.textContent = t('scan.helpShort', { n: this.ready.params.toLocaleString(uiLang()) });
    this.renderScan();
  }

  /** Tooltip of one dot: which part and matrix, what it does, its number. */
  private describeDot(box: string, name: string, index: number): string {
    const info = this.ready!.matrices.find((m) => m.name === name)!;
    const v = this.all.get(name)?.[index];
    const row = Math.floor(index / info.cols), col = index % info.cols;
    const part = t(`flow.${box}`), role = roleLabel(info);
    const lines = [part === role ? part : `${part} — ${role}`, tipText(info, row, col, this.ready!.tokens)];
    if (v !== undefined) lines.push(t(`inside.val.${this.mode}`, { v: this.expert ? fmt(v) : plainFmt(v) }));
    if (this.expert) lines.push(`${name}[${row}, ${col}]`);
    lines.push(t('scan.click'));
    return lines.filter(Boolean).join('\n');
  }

  private drawTile(tile: ScanTile): void {
    drawScan(tile.canvas, tile.layout, this.all, this.scanScale, tile.hover);
  }

  /** New numbers (or a new mode): one brightness scale for every part, shares, the busiest part. */
  private renderScan(): void {
    const values: Float32Array[] = [...this.all.values()];
    const n = values.reduce((a, d) => a + d.length, 0);
    const joined = new Float32Array(n);
    let o = 0;
    for (const d of values) { joined.set(d, o); o += d.length; }
    const any = joined.some((v) => v !== 0);
    this.scanScale = any ? symmetricScale(joined) : 1;
    const sums = this.tiles.map((tile) => {
      let s = 0;
      for (const p of tile.layout.pieces) { const d = this.all.get(p.name); if (d) for (let i = 0; i < d.length; i++) s += Math.abs(d[i]); }
      return s;
    });
    const total = sums.reduce((a, b) => a + b, 0);
    const shares = this.mode !== 'weights' && any && total > 0;
    this.tiles.forEach((tile, i) => {
      tile.part = shares ? sums[i] / total : null;
      const count = t('scan.count', { n: tile.layout.size.toLocaleString(uiLang()) });
      tile.share.textContent = tile.part === null ? count : `${count} · ${t(`scan.share.${this.mode}`, { pct: Math.round(tile.part * 100) })}`;
      this.drawTile(tile);
    });
    this.scanLegend.replaceChildren(
      h('span', { class: 'small' }, t(`scan.low.${this.mode}`)),
      h('span', { class: 'glow-bar', style: `background:${GLOW_GRADIENT}` }),
      h('span', { class: 'small' }, t(`scan.high.${this.mode}`)));
    this.scanVerdict.textContent = shares ? this.verdict(sums, total) : '';
  }

  /**
   * One or two sentences about the scan: how much a part's numbers move on
   * average compared with an average number of the whole model (a part's
   * share of the change mostly reflects its size, so it is not used here).
   */
  private verdict(sums: number[], total: number): string {
    const size = this.tiles.reduce((a, tl) => a + tl.layout.size, 0);
    const ratios = this.tiles.map((tl, i) => (sums[i] / tl.layout.size) / (total / size));
    const name = (i: number) => ({ part: t(`flow.${this.tiles[i].def.id}`), group: t(`inside.group.${this.tiles[i].def.column}`) });
    const x = (r: number) => r.toLocaleString(uiLang(), { maximumFractionDigits: 1, minimumFractionDigits: 1 });
    const hi = ratios.indexOf(Math.max(...ratios)), lo = ratios.indexOf(Math.min(...ratios));
    const out = [ratios[hi] >= BUSY ? t(`scan.busiest.${this.mode}`, { ...name(hi), x: x(ratios[hi]) }) : t(`scan.even.${this.mode}`)];
    if (ratios[lo] <= QUIET) out.push(t(`scan.quietest.${this.mode}`, { ...name(lo), x: x(ratios[lo]) }));
    return out.join(' ');
  }

  // -------------------------------------------------------------------------
  // "How fast each part changes" (expert)

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
      el.onclick = () => this.openNumbers(b.target);
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
    if (this.units && msg.step !== this.units.step) this.askUnits();
    if (msg.mode !== this.mode) return;
    const byName = new Map(msg.matrices.map((m) => [m.name, m.data]));
    this.all = byName;
    this.renderScan();
    for (const c of this.cards) {
      const d = byName.get(c.info.name);
      if (!d) continue;
      c.data = d;
      c.ownScale = symmetricScale(d);
    }
    this.rescale();
    this.modeNote.textContent = this.scanNote.textContent = msg.mode === 'weights' ? t('inside.note.weights', { step: msg.step })
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
