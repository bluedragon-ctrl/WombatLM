/**
 * "Inside the model" tab:
 * - "Where is it learning?": a "brain scan" of all the model's numbers, one
 *   dot each, grouped by part in the order the signal flows, glowing where
 *   they change (or are pushed, or are big); in expert mode a click on a part
 *   opens its heatmaps on "The numbers" tab;
 * - "Look inside one letter" (see explainView.ts);
 * - "What have the detectors learned?": the writer's thinking units, each
 *   with the moments of the learning data that switch it on most (see
 *   train/units.ts and unitCard.ts).
 * The heatmaps of every matrix, "how fast each part changes" and "follow the
 * signal" are on the expert-only "The numbers" tab (numbersTab.ts).
 */
import type { AxisKind } from '../nn/transformer';
import type { AnswerMsg, ExplainMsg, InspectMode, MatrixInfo, ReadyMsg, UnitsMsg, WeightsMsg } from '../train/protocol';
import { clearestFirst, matters, unitCard } from './unitCard';
import { symmetricScale } from '../viz/colors';
import type { HeatmapMarks } from '../viz/weightHeatmap';
import { drawScan, GLOW_GRADIENT, scanCellAt, scanLayout, ScanLayout, ScanPiece } from '../viz/brainScan';
import { h, showTip } from './dom';
import { ExplainView, Question } from './explainView';
import { t, uiLang } from './i18n';

const MODES: InspectMode[] = ['weights', 'delta', 'grad'];

/** A "?" that opens the full explanation (the panel itself keeps one short sentence). */
const help = (text: string) => h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, text));

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

interface ScanTile {
  def: (typeof BOXES)[number];
  el: HTMLElement;
  canvas: HTMLCanvasElement;
  layout: ScanLayout;
  share: HTMLElement;
  /** Share of all the change (or push) in this part, 0..1, or null in "Values" mode. */
  part: number | null;
  hover: { piece: ScanPiece } | null;
}

export class InsideTab {
  readonly el: HTMLElement;
  mode: InspectMode = 'delta';
  private ready?: ReadyMsg;
  private expert = false;
  private modeButtons: HTMLButtonElement[] = [];
  private scanNote: HTMLElement;
  private scanEl: HTMLElement;
  private scanFlow: HTMLElement;
  private scanLegend: HTMLElement;
  private scanVerdict: HTMLElement;
  private tiles: ScanTile[] = [];
  /** The latest numbers of every matrix (in the current mode), for the scan. */
  private all = new Map<string, Float32Array>();
  private scanScale = 1;
  private explain: ExplainView;
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
  /** Expert: a click on a part of the scan opens its numbers on "The numbers" tab (set by the app). */
  onOpenPart?: (selector: string) => void;
  /** The numbers used by the letter in "look inside one letter", for the heatmaps on "The numbers" tab (set by the app). */
  onMarks?: (marks: Map<string, HeatmapMarks>) => void;

  constructor(private onModeChange: () => void, requestExplain: (q: Question, position: number) => void) {
    this.scanNote = h('span', { class: 'small', 'data-test': 'scan-note' });
    const modes = h('div', { class: 'picker', role: 'group', 'aria-label': t('inside.mode') });
    for (const m of MODES) {
      const b = h('button', { 'data-mode': m }, t(`inside.mode.${m}`));
      b.onclick = () => { this.mode = m; this.renderModeButtons(); this.onModeChange(); };
      modes.append(b);
      this.modeButtons.push(b);
    }
    this.scanFlow = h('div', { class: 'flow scan-flow' });
    this.scanLegend = h('div', { class: 'legend scan-legend' });
    this.scanVerdict = h('p', { class: 'scan-verdict', 'data-test': 'scan-verdict' });
    this.scanEl = h('section', { class: 'panel scan', 'data-test': 'scan' },
      h('h2', {}, t('scan.title')),
      h('p', { class: 'small' }, h('span', { class: 'scan-help' }), ' ', help(t('scan.help'))),
      h('div', { class: 'row' }, modes, this.scanNote),
      this.scanFlow,
      this.scanLegend,
      this.scanVerdict,
      h('p', { class: 'small expert-only scan-expert' }, t('scan.expertNote')));
    this.explain = new ExplainView(requestExplain, (marks) => this.onMarks?.(marks), (code) => this.langName(code));
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
    this.el = h('div', { class: 'inside plain' },
      h('section', { class: 'panel inside-head' },
        h('p', { class: 'intro', 'data-test': 'inside-intro' }),
        h('p', { class: 'small' }, t('inside.guide'))),
      this.scanEl,
      this.explain.el,
      this.unitsEl);
    this.renderModeButtons();
  }

  /** Switch what the scan shows (e.g. "Change" from the learning-cycle link on the Model learning tab). */
  setMode(mode: InspectMode): void {
    this.mode = mode;
    this.renderModeButtons();
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    this.el.querySelector('.intro')!.textContent = t('inside.intro', { n: ready.params.toLocaleString(uiLang()) });
    this.all.clear();
    this.units = undefined;
    this.explain.setUnits(undefined);
    this.renderUnits();
    this.buildScan();
  }

  setExpert(on: boolean): void {
    this.expert = on;
    // Plain mode hides the engineering details: see style.css.
    this.el.classList.toggle('plain', !on);
    for (const tile of this.tiles) tile.el.title = on ? t('scan.click') : '';
    this.explain.setExpert(on);
    this.renderUnits();
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
      // Expert: a click opens this part's numbers on "The numbers" tab. Plain mode has no such tab.
      const el = h('button', { class: 'scan-tile', 'data-box': def.id, title: this.expert ? t('scan.click') : '' },
        h('span', { class: 'name' }, t(`flow.${def.id}`)),
        canvas,
        share);
      const tile: ScanTile = { def, el, canvas, layout, share, part: null, hover: null };
      el.onclick = () => { if (this.expert) this.onOpenPart?.(def.target); };
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
    if (this.expert) lines.push(t('scan.click'));
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

  onWeights(msg: WeightsMsg): void {
    if (this.units && msg.step !== this.units.step) this.askUnits();
    if (msg.mode !== this.mode) return;
    this.all = new Map(msg.matrices.map((m) => [m.name, m.data]));
    this.renderScan();
    this.scanNote.textContent = modeNote(msg);
  }

  setAnswer(a: AnswerMsg | undefined): void {
    this.explain.setAnswer(a);
  }

  private renderModeButtons(): void {
    for (const b of this.modeButtons) b.classList.toggle('on', b.dataset.mode === this.mode);
  }
}

/** The line next to the mode buttons: which step, or how many steps the change covers. */
export function modeNote(msg: WeightsMsg): string {
  return msg.mode === 'weights' ? t('inside.note.weights', { step: msg.step })
    : msg.steps === 0 ? t(`inside.note.${msg.mode}None`)
      : msg.mode === 'delta' ? t(msg.steps === 1 ? 'inside.note.delta1' : 'inside.note.delta', { n: msg.steps }) : t('inside.note.grad', { step: msg.step });
}

export function fmt(v: number): string {
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
