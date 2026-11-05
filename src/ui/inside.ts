/**
 * "Inside the brain" tab: a heatmap per weight matrix (weights, their
 * change in the last steps, or the gradient), and in expert mode
 * "follow the signal" for the current word.
 */
import type { AxisKind } from '../nn/transformer';
import type { AnswerMsg, InspectMode, MatrixInfo, ReadyMsg, WeightsMsg } from '../train/protocol';
import { DIVERGING_ENDS, symmetricScale } from '../viz/colors';
import { drawHeatmap, heatmapCellAt, heatmapLayout, HeatmapLayout } from '../viz/weightHeatmap';
import { h, showTip } from './dom';
import { t, uiLang } from './i18n';

const MODES: InspectMode[] = ['weights', 'delta', 'grad'];

/** Which part of the brain a matrix belongs to, for grouping. */
function group(name: string): string {
  if (/^enc(\d|Norm)/.test(name)) return 'encoder';
  if (/^dec(\d|Norm)/.test(name)) return 'decoder';
  if (name.startsWith('copy')) return 'copy';
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
  layout: HeatmapLayout;
  data: Float32Array | null;
  scale: number;
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

  constructor(private onModeChange: () => void) {
    this.modeNote = h('span', { class: 'small', 'data-test': 'mode-note' });
    const modes = h('div', { class: 'picker', role: 'group', 'aria-label': t('inside.mode') });
    for (const m of MODES) {
      const b = h('button', { 'data-mode': m }, t(`inside.mode.${m}`));
      b.onclick = () => { this.mode = m; this.renderModeButtons(); this.onModeChange(); };
      modes.append(b);
      this.modeButtons.push(b);
    }
    const legend = h('div', { class: 'legend' },
      h('span', { class: 'swatch-box', style: `background:${DIVERGING_ENDS.neg}` }), t('inside.negative'),
      h('span', { class: 'swatch-box', style: 'background:#fff' }), t('inside.zero'),
      h('span', { class: 'swatch-box', style: `background:${DIVERGING_ENDS.pos}` }), t('inside.positive'));
    this.body = h('div', { class: 'inside-body', 'data-test': 'heatmaps' });
    this.signal = h('section', { class: 'panel signal', hidden: true, 'data-test': 'signal' });
    this.el = h('div', { class: 'inside' },
      h('section', { class: 'panel inside-head' },
        h('p', { class: 'intro', 'data-test': 'inside-intro' }),
        h('div', { class: 'row' }, modes, this.modeNote),
        h('p', { class: 'small' }, t(`inside.help`)),
        legend),
      this.body,
      this.signal);
    this.renderModeButtons();
  }

  /** Switch what the heatmaps show (e.g. "Change" from the learning-cycle link on the Forms tab). */
  setMode(mode: InspectMode): void {
    this.mode = mode;
    this.renderModeButtons();
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    this.el.querySelector('.intro')!.textContent = t('inside.intro', { n: ready.params.toLocaleString(uiLang()) });
    this.build();
  }

  setExpert(on: boolean): void {
    this.expert = on;
    this.signal.hidden = !on;
    this.build();
    this.renderSignal();
  }

  /** Matrix names to request (only visible ones). */
  wanted(): string[] {
    return this.cards.map((c) => c.info.name);
  }

  private visible(m: MatrixInfo): boolean {
    return this.expert || m.rows > 1;
  }

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
    // Letters and copying are small: they share the first row.
    const order = ['letters', 'copy', 'encoder', 'decoder'];
    const sorted = [...groups].sort((a, b) => order.indexOf(a[0]) - order.indexOf(b[0]));
    this.body.replaceChildren(...sorted.map(([g, parts]) => h('section', { class: `panel group ${g}` },
      h('h2', {}, t(`inside.group.${g}`)),
      h('p', { class: 'small' }, t(`inside.groupHelp.${g}`)),
      h('div', { class: 'parts' }, ...[...parts].map(([p, cards]) => h('div', { class: 'part' },
        p ? h('h3', {}, t(`inside.part.${p}`)) : null,
        h('div', { class: 'cards' }, ...cards)))))));
    for (const c of this.cards) this.draw(c);
  }

  private card(info: MatrixInfo, prev?: Card): HTMLElement {
    const canvas = h('canvas', { class: 'heat', 'data-matrix': info.name });
    const scaleLabel = h('span', { class: 'scale' });
    const c: Card = { info, canvas, scaleLabel, layout: heatmapLayout(info.rows, info.cols), data: prev?.data ?? null, scale: prev?.scale ?? 1 };
    this.cards.push(c);
    canvas.addEventListener('mousemove', (e) => {
      const cell = heatmapCellAt(c.layout, e.offsetX, e.offsetY);
      showTip(e, cell && c.data ? this.describe(c, cell.row, cell.col) : null);
      this.draw(c, cell);
    });
    canvas.addEventListener('mouseleave', (e) => { showTip(e, null); this.draw(c); });
    const label = h('div', { class: 'label' }, roleLabel(info));
    if (this.expert) label.append(h('code', {}, ` ${info.name}`));
    return h('figure', { class: 'card' }, label, canvas,
      h('figcaption', { class: 'small' }, `${info.rows} × ${info.cols}`, scaleLabel));
  }

  private describe(c: Card, row: number, col: number): string {
    const v = c.data![row * c.info.cols + col];
    const r = axisText(c.info.rowKind, row, this.ready!.tokens), k = axisText(c.info.colKind, col, this.ready!.tokens);
    // A layer's matrix connects an incoming number (row) to an outgoing one (column).
    const layer = /^(dim|hidden)$/.test(c.info.rowKind) && /^(dim|hidden|none)$/.test(c.info.colKind);
    const where = layer ? `${r} → ${k || t('axis.single')}` : [r, k].filter(Boolean).join(', ');
    return `${roleLabel(c.info)} — ${where}: ${fmt(v)}`;
  }

  private draw(c: Card, highlight: { row: number; col: number } | null = null): void {
    drawHeatmap(c.canvas, c.data ?? new Float32Array(c.info.rows * c.info.cols), c.layout, c.scale, highlight);
  }

  onWeights(msg: WeightsMsg): void {
    if (msg.mode !== this.mode) return;
    const byName = new Map(msg.matrices.map((m) => [m.name, m.data]));
    for (const c of this.cards) {
      const d = byName.get(c.info.name);
      if (!d) continue;
      c.data = d;
      c.scale = symmetricScale(d);
      c.scaleLabel.textContent = d.some((v) => v !== 0) ? ` · ±${fmt(c.scale)}` : '';
      this.draw(c);
    }
    this.modeNote.textContent = msg.mode === 'weights' ? t('inside.note.weights', { step: msg.step })
      : msg.steps === 0 ? t(`inside.note.${msg.mode}None`)
        : msg.mode === 'delta' ? t(msg.steps === 1 ? 'inside.note.delta1' : 'inside.note.delta', { n: msg.steps }) : t('inside.note.grad', { step: msg.step });
  }

  setAnswer(a: AnswerMsg | undefined): void {
    this.answer = a;
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
    const decIn = ['✱', ...a.steps.map((s) => s.chosen || '⏎')];
    const strips = a.activations.map((act) => {
      const labels = (act.name.startsWith('enc') ? input : decIn).slice(0, act.T);
      const layout = heatmapLayout(act.T, act.d, 12);
      const canvas = h('canvas', { class: 'heat' });
      const scale = symmetricScale(act.data);
      drawHeatmap(canvas, act.data, layout, scale);
      canvas.addEventListener('mousemove', (e) => {
        const cell = heatmapCellAt(layout, e.offsetX, e.offsetY);
        showTip(e, cell ? `'${labels[cell.row]}', ${t('axis.dim', { n: cell.col + 1 })}: ${fmt(act.data[cell.row * act.d + cell.col])}` : null);
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

/** Plain-language name of a matrix (technical name in expert mode is added by the caller). */
export function roleLabel(m: MatrixInfo): string {
  if (m.name === 'encPos') return t('role.encPos');
  if (m.name === 'decPos') return t('role.decPos');
  const key = m.role.endsWith('.bias') ? 'role.bias' : `role.${m.role}`;
  const s = t(key);
  return s === key ? m.role : s;
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
