/**
 * Matrix tab: the answer on the Ask tab, shown as the matrix of numbers it
 * really is inside the model (one row per token, one column per number),
 * station by station. Rows fly in from the letter table, position rows are
 * added, arrows show which rows a row takes from, every square changes colour
 * from its old to its new value, and one clicked number can be followed all
 * the way. It shows how the numbers move and change, not why.
 *
 * The stations come from `matrixModel.ts`; the picture from `viz/matrixStage.ts`.
 */
import type { AnswerMsg } from '../train/protocol';
import { buildMatrixModel, meanAttention, MatrixModel, Station, StationKind } from './matrixModel';
import { ColourMode, drawStage, sideKind, StageHit, stageHit, StageLayout } from '../viz/matrixStage';
import { DIVERGING_ENDS } from '../viz/colors';
import { h, showTip } from './dom';
import { t } from './i18n';
import { shortLabel, tokenName } from './tokens';
import { EOS } from '../nn/vocab';

/** Animation: how long one station takes, and the pause before Play goes on. */
const STATION_MS = 1100;
const PAUSE_MS = 700;

export class MatrixTab {
  readonly el: HTMLElement;
  private a?: AnswerMsg;
  private m?: MatrixModel;
  private tokens: string[] = [];
  private expert = false;
  private visible = false;
  private key = '';
  private idx = 0;
  /** Animation time of the current station, 0..1. */
  private time = 1;
  private animStart = 0;
  private raf = 0;
  private playing = false;
  private playTimer = 0;
  private mode: ColourMode = 'values';
  private pinned: { lane: 'read' | 'write'; row: number; col: number } | null = null;
  private hover: { row: number; col: number } | null = null;
  /** Reading, "look at each other": whose arrows are drawn (index into the active rows). */
  private focus = -1;
  private layout?: StageLayout;

  private empty: HTMLElement;
  private body: HTMLElement;
  private question: HTMLElement;
  private passes: HTMLElement;
  private stationPills: HTMLElement;
  private caption: HTMLElement;
  private canvas: HTMLCanvasElement;
  private trail: HTMLElement;
  private back: HTMLButtonElement;
  private next: HTMLButtonElement;
  private play: HTMLButtonElement;
  private modeButtons: HTMLButtonElement[];

  constructor(private langName: (code: string) => string, goAsk: () => void) {
    const ask = h('button', { class: 'primary', 'data-test': 'matrix-go-ask' }, t('mx.goAsk'));
    ask.onclick = goAsk;
    this.empty = h('div', { class: 'mx-empty', 'data-test': 'matrix-empty' }, h('p', {}, t('mx.empty')), ask);
    this.question = h('div', { class: 'mx-question', 'data-test': 'matrix-question' });
    this.back = h('button', { 'data-test': 'matrix-back' }, t('jr.back'));
    this.next = h('button', { class: 'primary', 'data-test': 'matrix-next' }, t('jr.next'));
    this.play = h('button', { 'data-test': 'matrix-play' }, t('mx.play'));
    this.back.onclick = () => { this.stop(); this.go(this.idx - 1, false); };
    this.next.onclick = () => { this.stop(); this.go(this.idx + 1, true); };
    this.play.onclick = () => (this.playing ? this.stop() : this.start());
    this.modeButtons = (['values', 'added'] as ColourMode[]).map((mode) => {
      const b = h('button', { class: `seg${mode === this.mode ? ' on' : ''}`, 'data-mode': mode, 'data-test': `matrix-mode-${mode}` }, t(`mx.mode.${mode}`));
      b.onclick = () => { this.mode = mode; for (const x of this.modeButtons) x.classList.toggle('on', x.dataset.mode === mode); this.draw(); };
      return b;
    });
    this.passes = h('div', { class: 'jr-pills mx-passes', role: 'group', 'aria-label': t('mx.passes') });
    this.stationPills = h('div', { class: 'jr-pills mx-stations', role: 'group', 'aria-label': t('mx.stations') });
    this.caption = h('div', { class: 'jr-caption mx-caption', 'data-test': 'matrix-caption', 'aria-live': 'polite' });
    this.canvas = h('canvas', { class: 'mx-canvas', 'data-test': 'matrix-canvas' });
    this.trail = h('div', { class: 'mx-trail', 'data-test': 'matrix-trail' });
    this.canvas.addEventListener('mousemove', (e) => this.onMove(e));
    this.canvas.addEventListener('mouseleave', (e) => { this.hover = null; showTip(e, null); this.draw(); });
    this.canvas.addEventListener('click', (e) => this.onClick(e));
    const legend = h('div', { class: 'mx-legend small' },
      h('span', { class: 'mx-swatch', style: `background:${DIVERGING_ENDS.neg}` }), t('mx.legend.neg'),
      h('span', { class: 'mx-swatch', style: 'background:#fff' }), t('mx.legend.zero'),
      h('span', { class: 'mx-swatch', style: `background:${DIVERGING_ENDS.pos}` }), t('mx.legend.pos'),
      h('span', { class: 'mx-swatch mark' }), t('mx.legend.mark'),
      h('span', { class: 'mx-swatch pin' }), t('mx.legend.pin'));
    this.body = h('div', { class: 'mx-body', hidden: true },
      this.question,
      h('div', { class: 'row jr-controls' }, this.back, this.next, this.play,
        h('span', { class: 'mx-mode small' }, t('mx.mode'), ...this.modeButtons)),
      h('div', { class: 'mx-pillrow' }, h('span', { class: 'small' }, t('mx.passes')), this.passes),
      h('div', { class: 'mx-pillrow' }, h('span', { class: 'small' }, t('mx.stations')), this.stationPills),
      this.caption,
      h('div', { class: 'mx-canvas-wrap' }, this.canvas),
      legend,
      this.trail);
    this.el = h('section', { class: 'panel matrix', 'data-test': 'matrix' },
      h('div', { class: 'phase-head' }, h('h2', {}, t('mx.title')),
        h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, t('mx.help')), h('p', {}, t('mx.helpMore')))),
      h('p', { class: 'small' }, t('mx.intro')),
      this.empty,
      this.body);
  }

  setTokens(tokens: string[]): void {
    this.tokens = tokens;
    if (this.a) this.setAnswer(this.a);
  }

  setExpert(on: boolean): void {
    this.expert = on;
    this.render();
  }

  setAnswer(a: AnswerMsg | undefined): void {
    this.a = a;
    this.m = a && this.tokens.length ? buildMatrixModel(a, this.tokens) : undefined;
    const key = a ? `${a.text}|${a.lang}|${a.form}|${a.output}` : '';
    if (key !== this.key) {
      // Another question (or another answer): start from the beginning.
      this.key = key;
      this.idx = 0;
      this.time = 0;
      this.pinned = null;
      this.focus = -1;
      this.stop();
    }
    if (this.m) this.idx = Math.min(this.idx, this.m.stations.length - 1);
    this.render();
    if (this.visible && this.time < 1) this.animate();
  }

  /** The tab became visible or hidden: draw only while visible. */
  setVisible(on: boolean): void {
    this.visible = on;
    if (!on) { this.stop(); cancelAnimationFrame(this.raf); this.time = 1; return; }
    this.render();
    if (this.time < 1) this.animate();
  }

  // -------------------------------------------------------------------------

  private station(): Station | undefined {
    return this.m?.stations[this.idx];
  }

  private go(idx: number, animate: boolean): void {
    if (!this.m) return;
    const to = Math.max(0, Math.min(this.m.stations.length - 1, idx));
    if (to === this.idx && !animate) return;
    const lanePrev = this.station()?.lane;
    this.idx = to;
    if (this.station()?.lane !== lanePrev) this.focus = -1;
    this.time = animate ? 0 : 1;
    this.render();
    if (animate) this.animate();
  }

  private animate(): void {
    cancelAnimationFrame(this.raf);
    this.animStart = performance.now() - this.time * STATION_MS;
    const frame = (now: number) => {
      this.time = Math.min(1, (now - this.animStart) / STATION_MS);
      this.draw();
      if (this.time < 1) this.raf = requestAnimationFrame(frame);
      else this.renderTrail();
    };
    this.raf = requestAnimationFrame(frame);
  }

  private start(): void {
    if (!this.m) return;
    if (this.idx >= this.m.stations.length - 1) this.go(0, true);
    else if (this.time >= 1) this.go(this.idx + 1, true);
    this.playing = true;
    this.play.textContent = t('mx.stop');
    this.play.classList.add('on');
    const tick = () => {
      if (!this.playing || !this.m) return;
      if (this.time < 1) { this.playTimer = window.setTimeout(tick, 100); return; }
      if (this.idx >= this.m.stations.length - 1) { this.stop(); return; }
      this.playTimer = window.setTimeout(() => {
        if (!this.playing) return;
        this.go(this.idx + 1, true);
        tick();
      }, PAUSE_MS);
    };
    tick();
  }

  private stop(): void {
    this.playing = false;
    clearTimeout(this.playTimer);
    this.play.textContent = t('mx.play');
    this.play.classList.remove('on');
  }

  // -------------------------------------------------------------------------

  private render(): void {
    const a = this.a, m = this.m, st = this.station();
    this.empty.hidden = !!a;
    this.body.hidden = !a;
    if (!a || !m || !st) return;
    this.question.replaceChildren(
      h('b', {}, a.text), h('span', { class: 'op' }, '→'), h('b', {}, a.output || '∅'),
      h('span', { class: 'small' }, ` · ${this.full(a.input[0])} · ${t('mx.atStep', { n: a.step.toLocaleString() })}`));
    this.back.disabled = this.idx === 0;
    this.next.disabled = this.idx === m.stations.length - 1;

    // Passes: reading, then one per written token.
    const passNames = [t('mx.pass.read'), ...a.steps.map((s, k) => t('jr.p.write', { k: k + 1, tok: shortLabel(s.chosen) }))];
    const curPass = st.pass + 1;
    this.passes.replaceChildren(...passNames.map((n, i) => {
      const b = h('button', { class: `jr-pill${i === curPass ? ' on' : ''}${i < curPass ? ' done' : ''}${i > 0 ? ' write' : ''}`, 'data-pass': String(i) }, n);
      b.onclick = () => { this.stop(); this.go(m.passStart[i], true); };
      return b;
    }));
    const start = m.passStart[curPass];
    const end = curPass + 1 < m.passStart.length ? m.passStart[curPass + 1] : m.stations.length;
    this.stationPills.replaceChildren(...m.stations.slice(start, end).map((s, j) => {
      const i = start + j;
      const b = h('button', { class: `jr-pill${i === this.idx ? ' on' : ''}${i < this.idx ? ' done' : ''}`, 'data-station': s.kind }, `${j + 1}. ${this.stationName(s)}`);
      b.onclick = () => { this.stop(); this.go(i, true); };
      return b;
    }));
    this.caption.replaceChildren(...this.captionFor(st));
    this.draw();
    this.renderTrail();
  }

  private stationName(s: Station): string {
    const layer = this.layerCount(s) > 1 && (s.kind === 'self' || s.kind === 'cross' || s.kind === 'think') ? ` ${s.layer + 1}` : '';
    const key = s.kind === 'self' && s.lane === 'write' ? 'mx.st.back' : `mx.st.${s.kind}`;
    return t(key as 'mx.st.letters') + layer;
  }

  private layerCount(s: Station): number {
    return (s.lane === 'read' ? this.m!.laneKinds.read : this.m!.laneKinds.write).filter((x) => x.kind === 'think').length;
  }

  private full(token: string): string {
    return tokenName(token || EOS, { expert: this.expert, langName: this.langName });
  }

  /** What happens at this station, in words, with the real values. */
  private captionFor(st: Station): (HTMLElement | string)[] {
    const a = this.a!, m = this.m!;
    const laneStations = st.lane === 'read' ? m.laneKinds.read.length : m.laneKinds.write.length;
    const head = st.lane === 'read'
      ? t('mx.c.readHead', { i: st.laneIndex + 1, n: laneStations, name: this.stationName(st) })
      : t('mx.c.writeHead', { k: st.pass + 1, n: a.steps.length, i: st.laneIndex + 1, s: laneStations, name: this.stationName(st) });
    const row = st.active[st.active.length - 1];
    const tok = st.tokens[row];
    const id = this.tokens.indexOf(tok);
    const vars = { n: st.tokens.length, d: st.d, tok: this.full(tok), id, pos: row + 1, v: a.vocabSize, top: this.topText(st) };
    let body: string;
    switch (st.kind) {
      case 'letters':
        body = st.lane === 'read'
          ? t('mx.c.read.letters', { ...vars, first: this.full(st.tokens[1] ?? tok), firstId: this.tokens.indexOf(st.tokens[1] ?? tok) })
          : t(st.pass === 0 ? 'mx.c.write.lettersFirst' : 'mx.c.write.letters', vars);
        break;
      case 'position': body = t(st.lane === 'read' ? 'mx.c.read.position' : 'mx.c.write.position', vars); break;
      case 'self': body = st.lane === 'read'
        ? t('mx.c.read.self', { ...vars, tok: this.full(st.tokens[st.active[this.focusIndex(st)]]) })
        : t(st.pass === 0 ? 'mx.c.write.selfFirst' : 'mx.c.write.self', vars); break;
      case 'cross': body = t('mx.c.write.cross', vars); break;
      case 'think': body = t(st.lane === 'read' ? 'mx.c.read.think' : 'mx.c.write.think', vars); break;
      case 'knobs': body = t(st.lane === 'read' ? 'mx.c.read.knobs' : 'mx.c.write.knobs', vars); break;
      case 'guess': {
        const g = st.guess!;
        const p = Math.round((g.probs[g.chosenId] ?? 0) * 100);
        body = g.chosen === EOS
          ? t('mx.c.write.guessEnd', { ...vars, p, word: a.output || '∅' })
          : t('mx.c.write.guess', { ...vars, chosen: this.full(g.chosen), p, next: st.pass + 2 });
        break;
      }
    }
    const parts: (HTMLElement | string)[] = [h('strong', {}, head), h('span', {}, body)];
    if (st.kind === 'think' || st.kind === 'knobs' || st.kind === 'position' || st.kind === 'self' || st.kind === 'cross') parts.push(h('span', { class: 'small' }, t('mx.c.marks')));
    return parts;
  }

  private focusIndex(st: Station): number {
    return this.focus >= 0 && this.focus < st.active.length ? this.focus : st.active.length - 1;
  }

  /** "t 52 % · y 16 % · …": where the arrows of the focused row go. */
  private topText(st: Station): string {
    const at = st.attention;
    if (!at) return '';
    const qi = this.focusIndex(st);
    const keys = at.from === 'memory' ? this.m!.inputTokens : st.tokens;
    const ws = Array.from({ length: at.keys }, (_, k) => ({ k, w: meanAttention(at, st.active.length, qi, k) }))
      .sort((x, y) => y.w - x.w).slice(0, 3);
    return ws.map(({ k, w }) => `${shortLabel(keys[k])} ${Math.round(w * 100)} %`).join(' · ');
  }

  private sideTitle(st: Station): string {
    switch (sideKind(st)) {
      case 'table': return t('mx.side.table', { v: this.a!.vocabSize });
      case 'pos': return t('mx.side.pos');
      case 'memory': return t('mx.side.memory');
      default: return '';
    }
  }

  private draw(): void {
    const a = this.a, m = this.m, st = this.station();
    if (!a || !m || !st || !this.visible) return;
    const d = st.d;
    const posRows = st.lane === 'read' ? a.lookup.encPos : a.lookup.decPos.slice(0, st.tokens.length * d);
    const pin = this.pinned && this.pinned.lane === st.lane ? { row: this.pinned.row, col: this.pinned.col } : null;
    this.layout = drawStage(this.canvas, {
      station: st, t: this.time, scale: st.lane === 'read' ? m.scale.read : m.scale.write, mode: this.mode,
      labels: st.tokens.map(shortLabel), tableLabels: this.tokens.map(shortLabel), tableTokens: this.tokens,
      letterTable: a.letterTable, posRows, memory: m.memory, memoryLabels: m.inputTokens.map(shortLabel),
      sideTitle: this.sideTitle(st), pinned: pin, hover: this.hover, focus: this.focusIndex(st),
    });
  }

  private hit(e: MouseEvent): StageHit | null {
    return this.layout ? stageHit(this.layout, e.offsetX, e.offsetY) : null;
  }

  private onMove(e: MouseEvent): void {
    const st = this.station(), hit = this.hit(e), m = this.m;
    if (!st || !m) return;
    const prev = this.hover;
    this.hover = hit?.area === 'grid' ? { row: hit.row, col: hit.col } : null;
    if (prev?.row !== this.hover?.row || prev?.col !== this.hover?.col) this.draw();
    if (!hit) { showTip(e, null); return; }
    const d = st.d;
    if (hit.area === 'grid') {
      const i = hit.row * d + hit.col;
      const now = st.grid[i];
      const lines = [t('mx.tip.cell', { tok: this.full(st.tokens[hit.row]), n: hit.col + 1 })];
      if (st.before && st.kind !== 'guess') {
        const b = st.before[i];
        lines.push(t('mx.tip.change', { before: fmt(b), now: fmt(now), diff: (now - b >= 0 ? '+' : '') + fmt(now - b) }));
      } else lines.push(t('mx.tip.value', { now: fmt(now) }));
      if (!st.active.includes(hit.row)) lines.push(t('mx.tip.remembered'));
      lines.push(t('mx.tip.pin'));
      showTip(e, lines.join('\n'));
    } else if (hit.area === 'label') {
      showTip(e, st.kind === 'self' && st.active.length > 1 ? t('mx.tip.focus', { tok: this.full(st.tokens[hit.row]) }) : this.full(st.tokens[hit.row]));
    } else {
      const side = sideKind(st);
      const v = side === 'table' ? this.a!.letterTable[hit.row * d + hit.col]
        : side === 'memory' ? m.memory[hit.row * d + hit.col]
        : (st.lane === 'read' ? this.a!.lookup.encPos : this.a!.lookup.decPos)[hit.row * d + hit.col];
      const who = side === 'table' ? t('mx.tip.tableRow', { id: hit.row, tok: this.full(this.tokens[hit.row]) })
        : side === 'memory' ? t('mx.tip.memoryRow', { tok: this.full(m.inputTokens[hit.row]) })
        : t('mx.tip.posRow', { n: hit.row + 1 });
      let extra = '';
      if (side === 'table' && st.guess) extra = `\n${t('mx.tip.score', { s: fmt(st.guess.scores[hit.row]), p: (st.guess.probs[hit.row] * 100).toFixed(1) })}`;
      showTip(e, `${who}, ${t('axis.dim', { n: hit.col + 1 })}: ${fmt(v ?? 0)}${extra}`);
    }
  }

  private onClick(e: MouseEvent): void {
    const st = this.station(), hit = this.hit(e);
    if (!st || !hit) return;
    if (hit.area === 'grid') {
      const p = this.pinned;
      this.pinned = p && p.lane === st.lane && p.row === hit.row && p.col === hit.col ? null : { lane: st.lane, row: hit.row, col: hit.col };
      this.draw();
      this.renderTrail();
    } else if (hit.area === 'label' && st.kind === 'self' && st.active.length > 1) {
      this.focus = st.active.indexOf(hit.row);
      this.caption.replaceChildren(...this.captionFor(st));
      this.time = Math.max(this.time, 0.999);
      this.draw();
    }
  }

  /** Follow one number: its value at every station of its lane. */
  private renderTrail(): void {
    const m = this.m, st = this.station();
    if (!m || !st) { this.trail.replaceChildren(); return; }
    const p = this.pinned;
    if (!p || p.lane !== st.lane) {
      this.trail.replaceChildren(h('span', { class: 'small' }, t('mx.trail.hint')));
      return;
    }
    const kinds = p.lane === 'read' ? m.laneKinds.read : m.laneKinds.write;
    const values = m.trail(p.lane, p.row, p.col);
    const token = p.lane === 'read' ? m.inputTokens[p.row] : (st.tokens[p.row] ?? this.writeToken(p.row));
    const notYet = p.lane === 'write' && p.row > st.pass;
    const items: HTMLElement[] = [];
    kinds.forEach((k, i) => {
      if (k.kind === 'guess') return;
      const here = i === st.laneIndex || (st.kind === 'guess' && k.kind === 'knobs');
      const name = this.stationName({ ...st, kind: k.kind as StationKind, layer: k.layer, lane: p.lane });
      const v = values[i];
      const changed = i > 0 ? v - values[i - 1] : 0;
      const b = h('button', { class: `mx-trail-step${here ? ' on' : ''}`, title: i > 0 ? t('mx.trail.changed', { diff: (changed >= 0 ? '+' : '') + fmt(changed) }) : '' },
        h('small', {}, name), h('b', {}, fmt(v)));
      b.onclick = () => {
        this.stop();
        const pass = p.lane === 'read' ? 0 : Math.max(p.row, st.pass) + 1;
        this.go(m.passStart[pass] + i, true);
      };
      if (items.length) items.push(h('span', { class: 'op' }, '→'));
      items.push(b);
    });
    this.trail.replaceChildren(
      h('div', { class: 'small' }, t('mx.trail.head', { tok: this.full(token), n: p.col + 1 }),
        notYet ? ` ${t('mx.trail.notYet')}` : '',
        ' ', Object.assign(h('button', { class: 'link' }, t('mx.trail.clear')), { onclick: () => { this.pinned = null; this.draw(); this.renderTrail(); } })),
      h('div', { class: 'mx-trail-steps' }, ...items));
  }

  private writeToken(row: number): string {
    const a = this.a!;
    return row === 0 ? a.input[0] : a.steps[row - 1]?.chosen || EOS;
  }
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a === 0) return '0';
  return a >= 10 ? v.toFixed(1) : v.toFixed(2);
}
