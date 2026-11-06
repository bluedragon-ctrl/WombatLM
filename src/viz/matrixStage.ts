/**
 * The matrix view's picture (Canvas 2D): the model's numbers as a matrix
 * (one row per token, one column per number), with a side panel on the left
 * for what the current station uses: the letter table (letters, guess), the
 * position rows (+ position) or the reading result (look at the animal).
 *
 * One frame is drawn from a station, the station shown before it and an
 * animation time t (0..1): rows fly in from the letter table, position rows
 * fly in and are added, arrows show which rows a row takes from, cells change
 * colour from their old to their new value, and the biggest changes are
 * outlined. Only "how" the numbers move, never "why".
 */
import type { Station } from '../ui/matrixModel';
import { meanAttention } from '../ui/matrixModel';
import { prepare } from './canvas';
import { ACCENT, diverging, INK, MUTED } from './colors';
import { MARK } from './weightHeatmap';

export type ColourMode = 'values' | 'added';

export interface StageFrame {
  station: Station;
  /** Animation time 0..1 (1 = the station's work is done). */
  t: number;
  scale: number;
  mode: ColourMode;
  /** Short label per matrix row and per letter-table row. */
  labels: string[];
  tableLabels: string[];
  /** Raw tokens of the letter table (row i = token number i). */
  tableTokens: readonly string[];
  letterTable: Float32Array;
  /** The position rows used (one per matrix row), for "+ position". */
  posRows: Float32Array;
  /** Reading result and its row labels, for "look at the animal". */
  memory: Float32Array;
  memoryLabels: string[];
  /** Title above the side panel (letter table, position rows, reading result). */
  sideTitle: string;
  pinned: { row: number; col: number } | null;
  hover: { row: number; col: number } | null;
  /** Index into `station.active` of the row whose arrows are drawn. */
  focus: number;
  /** The comment being told now: a speech bubble below the matrix, pointing at what it talks about. */
  callout?: Callout | null;
}

/** What a comment points at: one number of the matrix, a row of the matrix, or a row of the side panel. */
export type Anchor = { area: 'grid'; row: number; col: number } | { area: 'row'; row: number } | { area: 'side'; row: number };

export interface Callout {
  text: string;
  anchor?: Anchor;
}

/** Room kept below the matrix for the speech bubble. */
const BUBBLE_ROOM = 120;
const BUBBLE_W = 560;

export interface StageLayout {
  width: number;
  height: number;
  rows: number;
  d: number;
  top: number;
  sideX: number;
  sideCellW: number;
  sideRowH: number;
  sideRows: number;
  labelX: number;
  gridX: number;
  cellW: number;
  rowH: number;
}

const SIDE_LABEL_W = 26;
const SIDE_CELL_W = 6;
const TABLE_ROW_H = 6;
const GAP = 140;
const LABEL_W = 58;
const CELL_W = 18;
const ROW_H = 22;
const TOP = 34;

export type SideKind = 'table' | 'pos' | 'memory' | 'none';

export function sideKind(st: Station): SideKind {
  if (st.kind === 'letters' || st.kind === 'guess') return 'table';
  if (st.kind === 'position') return 'pos';
  if (st.kind === 'cross') return 'memory';
  return 'none';
}

export function stageLayout(f: Pick<StageFrame, 'station' | 'tableLabels' | 'memoryLabels'>): StageLayout {
  const st = f.station;
  const rows = st.tokens.length;
  const side = sideKind(st);
  const sideRows = side === 'table' ? f.tableLabels.length : side === 'memory' ? f.memoryLabels.length : side === 'pos' ? rows : 0;
  const sideRowH = side === 'table' ? TABLE_ROW_H : ROW_H;
  const sideX = 8 + SIDE_LABEL_W;
  const labelX = sideX + st.d * SIDE_CELL_W + GAP;
  const gridX = labelX + LABEL_W;
  const width = gridX + st.d * CELL_W + 12;
  // Height: never shrink below the letter table, so the picture does not jump between stations.
  const tableH = f.tableLabels.length * TABLE_ROW_H;
  const height = TOP + Math.max(rows * ROW_H + BUBBLE_ROOM, sideRows * sideRowH, tableH) + 14;
  return { width, height, rows, d: st.d, top: TOP, sideX, sideCellW: SIDE_CELL_W, sideRowH, sideRows, labelX, gridX, cellW: CELL_W, rowH: ROW_H };
}

export type StageHit =
  | { area: 'grid'; row: number; col: number }
  | { area: 'label'; row: number }
  | { area: 'side'; row: number; col: number };

export function stageHit(L: StageLayout, x: number, y: number): StageHit | null {
  const r = Math.floor((y - L.top) / L.rowH);
  if (x >= L.gridX && x < L.gridX + L.d * L.cellW && r >= 0 && r < L.rows) {
    return { area: 'grid', row: r, col: Math.floor((x - L.gridX) / L.cellW) };
  }
  if (x >= L.labelX && x < L.gridX && r >= 0 && r < L.rows) return { area: 'label', row: r };
  const sr = Math.floor((y - L.top) / L.sideRowH);
  if (x >= L.sideX && x < L.sideX + L.d * L.sideCellW && sr >= 0 && sr < L.sideRows) {
    return { area: 'side', row: sr, col: Math.floor((x - L.sideX) / L.sideCellW) };
  }
  return null;
}

const ease = (t: number) => (t <= 0 ? 0 : t >= 1 ? 1 : t * t * (3 - 2 * t));
const phase = (t: number, from: number, to: number) => ease((t - from) / (to - from));

/**
 * Colour of a number. Square-root mapping: the small numbers of the letter
 * table stay visible on the same scale as the bigger ones later on (the sign
 * and the order of the numbers are kept).
 */
function colour(v: number, scale: number, alpha = 1): string {
  const x = v / scale;
  const [r, g, b] = diverging(Math.sign(x) * Math.sqrt(Math.min(1, Math.abs(x))));
  return alpha >= 1 ? `rgb(${r},${g},${b})` : `rgba(${r},${g},${b},${alpha})`;
}

/** Cells of the active rows with the biggest change at this station (index row*d+col). */
export function biggestChanges(st: Station, perRow = 4): Set<number> {
  const out = new Set<number>();
  if (!st.before || st.kind === 'guess') return out;
  for (const r of st.active) {
    const deltas = Array.from({ length: st.d }, (_, c) => ({ c, v: Math.abs(st.grid[r * st.d + c] - st.before![r * st.d + c]) }));
    deltas.sort((a, b) => b.v - a.v);
    for (const { c, v } of deltas.slice(0, perRow)) if (v > 1e-6) out.add(r * st.d + c);
  }
  return out;
}

export function drawStage(canvas: HTMLCanvasElement, f: StageFrame): StageLayout {
  const L = stageLayout(f);
  const ctx = prepare(canvas, L.width, L.height);
  const st = f.station;
  const d = st.d;
  const t = f.t;
  const active = new Set(st.active);
  const side = sideKind(st);
  const rowY = (r: number) => L.top + r * L.rowH;
  const sideY = (r: number) => L.top + r * L.sideRowH;

  // How far the cells have changed from "before" to "after": the second half
  // when something has to arrive first (position rows, arrows), else all of t.
  const cellT = st.kind === 'position' || st.kind === 'self' || st.kind === 'cross' ? phase(t, 0.45, 1) : ease(t);

  // --- column numbers -------------------------------------------------------
  ctx.font = '9px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillStyle = MUTED;
  for (let c = 0; c < d; c++) ctx.fillText(String(c + 1), L.gridX + c * L.cellW + L.cellW / 2, L.top - 8);

  // --- side panel -----------------------------------------------------------
  const sideData = side === 'table' ? f.letterTable : side === 'pos' ? f.posRows : side === 'memory' ? f.memory : null;
  const sideLabels = side === 'table' ? f.tableLabels : side === 'memory' ? f.memoryLabels : side === 'pos' ? st.tokens.map((_, i) => String(i + 1)) : [];
  // Rows of the side panel that this station uses.
  const used = new Map<number, number>(); // side row -> strength 0..1
  if (side === 'table' && st.kind === 'letters') {
    for (const r of st.active) used.set(tableRowOf(st.tokens[r], f), 1);
  } else if (side === 'table' && st.guess) {
    const top = topK(st.guess.probs, 5);
    for (const i of top) used.set(i, Math.max(0.25, st.guess.probs[i]));
  } else if (side === 'pos') {
    for (const r of st.active) used.set(r, 1);
  } else if (side === 'memory' && st.attention) {
    for (let k = 0; k < st.attention.keys; k++) used.set(k, meanAttention(st.attention, st.active.length, f.focus, k));
  }
  if (sideData) {
    for (let r = 0; r < L.sideRows; r++) {
      const strength = used.get(r) ?? 0;
      const dim = used.size && strength === 0 ? 0.35 : 1;
      for (let c = 0; c < d; c++) {
        ctx.fillStyle = colour(sideData[r * d + c], f.scale, dim);
        ctx.fillRect(L.sideX + c * L.sideCellW, sideY(r), L.sideCellW - (L.sideRowH > 8 ? 1 : 0), L.sideRowH - (L.sideRowH > 8 ? 2 : 0));
      }
      const label = sideLabels[r] ?? '';
      if (L.sideRowH > 8 || strength > 0) {
        ctx.font = strength > 0 ? 'bold 12px system-ui, sans-serif' : '11px system-ui, sans-serif';
        ctx.textAlign = 'right';
        ctx.fillStyle = strength > 0 ? INK : MUTED;
        ctx.fillText(label, L.sideX - 4, sideY(r) + L.sideRowH / 2);
      }
      if (strength > 0) {
        ctx.strokeStyle = st.guess && st.guess.chosenId === r ? '#15803d' : ACCENT;
        ctx.lineWidth = 2;
        ctx.strokeRect(L.sideX - 1, sideY(r) - 1, d * L.sideCellW + 1, L.sideRowH + 1);
      }
    }
    ctx.font = 'bold 11px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillStyle = MUTED;
    ctx.fillText(f.sideTitle, L.sideX - SIDE_LABEL_W + 4, L.top - 22);
  }

  // --- the matrix -----------------------------------------------------------
  const before = st.before;
  const marks = cellT >= 1 ? biggestChanges(st) : new Set<number>();
  for (let r = 0; r < L.rows; r++) {
    const isActive = active.has(r);
    // Letters: an active row still on its way from the letter table is drawn flying, below.
    if (st.kind === 'letters' && isActive && flight(st, r, t) < 1) continue;
    const alpha = isActive ? 1 : 0.35;
    for (let c = 0; c < d; c++) {
      const i = r * d + c;
      const after = st.grid[i];
      const b = before ? before[i] : after;
      const v = isActive ? b + (after - b) * cellT : after;
      const shown = f.mode === 'added' ? (before ? (isActive ? (after - b) * cellT : after - b) : v) : v;
      ctx.fillStyle = colour(shown, f.scale, alpha);
      ctx.fillRect(L.gridX + c * L.cellW, rowY(r), L.cellW - 1, L.rowH - 2);
    }
    // Row label.
    ctx.textAlign = 'right';
    ctx.font = isActive ? 'bold 14px system-ui, sans-serif' : '13px system-ui, sans-serif';
    ctx.fillStyle = isActive ? INK : MUTED;
    ctx.fillText(f.labels[r] ?? '', L.gridX - 8, rowY(r) + L.rowH / 2 - 1);
    if (isActive && st.active.length > 1 && st.active[f.focus] === r && (st.kind === 'self')) {
      ctx.fillStyle = ACCENT;
      ctx.fillText('▸', L.gridX - 24, rowY(r) + L.rowH / 2 - 1);
    }
  }
  // Active rows framed.
  for (const r of st.active) {
    if (st.kind === 'letters' && flight(st, r, t) < 1) continue;
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(L.gridX - 1.5, rowY(r) - 1.5, d * L.cellW + 2, L.rowH + 1);
  }

  // --- flying rows ----------------------------------------------------------
  if (st.kind === 'letters' && t < 1) {
    for (const r of st.active) {
      const e = flight(st, r, t);
      if (e <= 0 || e >= 1) continue;
      const from = { x: L.sideX, y: sideY(tableRowOf(st.tokens[r], f)), w: L.sideCellW, h: L.sideRowH };
      const to = { x: L.gridX, y: rowY(r), w: L.cellW, h: L.rowH - 2 };
      flyRow(ctx, st.grid.subarray(r * d, (r + 1) * d), from, to, e, f.scale, 1);
    }
  }
  if (st.kind === 'position' && t < 0.6) {
    const e = phase(t, 0, 0.45);
    for (const r of st.active) {
      const from = { x: L.sideX, y: sideY(r), w: L.sideCellW, h: L.sideRowH - 2 };
      const to = { x: L.gridX, y: rowY(r), w: L.cellW, h: L.rowH - 2 };
      flyRow(ctx, f.posRows.subarray(r * d, (r + 1) * d), from, to, e, f.scale, t < 0.45 ? 0.9 : Math.max(0, 1 - (t - 0.45) / 0.15));
    }
  }

  // --- arrows ---------------------------------------------------------------
  if (st.attention && (st.kind === 'self' || st.kind === 'cross')) {
    const show = phase(t, 0, 0.4);
    const qi = Math.min(f.focus, st.active.length - 1);
    const q = st.active[qi];
    const yq = rowY(q) + L.rowH / 2;
    for (let k = 0; k < st.attention.keys; k++) {
      const w = meanAttention(st.attention, st.active.length, qi, k);
      if (w < 0.01) continue;
      ctx.strokeStyle = `rgba(37,99,235,${(0.15 + 0.85 * w) * show})`;
      ctx.lineWidth = 1 + 9 * w;
      ctx.beginPath();
      if (st.kind === 'cross') {
        const yk = sideY(k) + L.sideRowH / 2 - 1;
        const x0 = L.sideX + d * L.sideCellW + 4, x1 = L.labelX + 4;
        ctx.moveTo(x0, yk);
        ctx.bezierCurveTo(x0 + 60, yk, x1 - 60, yq, x1, yq);
      } else {
        const yk = rowY(k) + L.rowH / 2;
        const x = L.labelX + 6;
        const bend = 24 + 100 * Math.abs(k - q) / Math.max(1, L.rows);
        if (k === q) ctx.arc(x - 8, yq, 7, 0.25 * Math.PI, 1.75 * Math.PI);
        else { ctx.moveTo(x, yk); ctx.bezierCurveTo(x - bend, yk, x - bend, yq, x, yq); }
      }
      ctx.stroke();
      // How much, next to where it comes from (on white, so arrows never hide it).
      const text = `${Math.round(w * 100)} %`;
      ctx.font = '11px system-ui, sans-serif';
      const tw = ctx.measureText(text).width;
      let tx: number, ty: number;
      if (st.kind === 'cross') { tx = L.sideX + d * L.sideCellW + 8; ty = sideY(k) + L.sideRowH / 2 - 1; }
      else {
        const bend = 24 + 100 * Math.abs(k - q) / Math.max(1, L.rows);
        tx = L.labelX + (k === q ? -20 : 2 - bend) - tw; ty = rowY(k) + L.rowH / 2 - 1;
      }
      ctx.fillStyle = `rgba(255,255,255,${0.9 * show})`;
      ctx.fillRect(tx - 2, ty - 7, tw + 4, 14);
      ctx.textAlign = 'left';
      ctx.fillStyle = `rgba(30,58,138,${show})`;
      ctx.fillText(text, tx, ty);
    }
  }

  // --- guess: the last row compared with every letter-table row -------------
  if (st.guess) {
    // First the comparison (lines), then the chances (bars).
    const lines = phase(t, 0, 0.5), e = phase(t, 0.5, 1);
    const q = st.active[0];
    const yq = rowY(q) + L.rowH / 2;
    const top = topK(st.guess.probs, 5);
    const x0 = L.sideX + d * L.sideCellW + 4;
    for (const i of top) {
      const p = st.guess.probs[i];
      const yk = sideY(i) + L.sideRowH / 2;
      const chosen = i === st.guess.chosenId;
      ctx.strokeStyle = chosen && e > 0 ? `rgba(21,128,61,${0.3 + 0.7 * e})` : `rgba(37,99,235,${(0.15 + 0.6 * Math.max(p, 0.3)) * lines})`;
      ctx.lineWidth = 1 + 6 * p;
      ctx.beginPath();
      ctx.moveTo(L.labelX + 4, yq);
      ctx.bezierCurveTo(L.labelX - 50, yq, x0 + 60, yk, x0 + 64, yk);
      ctx.stroke();
      // Chance bar and label.
      const bw = 56 * p * e;
      ctx.fillStyle = chosen ? '#15803d' : ACCENT;
      if (e > 0) ctx.fillRect(x0, yk - 4, Math.max(1, bw), 8);
      ctx.font = chosen ? 'bold 11px system-ui, sans-serif' : '11px system-ui, sans-serif';
      ctx.textAlign = 'left';
      ctx.fillStyle = chosen && e > 0 ? '#15803d' : INK;
      ctx.fillText(e > 0 ? `${f.tableLabels[i] ?? '?'} ${(p * 100).toFixed(p < 0.1 ? 1 : 0)} %` : (f.tableLabels[i] ?? '?'), x0 + 66, yk);
    }
  }

  // --- outlines: biggest changes, hover, pinned ------------------------------
  const outline = (r: number, c: number, color: string, w: number) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.strokeRect(L.gridX + c * L.cellW - 0.5, rowY(r) - 0.5, L.cellW, L.rowH - 1);
  };
  for (const i of marks) outline(Math.floor(i / d), i % d, MARK, 2);
  if (f.hover && f.hover.row < L.rows) outline(f.hover.row, f.hover.col, '#6b7280', 1.5);
  if (f.pinned && f.pinned.row < L.rows) {
    outline(f.pinned.row, f.pinned.col, '#111827', 3);
    // Same number in the column header, so it can be found again.
    ctx.fillStyle = '#111827';
    ctx.fillRect(L.gridX + f.pinned.col * L.cellW + 2, L.top - 3, L.cellW - 5, 2);
  }
  if (f.callout) drawCallout(ctx, L, f.callout);
  return L;
}

/** How far active row `r` has flown in from the letter table (0..1): one after another. */
function flight(st: Station, r: number, t: number): number {
  const n = st.active.length, j = st.active.indexOf(r);
  if (j < 0) return 1;
  return phase(t, j / n, (j + 1) / n);
}

/** Where an anchor is on the canvas. */
function anchorPoint(L: StageLayout, a: Anchor): { x: number; y: number } {
  if (a.area === 'grid') return { x: L.gridX + a.col * L.cellW + L.cellW / 2, y: L.top + (a.row + 1) * L.rowH - 2 };
  if (a.area === 'row') return { x: L.gridX - 20, y: L.top + (a.row + 1) * L.rowH - 4 };
  return { x: L.sideX + L.d * L.sideCellW + 2, y: L.top + a.row * L.sideRowH + L.sideRowH / 2 };
}

/** The comment told now: a speech bubble under the matrix with a pointer to what it is about. */
function drawCallout(ctx: CanvasRenderingContext2D, L: StageLayout, c: Callout): void {
  ctx.font = '14px system-ui, sans-serif';
  const words = c.text.split(/\s+/);
  const lines: string[] = [];
  let line = '';
  for (const w of words) {
    const next = line ? `${line} ${w}` : w;
    if (ctx.measureText(next).width > BUBBLE_W - 24 && line) { lines.push(line); line = w; } else line = next;
  }
  if (line) lines.push(line);
  const bh = lines.length * 19 + 16;
  const bx = L.labelX - 30, by = L.top + L.rows * L.rowH + 26;
  if (c.anchor) {
    // A highlighted target and a pointer from the bubble to it.
    const p = anchorPoint(L, c.anchor);
    if (c.anchor.area === 'grid') {
      ctx.strokeStyle = '#15803d';
      ctx.lineWidth = 3;
      ctx.strokeRect(L.gridX + c.anchor.col * L.cellW - 1.5, L.top + c.anchor.row * L.rowH - 1.5, L.cellW + 2, L.rowH + 1);
    }
    const sx = Math.max(bx + 16, Math.min(bx + BUBBLE_W - 16, p.x));
    ctx.strokeStyle = '#15803d';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx, by);
    if (c.anchor.area === 'side') ctx.bezierCurveTo(sx, by - 40, p.x + 60, p.y, p.x, p.y);
    else ctx.lineTo(p.x, p.y + 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(p.x, p.y + (c.anchor.area === 'side' ? 0 : 2), 3, 0, 2 * Math.PI);
    ctx.fillStyle = '#15803d';
    ctx.fill();
  }
  ctx.fillStyle = '#f0fdf4';
  ctx.strokeStyle = '#15803d';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(bx, by, BUBBLE_W, bh, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = '#14532d';
  ctx.textAlign = 'left';
  lines.forEach((l, i) => ctx.fillText(l, bx + 12, by + 17 + i * 19));
}

/** Row of the letter table for a token label (its id = its index in the table). */
function tableRowOf(token: string, f: StageFrame): number {
  return Math.max(0, f.tableTokens.indexOf(token));
}

function topK(p: Float32Array, k: number): number[] {
  return Array.from(p.keys()).sort((a, b) => p[b] - p[a]).slice(0, k);
}

/** A row of d coloured cells, on its way from one rectangle (one cell) to another. */
function flyRow(
  ctx: CanvasRenderingContext2D, values: Float32Array,
  from: { x: number; y: number; w: number; h: number }, to: { x: number; y: number; w: number; h: number },
  e: number, scale: number, alpha: number,
): void {
  const x = from.x + (to.x - from.x) * e, y = from.y + (to.y - from.y) * e;
  const w = from.w + (to.w - from.w) * e, hgt = from.h + (to.h - from.h) * e;
  for (let c = 0; c < values.length; c++) {
    ctx.fillStyle = colour(values[c], scale, alpha);
    ctx.fillRect(x + c * w, y, Math.max(1, w - 1), Math.max(1, hgt));
  }
  ctx.strokeStyle = `rgba(37,99,235,${alpha})`;
  ctx.lineWidth = 1.5;
  ctx.strokeRect(x - 1, y - 1, values.length * w + 1, hgt + 2);
}
