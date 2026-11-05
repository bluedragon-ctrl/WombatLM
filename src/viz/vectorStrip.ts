/**
 * Vector strip: one list of numbers (e.g. the 32 numbers a letter carries)
 * as a row of coloured squares, optionally with the value written inside.
 * Same diverging palette as the weight heatmaps.
 */
import { prepare } from './canvas';
import { diverging, INK } from './colors';
import { MARK } from './weightHeatmap';

export interface StripLayout {
  n: number;
  perRow: number;
  cellW: number;
  cellH: number;
  numbers: boolean;
}

/** Up to 32 squares per row; with `numbers`, wider squares (16 per row) with the value inside. */
export function stripLayout(n: number, numbers = false): StripLayout {
  return numbers
    ? { n, perRow: Math.min(n, 16), cellW: 40, cellH: 20, numbers }
    : { n, perRow: Math.min(n, 32), cellW: 14, cellH: 14, numbers };
}

export interface StripOptions {
  /** Squares to outline (e.g. the biggest changes). */
  marks?: ReadonlySet<number>;
  /** The clicked square (dark outline). */
  selected?: number | null;
  /** The square under the mouse. */
  hover?: number | null;
}

export function drawStrip(canvas: HTMLCanvasElement, values: Float32Array, layout: StripLayout, scale: number, opts: StripOptions = {}): void {
  const { n, perRow, cellW, cellH } = layout;
  const rows = Math.ceil(n / perRow);
  const ctx = prepare(canvas, perRow * cellW + 2, rows * cellH + 2);
  ctx.textAlign = 'center';
  ctx.font = '10px system-ui, sans-serif';
  for (let i = 0; i < n; i++) {
    const x = 1 + (i % perRow) * cellW, y = 1 + Math.floor(i / perRow) * cellH;
    const v = values[i] / scale;
    const [r, g, b] = diverging(v);
    ctx.fillStyle = `rgb(${r},${g},${b})`;
    ctx.fillRect(x, y, cellW - 1, cellH - 1);
    if (layout.numbers) {
      ctx.fillStyle = Math.abs(v) > 0.6 ? '#fff' : INK;
      ctx.fillText(short(values[i]), x + cellW / 2, y + cellH / 2);
    }
  }
  const outline = (i: number, color: string, w: number) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = w;
    ctx.strokeRect(1 + (i % perRow) * cellW - 0.5, 1 + Math.floor(i / perRow) * cellH - 0.5, cellW, cellH);
  };
  for (const i of opts.marks ?? []) outline(i, MARK, 2);
  if (opts.hover !== null && opts.hover !== undefined) outline(opts.hover, '#9ca3af', 1.5);
  if (opts.selected !== null && opts.selected !== undefined) outline(opts.selected, '#111827', 2);
}

export function stripCellAt(layout: StripLayout, x: number, y: number): number | null {
  const c = Math.floor((x - 1) / layout.cellW), r = Math.floor((y - 1) / layout.cellH);
  if (c < 0 || c >= layout.perRow || r < 0) return null;
  const i = r * layout.perRow + c;
  return i < layout.n ? i : null;
}

/** A value short enough for a 40 px square. */
function short(v: number): string {
  const a = Math.abs(v);
  if (a >= 9.95) return v.toFixed(0);
  if (a >= 0.995) return v.toFixed(1);
  return v.toFixed(2).replace(/^(-?)0\./, '$1.');
}

/** Indices of the `k` largest |values|. */
export function topIndices(values: Float32Array, k: number): number[] {
  return Array.from(values.keys()).sort((a, b) => Math.abs(values[b]) - Math.abs(values[a])).slice(0, k);
}
