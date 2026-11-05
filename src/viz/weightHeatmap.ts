/**
 * Weight heatmap: one coloured cell per number of a weight matrix. Blue is
 * negative, white zero, red positive; the scale is symmetric (the caller
 * picks it: per matrix, or shared by all pictures).
 *
 * Optional row / column labels (letters, positions) are drawn in a margin,
 * and rows / columns can be outlined ("used for this letter").
 */
import { prepare } from './canvas';
import { diverging } from './colors';

export interface HeatmapLayout {
  rows: number;
  cols: number;
  cellW: number;
  cellH: number;
  /** Margin for row labels (left) and column labels (top), in CSS pixels. */
  left: number;
  top: number;
  rowLabels?: readonly string[];
  colLabels?: readonly string[];
}

/** Rows / columns to outline. */
export interface HeatmapMarks {
  rows?: readonly number[];
  cols?: readonly number[];
}

export const MARK = '#f59e0b';

/**
 * Cell size: `cell` for big matrices, larger for thin ones (vectors). With
 * labels, cells along a labelled axis get at least 9 px so the labels can be read.
 */
export function heatmapLayout(rows: number, cols: number, cell = 5, labels: { rows?: readonly string[]; cols?: readonly string[] } = {}): HeatmapLayout {
  let cellW = cols === 1 ? 12 : cell, cellH = rows === 1 ? 12 : cell;
  if (labels.rows) cellH = Math.max(cellH, 9);
  if (labels.cols) cellW = Math.max(cellW, 9);
  return {
    rows, cols, cellW, cellH,
    left: labels.rows ? 16 : 0,
    top: labels.cols ? 13 : 0,
    rowLabels: labels.rows, colLabels: labels.cols,
  };
}

export function heatmapSize(l: HeatmapLayout): { width: number; height: number } {
  return { width: l.left + l.cols * l.cellW, height: l.top + l.rows * l.cellH };
}

let scratch: HTMLCanvasElement | null = null;

export function drawHeatmap(canvas: HTMLCanvasElement, data: Float32Array, layout: HeatmapLayout, scale: number,
  highlight?: { row: number; col: number } | null, marks?: HeatmapMarks | null): void {
  const { rows, cols, cellW, cellH, left, top } = layout;
  const { width, height } = heatmapSize(layout);
  const ctx = prepare(canvas, width, height);
  // Paint one pixel per number into a scratch canvas, then scale it up without smoothing.
  scratch ??= document.createElement('canvas');
  scratch.width = cols;
  scratch.height = rows;
  const sctx = scratch.getContext('2d')!;
  const img = sctx.createImageData(cols, rows);
  for (let i = 0; i < rows * cols; i++) {
    const [r, g, b] = diverging(data[i] / scale);
    img.data[4 * i] = r;
    img.data[4 * i + 1] = g;
    img.data[4 * i + 2] = b;
    img.data[4 * i + 3] = 255;
  }
  sctx.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(scratch, left, top, cols * cellW, rows * cellH);
  ctx.fillStyle = '#374151';
  ctx.textAlign = 'center';
  if (layout.rowLabels) {
    ctx.font = `${Math.min(11, cellH)}px system-ui, sans-serif`;
    layout.rowLabels.forEach((l, r) => ctx.fillText(l, left / 2, top + (r + 0.5) * cellH));
  }
  if (layout.colLabels) {
    ctx.font = `${Math.min(11, cellW)}px system-ui, sans-serif`;
    layout.colLabels.forEach((l, c) => ctx.fillText(l, left + (c + 0.5) * cellW, top / 2));
  }
  if (marks) {
    ctx.strokeStyle = MARK;
    ctx.lineWidth = 2;
    for (const r of marks.rows ?? []) if (r >= 0 && r < rows) ctx.strokeRect(left - 1, top + r * cellH - 1, cols * cellW + 2, cellH + 2);
    for (const c of marks.cols ?? []) if (c >= 0 && c < cols) ctx.strokeRect(left + c * cellW - 1, top - 1, cellW + 2, rows * cellH + 2);
  }
  if (highlight) {
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(left + highlight.col * cellW - 1, top + highlight.row * cellH - 1, cellW + 2, cellH + 2);
  }
}

export function heatmapCellAt(layout: HeatmapLayout, x: number, y: number): { row: number; col: number } | null {
  const row = Math.floor((y - layout.top) / layout.cellH), col = Math.floor((x - layout.left) / layout.cellW);
  return x >= layout.left && y >= layout.top && row >= 0 && row < layout.rows && col >= 0 && col < layout.cols ? { row, col } : null;
}
