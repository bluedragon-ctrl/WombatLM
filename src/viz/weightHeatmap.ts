/**
 * Weight heatmap: one coloured cell per number of a weight matrix. Blue is
 * negative, white zero, red positive; the scale is symmetric per matrix, so
 * the strongest numbers (99th percentile) get the deepest colour.
 */
import { prepare } from './canvas';
import { diverging } from './colors';

export interface HeatmapLayout {
  rows: number;
  cols: number;
  cellW: number;
  cellH: number;
}

/** Cell size: small for big matrices, larger for thin ones (vectors, the copy gate). */
export function heatmapLayout(rows: number, cols: number, cell = 5): HeatmapLayout {
  return { rows, cols, cellW: cols === 1 ? 12 : cell, cellH: rows === 1 ? 12 : cell };
}

let scratch: HTMLCanvasElement | null = null;

export function drawHeatmap(canvas: HTMLCanvasElement, data: Float32Array, layout: HeatmapLayout, scale: number, highlight?: { row: number; col: number } | null): void {
  const { rows, cols, cellW, cellH } = layout;
  const ctx = prepare(canvas, cols * cellW, rows * cellH);
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
  ctx.drawImage(scratch, 0, 0, cols * cellW, rows * cellH);
  if (highlight) {
    ctx.strokeStyle = '#111827';
    ctx.lineWidth = 1.5;
    ctx.strokeRect(highlight.col * cellW - 1, highlight.row * cellH - 1, cellW + 2, cellH + 2);
  }
}

export function heatmapCellAt(layout: HeatmapLayout, x: number, y: number): { row: number; col: number } | null {
  const row = Math.floor(y / layout.cellH), col = Math.floor(x / layout.cellW);
  return row >= 0 && row < layout.rows && col >= 0 && col < layout.cols ? { row, col } : null;
}
