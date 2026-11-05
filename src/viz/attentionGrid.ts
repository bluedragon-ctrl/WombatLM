/**
 * Attention grid: one row per position that is "looking" (an output letter),
 * one column per position it can look at (an input letter). A dark cell means
 * "looked here a lot". Numbers are percentages; each row adds up to 100 %.
 */
import { prepare } from './canvas';
import { ACCENT, INK, MUTED, sequential } from './colors';

export interface GridSpec {
  /** Labels of the rows (who is looking) and columns (where it can look). */
  rows: string[];
  cols: string[];
  /** Flattened attention, rows of length `stride`, starting at `offset`. */
  data: Float32Array;
  offset: number;
  stride: number;
}

export const ATT_CELL = 26;
const LEFT = 24, TOP = 22;

export function attentionGridSize(spec: GridSpec): { width: number; height: number } {
  return { width: LEFT + spec.cols.length * ATT_CELL + 2, height: TOP + spec.rows.length * ATT_CELL + 2 };
}

/** Which cell is at (x, y) in CSS pixels, or null. Hovering a row label counts as that row. */
export function attentionCellAt(spec: GridSpec, x: number, y: number): { row: number; col: number | null } | null {
  const row = Math.floor((y - TOP) / ATT_CELL);
  if (row < 0 || row >= spec.rows.length) return null;
  const col = Math.floor((x - LEFT) / ATT_CELL);
  return { row, col: col >= 0 && col < spec.cols.length ? col : null };
}

/** `shownRows`: rows from this one on are drawn empty (not written yet, "write it slowly"). */
export function drawAttentionGrid(canvas: HTMLCanvasElement, spec: GridSpec, highlightRow: number | null, shownRows = spec.rows.length): void {
  const { width, height } = attentionGridSize(spec);
  const ctx = prepare(canvas, width, height);
  ctx.textAlign = 'center';
  // Column labels (where it can look).
  ctx.fillStyle = INK;
  ctx.font = '14px system-ui, sans-serif';
  spec.cols.forEach((c, j) => ctx.fillText(c, LEFT + j * ATT_CELL + ATT_CELL / 2, TOP / 2));
  spec.rows.forEach((r, i) => {
    const y = TOP + i * ATT_CELL;
    if (i >= shownRows) {
      ctx.fillStyle = '#cbd5e1';
      ctx.font = '14px system-ui, sans-serif';
      ctx.fillText('·', LEFT / 2, y + ATT_CELL / 2);
      ctx.fillStyle = '#f8fafc';
      spec.cols.forEach((_, j) => ctx.fillRect(LEFT + j * ATT_CELL, y, ATT_CELL - 1, ATT_CELL - 1));
      return;
    }
    if (i === highlightRow) {
      ctx.fillStyle = '#fde68a';
      ctx.fillRect(0, y, LEFT - 2, ATT_CELL);
    }
    ctx.fillStyle = INK;
    ctx.font = `${i === highlightRow ? 'bold ' : ''}14px system-ui, sans-serif`;
    ctx.fillText(r, LEFT / 2, y + ATT_CELL / 2);
    spec.cols.forEach((_, j) => {
      const v = spec.data[spec.offset + i * spec.stride + j];
      const x = LEFT + j * ATT_CELL;
      ctx.fillStyle = sequential(v);
      ctx.fillRect(x, y, ATT_CELL - 1, ATT_CELL - 1);
      if (v >= 0.1) {
        ctx.fillStyle = v > 0.6 ? '#fff' : MUTED;
        ctx.font = '10px system-ui, sans-serif';
        ctx.fillText(String(Math.round(v * 100)), x + ATT_CELL / 2, y + ATT_CELL / 2);
      }
    });
  });
  if (highlightRow !== null && highlightRow < spec.rows.length) {
    ctx.strokeStyle = ACCENT;
    ctx.lineWidth = 2;
    ctx.strokeRect(LEFT - 1, TOP + highlightRow * ATT_CELL - 1, spec.cols.length * ATT_CELL + 1, ATT_CELL + 1);
  }
}
