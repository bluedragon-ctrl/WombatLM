/**
 * "Brain scan": every number of one part of the model as one dot, glowing
 * where it is big (how much it moved, how hard it is pushed, how big it is).
 *
 * All parts use the same dot size and the same number of dots per row, so a
 * part's area is its share of the model and the same brightness means the
 * same size in every part. The matrices of a part are packed one after the
 * other (their boundaries are named on hover, not drawn).
 */
import { prepare } from './canvas';

/** One matrix packed into a tile: dots start..start+size-1. */
export interface ScanPiece {
  name: string;
  start: number;
  size: number;
}

export interface ScanLayout {
  /** Dots per row and rows. */
  cols: number;
  rows: number;
  /** Dot size in CSS pixels. */
  dot: number;
  pieces: ScanPiece[];
  /** Total number of dots (numbers). */
  size: number;
}

export function scanLayout(matrices: { name: string; size: number }[], cols = 96, dot = 2): ScanLayout {
  const pieces: ScanPiece[] = [];
  let start = 0;
  for (const m of matrices) { pieces.push({ name: m.name, start, size: m.size }); start += m.size; }
  return { cols, rows: Math.max(1, Math.ceil(start / cols)), dot, pieces, size: start };
}

/** Which matrix and which of its numbers is under (x, y) CSS pixels, or null. */
export function scanCellAt(l: ScanLayout, x: number, y: number): { piece: ScanPiece; index: number } | null {
  const c = Math.floor(x / l.dot), r = Math.floor(y / l.dot);
  if (c < 0 || c >= l.cols || r < 0 || r >= l.rows) return null;
  const i = r * l.cols + c;
  const piece = l.pieces.find((p) => i >= p.start && i < p.start + p.size);
  return piece ? { piece, index: i - piece.start } : null;
}

/** Glow palette for 0..1: near black → purple → red → orange → yellow. */
const STOPS: [number, number, number, number][] = [
  [0, 17, 24, 39],
  [0.25, 76, 29, 149],
  [0.5, 190, 24, 60],
  [0.75, 249, 115, 22],
  [1, 253, 224, 71],
];

export function glow(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v));
  for (let i = 1; i < STOPS.length; i++) {
    const [p1, r1, g1, b1] = STOPS[i];
    if (x <= p1) {
      const [p0, r0, g0, b0] = STOPS[i - 1];
      const f = (x - p0) / (p1 - p0);
      return [Math.round(r0 + f * (r1 - r0)), Math.round(g0 + f * (g1 - g0)), Math.round(b0 + f * (b1 - b0))];
    }
  }
  return [253, 224, 71];
}

/** CSS gradient of the glow palette, for the legend. */
export const GLOW_GRADIENT = `linear-gradient(to right, ${STOPS.map(([p, r, g, b]) => `rgb(${r},${g},${b}) ${p * 100}%`).join(', ')})`;

/** Brightness of one number: |v| / scale (the same scale for every part), clamped to 1. */
export function brightness(v: number, scale: number): number {
  return Math.min(1, Math.abs(v) / scale);
}

/**
 * Draw a tile. `data` gives each matrix's numbers by name (missing = dark).
 * Drawn one pixel per dot into a buffer, then scaled up without smoothing.
 */
export function drawScan(canvas: HTMLCanvasElement, l: ScanLayout, data: ReadonlyMap<string, Float32Array>, scale: number,
  hover: { piece: ScanPiece } | null = null): void {
  const W = l.cols * l.dot, H = l.rows * l.dot;
  const ctx = prepare(canvas, W, H);
  const img = new ImageData(l.cols, l.rows);
  const px = img.data;
  const dark = glow(0);
  for (let i = 0; i < l.cols * l.rows; i++) {
    px[i * 4] = 255; px[i * 4 + 1] = 255; px[i * 4 + 2] = 255; px[i * 4 + 3] = 255; // after the last dot: page white
  }
  for (const p of l.pieces) {
    const d = data.get(p.name);
    const dim = hover && hover.piece !== p;
    for (let k = 0; k < p.size; k++) {
      const [r, g, b] = d ? glow(brightness(d[k], scale)) : dark;
      const o = (p.start + k) * 4;
      // While hovering a matrix, the others are dimmed so it stands out.
      px[o] = dim ? r * 0.45 : r; px[o + 1] = dim ? g * 0.45 : g; px[o + 2] = dim ? b * 0.45 : b; px[o + 3] = 255;
    }
  }
  const buf = document.createElement('canvas');
  buf.width = l.cols; buf.height = l.rows;
  buf.getContext('2d')!.putImageData(img, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(buf, 0, 0, W, H);
}
