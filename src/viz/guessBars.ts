/**
 * Guess bars: for every letter of the answer, the model's top 5 candidates
 * with their probabilities; the chosen one is highlighted.
 */
import type { StepGuess } from '../train/protocol';
import { prepare } from './canvas';
import { ACCENT, INK, MUTED, sequential } from './colors';

/** Width of one letter column (exported so hover can find the column under the mouse). */
export const GUESS_COL = 78;
const COL = GUESS_COL, ROW = 20, HEAD = 26;

/** Which column (output step) is at x, or null. */
export function guessColumnAt(x: number, steps: number): number | null {
  const c = Math.floor((x - 2) / COL);
  return c >= 0 && c < steps ? c : null;
}

export function guessBarsWidth(steps: number): number {
  return Math.max(COL, steps * COL) + 4;
}

/**
 * `highlight`: the column linked to the attention grid (hovered there or here).
 * `shown`: only the first `shown` columns are drawn, the rest are empty
 * placeholders ("write it slowly").
 */
export function drawGuessBars(canvas: HTMLCanvasElement, steps: StepGuess[], highlight: number | null = null, shown = steps.length): void {
  const height = HEAD + 5 * ROW + 6;
  const ctx = prepare(canvas, guessBarsWidth(steps.length), height);
  steps.forEach((s, c) => {
    const x = c * COL + 2;
    if (c >= shown) {
      ctx.strokeStyle = '#e5e7eb';
      ctx.setLineDash([4, 4]);
      ctx.lineWidth = 1;
      ctx.strokeRect(x + 1, HEAD, COL - 6, 5 * ROW);
      ctx.setLineDash([]);
      ctx.fillStyle = '#cbd5e1';
      ctx.textAlign = 'center';
      ctx.font = 'bold 18px system-ui, sans-serif';
      ctx.fillText('?', x + COL / 2, 12);
      return;
    }
    if (c === highlight) {
      ctx.fillStyle = '#fef3c7';
      ctx.fillRect(x - 2, 0, COL, height);
    }
    ctx.fillStyle = INK;
    ctx.textAlign = 'center';
    ctx.font = 'bold 18px system-ui, sans-serif';
    ctx.fillText(s.chosen || '⏎', x + COL / 2, 12);
    ctx.font = '13px system-ui, sans-serif';
    s.top.forEach((g, r) => {
      const y = HEAD + r * ROW;
      const chosen = g.char === s.chosen;
      ctx.fillStyle = sequential(g.p);
      ctx.fillRect(x + 22, y + 2, Math.max(1, (COL - 28) * g.p), ROW - 4);
      if (chosen) {
        ctx.strokeStyle = ACCENT;
        ctx.lineWidth = 2;
        ctx.strokeRect(x + 1, y + 1, COL - 6, ROW - 2);
      }
      ctx.fillStyle = INK;
      ctx.textAlign = 'center';
      ctx.font = `${chosen ? 'bold ' : ''}13px system-ui, sans-serif`;
      ctx.fillText(g.char === '' ? '⏎' : g.char, x + 11, y + ROW / 2);
      ctx.textAlign = 'right';
      ctx.fillStyle = g.p > 0.85 ? '#fff' : MUTED; // white only where the bar is under the text
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(`${Math.round(g.p * 100)}%`, x + COL - 8, y + ROW / 2);
    });
  });
  ctx.font = '13px system-ui, sans-serif';
}
