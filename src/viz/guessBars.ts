/**
 * Guess bars: for every letter of the answer, the model's top 5 candidates
 * with their probabilities; the chosen one is highlighted. Under each column,
 * how much of the letter was copied from the input vs written by the model.
 */
import type { StepGuess } from '../train/protocol';
import { prepare } from './canvas';
import { ACCENT, INK, MUTED, sequential } from './colors';

export interface GuessBarLabels {
  copied: string;
  wrote: string;
}

const COL = 78, ROW = 20, HEAD = 26;

export function guessBarsWidth(steps: number): number {
  return Math.max(COL, steps * COL) + 4;
}

export function drawGuessBars(canvas: HTMLCanvasElement, steps: StepGuess[], labels: GuessBarLabels): void {
  const hasWrite = steps.some((s) => s.write !== undefined);
  const height = HEAD + 5 * ROW + (hasWrite ? 30 : 6);
  const ctx = prepare(canvas, guessBarsWidth(steps.length), height);
  steps.forEach((s, c) => {
    const x = c * COL + 2;
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
    if (hasWrite && s.write !== undefined) {
      const y = HEAD + 5 * ROW + 6;
      const w = COL - 8;
      ctx.fillStyle = '#c7d2fe';
      ctx.fillRect(x, y, w * (1 - s.write), 8);
      ctx.fillStyle = '#fbbf24';
      ctx.fillRect(x + w * (1 - s.write), y, w * s.write, 8);
      ctx.fillStyle = MUTED;
      ctx.textAlign = 'center';
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(s.write < 0.5 ? labels.copied : labels.wrote, x + w / 2, y + 17);
    }
  });
  ctx.font = '13px system-ui, sans-serif';
}
