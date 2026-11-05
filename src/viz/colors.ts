/** Shared colours: one sequential palette, status and form-kind categories. */

export const INK = '#1f2430';
export const MUTED = '#6b7280';
export const GRID = '#e5e7eb';
export const ACCENT = '#2563eb';

/** Sequential palette (probabilities, attention): white -> deep blue. */
export function sequential(v: number): string {
  const x = Math.max(0, Math.min(1, v));
  const r = Math.round(255 - x * (255 - 30));
  const g = Math.round(255 - x * (255 - 64));
  const b = Math.round(255 - x * (255 - 175));
  return `rgb(${r},${g},${b})`;
}

export const STATUS_COLORS = {
  correct: '#15803d',
  close: '#b45309',
  wrong: '#b91c1c',
  invented: '#7c3aed',
} as const;

export const STATUS_ICONS = { correct: '✓', close: '≈', wrong: '✗', invented: '✨' } as const;

/** Scoreboard lines. Never-seen lines are dashed (colour is never the only cue). */
export const SCORE_STYLE = {
  known: { color: '#111827', dash: [] as number[] },
  pattern: { color: '#0d9488', dash: [6, 3] },
  partly: { color: '#d97706', dash: [6, 3] },
  irregular: { color: '#be185d', dash: [2, 3] },
} as const;

/**
 * Diverging palette (weights): blue negative, white zero, red positive.
 * `v` is already scaled to -1..1 (clamped).
 */
export function diverging(v: number): [number, number, number] {
  const x = Math.max(-1, Math.min(1, v));
  const [r, g, b] = x < 0 ? NEG_RGB : POS_RGB;
  const a = Math.abs(x);
  return [Math.round(255 - a * (255 - r)), Math.round(255 - a * (255 - g)), Math.round(255 - a * (255 - b))];
}
const NEG_RGB = [33, 102, 172] as const;
const POS_RGB = [178, 24, 43] as const;
export const DIVERGING_ENDS = { neg: `rgb(${NEG_RGB.join(',')})`, pos: `rgb(${POS_RGB.join(',')})` };

/** Sequential palette as RGB (for pixel buffers). */
export function sequentialRgb(v: number): [number, number, number] {
  const x = Math.max(0, Math.min(1, v));
  return [Math.round(255 - x * 225), Math.round(255 - x * 191), Math.round(255 - x * 80)];
}

/** Letter-map categories: colour and marker shape (colour is never the only cue). */
export const LETTER_KINDS = {
  vowel: { color: '#dc2626', shape: 'circle' },
  consonant: { color: '#2563eb', shape: 'square' },
  accented: { color: '#9333ea', shape: 'diamond' },
  other: { color: '#6b7280', shape: 'triangle' },
  task: { color: '#0d9488', shape: 'star' },
} as const;
export type LetterKind = keyof typeof LETTER_KINDS;

/** Robust symmetric scale: the 99th percentile of |values| (never 0). */
export function symmetricScale(data: Float32Array, q = 0.99): number {
  if (data.length === 0) return 1;
  const abs = Float32Array.from(data, Math.abs).sort();
  const v = abs[Math.min(abs.length - 1, Math.floor(q * (abs.length - 1)))];
  return v > 0 ? v : abs[abs.length - 1] || 1;
}
