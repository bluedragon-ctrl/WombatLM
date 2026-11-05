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
