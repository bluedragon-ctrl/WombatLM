/** Number formats shared by the Inside-the-model views. */

/** A chance as a whole percentage; below 1 % as "<1 %" rather than a misleading 0, likewise ">99 %". */
export function pctText(p: number): string {
  if (p > 0 && p < 0.005) return '<1 %';
  if (p > 0.995 && p < 1) return '>99 %';
  return `${Math.round(p * 100)} %`;
}
