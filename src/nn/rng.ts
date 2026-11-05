/**
 * Seeded pseudo-random numbers (mulberry32). The same seed always gives the
 * same sequence, so a training run can be repeated bit for bit.
 */
export class Rng {
  private state: number;
  private spareNormal: number | null = null;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Uniform float in [0, 1). */
  next(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.next() * n);
  }

  /** Standard normal sample (Box–Muller). */
  normal(): number {
    if (this.spareNormal !== null) {
      const s = this.spareNormal;
      this.spareNormal = null;
      return s;
    }
    let u = 0;
    while (u === 0) u = this.next();
    const v = this.next();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spareNormal = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  /** Pick a random element. */
  pick<T>(items: readonly T[]): T {
    return items[this.int(items.length)];
  }

  /** Shuffle in place (Fisher–Yates). */
  shuffle<T>(items: T[]): T[] {
    for (let i = items.length - 1; i > 0; i--) {
      const j = this.int(i + 1);
      const tmp = items[i];
      items[i] = items[j];
      items[j] = tmp;
    }
    return items;
  }

  /** Current internal state, so a run can be resumed exactly. */
  getState(): number {
    return this.state;
  }

  setState(state: number): void {
    this.state = state >>> 0;
    this.spareNormal = null;
  }
}
