import { describe, expect, it } from 'vitest';
import { alignTo, nearestRows, pca2 } from '../../src/nn/pca';
import { Rng } from '../../src/nn/rng';

describe('pca2', () => {
  it('finds the two directions with the most spread', () => {
    // Points spread along dims 2 (a lot) and 0 (less) of a 5-D space, plus a little noise.
    const rng = new Rng(4);
    const n = 40, d = 5;
    const data = new Float32Array(n * d);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < d; j++) data[i * d + j] = rng.normal() * 0.01;
      data[i * d + 2] += rng.normal() * 3;
      data[i * d + 0] += rng.normal() * 1;
    }
    const rows = Array.from({ length: n }, (_, i) => i);
    const { coords, explained } = pca2(data, d, rows);
    expect(explained[0]).toBeGreaterThan(explained[1]);
    expect(explained[0] + explained[1]).toBeGreaterThan(0.99);
    // x follows dim 2 and y follows dim 0 (up to sign), and the axes are centred.
    let cx = 0, cy = 0, sx = 0;
    for (let i = 0; i < n; i++) {
      cx += coords[2 * i]; cy += coords[2 * i + 1];
      sx += Math.abs(Math.abs(coords[2 * i]) - Math.abs(data[i * d + 2] - mean(data, d, 2, n)));
    }
    expect(Math.abs(cx / n)).toBeLessThan(1e-4);
    expect(Math.abs(cy / n)).toBeLessThan(1e-4);
    expect(sx / n).toBeLessThan(0.05);
  });

  it('only uses the given rows and is deterministic', () => {
    const data = Float32Array.from({ length: 6 * 3 }, (_, i) => Math.sin(i * 1.7));
    const a = pca2(data, 3, [0, 2, 3, 5]);
    const b = pca2(data, 3, [0, 2, 3, 5]);
    expect(a.coords.length).toBe(8);
    expect(Array.from(a.coords)).toEqual(Array.from(b.coords));
  });
});

describe('alignTo', () => {
  const prev = Float32Array.from([0, 0, 2, 0, 2, 1, 0, 3]);
  const transform = (p: Float32Array, th: number, mirror: number) => {
    const c = Math.cos(th), s = Math.sin(th);
    return p.map((_, i) => {
      const x = p[i - (i % 2)], y = mirror * p[i - (i % 2) + 1];
      return i % 2 === 0 ? c * x - s * y + 5 : s * x + c * y - 1;
    });
  };
  const centred = (p: Float32Array) => {
    const n = p.length / 2;
    let x = 0, y = 0;
    for (let i = 0; i < n; i++) { x += p[2 * i]; y += p[2 * i + 1]; }
    return Array.from(p, (v, i) => v - (i % 2 === 0 ? x / n : y / n));
  };

  it('undoes a rotation', () => {
    const out = alignTo(prev, transform(prev, 2.1, 1));
    centred(out).forEach((v, i) => expect(v).toBeCloseTo(centred(prev)[i], 4));
  });

  it('undoes a mirror image (a PCA sign flip)', () => {
    const out = alignTo(prev, transform(prev, 0.7, -1));
    centred(out).forEach((v, i) => expect(v).toBeCloseTo(centred(prev)[i], 4));
  });

  it('leaves mismatched frames alone', () => {
    const next = Float32Array.from([1, 2]);
    expect(alignTo(prev, next)).toBe(next);
  });
});

describe('nearestRows', () => {
  it('lists the rows pointing the most the same way, closest first, among candidates only', () => {
    // Candidates average to (0,0): 0 → (2,0), 1 → (1.8,0.4), 2 → (-2,0), 3 → (-1.8,-0.4), 4 → (0,1) not a candidate.
    const data = new Float32Array([2, 0, 1.8, 0.4, -2, 0, -1.8, -0.4, 0, 1, 2, 0.01]);
    const rows = [0, 1, 2, 3, 4, 5];
    const near = nearestRows(data, 2, rows, [true, true, true, true, false, false], 2);
    expect(near[0]).toEqual([1, 3]); // row 5 is closer but not a candidate
    expect(near[2]).toEqual([3, 1]);
    expect(near[4]).toEqual([]);
    // Length doesn't matter, only direction.
    const scaled = nearestRows(new Float32Array([1, 0, 9, 1, -1, 0, -9, -1]), 2, [0, 1, 2, 3], [true, true, true, true], 1);
    expect(scaled[0]).toEqual([1]);
  });

  it('measures from the average letter, so a shared offset does not count', () => {
    // All rows share a big (10, 10); only the small part tells them apart.
    const data = new Float32Array([11, 10, 10.9, 10.2, 9, 10, 9.1, 9.8]);
    expect(nearestRows(data, 2, [0, 1, 2, 3], [true, true, true, true], 1)).toEqual([[1], [0], [3], [2]]);
  });
});

function mean(data: Float32Array, d: number, j: number, n: number): number {
  let s = 0;
  for (let i = 0; i < n; i++) s += data[i * d + j];
  return s / n;
}
