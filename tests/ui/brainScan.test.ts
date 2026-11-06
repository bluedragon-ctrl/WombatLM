import { describe, expect, it } from 'vitest';
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { Engine } from '../../src/train/engine';
import type { FromWorker, ReadyMsg } from '../../src/train/protocol';
import { BOXES } from '../../src/ui/inside';
import { brightness, glow, scanCellAt, scanLayout } from '../../src/viz/brainScan';

describe('brain scan', () => {
  it('every number of the model is in exactly one part of the scan', () => {
    const sent: FromWorker[] = [];
    const e = new Engine({ post: (m: FromWorker) => sent.push(m) } as never, new Dataset(baseBundle()));
    e.handle({ type: 'init', seed: 1 });
    const ready = sent.find((m) => m.type === 'ready') as ReadyMsg;
    for (const m of ready.matrices) expect(BOXES.filter((b) => b.match.test(m.name)).map((b) => b.id), m.name).toHaveLength(1);
    const total = BOXES.reduce((a, b) => a + scanLayout(ready.matrices.filter((m) => b.match.test(m.name)).map((m) => ({ name: m.name, size: m.rows * m.cols }))).size, 0);
    expect(total).toBe(ready.params);
  });

  it('packs the matrices one after the other and finds the number under a dot', () => {
    const l = scanLayout([{ name: 'a', size: 5 }, { name: 'b', size: 7 }], 4, 2);
    expect(l.rows).toBe(3);
    expect(l.size).toBe(12);
    expect(scanCellAt(l, 0, 0)).toEqual({ piece: l.pieces[0], index: 0 });
    expect(scanCellAt(l, 1, 2)).toEqual({ piece: l.pieces[0], index: 4 }); // row 1, column 0 = dot 4
    expect(scanCellAt(l, 2, 2)).toEqual({ piece: l.pieces[1], index: 0 }); // dot 5
    expect(scanCellAt(l, 6, 4)?.index).toBe(6); // dot 11 = last of b
    expect(scanCellAt(l, 8, 0)).toBeNull();
  });

  it('brightness: same scale for all, bigger is never darker', () => {
    expect(brightness(0, 1)).toBe(0);
    expect(brightness(-0.5, 1)).toBe(0.5);
    expect(brightness(3, 1)).toBe(1);
    let prev = -1;
    for (let v = 0; v <= 1; v += 0.05) {
      const [r, g, b] = glow(v);
      const lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      expect(lum).toBeGreaterThan(prev);
      prev = lum;
    }
  });
});
