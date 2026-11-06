/**
 * The matrix view's data: one answer as a list of stations. At every station
 * the model's numbers form a matrix, one row per token and `d` columns (32
 * numbers). Each station turns this matrix into a new one of the same shape.
 *
 * Reading (once, all input tokens together):
 *   letters → + position → [look at each other → think] per layer → volume knobs
 * Writing (one pass per token; pass k has rows 0..k, only row k is new):
 *   letters → + position → [look back → look at the animal → think] per layer
 *   → volume knobs → guess (the last row is compared with every letter-table row)
 *
 * In writing pass k the earlier rows are the numbers the model remembers from
 * the earlier passes at the same station (they do not change any more).
 * Everything comes with the answer (`AnswerMsg`); nothing about the model is
 * computed here except sums of what each station adds, and softmax.
 */
import type { AnswerMsg, AttentionMap } from '../train/protocol';
import { EOS } from '../nn/vocab';

export type StationKind = 'letters' | 'position' | 'self' | 'cross' | 'think' | 'knobs' | 'guess';

/** One grid (or list) of learned numbers a station uses. */
export type MachineKey = 'q' | 'k' | 'v' | 'o' | 'up' | 'down' | 'gamma' | 'beta';
export interface MachinePart {
  key: MachineKey;
  rows: number;
  cols: number;
  data: Float32Array;
}

export interface StationAttention {
  /** Rows (queries) that look, and how many rows (keys) they look at. */
  heads: number;
  /** Per head, per active row (in `active` order), per key: weight 0..1. */
  weights: Float32Array;
  keys: number;
  /** The keys are rows of the same matrix ('self') or of the reading result ('memory'). */
  from: 'self' | 'memory';
}

export interface Station {
  lane: 'read' | 'write';
  /** -1 for reading, k for writing pass k. */
  pass: number;
  kind: StationKind;
  /** Layer number (0-based) for self / cross / think. */
  layer: number;
  /** Row labels (tokens), one per row. */
  tokens: string[];
  /** Rows that travel through this station now (reading: all; writing: the newest). */
  active: number[];
  d: number;
  /** The matrix after this station, [rows × d]. */
  grid: Float32Array;
  /** The matrix before this station (same shape), null at "letters". */
  before: Float32Array | null;
  attention?: StationAttention;
  /** Guess: the score and chance of every token, and the token written. */
  guess?: { scores: Float32Array; probs: Float32Array; chosen: string; chosenId: number };
  /** Index of this station in its lane's list of station kinds (for the trail of one number). */
  laneIndex: number;
  /**
   * The learned numbers this station passes the rows through (fixed while it
   * answers). Letters, position and guess use the table on the left instead.
   */
  machine: MachinePart[];
}

/** One lane's matrices at every station, all rows (writing passes take the first k+1). */
interface LaneGrid {
  kind: StationKind;
  layer: number;
  grid: Float32Array;
  att?: AttentionMap;
}

export interface MatrixModel {
  stations: Station[];
  /** Index of the first station of reading (0) and of every writing pass. */
  passStart: number[];
  /** One colour scale per lane, so colours can be compared station by station. */
  scale: { read: number; write: number };
  /** Station kinds of each lane in order (for the trail of one number). */
  laneKinds: { read: { kind: StationKind; layer: number }[]; write: { kind: StationKind; layer: number }[] };
  /** The values of one number (row, column) at every station of its lane. */
  trail(lane: 'read' | 'write', row: number, col: number): number[];
  /** Reading result: the matrix "look at the animal" looks at. */
  memory: Float32Array;
  inputTokens: string[];
  d: number;
}

const add = (a: Float32Array, b: Float32Array | undefined): Float32Array<ArrayBufferLike> => {
  const out = a.slice();
  if (b) for (let i = 0; i < out.length; i++) out[i] += b[i] ?? 0;
  return out;
};

/** Robust scale for a set of matrices: the 98th percentile of |values|. */
function laneScale(grids: Float32Array[]): number {
  let n = 0;
  for (const g of grids) n += g.length;
  const abs = new Float32Array(n);
  let o = 0;
  for (const g of grids) for (let i = 0; i < g.length; i++) abs[o++] = Math.abs(g[i]);
  abs.sort();
  const v = abs[Math.min(n - 1, Math.floor(0.98 * (n - 1)))];
  return v > 0 ? v : 1;
}

export function softmax(scores: Float32Array): Float32Array {
  let max = -Infinity;
  for (const s of scores) max = Math.max(max, s);
  const out = new Float32Array(scores.length);
  let sum = 0;
  for (let i = 0; i < scores.length; i++) { out[i] = Math.exp(scores[i] - max); sum += out[i]; }
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** Number of layers in a lane: enc0, enc1, … found in the adds. */
function layers(a: AnswerMsg, prefix: 'enc' | 'dec'): number {
  let n = 0;
  while (a.adds.some((x) => x.name === `${prefix}${n}.ffn.add`)) n++;
  return n;
}

export function buildMatrixModel(a: AnswerMsg, tokens: readonly string[]): MatrixModel {
  const act = (name: string) => a.activations.find((x) => x.name === name)?.data;
  const addOf = (name: string) => a.adds.find((x) => x.name === name)?.data;
  const att = (name: string) => a.attention.find((x) => x.name === name);
  const d = a.activations.find((x) => x.name === 'enc.in')?.d ?? a.lookup.encTok.length / Math.max(1, a.input.length);
  const nSteps = a.steps.length;

  // Reading lane (T input rows).
  const read: LaneGrid[] = [{ kind: 'letters', layer: 0, grid: a.lookup.encTok }];
  let x: Float32Array = act('enc.in') ?? add(a.lookup.encTok, a.lookup.encPos);
  read.push({ kind: 'position', layer: 0, grid: x });
  for (let l = 0; l < layers(a, 'enc'); l++) {
    x = add(x, addOf(`enc${l}.self.add`));
    read.push({ kind: 'self', layer: l, grid: x, att: att(`enc${l}.self`) });
    x = add(x, addOf(`enc${l}.ffn.add`));
    read.push({ kind: 'think', layer: l, grid: x });
  }
  const memory = act('memory') ?? x;
  read.push({ kind: 'knobs', layer: 0, grid: memory });

  // Writing lane (all teacher-forced rows; only the first nSteps are real passes).
  const cut = (g: Float32Array | undefined) => (g ?? new Float32Array(nSteps * d)).slice(0, nSteps * d);
  const write: LaneGrid[] = [{ kind: 'letters', layer: 0, grid: cut(a.lookup.decTok) }];
  let y: Float32Array = cut(act('dec.in') ?? add(a.lookup.decTok, a.lookup.decPos));
  write.push({ kind: 'position', layer: 0, grid: y });
  for (let l = 0; l < layers(a, 'dec'); l++) {
    y = add(y, cut(addOf(`dec${l}.self.add`)));
    write.push({ kind: 'self', layer: l, grid: y, att: att(`dec${l}.self`) });
    y = add(y, cut(addOf(`dec${l}.cross.add`)));
    write.push({ kind: 'cross', layer: l, grid: y, att: att(`dec${l}.cross`) });
    y = add(y, cut(addOf(`dec${l}.ffn.add`)));
    write.push({ kind: 'think', layer: l, grid: y });
  }
  const final = act('dec.final');
  write.push({ kind: 'knobs', layer: 0, grid: final ? cut(final) : y });
  write.push({ kind: 'guess', layer: 0, grid: write[write.length - 1].grid });

  const w = (name: string) => a.weights.find((x) => x.name === name);
  const parts = (prefix: string, keys: [MachineKey, string][]): MachinePart[] =>
    keys.flatMap(([key, suffix]) => { const m = w(`${prefix}${suffix}`); return m ? [{ key, rows: m.rows, cols: m.cols, data: m.data }] : []; });
  const machine = (lane: 'enc' | 'dec', g: LaneGrid): MachinePart[] => {
    const att: [MachineKey, string][] = [['q', '.q.W'], ['k', '.k.W'], ['v', '.v.W'], ['o', '.o.W']];
    if (g.kind === 'self') return parts(`${lane}${g.layer}.self`, att);
    if (g.kind === 'cross') return parts(`${lane}${g.layer}.cross`, att);
    if (g.kind === 'think') return parts(`${lane}${g.layer}.ffn`, [['up', '.up.W'], ['down', '.down.W']]);
    if (g.kind === 'knobs') return parts(`${lane}Norm`, [['gamma', '.gamma'], ['beta', '.beta']]);
    return [];
  };

  const stations: Station[] = [];
  const passStart: number[] = [0];
  const T = a.input.length;
  const allRead = Array.from({ length: T }, (_, i) => i);
  read.forEach((g, i) => stations.push({
    lane: 'read', pass: -1, kind: g.kind, layer: g.layer, tokens: a.input, active: allRead, d,
    grid: g.grid, before: i > 0 ? read[i - 1].grid : null, laneIndex: i, machine: machine('enc', g),
    attention: g.att ? attention(g.att, allRead, T, 'self') : undefined,
  }));

  const decIn = [a.input[0], ...a.steps.map((s) => s.chosen || EOS)].slice(0, nSteps);
  const V = a.vocabSize;
  for (let k = 0; k < nSteps; k++) {
    passStart.push(stations.length);
    const rows = k + 1;
    const sl = (g: Float32Array) => g.slice(0, rows * d);
    write.forEach((g, i) => {
      const st: Station = {
        lane: 'write', pass: k, kind: g.kind, layer: g.layer, tokens: decIn.slice(0, rows), active: [k], d,
        grid: sl(g.grid), before: i > 0 ? sl(write[i - 1].grid) : null, laneIndex: i, machine: machine('dec', g),
      };
      if (g.att) st.attention = attention(g.att, [k], g.kind === 'cross' ? T : rows, g.kind === 'cross' ? 'memory' : 'self');
      if (g.kind === 'guess') {
        const scores = a.logits.slice(k * V, (k + 1) * V);
        const chosen = a.steps[k].chosen || EOS;
        st.guess = { scores, probs: softmax(scores), chosen, chosenId: tokens.indexOf(chosen) };
      }
      stations.push(st);
    });
  }

  return {
    stations, passStart, memory, inputTokens: a.input, d,
    scale: { read: laneScale(read.map((g) => g.grid)), write: laneScale(write.map((g) => g.grid)) },
    laneKinds: { read: read.map((g) => ({ kind: g.kind, layer: g.layer })), write: write.map((g) => ({ kind: g.kind, layer: g.layer })) },
    trail(lane, row, col) {
      return (lane === 'read' ? read : write).map((g) => g.grid[row * d + col] ?? 0);
    },
  };
}

/** The attention rows of the active queries over the first `keys` keys. */
function attention(m: AttentionMap, active: number[], keys: number, from: 'self' | 'memory'): StationAttention {
  const weights = new Float32Array(m.heads * active.length * keys);
  for (let hd = 0; hd < m.heads; hd++) {
    active.forEach((q, qi) => {
      for (let k = 0; k < keys; k++) weights[(hd * active.length + qi) * keys + k] = m.data[hd * m.Tq * m.Tk + q * m.Tk + k] ?? 0;
    });
  }
  return { heads: m.heads, weights, keys, from };
}

/** Average over heads: how much active row `qi` takes from key `k`. */
export function meanAttention(at: StationAttention, nActive: number, qi: number, k: number): number {
  let s = 0;
  for (let hd = 0; hd < at.heads; hd++) s += at.weights[(hd * nActive + qi) * at.keys + k];
  return s / at.heads;
}
