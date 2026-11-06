/**
 * The commentary of the matrix animation: for every station a short list of
 * comments ("beats"). Each beat plays one part of the station's animation
 * (`from` → `to`, 0..1; equal = the picture holds still while it is read)
 * and can point at what it talks about: one number, a row, or a row of the
 * side panel. The comments say what happens, with the real numbers of this
 * answer, never why.
 */
import type { Anchor } from '../viz/matrixStage';
import type { MachineKey, MatrixModel, Station } from './matrixModel';
import { meanAttention } from './matrixModel';
import { biggestChanges } from '../viz/matrixStage';
import { t } from './i18n';
import { shortLabel } from './tokens';
import { EOS } from '../nn/vocab';

export interface Beat {
  text: string;
  from: number;
  to: number;
  anchor?: Anchor;
  /** Grids of learned numbers (right panel) this comment is about; none = all of the station's. */
  uses?: MachineKey[];
}

export interface ScriptContext {
  m: MatrixModel;
  /** Token list (row i of the letter table = token i). */
  tokens: readonly string[];
  /** A token as the UI names it ("Czech, female", "t", "end"). */
  full: (token: string) => string;
  /** The answer's input word and output word. */
  word: string;
  output: string;
  /** The number the user follows, if it is in this station's matrix. */
  pinned: { row: number; col: number } | null;
  /** Reading, "look at each other": index into `active` of the row whose arrows are shown. */
  focus: number;
}

const f2 = (v: number) => (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));

/** The number a comment uses as its example: the followed one, else the biggest change of the newest row. */
function exampleCell(st: Station, pinned: ScriptContext['pinned']): { row: number; col: number } {
  if (pinned && st.active.includes(pinned.row)) return pinned;
  const row = st.active[st.active.length - 1];
  const big = [...biggestChanges(st, 1)].find((i) => Math.floor(i / st.d) === row);
  return { row, col: big !== undefined ? big % st.d : 0 };
}

export function beatsFor(st: Station, c: ScriptContext): Beat[] {
  const d = st.d;
  const tok = (r: number) => c.full(st.tokens[r]);
  const idOf = (r: number) => c.tokens.indexOf(st.tokens[r]);
  const last = st.active[st.active.length - 1];
  const before = (r: number, col: number) => st.before?.[r * d + col] ?? 0;
  const after = (r: number, col: number) => st.grid[r * d + col];
  const read = st.lane === 'read';
  const ex = exampleCell(st, c.pinned);
  const exVars = { c: ex.col + 1, tok: tok(ex.row), a: f2(before(ex.row, ex.col)), b: f2(after(ex.row, ex.col)) };
  const exAnchor: Anchor = { area: 'grid', row: ex.row, col: ex.col };

  /** "g 27 % · r 24 % · ✱ 20 %": where the arrows of one row go. */
  const arrows = (qi: number) => {
    const at = st.attention!;
    const keys = at.from === 'memory' ? c.m.inputTokens : st.tokens;
    return Array.from({ length: at.keys }, (_, k) => ({ k, w: meanAttention(at, st.active.length, qi, k) }))
      .sort((x, y) => y.w - x.w).slice(0, 3)
      .map(({ k, w }) => `${shortLabel(keys[k])} ${Math.round(w * 100)} %`).join(' · ');
  };

  switch (st.kind) {
    case 'letters': {
      if (read) {
        const n = st.active.length;
        return [
          { text: t('mx.b.read.tokens', { word: c.word, n, task: tok(0), list: st.tokens.map(shortLabel).join(' ') }), from: 0, to: 0 },
          ...st.active.map((r, j) => ({
            text: t('mx.b.read.row', { tok: tok(r), id: idOf(r), pos: r + 1 }), from: j / n, to: (j + 1) / n, anchor: { area: 'row', row: r } as Anchor,
          })),
          { text: t('mx.b.read.lettersDone', { n, d }), from: 1, to: 1 },
        ];
      }
      const beats: Beat[] = [{
        text: t(st.pass === 0 ? 'mx.b.write.first' : 'mx.b.write.row', { tok: tok(last), id: idOf(last), pos: last + 1 }),
        from: 0, to: 1, anchor: { area: 'row', row: last },
      }];
      if (st.pass === 1) beats.push({ text: t('mx.b.write.remembered'), from: 1, to: 1, anchor: { area: 'row', row: 0 } });
      return beats;
    }
    case 'position': {
      const pos = before(ex.row, ex.col), sum = after(ex.row, ex.col);
      return [
        { text: read ? t('mx.b.read.pos') : t('mx.b.write.pos', { pos: last + 1 }), from: 0, to: 0.45, anchor: read ? undefined : { area: 'side', row: last } },
        { text: t('mx.b.add', { c: ex.col + 1, tok: tok(ex.row), a: f2(pos), p: sum - pos < 0 ? `(${f2(sum - pos)})` : f2(sum - pos), sum: f2(sum) }), from: 0.45, to: 1, anchor: exAnchor },
      ];
    }
    case 'self': {
      const qi = read ? (c.focus >= 0 && c.focus < st.active.length ? c.focus : st.active.length - 1) : 0;
      const q = st.active[qi];
      const look: Beat = read
        ? { text: t('mx.b.read.look', { tok: tok(q), top: arrows(qi) }), from: 0, to: 0.45, anchor: { area: 'row', row: q } }
        : { text: t(st.pass === 0 ? 'mx.b.write.backFirst' : 'mx.b.write.back', { top: arrows(0) }), from: 0, to: 0.45, anchor: { area: 'row', row: q } };
      look.uses = ['q', 'k'];
      const beats: Beat[] = [];
      // The grids of learned numbers: told the first time each lane comes here.
      if (st.pass <= 0) beats.push({ text: t('mx.b.machine.att'), from: 0, to: 0 });
      beats.push(look, { text: t('mx.b.mix', exVars), from: 0.45, to: 1, anchor: exAnchor, uses: ['v', 'o'] });
      if (read) beats.push({ text: t('mx.b.read.lookAll'), from: 1, to: 1 });
      return beats;
    }
    case 'cross':
      return [
        ...(st.pass === 0 ? [{ text: t('mx.b.machine.cross'), from: 0, to: 0 }] : []),
        { text: t('mx.b.write.cross', { top: arrows(0) }), from: 0, to: 0.45, anchor: { area: 'row', row: last }, uses: ['q', 'k'] },
        { text: t('mx.b.mix', exVars), from: 0.45, to: 1, anchor: exAnchor, uses: ['v', 'o'] },
      ];
    case 'think':
      return [
        { text: t('mx.b.think.up', { tok: tok(last) }), from: 0, to: 0.5, anchor: { area: 'row', row: last }, uses: ['up'] },
        { text: t('mx.b.think.down'), from: 0.5, to: 1, uses: ['down'] },
        { text: t('mx.b.biggest', exVars), from: 1, to: 1, anchor: exAnchor },
      ];
    case 'knobs': {
      const range = (g: Float32Array | null, r: number) => {
        let lo = Infinity, hi = -Infinity;
        for (let i = 0; i < d; i++) { const v = g?.[r * d + i] ?? 0; lo = Math.min(lo, v); hi = Math.max(hi, v); }
        return [f2(lo), f2(hi)];
      };
      const [min, max] = range(st.before, last), [min2, max2] = range(st.grid, last);
      return [
        { text: t('mx.b.knobs', { tok: tok(last), min, max, min2, max2 }), from: 0, to: 1, anchor: { area: 'row', row: last } },
        { text: t(read ? 'mx.b.read.done' : 'mx.b.write.final'), from: 1, to: 1 },
        { text: t('mx.b.machine.fixed'), from: 1, to: 1 },
      ];
    }
    case 'guess': {
      const g = st.guess!;
      const top = Array.from(g.probs.keys()).sort((a, b) => g.probs[b] - g.probs[a]).slice(0, 3)
        .map((i) => `${shortLabel(c.tokens[i])} ${(g.probs[i] * 100).toFixed(g.probs[i] < 0.1 ? 1 : 0)} %`).join(' · ');
      const end = g.chosen === EOS;
      return [
        { text: t('mx.b.guess.compare', { v: g.scores.length, s: f2(g.scores[g.chosenId] ?? 0) }), from: 0, to: 0.5, anchor: { area: 'row', row: last } },
        { text: t('mx.b.guess.chances', { top }), from: 0.5, to: 1, anchor: { area: 'side', row: g.chosenId } },
        {
          text: end ? t('mx.b.guess.end', { word: c.output || '∅' }) : t('mx.b.guess.write', { chosen: c.full(g.chosen), next: st.pass + 2 }),
          from: 1, to: 1, anchor: { area: 'side', row: g.chosenId },
        },
      ];
    }
  }
}
