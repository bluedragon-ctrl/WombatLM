/**
 * "What have the thinking units learned?": each thinking unit (detector) of
 * the writing part explained by the moments that switch it on the most.
 *
 * The model reads every learning example with its right answer (as in
 * learning, without dropout). At every letter it writes, each unit of the
 * writer's feed-forward step has a value (after GELU). For each unit we keep:
 * - how often it is on at all;
 * - its strongest moments: which animal, which form, how much of the answer
 *   was written and which letter comes next;
 * - a "theme", if those moments clearly share something (the next letter,
 *   the letter just written, the language and form, ...), much more often
 *   than chance;
 * - what it really does there: at each of those moments the unit is switched
 *   off (its part of the numbers taken away) and the guess is made again.
 *   How much the right letter's chance falls, and which letters lose the
 *   most, is measured, not estimated: in the last writing layer, the unit
 *   only adds its pattern to the numbers that go to the last volume knobs and
 *   the letter table;
 * - the letters its pattern points to by itself (a straight-line hint, for
 *   Expert).
 *
 * Small models often have units without a clear theme (they help with many
 * things at once). They are reported as such, not hidden.
 */
import type { Dataset, Task } from '../data/dataset';
import { makeBatch, type Example, type SpecialIds, type Translator } from '../nn/translator';
import { rowSoftmax, type Trace } from '../nn/transformer';
import { lensLogits } from '../nn/explain';
import type { UnitInfo, UnitMoment, UnitTheme } from './protocol';

/** A unit is "on" above this value (as in "look inside one letter"). */
export const ON = 0.05;
/** Strongest moments kept per unit. */
export const TOP = 8;
/** A theme needs at least this many of the TOP moments, and this many times its share among all moments. */
const THEME_MIN = 6;
/** A letter counts as helped if switching the unit off lowers its chance by more than this (on average). */
const HELPS_MIN = 0.005;
const THEME_LIFT = 2;

interface Moment {
  task: number;
  /** Letters of the answer already written. */
  at: number;
  /** Features of the moment, for themes. */
  next: string;
  prev: string;
  /** Token id of the right letter (or the end). */
  right: number;
}

/** Features a theme can be about, from most to least specific. */
const KINDS: UnitTheme['kind'][] = ['next', 'prev', 'task', 'form', 'lang'];

function feature(kind: UnitTheme['kind'], m: Moment, tasks: readonly Task[]): string {
  const t = tasks[m.task];
  switch (kind) {
    case 'next': return m.next;
    case 'prev': return m.prev;
    case 'task': return `${t.to}:${t.form}`;
    case 'form': return t.form;
    case 'lang': return t.to;
  }
}

export function profileUnits(model: Translator, dataset: Dataset, tasks: readonly Task[], ids: SpecialIds, layer = 'dec0', batchSize = 64): { units: UnitInfo[]; moments: number } {
  const vocab = dataset.vocab;
  const charOf = (id: number) => (id === ids.eos ? '' : vocab.tokens[id]);
  const examples: Example[] = tasks.map((t) => ({
    target: dataset.targetId(t.to, t.form),
    src: vocab.encodeWord(t.source).slice(0, model.config.maxLen - 2),
    tgt: vocab.encodeWord(t.answer).slice(0, model.config.maxLen - 2),
  }));
  const moments: Moment[] = [];
  const rows: Float32Array[] = []; // one per moment: every unit's value
  const res: Float32Array[] = []; // one per moment: the letter's numbers after the layer
  for (let s = 0; s < examples.length; s += batchSize) {
    const part = examples.slice(s, s + batchSize);
    const batch = makeBatch(part, ids);
    const trace: Trace = new Map();
    model.logits(batch, trace);
    const hidden = trace.get(`${layer}.ffn.hidden`);
    if (!hidden) throw new Error(`units: missing trace ${layer}.ffn.hidden`);
    const units = hidden.shape[hidden.shape.length - 1];
    const out = trace.get(`${layer}.out`)!;
    const dm = out.shape[out.shape.length - 1];
    part.forEach((ex, b) => {
      // Letter t of the answer is written after t letters (t = tgt.length: the end of the word).
      for (let t = 0; t <= ex.tgt.length; t++) {
        const r = b * batch.Tt + t;
        rows.push(hidden.data.slice(r * units, (r + 1) * units));
        res.push(out.data.slice(r * dm, (r + 1) * dm));
        moments.push({
          task: s + b, at: t,
          next: t < ex.tgt.length ? charOf(ex.tgt[t]) : '',
          prev: t === 0 ? '<start>' : charOf(ex.tgt[t - 1]),
          right: t < ex.tgt.length ? ex.tgt[t] : ids.eos,
        });
      }
    });
  }
  const nUnits = rows[0]?.length ?? 0;
  // How common each feature value is among all moments (the "chance" a theme must beat).
  const base = new Map<string, number>();
  for (const m of moments) for (const k of KINDS) { const key = `${k}|${feature(k, m, tasks)}`; base.set(key, (base.get(key) ?? 0) + 1); }

  const block = model.dec.find((b) => b.name === layer)!;
  const down = block.ffn.down.W; // [units, d]
  const d = down.shape[1];
  const { gamma } = model.decNorm;
  const V = model.tokEmb.shape[0];
  const letters: number[] = [];
  for (let i = 0; i < V; i++) if (i === ids.eos || vocab.isChar(i)) letters.push(i);

  // Switching a unit off is exact only in the last writing layer (later layers would react to it).
  const last = model.dec[model.dec.length - 1].name === layer;
  const units: UnitInfo[] = [];
  for (let k = 0; k < nUnits; k++) {
    let on = 0;
    const order: number[] = [];
    for (let i = 0; i < rows.length; i++) { if (rows[i][k] > ON) on++; order.push(i); }
    order.sort((a, b) => rows[b][k] - rows[a][k]);
    // The strongest moments, each shown once (several animals can share an answer, e.g. "calf").
    const top: number[] = [];
    const seen = new Set<string>();
    for (const i of order) {
      if (top.length === TOP || rows[i][k] <= ON) break;
      const m = moments[i], t = tasks[m.task];
      const key = `${t.answer}|${m.at}|${t.to}|${t.form}`;
      if (seen.has(key)) continue;
      seen.add(key);
      top.push(i);
    }
    const best: UnitMoment[] = top.map((i) => {
      const m = moments[i], t = tasks[m.task];
      return { source: t.source, answer: t.answer, lang: t.to, form: t.form, at: m.at, next: m.next, value: rows[i][k] };
    });
    // Theme: the feature value shared by most of the strongest moments, far more often than by chance.
    let theme: UnitTheme | null = null;
    if (top.length === TOP) {
      for (const kind of KINDS) {
        const counts = new Map<string, number>();
        for (const i of top) { const v = feature(kind, moments[i], tasks); counts.set(v, (counts.get(v) ?? 0) + 1); }
        const [value, count] = [...counts].sort((a, b) => b[1] - a[1])[0];
        const share = (base.get(`${kind}|${value}`) ?? 0) / moments.length;
        if (count >= THEME_MIN && count / TOP >= THEME_LIFT * share) { theme = { kind, value, count, of: TOP }; break; }
      }
    }
    // Letters it pushes towards: its pattern, centred and scaled by the last volume knobs, against the letter table.
    const row = down.data.subarray(k * d, (k + 1) * d);
    let mean = 0;
    for (let j = 0; j < d; j++) mean += row[j];
    mean /= d;
    const dir = Float32Array.from(row, (v, j) => (v - mean) * gamma.data[j]);
    const scores = letters.map((tok) => {
      let s = 0;
      for (let j = 0; j < d; j++) s += dir[j] * model.tokEmb.data[tok * d + j];
      return { char: charOf(tok), score: s };
    }).sort((a, b) => b.score - a.score);
    // Switched off at its strongest moments: the right letter's chance with and without it, and who loses most.
    let off: UnitInfo['off'] = null;
    if (last && top.length) {
      let withP = 0, withoutP = 0;
      const drop = new Float64Array(V);
      for (const i of top) {
        const x = res[i];
        const p1 = rowSoftmax(lensLogits(model, x), 0, V);
        const xo = Float32Array.from(x, (v, j) => v - rows[i][k] * row[j]);
        const p0 = rowSoftmax(lensLogits(model, xo), 0, V);
        withP += p1[moments[i].right];
        withoutP += p0[moments[i].right];
        for (const tok of letters) drop[tok] += p1[tok] - p0[tok];
      }
      const n = top.length;
      const helps = letters.map((tok) => ({ char: charOf(tok), drop: drop[tok] / n }))
        .filter((x) => x.drop > HELPS_MIN).sort((a, b) => b.drop - a.drop).slice(0, 3);
      off = { withP: withP / n, withoutP: withoutP / n, helps };
    }
    units.push({ unit: k, on: rows.length ? on / rows.length : 0, top: best, theme, pushes: scores.slice(0, 3), off });
  }
  return { units, moments: moments.length };
}
