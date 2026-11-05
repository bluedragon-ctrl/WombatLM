/**
 * cs <-> en translation of animal names: what does the model learn, and what
 * does it show? Results and reasoning: docs/DECISIONS.md ("Translation as a
 * teaching mode"), plan: docs/TRANSLATION_PLAN.md.
 *
 *   npx tsx scripts/experiments/translate-check.ts [setup] [steps] [seed] [out.json]
 *
 * setup:
 *   plain  translation only, no dropout
 *   copy   translation + "copy the name" practice 1:1, dropout 0.1
 *   mixed  the app's forms training (forms + copy practice) plus translation, dropout 0.1
 *   forms  word forms only, as the app trains them (forms + copy practice 1:1), dropout 0.1
 *          (copy practice without never-seen animals and without translation held-out names)
 *   shared one model for both: translation 2 : forms 1 : copy practice 1, dropout 0.1.
 *          Run it for twice the steps of `forms` / `plain` and every task has seen
 *          exactly as many examples as when trained alone.
 * DROPOUT=x overrides the dropout of any setup.
 *
 * Data: every pack (205 animals). Held out from translation: every 5th animal
 * of each similarity kind (lookalike / partial / different, `classify()`),
 * sorted by id. Their names are not in the copy practice either.
 */
import { writeFileSync } from 'node:fs';
import { Tape } from '../../src/nn/autograd';
import { Adam } from '../../src/nn/optim';
import { Rng } from '../../src/nn/rng';
import { TRANSLATOR_DEFAULTS, Translator, makeBatch } from '../../src/nn/translator';
import { allPacks, baseBundle } from '../../src/data/bundle';
import { Dataset, Kind, Task, classify } from '../../src/data/dataset';
import { confidence, judge } from '../../src/train/trainer';

const setup = process.argv[2] ?? 'copy';
const steps = Number(process.argv[3] ?? 8000);
const seed = Number(process.argv[4] ?? 1);
const outFile = process.argv[5];
if (!['plain', 'copy', 'mixed', 'forms', 'shared'].includes(setup)) throw new Error(`unknown setup ${setup}`);

const ds = new Dataset({ ...baseBundle(), packs: allPacks() });
const v = ds.vocab;
const ids = { pad: v.pad, bos: v.bos, eos: v.eos };
const KINDS: Kind[] = ['lookalike', 'partial', 'different'];

const kindOf = new Map(ds.animals.map((a) => [a.id, classify(a.names.cs, a.names.en)]));
const held = new Set<string>();
for (const k of KINDS) {
  ds.animals.filter((a) => kindOf.get(a.id) === k).map((a) => a.id).sort()
    .forEach((id, i) => { if (i % 5 === 2) held.add(id); });
}
const trans = ds.nameTasks();
const trainT = trans.filter((t) => !held.has(t.animal.id));
const testT = trans.filter((t) => held.has(t.animal.id));
const copies = ds.nameTasks((a) => !held.has(a.id), true);
const formCopies = ds.nameTasks((a) => !a.unseen && !held.has(a.id), true);
const forms = ds.formTasks((a) => !a.unseen);
const formsUnseen = ds.formTasks((a) => a.unseen);

const pools: { tasks: Task[]; w: number }[] =
  setup === 'plain' ? [{ tasks: trainT, w: 1 }]
  : setup === 'copy' ? [{ tasks: trainT, w: 1 }, { tasks: copies, w: 1 }]
  : setup === 'forms' ? [{ tasks: forms, w: 1 }, { tasks: formCopies, w: 1 }]
  : setup === 'shared' ? [{ tasks: trainT, w: 2 }, { tasks: forms, w: 1 }, { tasks: formCopies, w: 1 }]
  : [{ tasks: trainT, w: 1 }, { tasks: forms, w: 1 }, { tasks: copies, w: 1 }];
const dropout = process.env.DROPOUT !== undefined ? Number(process.env.DROPOUT) : setup === 'plain' ? 0 : 0.1;
const model = new Translator({ ...TRANSLATOR_DEFAULTS, vocabSize: v.size, dropout, targetStart: true }, seed);
const opt = new Adam(model.params.list, { lr: 3e-3 });
const rng = new Rng(seed + 7);
const wsum = pools.reduce((s, p) => s + p.w, 0);
const pick = (): Task => {
  let r = rng.next() * wsum;
  for (const p of pools) if ((r -= p.w) < 0) return p.tasks[rng.int(p.tasks.length)];
  return pools[0].tasks[0];
};
const ex = (t: Task) => ({ target: ds.targetId(t.to, t.form), src: v.encodeWord(t.source), tgt: v.encodeWord(t.answer) });
const run = (ts: Task[]) => model.greedy(ts.map((t) => ({ target: ds.targetId(t.to, t.form), src: v.encodeWord(t.source) })), ids).map((o) => v.decode(o));
const acc = (ts: Task[]) => { if (!ts.length) return null; const o = run(ts); return ts.filter((t, i) => o[i] === t.answer).length / ts.length; };
const pct = (x: number | null | undefined) => (x == null || Number.isNaN(x) ? '   -' : `${Math.round(x * 100)}%`.padStart(4));
const tag = `[${setup}${process.env.DROPOUT !== undefined ? ` d${dropout}` : ''} s${seed}]`;
const withForms = setup === 'mixed' || setup === 'forms' || setup === 'shared';

console.log(`${tag} params ${model.params.count()} | translation train ${trainT.length}, held-out ${testT.length} ` +
  `(${KINDS.map((k) => `${k} ${testT.filter((t) => kindOf.get(t.animal.id) === k).length}`).join(', ')}) | copy ${copies.length} | forms ${forms.length}`);

const curve: Record<string, number | null>[] = [];
const t0 = performance.now();
for (let s = 1; s <= steps; s++) {
  const batch = makeBatch(Array.from({ length: 16 }, () => ex(pick())), ids);
  opt.zeroGrad();
  const tape = new Tape();
  const loss = tape.run(() => model.loss(batch, v.pad));
  tape.backward(loss);
  opt.step();
  if (s % 500 === 0 || s === steps) {
    const row: Record<string, number | null> = { step: s, known: acc(trainT), held: acc(testT) };
    for (const k of KINDS) row[k] = acc(testT.filter((t) => kindOf.get(t.animal.id) === k));
    if (withForms) { row.formsKnown = acc(forms); row.formsUnseen = acc(formsUnseen); }
    curve.push(row);
    console.log(`${tag} step ${String(s).padStart(5)} known ${pct(row.known)} | held-out ${pct(row.held)} ` +
      `(look ${pct(row.lookalike)} part ${pct(row.partial)} diff ${pct(row.different)})` +
      (withForms ? ` | forms known ${pct(row.formsKnown)} unseen ${pct(row.formsUnseen)}` : '') +
      ` ${((performance.now() - t0) / 1000).toFixed(0)}s`);
  }
}

// ---------------------------------------------------------------------------
// Every word once more, with confidence and the shape of the cross-attention:
//   diag  correlation between output step and the attention-weighted input
//         position (1 = reads the input left to right, like copying)
//   peak  share of the letters' attention on the single strongest letter
//   task  share of attention on the task token
// ---------------------------------------------------------------------------
function pearson(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 3) return 0;
  const mx = x.reduce((s, a) => s + a, 0) / n, my = y.reduce((s, a) => s + a, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) { sxy += (x[i] - mx) * (y[i] - my); sxx += (x[i] - mx) ** 2; syy += (y[i] - my) ** 2; }
  return sxx && syy ? sxy / Math.sqrt(sxx * syy) : 0;
}

function look(t: Task) {
  const tr = model.inspect(ds.targetId(t.to), v.encodeWord(t.source), ids);
  const a = tr.attention[Object.keys(tr.attention).find((n) => n.includes('cross'))!];
  const Ts = a.Tk - 1; // key 0 is the task token
  const Tq = Math.max(1, a.Tq - 1); // letters only, not the final end-of-word step
  let task = 0, peak = 0, diag = 0, n = 0;
  for (let h = 0; h < a.heads; h++) {
    const xs: number[] = [], ys: number[] = [];
    for (let q = 0; q < Tq; q++) {
      const row = a.data.subarray((h * a.Tq + q) * a.Tk, (h * a.Tq + q + 1) * a.Tk);
      let m = 0, pos = 0, mx = 0;
      for (let k = 1; k < a.Tk; k++) { m += row[k]; pos += row[k] * (k - 1); mx = Math.max(mx, row[k]); }
      task += row[0];
      peak += mx / Math.max(1e-9, m);
      xs.push(Tq > 1 ? q / (Tq - 1) : 0);
      ys.push(Ts > 1 ? pos / Math.max(1e-9, m) / (Ts - 1) : 0);
      n++;
    }
    diag += pearson(xs, ys);
  }
  return { out: v.decode(tr.output), conf: confidence(tr, v.eos), task: task / n, peak: peak / n, diag: diag / a.heads };
}

if (setup === 'forms') {
  if (outFile) writeFileSync(outFile, JSON.stringify({ setup, seed, steps, curve }, null, 1));
  process.exit(0);
}
const words = [...trainT.map((t) => [t, false] as const), ...testT.map((t) => [t, true] as const)].map(([t, isHeld]) => {
  const s = look(t);
  return { id: t.animal.id, from: t.from, to: t.to, source: t.source, answer: t.answer, kind: kindOf.get(t.animal.id)!, held: isHeld,
    out: s.out, status: judge(s.out, t.answer), conf: s.conf, task: s.task, peak: s.peak, diag: s.diag };
});
type Word = (typeof words)[number];
const mean = (xs: number[]) => (xs.length ? xs.reduce((s, x) => s + x, 0) / xs.length : NaN);
const f2 = (x: number) => (Number.isNaN(x) ? '  - ' : x.toFixed(2));

const heldW = words.filter((w) => w.held);
const wrong = heldW.filter((w) => w.status === 'wrong');
console.log(`\n${tag} held-out right or one letter off: ` +
  KINDS.map((k) => `${k} ${heldW.filter((w) => w.kind === k && w.status !== 'wrong').length}/${heldW.filter((w) => w.kind === k).length}`).join(', '));
console.log(`${tag} confidence: known right ${f2(mean(words.filter((w) => !w.held && w.status === 'correct').map((w) => w.conf)))}` +
  ` | held-out right ${f2(mean(heldW.filter((w) => w.status === 'correct').map((w) => w.conf)))}` +
  ` | held-out wrong ${f2(mean(wrong.map((w) => w.conf)))} (above 0.5: ${wrong.filter((w) => w.conf > 0.5).length}/${wrong.length})`);
const knownAnswers = new Set(words.filter((w) => !w.held).map((w) => w.answer));
const recalled = wrong.filter((w) => knownAnswers.has(w.out));
console.log(`${tag} answers with a word it learned for another animal: ${recalled.length}/${wrong.length} ${recalled.map((w) => `${w.source}→${w.out}`).join(', ')}`);
const czechInEnglish = wrong.filter((w) => w.to === 'en' && /[áčďéěíňóřšťúůýž]/.test(w.out));
console.log(`${tag} English answers with Czech letters: ${czechInEnglish.length} ${czechInEnglish.map((w) => `${w.source}→${w.out}`).join(', ')}`);
if (withForms) {
  // Does one task leak into the other?
  const formWords = new Set(forms.map((t) => t.answer));
  const asForm = words.filter((w) => w.status === 'wrong' && formWords.has(w.out) && !knownAnswers.has(w.out));
  console.log(`${tag} translations answered with a word form: ${asForm.length}/${words.filter((w) => w.status === 'wrong').length} ${asForm.slice(0, 8).map((w) => `${w.source}→${w.out}`).join(', ')}`);
  const fo = run(forms);
  const otherName = forms.map((t, i) => [t, fo[i]] as const)
    .filter(([t, o]) => o !== t.answer && Object.entries(t.animal.names).some(([l, n]) => l !== t.to && n === o));
  console.log(`${tag} known word forms answered with the other language's name: ${otherName.length}/${fo.filter((o, i) => o !== forms[i].answer).length} ` +
    otherName.slice(0, 8).map(([t, o]) => `${t.to}:${t.form} ${t.source}→${o}`).join(', '));
}
for (const isHeld of [false, true]) {
  console.log(`${tag} attention ${isHeld ? 'held-out' : 'known   '}: ` + KINDS.map((k) => {
    const s = words.filter((w: Word) => w.held === isHeld && w.kind === k && w.out.length > 2);
    return `${k.slice(0, 4)} diag ${f2(mean(s.map((w) => w.diag)))} peak ${f2(mean(s.map((w) => w.peak)))} task ${f2(mean(s.map((w) => w.task)))}`;
  }).join(' | '));
}
const mark = { correct: '✓', close: '≈', wrong: '✗', invented: '✨' } as const;
console.log(`\n${tag} held-out words:`);
for (const w of [...heldW].sort((a, b) => a.kind.localeCompare(b.kind) || a.from.localeCompare(b.from) || a.source.localeCompare(b.source))) {
  console.log(`  ${mark[w.status]} ${w.kind.slice(0, 4)} ${w.from}→${w.to} ${w.source.padEnd(14)} → ${w.out.padEnd(14)} (${w.answer})  conf ${w.conf.toFixed(2)}`);
}
if (outFile) writeFileSync(outFile, JSON.stringify({ setup, seed, steps, curve, words }, null, 1));
