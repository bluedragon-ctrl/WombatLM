/**
 * Train the word-form model in Node with the app's trainer and print progress.
 *   npx tsx scripts/train-node.ts [steps] [seed] [copyWeight] [dropout] [quiet]
 */
import { baseBundle } from '../src/data/bundle';
import { Dataset } from '../src/data/dataset';
import { FormTrainer, Scores } from '../src/train/trainer';

const steps = Number(process.argv[2] ?? 5000);
const seed = Number(process.argv[3] ?? 1);
const copyWeight = Number(process.argv[4] ?? 1);
const dropout = Number(process.argv[5] ?? 0.1);
const quiet = process.argv[6] === 'quiet';
const ds = new Dataset(baseBundle());
const tr = new FormTrainer(ds, { seed, copyWeight, config: { dropout } });
console.log(`params ${tr.model.params.count()}, form tasks ${tr.trainTasks().length}, copy tasks ${tr.copyTasks().length}`);
const pct = (v: number | null) => (v === null ? '  - ' : `${Math.round(v * 100)}%`.padStart(4));
const line = (s: Scores) =>
  `known ${pct(s.known)} | never-seen ${pct(s.unseen)} (close ${pct(s.close)}; pattern ${pct(s.pattern)} partly ${pct(s.partly)} irregular ${pct(s.irregular)})`;
const t0 = performance.now();
let evalMs = 0;
for (let s = 1; s <= steps; s++) {
  tr.trainStep();
  if (s % 1000 === 0 || s === steps) {
    const e0 = performance.now();
    const ev = tr.evaluate();
    evalMs += performance.now() - e0;
    const sps = (s * 1000) / (performance.now() - t0 - evalMs);
    console.log(`step ${String(s).padStart(5)} loss ${tr.lossEma!.toFixed(3)} ${sps.toFixed(1)} steps/s`);
    for (const lang of ds.languageCodes) console.log(`  ${lang}: ${line(ev.scores[lang])}`);
  }
}
const e0 = performance.now();
const ev = tr.evaluate({ inventions: true });
console.log(`full evaluation with inventions: ${ev.results.length} words in ${(performance.now() - e0).toFixed(0)} ms`);
if (!quiet) {
  const mark = { correct: '✓', close: '≈', wrong: '✗', invented: '✨' } as const;
  for (const r of ev.results) {
    if (r.split === 'unseen' || r.status === 'invented') {
      console.log(`  ${mark[r.status]} ${r.lang} ${r.form.padEnd(6)} ${r.source.padEnd(12)} -> ${r.output.padEnd(14)} ${r.answer ? `(${r.answer})` : ''}`);
    }
  }
}
