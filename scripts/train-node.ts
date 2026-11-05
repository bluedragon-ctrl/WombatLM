/**
 * Train the translator in Node and print progress. For experiments:
 *   npx tsx scripts/train-node.ts [steps] [seed] [lr] [batch]
 */
import { baseBundle } from '../src/data/bundle';
import { Dataset } from '../src/data/dataset';
import { TranslatorTrainer } from '../src/train/trainer';

const steps = Number(process.argv[2] ?? 3000);
const seed = Number(process.argv[3] ?? 1);
const lr = Number(process.argv[4] ?? 3e-3);
const batch = Number(process.argv[5] ?? 16);
const identityWeight = Number(process.argv[6] ?? 0);
const weightDecay = Number(process.argv[7] ?? 0);
const quiet = process.argv[8] === 'quiet';
const dropoutP = Number(process.argv[9] ?? 0);
const dModel = Number(process.argv[10] ?? 32);
const ffn = Number(process.argv[11] ?? dModel * 2);
const ds = new Dataset(baseBundle());
const tr = new TranslatorTrainer(ds, { seed, lr, batchSize: batch, identityWeight, weightDecay, config: { dropout: dropoutP, dModel, ffn } });
console.log(`params ${tr.model.params.count()}, train tasks ${tr.trainTasks().length}`);
const t0 = performance.now();
const pct = (v: number | null) => (v === null ? '  - ' : `${Math.round(v * 100)}%`.padStart(4));
const closeRate = (ev: ReturnType<typeof tr.evaluate>, split: string) => {
  const rs = ev.results.filter((r) => ds.animals.find((a) => a.id === r.id)!.split === split);
  return pct(rs.filter((r) => r.status !== 'wrong').length / rs.length);
};
for (let s = 1; s <= steps; s++) {
  tr.trainStep();
  if (s % 1000 === 0 || s === steps) {
    const ev = tr.evaluate();
    const sc = ev.scores['cs-en'];
    const sps = (s * 1000) / (performance.now() - t0);
    console.log(
      `step ${String(s).padStart(5)} loss ${tr.lossEma!.toFixed(3)} | known ${pct(sc.known)} lookalike ${pct(sc.lookalike)} partial ${pct(sc.partial)} different ${pct(sc.different)} | unseen correct-or-close ${closeRate(ev, 'unseen')} | ${sps.toFixed(1)} steps/s`,
    );
  }
}
const ev = tr.evaluate();
for (const r of ev.results) {
  const a = ds.animals.find((x) => x.id === r.id)!;
  if (a.split === 'unseen' && !quiet) console.log(`  unseen ${r.from}>${r.to} ${a.names[r.from]} -> ${r.output} (${a.names[r.to]}) ${r.status}`);
}
const wrongKnown = ev.results.filter((r) => ds.animals.find((x) => x.id === r.id)!.split === 'train' && r.status !== 'correct');
console.log('known but wrong:', wrongKnown.map((r) => `${r.id}:${r.from}>${r.to}=${r.output}`).join(', '));
