/**
 * Does the model need to be bigger for more languages? Trains the app's
 * FormTrainer on a chosen set of data languages and prints known / never-seen
 * accuracy per language at checkpoints.
 *
 *   npx tsx scripts/experiments/languages.ts [langs] [steps] [seed] [dModel] [batch] [lr]
 *   npx tsx scripts/experiments/languages.ts cs,en 10000 1 32 16 3e-3     # the old data
 *   npx tsx scripts/experiments/languages.ts cs,en,sk,de 10000 1 48 16    # all languages, wider
 *
 * ffn is always 2 × dModel (as in the default 32 / 64).
 */
import { baseBundle } from '../../src/data/bundle';
import { Dataset } from '../../src/data/dataset';
import { FormTrainer } from '../../src/train/trainer';

const langs = (process.argv[2] ?? 'cs,en,sk,de').split(',');
const steps = Number(process.argv[3] ?? 10000);
const seed = Number(process.argv[4] ?? 1);
const dModel = Number(process.argv[5] ?? 32);
const batchSize = Number(process.argv[6] ?? 16);
const lr = Number(process.argv[7] ?? 3e-3);
const every = Math.max(1, Math.round(steps / 4));

const bundle = baseBundle();
bundle.languages = bundle.languages.filter((l) => langs.includes(l.code));
const ds = new Dataset(bundle);
const tr = new FormTrainer(ds, { seed, batchSize, lr, config: { dModel, ffn: 2 * dModel } });
const tag = `langs=${langs.join('+')} d=${dModel} batch=${batchSize} lr=${lr} seed=${seed}`;
console.log(`${tag} | params ${tr.model.params.count()} | form tasks ${tr.trainTasks().length}, copy tasks ${tr.copyTasks().length}`);

const pct = (v: number | null) => (v === null ? '  -' : `${Math.round(v * 100)}`.padStart(3));
const t0 = performance.now();
let evalMs = 0;
for (let s = 1; s <= steps; s++) {
  tr.trainStep();
  if (s % every === 0) {
    const e0 = performance.now();
    const ev = tr.evaluate();
    evalMs += performance.now() - e0;
    const sps = (s * 1000) / (performance.now() - t0 - evalMs);
    const cells = langs.map((l) => `${l} ${pct(ev.scores[l].known)}/${pct(ev.scores[l].unseen)}/${pct(ev.scores[l].close)}`);
    console.log(`step ${String(s).padStart(6)} | known/never-seen/close % | ${cells.join(' | ')} | ${sps.toFixed(1)} steps/s`);
  }
}
