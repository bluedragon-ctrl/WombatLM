/**
 * Word forms on the full base dataset, with "copy the name" practice.
 *   npx tsx scripts/experiments/forms-full.ts [steps] [seed] [dropout] [copyWeight]
 */
import { Tape } from '../../src/nn/autograd';
import { Adam } from '../../src/nn/optim';
import { Rng } from '../../src/nn/rng';
import { TRANSLATOR_DEFAULTS, Translator, makeBatch } from '../../src/nn/translator';
import { baseBundle } from '../../src/data/bundle';
import { Dataset, FORM_KINDS, Task, formKind } from '../../src/data/dataset';
import { judge } from '../../src/train/trainer';

const steps = Number(process.argv[2] ?? 5000);
const seed = Number(process.argv[3] ?? 1);
const drop = Number(process.argv[4] ?? 0.1);
const copyWeight = Number(process.argv[5] ?? 1);
const ds = new Dataset(baseBundle());
const v = ds.vocab;
const ids = { pad: v.pad, bos: v.bos, eos: v.eos };
const train = ds.formTasks((a) => a.split === 'train');
const test = ds.formTasks((a) => a.split === 'unseen');
const copies = ds.nameTasks((a) => a.split === 'train', true);
const model = new Translator({ ...TRANSLATOR_DEFAULTS, vocabSize: v.size, dropout: drop }, seed);
const opt = new Adam(model.params.list, { lr: 3e-3 });
const rng = new Rng(seed + 7);
const ex = (t: Task) => ({ target: ds.targetId(t.to, t.form), src: v.encodeWord(t.source), tgt: v.encodeWord(t.answer) });
const pick = () => (rng.next() < copyWeight / (1 + copyWeight) ? copies[rng.int(copies.length)] : train[rng.int(train.length)]);
const run = (ts: Task[]) => model.greedy(ts.map((t) => ({ target: ds.targetId(t.to, t.form), src: v.encodeWord(t.source) })), ids).map((o) => v.decode(o));
const score = (ts: Task[]) => { const o = run(ts); return { ok: ts.filter((t, i) => judge(o[i], t.accepted) === 'correct').length, close: ts.filter((t, i) => judge(o[i], t.accepted) === 'close').length, n: ts.length, o }; };
console.log(`form tasks: train ${train.length}, never-seen ${test.length}; copy practice ${copies.length}`);
for (let s = 1; s <= steps; s++) {
  const batch = makeBatch(Array.from({ length: 16 }, () => ex(pick())), ids);
  opt.zeroGrad(); const tape = new Tape(); const loss = tape.run(() => model.loss(batch, v.pad)); tape.backward(loss); opt.step();
  if (s % 1000 === 0) {
    const parts = ds.languageCodes.map((l) => {
      const k = score(train.filter((t) => t.to === l)), u = score(test.filter((t) => t.to === l));
      return `${l}: known ${k.ok}/${k.n}, never-seen ${u.ok}/${u.n} (+${u.close} close)`;
    });
    console.log(`step ${s} | ${parts.join(' | ')}`);
  }
}
const res = score(test);
for (const l of ds.languageCodes) for (const k of FORM_KINDS) {
  const idx = test.map((t, i) => [t, i] as const).filter(([t]) => t.to === l && formKind(t.source, t.answer) === k);
  console.log(`  ${l} ${k.padEnd(9)} ${idx.filter(([t, i]) => judge(res.o[i], t.accepted) === 'correct').length}/${idx.length}`);
}
test.forEach((t, i) => console.log(`  ${t.to} ${t.form.padEnd(6)} ${t.source.padEnd(9)} -> ${res.o[i].padEnd(13)} ${judge(res.o[i], t.accepted) === 'correct' ? '✓' : judge(res.o[i], t.accepted) === 'close' ? `≈ (${t.answer})` : `✗ (${t.answer})`}`));
