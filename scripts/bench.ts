/** Quick speed check: translator training steps per second with batch 16 in Node. */
import { Tape } from '../src/nn/autograd';
import { Adam } from '../src/nn/optim';
import { Rng } from '../src/nn/rng';
import { Example, TRANSLATOR_DEFAULTS, Translator, makeBatch } from '../src/nn/translator';

const V = 49;
const model = new Translator({ ...TRANSLATOR_DEFAULTS, vocabSize: V }, 1);
console.log('params', model.params.count());
const rng = new Rng(2);
const word = () => Array.from({ length: 4 + rng.int(6) }, () => 4 + rng.int(43));
const ex = (): Example => ({ target: 47 + rng.int(2), src: word(), tgt: word() });
const opt = new Adam(model.params.list);
const ids = { pad: 0, bos: 1, eos: 2 };
const N = Number(process.argv[2] ?? 200);
const t0 = performance.now();
for (let s = 0; s < N; s++) {
  const batch = makeBatch(Array.from({ length: 16 }, ex), ids);
  opt.zeroGrad();
  const tape = new Tape();
  const loss = tape.run(() => model.loss(batch, 0));
  tape.backward(loss);
  opt.step();
}
const dt = performance.now() - t0;
console.log(`${N} steps in ${dt.toFixed(0)} ms -> ${((N * 1000) / dt).toFixed(1)} steps/s`);
const inputs = Array.from({ length: 300 }, () => ({ target: 47, src: word() }));
const t1 = performance.now();
model.greedy(inputs, ids);
console.log(`greedy eval of 300 words: ${(performance.now() - t1).toFixed(0)} ms`);
