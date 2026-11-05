/** Dataset linter: `npm run validate-data`. Also run by the test suite. */
import { allPacks, baseBundle } from '../src/data/bundle';
import { KINDS, mergePacks, validatePacks, classify } from '../src/data/dataset';
import { buildVocab } from '../src/nn/vocab';

const bundle = baseBundle();
const vocab = buildVocab(bundle.vocabSpec, bundle.languages, bundle.forms);
const codes = bundle.languages.map((l) => l.code);
const errors = validatePacks(allPacks(), vocab, codes);

for (const pack of allPacks()) {
  const animals = mergePacks([pack]);
  const counts = Object.fromEntries(KINDS.map((k) => [k, 0]));
  const [a, b] = codes;
  for (const an of animals) if (an.names[a] && an.names[b]) counts[classify(an.names[a], an.names[b])]++;
  const unseen = animals.filter((x) => x.split === 'unseen').length;
  const young = animals.filter((x) => Object.keys(x.young).length > 0).length;
  console.log(
    `${pack.pack.padEnd(12)} ${String(animals.length).padStart(4)} animals | ${unseen} unseen | ${young} with young names | ` +
      KINDS.map((k) => `${k} ${counts[k]} (${Math.round((100 * counts[k]) / animals.length)}%)`).join(', '),
  );
}
console.log(`vocabulary: ${vocab.size} tokens`);
if (errors.length) {
  console.error(`\n${errors.length} problem(s):`);
  for (const e of errors) console.error(' - ' + e);
  process.exit(1);
}
console.log('Data OK');
