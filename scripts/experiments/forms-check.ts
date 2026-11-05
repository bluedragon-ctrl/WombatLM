/**
 * Feasibility check: can the tiny translator learn animal word forms
 * (male / female / young) within one language and get never-seen animals right?
 *   npx tsx scripts/experiments/forms-check.ts [steps] [seed] [dropout]
 */
import { Tape } from '../../src/nn/autograd';
import { Adam } from '../../src/nn/optim';
import { Rng } from '../../src/nn/rng';
import { Example, TRANSLATOR_DEFAULTS, Translator, makeBatch } from '../../src/nn/translator';
import { buildVocab, targetToken } from '../../src/nn/vocab';
import { baseBundle } from '../../src/data/bundle';

// base | male | female | young   ('' = no such word). '*' marks never-seen animals.
const CS = `
tygr|tygr|tygřice|tygře
lev|lev|lvice|lvíče
slon|slon|slonice|slůně
medvěd|medvěd|medvědice|medvídě
orel|orel|orlice|orlíče
holub|holub|holubice|holoubě
osel|osel|oslice|osle
králík|králík|králice|králíče
zajíc|zajíc|zaječice|zajíče
šakal|šakal|šakalice|
hroch|hroch|hrošice|
bobr|bobr|bobřice|
had|had||hádě
krocan|krocan|krůta|krůtě
jelen|jelen|laň|kolouch
srna|srnec|srna|srnče
kočka|kocour|kočka|kotě
pes|pes|fena|štěně
kráva|býk|kráva|tele
prase|kanec|prasnice|sele
koza|kozel|koza|kůzle
kachna|kačer|kachna|kachně
husa|houser|husa|house
slepice|kohout|slepice|kuře
liška|lišák|liška|
opice|opičák|opice|
*vlk|vlk|vlčice|vlče
*velbloud|velbloud|velbloudice|velbloudě
*páv|páv|pávice|pávě
*čáp|čáp|čápice|čápě
*buvol|buvol|buvolice|
*kůň|hřebec|klisna|hříbě
*ovce|beran|ovce|jehně
`;
const EN = `
tiger|tiger|tigress|cub
lion|lion|lioness|cub
elephant|bull|cow|calf
bear|||cub
eagle|||eaglet
pigeon|||squab
donkey|jack|jenny|foal
deer|stag|doe|fawn
cat|tom|queen|kitten
dog|dog||puppy
cow|bull|cow|calf
pig|boar|sow|piglet
goat|||kid
duck|drake|duck|duckling
goose|gander|goose|gosling
hen|rooster|hen|chick
turkey|tom|hen|poult
rabbit|buck|doe|kit
fox|||kit
*wolf|||pup
*camel|bull|cow|calf
*peacock|peacock|peahen|peachick
*horse|stallion|mare|foal
*sheep|ram|ewe|lamb
`;
const FORMS = ['male', 'female', 'young'];
const steps = Number(process.argv[2] ?? 3000);
const seed = Number(process.argv[3] ?? 1);
const drop = Number(process.argv[4] ?? 0);
/** Weight of "copy the name" practice on all base-dataset names (0 = off). */
const copyWeight = Number(process.argv[5] ?? 0);

const b = baseBundle();
const vocab = buildVocab(b.vocabSpec, b.languages, ['name', ...FORMS]);
interface Item { lang: string; form: string; src: string; tgt: string; unseen: boolean }
const items: Item[] = [];
for (const [lang, table] of [['cs', CS], ['en', EN]] as const) {
  for (const line of table.trim().split('\n')) {
    const unseen = line.startsWith('*');
    const [base, ...forms] = line.replace('*', '').split('|');
    forms.forEach((w, i) => { if (w) items.push({ lang, form: FORMS[i], src: base, tgt: w, unseen }); });
  }
}
const train = items.filter((i) => !i.unseen), test = items.filter((i) => i.unseen);
// Copy practice: <cs> slon -> slon for every name in the base dataset, except never-seen animals.
const hidden = new Set(test.map((t) => t.src));
const copies: Item[] = b.packs[0].animals.flatMap((a) =>
  Object.entries(a.names).filter(([, w]) => !hidden.has(w)).map(([lang, w]) => ({ lang, form: 'name', src: w, tgt: w, unseen: false })));
const pick = () => (copyWeight > 0 && rng.next() < copyWeight / (1 + copyWeight) ? copies[rng.int(copies.length)] : train[rng.int(train.length)]);
const toEx = (i: Item): Example => ({ target: vocab.id(targetToken(i.lang, i.form)), src: vocab.encodeWord(i.src), tgt: vocab.encodeWord(i.tgt) });
const ids = { pad: vocab.pad, bos: vocab.bos, eos: vocab.eos };
const model = new Translator({ ...TRANSLATOR_DEFAULTS, vocabSize: vocab.size, dropout: drop }, seed);
const opt = new Adam(model.params.list, { lr: 3e-3 });
const rng = new Rng(seed + 7);
const run = (set: Item[]) => model.greedy(set.map((i) => ({ target: vocab.id(targetToken(i.lang, i.form)), src: vocab.encodeWord(i.src) })), ids).map((o) => vocab.decode(o));
const acc = (set: Item[], lang: string) => {
  const s = set.filter((i) => i.lang === lang);
  const out = run(s);
  return `${s.filter((x, k) => out[k] === x.tgt).length}/${s.length}`;
};
console.log(`copy practice ${copyWeight ? copies.length : 0} words, train examples ${train.length}, never-seen ${test.length}, vocab ${vocab.size}`);
for (let s = 1; s <= steps; s++) {
  const batch = makeBatch(Array.from({ length: 16 }, () => toEx(pick())), ids);
  opt.zeroGrad();
  const tape = new Tape();
  const loss = tape.run(() => model.loss(batch, vocab.pad));
  tape.backward(loss);
  opt.step();
  if (s % 500 === 0) console.log(`step ${s} loss ${loss.item().toFixed(3)} | known cs ${acc(train, 'cs')} en ${acc(train, 'en')} | never-seen cs ${acc(test, 'cs')} en ${acc(test, 'en')}`);
}
const out = run(test);
test.forEach((t, k) => console.log(`  ${t.lang} ${t.form.padEnd(6)} ${t.src.padEnd(9)} -> ${out[k].padEnd(12)} ${out[k] === t.tgt ? '✓' : `✗ (${t.tgt})`}`));
