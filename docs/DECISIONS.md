# Decisions log

Deviations from, and refinements of, `docs/DESIGN.md`. Newest last.

## M1 — brain library

- **Separate position tables for encoder and decoder** (`encPos`, `decPos`, 22 × 32 each). The spec says "learned positions, 22"; the encoder and decoder positions mean different things, so they get their own tables.
- **Output bias** (`outBias`, one number per token) on top of the tied output projection. Cheap (49 numbers) and lets the model learn letter frequencies without bending the letter map.
- **Initialisation:** linear layers N(0, 1/√d_in); attention-output and FFN-down layers scaled by an extra 1/√(2·layers); letter embeddings N(0, 0.3); positions N(0, 0.1). LayerNorm γ = 1, β = 0.
- **Parameter count:** 24 529 for the default config with a 49-token vocabulary.
- **Target-token order in the vocabulary is form-major** (`<cs> <en> <cs:young> <en:young>`), so adding a form only appends tokens. Models remap embeddings *by token text* (`growVocab`), so the order is not load-bearing.
- **Greedy decoding uses a key/value cache** (`Block.step`). Evaluating ~300 words every 50 steps would otherwise cost more than training itself. A test checks the cached path against full recomputation.
- **Speed (Node 22, 1 core):** ≈ 45 training steps/s at batch 16 with random 4–9 letter words (`npx tsx scripts/bench.ts`). Above the 30 steps/s target without buffer reuse, so the code stays allocation-per-op for readability.
- **Adam state after vocabulary growth** is restarted for the grown parameters only (their size changed); all other parameters keep their moments.

## M2 (work in progress) — generalisation experiments

- Translation on the ~150-animal base set memorises the known animals (94–100%) but scores **0% on never-seen animals**, even identical words (leopard → "kowpa"). Weight decay alone did not help.
- The same architecture copies never-seen random strings 100% correctly, so the network can learn letter-level rules; the data (≈240 pairs, ~40% unrelated words) is what pushes it to memorise.
- "Keep it as it is" practice (`<cs> tygr → tygr`, `identityWeight`) plus dropout 0.1 lifts never-seen lookalikes to ~20–30% but slows memorisation (known 40–75% at 6k steps).
- Dropout was added as an option (`TranslatorConfig.dropout`, default off) and AdamW-style weight decay as an option (`weightDecay`, default 0).
- **Open question (with the user):** switch the main task from cs↔en translation to within-language word forms (animal → baby / male / female), where most answers are "copy the stem, change the ending".

### Word-forms feasibility check (`scripts/experiments/forms-check.ts`)

Within-language forms (male / female / young), 26 Czech + 19 English training animals (118 examples), 7 + 5 never-seen animals (33 forms), 3–4k steps.

| Setup | known cs | known en | never-seen cs | never-seen en |
|---|---|---|---|---|
| plain | 72/72 | 46/46 | 0/20 | 0/13 |
| + dropout 0.1 | 72/72 | 45/46 | 0/20 | 0/13 |
| + copy practice on all 276 base names, dropout 0.1 | 66–72/72 | 43–46/46 | 5–7/20 | 0–1/13 |

With copy practice the model gets regular Czech forms right (buvol → buvolice, páv → pávice, vlk → vlk) and makes plausible over-regularised mistakes (vlk → *vlkice*, ovce → *ovcour*, kůň → *koň*). English stays at ~0, as expected for irregular forms. The signal is real but small with ~30 animals; the full dataset is needed to judge it properly.

## Dataset v2: word forms (male / female / young)

Agreed with the user: the main task becomes **within-language word forms** instead of cs↔en translation. Input `<cs:female> tygr` → `tygřice`.

- **Format:** each animal keeps `names` and gains optional `forms` (`{ cs: { male, female, young } }`), `formsAlt` (accepted alternatives, never trained) and `unsure` (entries waiting for native-speaker review, e.g. `"cs.female"`). The old `young`/`youngAlt` fields are gone.
- `data/forms.json` is now `["name", "male", "female", "young"]`. `name` is used for "copy the name" practice (`<cs> slon → slon`). Vocabulary: 4 + 43 + 2 × 4 = 55 tokens.
- **Missing forms are left empty** (žirafa, panda, …): they are what the model invents. Words unsuitable for an 11+ audience are left out.
- **Form kind** (for colouring and the scoreboard) replaces lookalike/partial/different within a language: the share of the name's letters found in order in the form (LCS / name length, without diacritics): ≥ 0.75 `pattern` (tygr → tygřice), ≥ 0.5 `partly` (lev → lvice), otherwise `irregular` (pes → štěně).
- **Draft content:** 151 animals; 92 have forms (Czech 49 animals / 128 forms, English 87 animals / 174 forms); 11 never-seen animals (53 forms); 19 entries marked unsure. Solid Czech male/female/young words exist for only ~50 common animals, not ~100 as hoped. `docs/DATA_REVIEW.md` lists everything for review.

### Full-data check (`scripts/experiments/forms-full.ts`, 5 000 steps)

| Setup | known cs | known en | never-seen cs exact (+ close) | never-seen en |
|---|---|---|---|---|
| plain, no dropout | 94/102 | 143/147 | 0/26 | 2/27 |
| copy practice 1:1 + dropout 0.1, seeds 1–3 | 89–97/102 | 109–123/147 | 3–7/26 (+5) | 2–4/27 |

Never-seen Czech "pattern" forms: 3–7 of 18 exact; most errors are plausible (vlk → *vlkice*, kamzík → *kamzíkice*, žába → *žábě*). English never-seen stays near zero, as expected. English known forms train more slowly with copy practice. Still short of the goal; the next lever is a copy-friendly model change.

## Copy mechanism and target-token start

- **Copy mechanism** (`TranslatorConfig.copy`): a single-head attention from the final decoder state over the input letters (never the target token or padding) plus a learned write-or-copy gate: `p = g·softmax(logits) + (1−g)·copy`. Loss is the NLL of the mixed distribution. New ops `pointerMix` and `nllFromProbs`, both gradient-checked. +2 145 parameters. `inspect()` returns the copy attention and a per-letter "wrote it itself" share for visualisation.
- **Target-token start** (`TranslatorConfig.targetStart`): the decoder starts from the target token (`<cs:female>`) instead of `<bos>`. Without it the model often mixed up forms (čáp → male *čápice*, female *čáp*).

Results (`forms-full.ts`, 5 000 steps; never-seen Czech forms are 26, 18 of them "pattern"):

| Setup | known cs / en | never-seen cs exact |
|---|---|---|
| copy mechanism only | 96–101/102, 137–146/147 | 1/26 |
| copy mechanism + copy practice + dropout 0.1 | 93/102, 139/147 | 7/26 |
| + target-token start (seeds 1, 2) | 96/102, 131–141/147 | 7–8/26 |
| target-token start + copy practice, no copy mechanism | 89/102, 124/147 | 8/26 |

Conclusions:
- **Copy practice is what makes never-seen forms work.** The copy mechanism mainly keeps known forms accurate while copy practice is on (English 131–141 vs 109–123 without it) and gives a useful visual; it does not by itself improve generalisation.
- What the model reliably generalises is **keeping the stem** (male vlk → vlk, female žába → žába). Adding an ending to a never-seen stem is hit-and-miss (buvol → buvolice in some runs, *bulice* in others; velbloud → *veliceud*). Consonant changes (vlk → vlčice, zajíc → zaječice) fail, with believable mistakes.
- English never-seen forms stay near zero, as expected.
- **Defaults chosen:** copy mechanism on, target-token start on, copy practice 1:1, dropout 0.1, LR 3e-3, batch 16. The never-seen set is small (26 Czech forms), so single-run numbers move by ±2.

## M2 — trainer on word forms

- `TranslatorTrainer` became **`FormTrainer`** (`src/train/trainer.ts`). Batches mix word-form tasks of train-split animals (`formTasks`) with "copy the name" practice (`nameTasks(…, true)`) at 1:1 (`copyWeight`). Defaults in `TRAINER_DEFAULTS`: copy mechanism, target-token start, dropout 0.1, Adam LR 3e-3, batch 16.
- **Scoreboard per language** (`Scores`): known · never-seen pattern / partly / irregular · never-seen exact (`unseen`) · never-seen right-or-close (`close`) · fresh / old (Teach mode).
- **Inventions:** `Dataset.missingFormTasks()` lists the forms the data lacks; `evaluate({ inventions: true })` asks for them too and marks them `invented` (never scored). `confidence(trace, eos)` = geometric mean of the chosen letters' probabilities (DESIGN §8.7).
- `unsure` entries are still trained on, as in the experiments; dropping them is a data-review decision.
- Check run (`npx tsx scripts/train-node.ts 5000 1`): known cs 98% / en 93%, never-seen cs 23% exact (35% incl. close), en 11%; 40 steps/s in Node.
- **Evaluation cost:** the 355 real forms take ~250 ms, all 906 forms incl. inventions ~640 ms (Node). Evaluating everything every 50 steps would halve training speed, so the worker evaluates real forms on a schedule and inventions less often (see the worker entry).
