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

## M2 — worker and protocol

- `src/train/protocol.ts` follows DESIGN §7 with the forms design: `ask{id, text, lang, form}` replaces `translate{text, to}`; snapshots, weights, letter map, Teach and namer messages come with M3–M6. Answers carry a **badge** (`real` / `hidden` / `invention` / `free`), the real answer for real and hidden words, top-5 guesses per letter (incl. "end of word"), the copy-vs-write share, attention maps (transferred) and the confidence.
- The loop lives in `src/train/engine.ts` (no worker globals; a host passes `post`, `schedule`, `now`), so tests drive it in Node with a fake clock. `worker.ts` only wires it up and yields through a `MessageChannel` (nested `setTimeout(0)` is clamped to 4 ms).
- **Loop:** slices of ~12 ms of training, then yield. `max` speed trains back-to-back; a numeric speed earns steps over time (at most a quarter second banked), and the first step happens immediately on Play.
- **Evaluation schedule:** real forms every 100 steps (or every 3 s at slow speeds), inventions every 500 steps, plus on init / reset / `step` / `evaluate`. At ~40 steps/s this costs about 10% of training time instead of ~50% for "everything every 50 steps".
- Ticks: at most every 50 ms, carrying every batch loss since the last tick (the UI draws the loss chart from these).

## M2 — minimal browser UI

- `src/ui/app.ts` (vanilla DOM) + `src/viz/` (Canvas 2D: guess bars, line chart, colours) + `src/ui/style.css`. Panels: controls (play/pause, step ×1/×10, speed 1/5/20/max steps/s, seed + reset, expert LR and batch), the question box with **language and form pickers built from `languages.json` × `forms.json`**, answer with badge and confidence meter, guess bars with a copied/wrote strip, loss chart (linear; log in expert mode), scoreboard (lines for the selected language: known / pattern / partly / irregular; table for all languages with the "right or one letter off" row in bold), animal list (per selected language, columns male / female / young, ✓ ≈ ✗ ✨, never-seen badge, filters all / known / never seen / wrong).
- Typing an animal name of another language switches the language (forms stay within one language); the hint says "looks like …".
- **A hidden answer is shown uncoloured until "Reveal"**: colouring it green/red would give the answer away. The animal list still shows ✓/≈/✗ for never-seen animals (it is the scoreboard view) but only shows real words of known animals in its tooltips.
- UI strings live in `src/ui/i18n/{en,cs}.json` (picked from the browser language, stored in localStorage). **The Czech copy needs the user's review** (M7 polishes it).
- **Browser speed (headless Chromium in the cloud container, 1 worker):** ~37–38 training steps/s, ~32 steps/s wall-clock including evaluation and UI. ≥ 30 target met. A 2-minute run (~3 800 steps) reached known cs 87% / en 87%, never-seen Czech pattern 44% (`buvol → buvolice`).
- `e2e/smoke.spec.ts` + `playwright.config.ts`: load, play 5 s, ask a question (`npm run e2e`). Not in CI (CI has no pre-installed browser).
- **Acceptance run in the browser** (seed 1, max speed): 5 000 steps in 160 s wall-clock (38 steps/s training). Known cs **98%**, en **93%**; never-seen Czech pattern 33% (35% right or one letter off), English never-seen 11%. Identical to the Node run with the same seed (`scripts/train-node.ts 5000 1`), so training is deterministic across Node and the worker. English known forms are below the 95% target at 5k steps; whether to train longer or accept this is open for the user.

## Project priority: teaching value over accuracy (user decision)

WombatLM is a teaching demo. Its value is in **showing how a model works**: how tokens are generated letter by letter, how weights change while it learns, what it copies and what it invents. Model accuracy does not need to be perfect.

- The M2 acceptance target "known ≥ 95%" is dropped. Known cs 98% / en 93% at 5 000 steps is accepted as is; no longer default training.
- Visible mistakes are welcome: believable near-misses and inventions (vlk → *vlkice*) are part of the lesson.
- When choosing what to work on next, prefer clearer visualisations and explanations over accuracy tuning.

## M3 — seeing inside

- **Protocol** (`src/train/protocol.ts`): `inspect{id, matrixIds, mode, letterMap?}` answers with `weights` (requested matrices, transferred) and, if asked, `letterMap`. One request covers both, so the UI keeps a single "look inside" request in flight. `ready` now lists every parameter as a matrix (`MatrixInfo`: shape, what rows/columns mean, role) so the UI builds the heatmap grid from data. `answer` also carries the activations (`enc.in`, `enc0.out`, `dec.in`, `dec0.out`) for "follow the signal"; `enc.in`/`dec.in` are new trace points in `Translator.encode/decode`.
- **Change mode semantics:** "change since the previous `inspect` that came after at least one training step". If nothing was trained in between, the previous change is shown again instead of a blank picture. The very first request after training shows the latest single step: the engine keeps a copy of the weights from before each step (reused buffers, ~27k floats; no measurable slowdown, ~36 steps/s in the browser). The UI asks at most every 250 ms, so while training a frame covers ~10 steps; the note under the mode buttons says how many.
- **Gradient mode** shows the gradient of the latest batch (the trainer zeroes gradients at the start of the next step, so they survive). Before any training it is all zeros and says so.
- **Plain vs expert:** plain mode shows matrices (vectors hidden); expert adds biases and LayerNorm gains/shifts, technical names, the self-attention grids and "follow the signal".
- **Letter map:** PCA in the worker (`src/nn/pca.ts`, power iteration on the 32×32 covariance, deterministic start), over characters, end-of-word and the task tokens (not `<pad> <bos> <unk>`, which barely train). The UI aligns each frame to the previous one with 2-D orthogonal Procrustes, trying the mirror image too (this subsumes PCA sign alignment), then glides points and the view. Categories: vowel ○, consonant □, accented letter ◇, end/space/hyphen △, task ☆ (shape + colour). The panel says how much of the variation the two axes show (after 2 000 steps ~23 % + 7 %; honest that the flat map is a squeeze).
- **Layout:** the Forms tab has three columns (430 px · flexible · 340 px letter map), the attention panel sits under the guess bars. Attention grids: one per head for cross-attention plus the copy attention; rows are the output letters, the task token is shown as ✱. Hovering a guess column highlights the matching attention rows and vice versa.
- **Screenshots:** `npm run screenshots` (opt-in Playwright spec, seed 1, 2 000 steps) writes `docs/screenshots/m3/` for PR review; `npm run e2e` stays fast and now also checks the tabs, attention grids, letter map and heatmap modes.

## GitHub Pages deploy (pulled forward from M7)

- `.github/workflows/deploy.yml` builds (after `npm test`) and publishes `dist/` to GitHub Pages on every push to `main`, or by hand (`workflow_dispatch`). The site lives at `https://bluedragon-ctrl.github.io/WombatLM/`; `vite.config.ts` already uses `base: './'` and the data is bundled, so no other change was needed.
- Pulled forward from M7 at the user's request so the current state can be tried online. Until M4's replay lands, visitors start from an untrained brain and press Play.
- One-time repository setting (done by the user): Settings → Pages → Source: **GitHub Actions**.

## Learning vs answering (user feedback after M3)

User feedback: "it is not clear when it learns (builds weights) and when it generates answers, passing through the net." Before this change the UI re-asked the current question after every tick, so the answer changed "by itself" while training and the two looked like one process.

- **Hybrid two-step split (user's choice among strict / hybrid / labels only).** The Forms tab has two numbered phases: ① Build the brain (learning) and ② Ask the brain (answering). While training runs the answer is **not** refreshed: it is greyed and says how many steps the brain has learned since ("Ask again" asks once). On pause and after Step it refreshes automatically. A "keep answering while it learns" checkbox (off by default) brings back the old live behaviour, which still shows an answer improving.
- **Practice strip** (`FormTrainer.practice()`, `tick.practice`): the latest batch's learning data (questions with their right answers) and what the brain answered **before** learning from them: `trainStep(answerFirst)` runs one greedy pass (no dropout, weights only read, no random numbers, so training is unchanged) over the first examples before the forward/backward pass. This makes the cycle honest: *pick → it answers them → measure the mistake → nudge*. Answering costs a greedy pass, so it happens on Step and at slow speeds always, at higher speeds only while the Practice tab is open and at most every 500 ms (the answered batch is kept until a tick sends it). The whole batch is sent (capped at 32).
- **Speed (headless Chromium, cloud container, seed 1, max speed, 20 s after a 3 s warm-up, two runs each):** `main` 34–38 training steps/s (mostly 36–37), wall-clock 30.9 / 31.0 steps/s; this branch 35–38 (mostly 36–37), wall-clock 31.3 / 31.1. No measurable slowdown.
- **"Numbers nudged"** is the full parameter count: Adam moves every parameter each step (momentum), even ones with a zero gradient this batch.
- **One-step cycle** is an explanation replayed after the fact: the worker does the whole step at once, then the UI lights *pick → it answers them → measure → nudge* 700 ms apart and reveals the strip in sync (right answers, then its answers, then ✓/≈/✗). It only animates for Step ×1 while paused; Step ×10 and Play show it finished.
- **Practice tab (user decision in the PR review):** the example list made the left column too tall (the animal list started below the fold at 1400×900), so it moved to its own tab with its own Play / Step ×1, badge and cycle. The Forms tab keeps the badge, the cycle and a "see all the examples it practises →" link. The list is only drawn while its tab is visible.
- **No teacher forcing in the UI (user decision in the PR review):** a first version showed the *teacher-forced* guesses the loss is computed from ("with help": each letter guessed while seeing the correct letters before it), then both those and free-running answers side by side. The user found the "help" idea did not help someone understand how AI works and asked for the simple story: learning data → what it answered → right or wrong. Teacher forcing stays an implementation detail (the batch loss in "measure the mistake" is still computed that way; the plain wording "mistake" is accurate enough).
- **Speed with answering first** (same setup as above, final version): Forms tab 35–37 training steps/s, wall-clock 31.5; Practice tab open 33–37, wall-clock 31.5. No slowdown.
- **Write it slowly** replays the `answer` already received (no new request); hovering is disabled during the replay, and changing the question stops it.
- **Frozen brain** test: `ask` ×5, `evaluate` with inventions and `inspect` leave every weight bit-identical, and the next training step equals that of a twin engine that never answered (so optimiser state and random numbers are untouched too).
