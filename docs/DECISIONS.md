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

- `TranslatorTrainer` became **`FormTrainer`** (`src/train/trainer.ts`). Batches mix word-form tasks of trained animals (`formTasks`) with "copy the name" practice (`nameTasks(…, true)`) at 1:1 (`copyWeight`). Defaults in `TRAINER_DEFAULTS`: copy mechanism, target-token start, dropout 0.1, Adam LR 3e-3, batch 16.
- **Scoreboard per language** (`Scores`): known · never-seen pattern / partly / irregular · never-seen exact (`unseen`) · never-seen right-or-close (`close`) · fresh / old (Teach mode).
- **Inventions:** `Dataset.missingFormTasks()` lists the forms the data lacks; `evaluate({ inventions: true })` asks for them too and marks them `invented` (never scored). `confidence(trace, eos)` = geometric mean of the chosen letters' probabilities (DESIGN §8.7).
- `unsure` entries are still trained on, as in the experiments; dropping them is a data-review decision. (Later removed altogether, see "One right word per answer".)
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

## Tab reorganisation, step 1: Learning data tab (user request)

User: "I do not fully understand tabs shown, so I would like to reorganise them a bit in several steps. 1. One tab with Learning data, so people can see what are actually source data engine is learning from."

- **New tab "Learning data"** (`src/ui/dataTab.ts`), placed after Forms: Forms · Learning data · Practice · Inside the brain. Forms stays the opening tab; further reordering is left for the next steps.
- **What it shows is exactly what the trainer samples from** (`FormTrainer.defaultMix`): word-form tasks (name → male / female / young, main answer only) and "copy the name" tasks of trained animals. A unit test (`tests/ui/dataTab.test.ts`) checks the tab's counts against `trainTasks()` / `copyTasks()`, in total and per language. Base pack: 529 examples (249 word forms + 280 copy the name) from 140 animals; 11 never-seen animals kept aside.
- **One table per data language** (language picker), animal name first (the copy practice), then the three forms. Learning examples are yellow (the same colour as "right answer (learning data)" on the Practice tab); a dash means no real word, so nothing to learn there. Filter all / learns from / never seen, and a search box.
- **Never-seen animals are listed, greyed, with their answers hidden** behind "show the never-seen answers" (like the reveal on the Forms tab), so the tab does not give away the test. Searching does not match hidden answers.
- **Clicking a word asks the brain** that question on the Forms tab, linking "learning data" to "answering".
- Alternative accepted spellings (`formsAlt`, `alt`) are not shown: they are used only for judging answers, never trained on.

## Tab reorganisation, step 2: the full test becomes an expert extra (user request)

User: "I do not understand table Animals on the Forms tab. Are those learning data, sample data or what?" — it was the brain's own answers for every animal (the scheduled evaluation), marked against the learning data, but it looked like data. User decision: "fully remove the full test for now or have it available on request. The story should be as simple as possible in the beginning: 1. learning data goes in, 2. the model is established, I can ask it questions about single animals. Extra steps for evaluation are advanced data for the beginner."

- **Forms tab** no longer has the Animals table or the Scoreboard. It keeps ① Build (with the loss chart: that is learning) and ② Ask (single animals).
- **New Test tab, shown only in expert mode** ("Expert (adds settings and the Test tab)"): the Animals table (with a ✓ ≈ ✗ ✨ legend, filters and its own data-language picker), the Scoreboard, "Test again now" and "Its answers at step N". It is introduced as "an extra for later" and the help says plainly that the words are the brain's answers, not learning data. Clicking a word asks it on the Forms tab. Turning expert mode off while on the tab goes back to Forms.
- **The engine tests only on request** (`watchTests`, like `watchPractice`): on init / reset / Step and in the training loop it evaluates only while the Test tab is open (same EVAL schedule: every 100 steps or 3 s, inventions every 500). Opening the tab tests the current brain right away (~0.7 s in the browser at step ~200, with inventions) unless it was already tested at this step; `evaluate` still works any time. So without the Test tab the training loop no longer spends time on evaluation.
- The scoreboard history therefore has points only from steps when the tab was open.

## Tab reorganisation, step 3: tabs follow the story (user request)

User: "Maybe there should be one tab Model learning, because this is really part of building the model. Model testing should go to a separate tab, Question mode, where you ask questions and see how the model answers. And the Test tab shows the automatic test evaluation."

- **Tabs, in story order:** Learning data · Model learning · Ask the model · Inside the brain · Test. The Forms and Practice tabs are gone.
- **Opens on Learning data** (user choice): a beginner sees the data first.
- **Model learning** = everything that changes the brain: ① Build (Play / Step / speed / seed, expert settings, status, learning badge, the one-step cycle), the mistakes (loss) chart, the latest batch (learning data and what it answered before learning from it), and the **letter map** (user choice: it is something you watch change while it learns). The Practice tab's duplicate Play / Step ×1, badge and cycle, and the "see all the examples" link, are gone. `watchPractice` is on while this tab is open.
- **Ask the model** = everything that only reads the brain: ② Ask (question, answer, step stamp, "keep answering while it learns", guesses, Write it slowly) and "where it looks" (attention). **No training controls** (user choice): a status line "The model is at step N, paused / learning right now" with a link to Model learning.
- **Test is always visible, last** (user choice; step 2 had made it expert-only). It still tests only while open.
- **Expert checkbox moved to the header:** it affects several tabs (settings on Model learning, extra attention grids on Ask, the Inside tab, the log-scale loss chart). Label back to "Expert" / "Pro pokročilé".
- Clicking a word on Learning data or Test asks it on the Ask tab.
- Tab names use "model" as the user wrote them; the panels still say "brain" (① Build the brain, ② Ask the brain, Inside the brain). Unifying the wording is a possible later step.

## "Model" instead of "brain" (user request)

User: "use 'model' everywhere instead of 'brain'". All UI text now says model: the subtitle ("A tiny model learns animal word forms"), ① Build the model, ② Ask the model, Inside the model, the step stamp, help texts and tooltips. Czech: mozek → model with the same declension (mozku → modelu, mozkem → modelem); both are masculine inanimate, so the adjectives and verbs around them stay as they were. Code comments and test names were changed too, so the codebase uses one word. Older entries in this file keep "brain" as written at the time.

## Tokens on screen (user request)

The user asked to show how tokenization works: how words become tokens, what tokens look like, and what they look like in an answer. Steps 1 and 2 of the proposal; step 3 (vocabulary list, a toy subword tokenizer for comparison with big models) is left for later.

- **Ask tab, token strip** (`renderTokens` in `app.ts`): "The model reads" (task token + letters) and "It writes" (letters + end), each chip with its token number. Built only from the answer message and `ready.tokens`, no new worker messages. Answer tokens are linked to the guess-bar columns by hover; copied letters are tinted like the "copied" label. "Show only the numbers" turns chips into plain ids. Letters without a token show a red `?` chip and a warning (they become `<unk>`).
- **Learning data tab, one learning example** (`renderExample` in `dataTab.ts`): clicking a word now shows it as tokens instead of jumping to the Ask tab (a button still asks it). Three rows: what it reads, what it is shown of the answer so far, what it must guess next: the same answer shifted by one ("predict the next token", teacher forcing). The note ties the number of guesses to the loss. Never-seen animals show only what it would read, with a note that it never learns from them. The panel starts with a word-form example, so it is never empty.
- **Plain names for special tokens** (defaults the user accepted): "end", "?", "Czech, female"; the raw token (`<eos>`, `<cs:female>`) is in the tooltip and on the chip in Expert mode. Numbers are shown small on every chip by default.
- The note says plainly that big chat models use pieces of words (~100,000 tokens) while this one uses single letters (55 tokens).
- Shared helpers in `src/ui/tokens.ts`; `tokenizeWord` is tested to give the same ids as `Vocab.encodeWord`.


## Build time: quiet learning, real speed and a step goal (user request)
The user reported 15–20 minutes to build the model on a ThinkPad T14s at 37–40 steps/s, i.e. ~35–45k steps.
- **GPU (WebGPU) not started.** At this size (24.5k weights, ~22 ms/step, ~80 % in 160×32·32×32 matrix products, ~240 tiny kernels per step) per-launch overhead dominates; it could pay off only with bigger batches (below). Still a non-goal of DESIGN §1 for now.
- **Quiet learning + real speed** (PR #7): a switch that skips testing, answering first, letter map / weights and live re-asking while running; the status shows steps per wall-clock second. Headless Chromium: ~30–34 → ~36 steps/s. On the user's T14s: practically no difference, so the pictures are not the cost.
- **Steps-to-quality (Node, seed 1, known forms cs / en):** batch 16: 2,500 steps 85 / 74 %, 5,000 98 / 93 %, 7,500 99 / 99 %, 15,000–22,500 100 / 100 %. Never-seen forms do not improve after ~5,000 steps (cs 12–27 %, en 0–11 %, noisy). So the long builds bought nothing visible.
- **Bigger batches (same examples seen, same CPU time):** batch 64 (LR 3e-3) 1,250 steps 98 / 95 %; batch 64 (LR 6e-3) also better on never-seen (cs 27–38 %, en 15–22 %, one seed); batch 256 (LR 1.2e-2) 624 steps 99 / 99 %. On a CPU a bigger step costs proportionally more; on a GPU it would not — the case for a later WebGPU experiment.
- **Step goal ("Learn until"):** training stops by itself at step 1,000 / 2,500 / **5,000 (default)** / 10,000 / no limit. Play is disabled once the goal is reached (Step still works; a higher goal enables Play again; Reset starts over). A progress bar and "about N left" (from the real speed) under the status. The final model is tested once if the Test tab is open. Protocol: `setGoal{steps}`, `tick.goal`.

## Mistake per example and per letter (user request)
"The loss number is a mystery." The practice table on the Model learning tab now shows, once the mistake is measured (cycle stage 3):
- a **Mistake (loss)** column per example: the average over its letters (incl. the word end), with a short bar;
- each letter of the right answer **tinted by its own mistake** (−ln of the probability the model gave the right letter, teacher-forced: it is shown the answer so far), hover shows "gave the right letter 12 % → mistake 2.12";
- a sentence under the table: 100 % → 0, 50 % → 0.69, 10 % → 2.3, 1 % → 4.6; a line is the average over its letters; the step's mistake is the average over **all letters of all examples** (so longer words weigh more — the per-line numbers do not average exactly to it); a right answer can still have a mistake (right, but not sure).
- The numbers come from the same forward pass as the training step (`Translator.loss(batch, pad, tokenP)` fills the probability of the right token per position; no extra pass, dropout included, so they are exactly what was learned from). `PracticeExample.letters`/`loss`, `PracticeBatch.letterCount`. Tested: the letters' mistakes average exactly to the batch loss.

## Letter map: what to look for, and closest letters (user request)
"It is not clear what the letter map means." Two changes on the Model learning tab:
- **Text says what to notice first.** The visible line: letters that play the same part in words end up close; find the vowels, which start scattered and gather. How the map is made (32 numbers per letter, squeezed flat) moved behind the "?". The "11 % + 9 %" note became plain words: the flat map keeps only about N % of the differences, so trust the list of closest letters more than distances on the map.
- **Click a letter (or focus the map and type it) to see its 4 closest letters**, measured on all 32 numbers, not on the flat picture: lines on the map, everything else faded, and clickable chips under the map ("Closest to e: á ě u o"); Escape / × clears. The selection is kept while training continues.
- Measure: cosine similarity **from the average letter** (`nearestRows` in `src/nn/pca.ts`), computed in the worker with the PCA (`LetterMapMsg.near`). Compared after 5,000 steps (seed 1): plain cosine and distance gave noisier lists; centred cosine gave the clearest vowel groups (a → á ý o ě, o → á a e, á → o a e u). The similarities are modest (≈ 0.3–0.4): the letter table only needs to be good enough for the task.
- **Rare letters are flagged.** A letter seen fewer than 30 times in the learning data (`LetterMapMsg.counts`, from `FormTrainer.letterCounts()`, source words + answers) gets a note that the model learned little about it and its closest letters are mostly chance (q 7×, ó 1×…). This also explains the pile of accented letters at the edge of the map.
- Considered, not done yet: start-position ghosts with trails; hiding task tokens by default; marker size by frequency.

## One right word per answer (user request)
"Flags for answers not checked by a Czech speaker, and alternatives, are unnecessary complications for such a project. If needed I will modify the data."
- Removed `unsure` (19 entries), `formsAlt` (16 accepted alternative forms, e.g. koťátko, cub) and `alt` (9 alternative names, e.g. kocour for kočka, rhino) from the data, the code and the Learning data tab. The main words are unchanged.
- Every question now has exactly one right answer: `judge(output, answer)` is right / one letter off ("close") / wrong. `Task.accepted` and `AnswerMsg.answer.accepted` are gone (`answer.word`). The Ask tab recognises an animal only by its name.
- The validator rejects any field it does not know, so old fields in a data file are reported, not silently ignored.

## Copy mechanism removed

- **Removed completely** (user decision): the pointer-style copy attention, the write-or-copy gate, the ops `pointerMix` and `nllFromProbs`, the `copy` option of `TranslatorConfig`, `TranslationTrace.writeShare`, `StepGuess.write`, and in the UI the "copied / wrote" strip under the guess bars, the tinted "copied" answer tokens, the "Where it copies from" attention grid and the "Copying" group on the Inside tab.
- **Why:** this is a teaching demo and should be as simple as possible. Real language models have no copy mechanism; plain attention learns to copy. The experiments above showed the mechanism does not help generalisation (never-seen Czech 8/26 without it vs 7–8/26 with it); copy practice is what makes never-seen forms work, and it stays.
- **Cost:** known-form accuracy drops a little (measured earlier at 5 000 steps: Czech 89/102 vs 96/102, English 124/147 vs 131–141/147). Accepted: accuracy is a guide, not a gate. Re-measured with the app's trainer (`scripts/train-node.ts 5000 1`, seed 1): known Czech 91% (was 98%), English 92% (was 93%); never-seen Czech 19% (was 23%), within the ±2-word noise of the small never-seen set. The cross-attention grid still shows the stem being copied.
- **Parameters:** 24 727 (was 26 872) with the 55-token vocabulary.

## `unseen: true` instead of `split`; no `kindOverride` (user request)
"Would other learning data simplification make sense? e.g. one simple flag hidden: true/false." Chose the name `unseen` to match the app's existing "never seen" wording (the records view already uses "hidden" for masked forms).
- **`"split": "train" | "unseen"` became an optional `"unseen": true`.** 182 entries said `"split": "train"`, which was already the default; they now carry nothing. The 19 never-seen entries (11 in the base pack, 8 in the extension packs) say `"unseen": true`. The validator rejects `split` and any `unseen` value other than `true`. In code, `Animal.unseen`, `AnimalInfo.unseen` and `TaskResult.unseen` are booleans; the `Split` type is gone.
- **`kindOverride` removed** from the format, the merge and the validator, together with `pairKey` and `animalKind`, which only served it. No animal in any pack used it.
- A data entry is now just `id`, `names`, `forms` and the optional `unseen`. A script checked that the data is otherwise identical.

## Learning data tab: data sooner, one scroll, one word for "never seen" (review)
- **The data starts higher** (table top ~455 px instead of ~590 px at 1400×900). The intro is shorter, the seven counts became two lines, **Learning** (529 examples: 249 word forms + 280 copy the name, from 140 animals, per language) and **Testing** (302 questions with a right answer, 53 of them never seen; 604 with no real word), the view note sits beside the view buttons, and the table legend shares a line with the "animals shown" count.
- **One scroll bar.** The data no longer has its own scroll box (it ended below the window, so it needed two scrolls); the page scrolls, the column headings stick to the top, and the example panel stays in view beside it.
- **The intro fits every view.** It no longer says "marked in yellow" (only the overview table has yellow) or that every question has a right answer (604 have none). The "?" help says the yellow is in the overview table.
- **One word: "never seen".** "Kept aside" / "stranou" is gone; the legend says "never seen (only tested)". `unseen` appears only where the data records show the field; "hidden" only for answers that are not revealed.
- Not changed: narrow screens. Mobile/tablet is a non-goal (DESIGN.md: desktop ≥ 1280×720); checked at 1280×720.

## Ask the model tab: fewer unclear controls (review)
- **"Keep answering while it learns" removed.** Training controls moved to the Model learning tab, so the toggle only mattered if training was left running while on the Ask tab, and with "Learn until" it stops by itself. While the model learns, the answer stays marked out of date with "Ask again"; it refreshes by itself when training pauses.
- **"copy the name" became "same name"** (cs "stejné jméno"). As a button it read like a clipboard action. It is the copy-practice task (`<cs> slon → slon`), kept because copying vs inventing is part of the story. Every form button now has a tooltip (`form.tip.*`) saying what it asks for.
- **Panel titles shortened** to "Learning" (Model learning tab) and "Answering" (Ask tab), cs "Učení" / "Odpovídání". The ① / ② numbering came from when both phases were on one tab; the tab names now say the rest.

## Inside the model: explain what the numbers mean (user review)
"It contains a lot of data, but it is not clear for the user what they mean and how they can understand them." The user picked three of the proposed changes (A, B, C):
- **A. "Where is it learning?"** A diagram of the model's parts in the order a word goes through them (letters and positions → reading: look at each other, think → writing: look back, look at the animal, think → guess). Each part shows its **change per step**, √ΣΔw² / √Σw² / steps over the time since the previous request, with **one scale for every part**, and a line of it over time (collected while the tab is open). Volume knobs count with the step they prepare. Clicking a part scrolls to its heatmaps. The worker sends the summary of *every* matrix with each `weights` reply (`WeightsMsg.summary`, its own baseline `summaryBase`, independent of the heatmap mode); each heatmap caption also shows its own rate.
- **B. Readable heatmaps.** The letter table, the position tables and the favourite letters have their letters / positions on the axes (cells ≥ 9 px). One **shared colour scale** for all heatmaps by default (the largest of the matrices' own 99th-percentile scales), so a quiet matrix looks quiet; a checkbox switches back to one scale per matrix. In plain mode the number × number matrices (Q/K/V/O, thinking) are small thumbnails, since their axes mean nothing to a beginner; expert mode shows them full size. Tooltips say in plain words what the number does ("How much number 4 of a letter adds to number 7 of its question"), and what the shown value is in this mode (value / change / gradient, "learning moves the number the opposite way").
- **C. "Look inside one letter",** computed by the worker only on request (`explain` message, `nn/explain.ts`), for the current Ask question and one answer letter (picked on the Inside tab, or by clicking a guess-bar column on the Ask tab). It shows the residual stream of that position: letter-table row + position row = what goes in; for each step (look back, look at the animal, think) the attention per head or the thinking units that are on, what the step *adds* (the 3 biggest changes outlined) and the numbers after it, with "changes the numbers by N %" (‖add‖ / ‖before‖); then the numbers after the last volume knobs and the top-5 guesses with their scores. All strips share one colour scale, so sizes compare from step to step; a checkbox writes the values in the squares. **Clicking a square shows the exact sum it comes from:** what an attention step adds to number j split by the position looked at (Σ_heads weight · V · Wo, plus the bias), what thinking adds split by thinking unit, and a guess score as Σ number × letter-table row + favourite-letter bonus. The weights the letter uses (its letter-table rows, the position row, and for a clicked square the matching row/column of Wo or the thinking matrices) are outlined in the heatmaps below. The parts sum exactly to the shown value (unit test `tests/nn/explain.test.ts`).
- Not done (proposal D): "what one training step did" (the same trace before and after a step); it can reuse `explain`.
- "Follow the signal" (expert): decoder rows are labelled with the letter going in and the letter it is used to guess ("b→u"), since row `b` produces `u`.
- `Block.forward` records more trace points (volume-knob outputs, what each step adds, the thinking units); `Translator.inspect` keeps only the `*.in` / `*.out` ones for "follow the signal".

## Slovak and German (user request: "what would adding Slovak and German need, should the model be bigger?")
- **Data, first draft (needs a fluent reviewer):** `data/animals.sk.json` (pack `base-sk`) and `data/animals.de.json` (pack `base-de`) add a name in each language for all 151 base animals and the male / female / young words where a solid word exists: Slovak 43 animals / 107 forms, German 52 / 128. They merge into the base animals by id (`LANGUAGE_PACKS` in `src/data/bundle.ts`), so never-seen marks come from the base pack: never-seen forms sk 19, de 26, every kind in both. Kept out when unsure, so the model invents them: Slovak female dog (*suka* is also an insult; *fenka* is colloquial), *myšiak* (it means buzzard), camel, mouflon, stork, capercaillie and snake forms. German: nouns lower-cased; for big animals the real compounds (*elefantenkuh*, *zebrafohlen*, *kamelhengst*) like English bull / cow / calf, but no *-junges* / *-baby* phrases (generic, like *baby tiger*).
- **Vocabulary:** 8 new letters `ä ĺ ľ ô ŕ ö ü ß` (`ŕ` is not used by any word yet; kept so typed Slovak is not `<unk>`) and 8 target tokens: 71 tokens, 25 255 numbers (+528). Tests that used `ä` / `ö` as an example of an unknown letter now use `ø`.
- **Validator:** the "trained and never-seen animals" check counts only animals a pack *adds*, so a pack that only adds languages keeps the earlier split (test added). The summary lists only a pack's own languages.
- **No other code changes:** language pickers, the scoreboard, the Learning data counts and the Test tab are built from the data and show four languages.
- **Training data doubles:** 439 word forms + 560 copy-the-name examples (was 249 + 280).

**Model size, measured** (`scripts/experiments/languages.ts`, Node, batch 16, LR 3e-3, known / never-seen %, ffn = 2 × d_model; 4 runs in parallel, so steps/s are lower than alone):

| run | steps | cs | en | sk | de | steps/s |
|---|---|---|---|---|---|---|
| cs+en, d 32 (before) | 5 000 | 94 / 27 | 92 / 11 | | | 43 |
| cs+en, d 32 (before) | 10 000 | 98 / 35 | 97 / 15 | | | 43 |
| 4 languages, d 32, seed 1 | 5 000 | 77 / 38 | 39 / 19 | 75 / 21 | 57 / 4 | 38 |
| 4 languages, d 32, seed 2 | 5 000 | 75 / 35 | 48 / 19 | 75 / 32 | 63 / 4 | 38 |
| 4 languages, d 32, seed 1 | 10 000 | 85 / 42 | 64 / 15 | 91 / 21 | 72 / 4 | 38 |
| 4 languages, d 32, seed 2 | 10 000 | 88 / 12 | 68 / 19 | 91 / 26 | 84 / 4 | 38 |
| 4 languages, d 32, seed 1 | 15 000 | 99 / 35 | 78 / 15 | 94 / 21 | 82 / 4 | 38 |
| 4 languages, d 32, seed 1 | 22 500 | 96 / 42 | 83 / 19 | 95 / 32 | 95 / 4 | 39 |
| 4 languages, d 32, seed 1 | **30 000** | **98 / 31** | **95 / 19** | **99 / 26** | **97 / 8** | 39 |
| 4 languages, d 32, batch 64, LR 6e-3 | 5 000 (= 20 000 × 16) | 93 / 23 | 75 / 11 | 95 / 21 | 92 / 8 | 8 |
| 4 languages, d 48, seed 1 | 10 000 | 96 / 23 | 91 / 19 | 94 / 32 | 91 / 4 | 20 |
| 4 languages, d 48, seed 2 | 10 000 | 95 / 27 | 76 / 7 | 99 / 21 | 90 / 4 | 20 |
| 4 languages, d 48, seed 2 | 20 000 | 100 / 31 | 90 / 7 | 100 / 26 | 96 / 8 | 20 |

- **d_model 32 is big enough.** With enough steps it learns all four languages (30 000 steps: 95–99 % known), as well as d 48 does. d 48 (53 207 numbers) needs fewer steps but each step costs twice as much, so per second of learning it is no better (d 32 at 22 500 steps ≈ d 48 at 10 000 steps in time). **Decision: keep d 32**, as DESIGN §15 planned ("raise to 48 if accuracy drops noticeably"): accuracy drops only for lack of steps, not for lack of numbers.
- **The real cost is learning time:** about 4× the steps for the same quality (≈ 30 000 instead of ≈ 7 500), because the data doubled and each language gets a quarter of each batch instead of half. English, which is all memorising, is the slowest. A bigger batch does not help on a CPU (same examples, same time).
- **Never-seen:** Slovak behaves like Czech (21–32 %, patterns carry over). German stays at 4–8 %: most of its never-seen forms keep the name (*giraffe → giraffenkuh*), but which ending (*-bulle/-kuh/-kalb* or *-hengst/-stute/-fohlen*) depends on the kind of animal, not on the letters, so the model cannot guess it. A good example for the memorise-vs-generalise lesson.
- **Not decided (user):** the default "Learn until" goal (5 000) and its options (up to 10 000) were set for two languages; with four, known English is ~40–50 % at 5 000 steps.

## Ask the model: a question form with two answer buttons (user request)
- **The question is a small form:** 1. Language, 2. Animal (text box), 3. Which word (same name / male / female / young), then **Answer now** and **✍️ Answer slowly**. Enter in the word box = Answer now; both buttons are off while the box is empty.
- **Answers only when asked.** Before, every keystroke or picker click answered at once. Now changing the language, word or form takes the old answer away (its tokens, guesses and attention too) and says which button to press; a reply still on its way for the old question is dropped. Exceptions, because they are already a deliberate click: the first question when the page loads, a word clicked on the Learning data or Test tab, and the automatic refresh of an answer on screen when the model pauses at a new step (unchanged).
- **No silent language switch.** Typing a word of another language used to switch the language by itself; with the language chosen first that was surprising. The line under the box now says what the word is (✓ an animal from the learning data / this word is in the German learning data, with a "Switch to German" link / not an animal from the data, the model will make one up).
- **Answer slowly** replaces "Write it slowly" under the guesses: it asks and then writes the fresh answer letter by letter, slower than before (1.1 s per letter, 1.4 s reading the animal; was 0.4 s / 0.7 s), and the same button becomes **■ Stop**. A box under the buttons says what the model is doing: "Reading the animal “tygr”… (6 tokens, looked at together)", then for each letter "Writing letter 3 of 7 — its best guesses: g 98 % · y 1 % · ž 0 % → it writes g", "Deciding the word is finished ⏎", "Done". The answer word, the written tokens, the guess column and the attention rows grow with it, as before. The letter count comes from the guesses (not from the output text), so it is right for an untrained model whose guesses are not letters.
- Screenshots: `docs/screenshots/ask-form/`.

## Performance review: faster kernels, background tests, shorter answering (user request)
- **Matrix multiplies work on 4 rows (or 4 columns) at once** (`nn/ops.ts`). Training spent ~85 % of its time in them. Training: 46 → 72 steps/s in `scripts/bench.ts` (Node), 27 → 16 ms per real step. `mmNN`/`mmNT` keep the order of each sum, so a word's answer does not depend on which other words are in the batch; `mmTN` (gradients only) may differ in the last bit.
- **Greedy answering drops words that have ended** (`Translator.greedy`). Before, a batch of 128 kept computing every word until the longest one ended: after training, half of that work was for finished words. Answers are bit-identical, evaluation with inventions ~1.8× faster once trained.
- **The test of every animal runs in the background** (`Engine.startEval`). It freezes a copy of the weights when it starts, then answers 32 questions at a time in ~12 ms turns between training slices and messages, and reports `eval` when done (for the step it started at). Before, it blocked the worker for up to ~1.4 s (with inventions), so Pause and questions waited. A new test replaces one still running (keeping its inventions); the training loop does not start another while one runs. Same EVAL schedule.
- With the Test tab open (Node, 20 s at full speed): 30 → 47 steps/s, longest blocking task 1 366 → 47 ms.

## One learning step in full, and the tokens' way through the model (user request after the transparency review)
User: "Before and after nudge would be nice, to show exactly what numbers were changed and why. Maybe it should be available only in single step. Continuous calculation could be too heavy. It is also not clear how exactly tokens are pushed through the model."
- **Only for Step ×1 while paused.** The engine records a step in detail only for `step` with `n = 1` while not running (`FormTrainer.trainStep(answerFirst, true)`, `src/train/stepDetail.ts`). Cost measured in Node: ~0.37 s for one recorded step vs ~0.04 s for an ordinary one (≈ 10–20 steps), so never while Play runs. The recorded step trains **bit for bit like an ordinary one** (test): the extra passes replay the dropout random numbers and restore them.
- **What is recorded:** weights before; the gradient; **each example's share of every number's gradient** (the batch loss is the mean over all letters, so the gradient of example *b*'s letters alone, × its letter count / all letters, with the *same* dropout masks, adds up exactly to the whole gradient: error ~3e-8); Adam's `m`/`v` before the step, the learning rate used (warm-up), the clip factor; the batch measured **the ordinary way (no dropout)** before and after, and the greedy answers after. Kept in the worker until the next step; per-example shares skipped for batches over 32.
- **Model learning tab:** a fifth cycle stage **check** ("mistake 0.84 → 0.81"; dashed "only with Step ×1" otherwise); an **After the step** column in the batch table (letters tinted by their new mistake, underlined green/red if the right letter became more/less likely, before → after on hover, the example's mistake before → after, and its new answer if it changed); a new panel **What this step changed**: the five stages with this step's numbers (mistake with dropout, length of all gradients, clipping factor, learning rate and warm-up, mistake before → after), how many numbers moved (biggest, median), the 10 numbers that **moved the most** or have the **biggest gradient**, and for the chosen number **the whole calculation**: gradient (with each example's share, they add up), clipping, Adam's m and v with the start corrections, the move = lr × m̂ / √v̂, before − move = after, checked against the model's number now.
- **Inside the model:** after a recorded step, clicking any square of a heatmap opens its "why" on the Model learning tab.
- **Dropout is now visible**: the step's own mistake (with 1 in 10 numbers switched off) and the ordinary before/after are different numbers; the texts say why.
- **Seen in practice (worth knowing when reading the panel):** with Adam the numbers that *move* the most are often ones that *rarely* get a gradient (e.g. the favourite-letter bonus of a task token), and the biggest gradient can give a small move (m̂/√v̂ = 0.17 in one screenshot). The panel's texts explain this instead of hiding it. About 1.5 % of the numbers do not move at all (e.g. position rows beyond the longest word: never any gradient, so Adam's memory is 0).
- **Ask the model tab: Follow the tokens through the model** (`src/ui/journey.ts`), a stepper below the answer: 1 tokens (word → letters → task token + letters → `[63, 23, 28, 10, 21]`), 2 letter table (each token's row + position row = what goes in, as real strips), 3 reading once (encoder self-attention tables, "after reading" strips), then **one writing pass per token**: the token going in (task token, then the token written last, marked ↩), its rows, look back / look at the animal (per head, also shown next to the reading lane), think, final numbers, the 71 scores → top 5 chances + the rest, the token written, and that it goes back in. "Answer slowly" drives the stages. Link to *Look inside one letter* for every number of a pass.
- **Data:** `AnswerMsg` now carries `inputIds`, `vocabSize` and `lookup` (letter-table and position rows of every token going in; their sums are `enc.in` / `dec.in`, tested), and two more activations, `memory` and `dec.final` (also labelled in Expert *Follow the signal*).
- Screenshots: `docs/screenshots/step-and-tokens/`. Czech strings `sd.*`, `jr.*`, `cycle.check*`, `practice.check*`, `practice.nowAnswers`, `inside.whyClick`, `signal.memory`, `signal.dec.final` need the user's review.
- **"same name" everywhere** (user, follow-up). The rename first touched only the Ask tab button; the Model learning tab (`practice.copy`) and the Learning data tab (`data.col.name`, `data.tip.copy`, `data.help`, `data.rec.names`, `data.sum.learn`) still said "copy the name". Now every label for this task says "same name" / "stejné jméno". Explanations of *why* it exists still say it teaches the model to copy letters.

## Inside the model for non-experts: "what would it write now" and plain mode (user request)
User: "Inside model page is extremely heavy on data and explanation … how to simplify it so even a person with limited math knowledge understands?" The analysis found: almost everything was rows of red/blue squares that look like noise, engineering numbers (`±1.9e-3`, `0.19 % per step`, `size of the change: 233 %`, `score 10.14`, `32 × 64`), long help paragraphs, ~11 new terms at once, and "How the model picks one letter" opening on the least interesting letter. Of the options proposed (A "what would it write now", B attention as lines, C one "brain scan" picture, D thinking units by examples, E plain words and numbers, F three levels), the user chose the recommended start: **E + A**. B, C, D and F are not done.
- **A. "What would it write now?" (the logit lens).** After the start and after each stage, the letter's numbers are put through the last volume knobs and the letter table exactly as the final numbers are (`lensLogits` in `nn/explain.ts`; after the last stage it equals the real guess, tested). `ExplainMsg.lens` carries the top 3 letters and the chance of the finally written letter for each point. Plain mode shows them as small guess bars per stage, a **"How sure it is about 'i', stage by stage"** ladder (`<1 % → <1 % → 18 % → 99 %`) and one sentence naming the stage with the biggest step ("…is stage 4: think"), and on the right of each stage "chance of 'i': 18 % → 99 % ▲". Seen in practice (buvol → buvolice, step ~2 800): after looking back it would end the word (⏎ 97 %), looking at the animal (90 % on the task token "Czech, female") brings "i" up to 18 %, thinking settles it. This is the story the old rows of squares could not tell.
- **Plain mode of "How the model picks one letter" has no rows of numbers.** Attention: both heads together (their average) in one row, with "mostly at 'l' (50 %)". Thinking: "26 of 64 small detectors switched on". Guesses without scores. One short sentence per stage, the full explanation behind "?". The "size of the change" meter, the scores, the strips, the clickable sums and the "write the numbers" checkbox are in Expert, which also shows the new "would write now" bars.
- **It opens on the most interesting letter:** the first one where the answer stops copying the animal (buvol → buvol**i**ce, tygr → tyg**ř**ice), and follows it while the model learns until the user picks a letter.
- **E. Plain words and numbers on the rest of the tab:** "Where is it learning?" shows bars without the `% per step` values (they are in the tooltip and in Expert); heatmap captions (shape, colour scale, rate) and the "same colour scale" checkbox are Expert only; tooltips write small numbers without scientific notation (0.0019; "almost 0" below 0.000001); chances below 1 % are "<1 %", not a misleading "0 %". Long help texts of the three sections are behind "?" with one short sentence in the panel.
- The page in plain mode is ~3 100 px tall instead of ~3 600.
- Screenshots: `docs/screenshots/plain-inside/`. New Czech strings (`explain.helpShort`, `explain.helpMore`, `explain.ladder`, `explain.jump*`, `explain.wouldWrite`, `explain.effect`, `explain.looksAt`, `explain.mostly`, `explain.unitsOn`, `explain.name.*`, `explain.*.short`, `flow.helpShort`, `inside.helpShort`, `inside.almostZero`) need the user's review.

### C. One "brain scan" picture (user: "continue with C")
- **"Where is it learning?" is now one picture of all 25,255 numbers** (`src/viz/brainScan.ts`): one dot per number, grouped into the same 7 parts as before, in signal order (letters and positions → reading the animal → writing the answer). It replaces, in plain mode, both the bar diagram and the 15 heatmaps.
- **Honest sizes:** every part uses the same dot size and 96 dots per row, so a part's area is its number of numbers. One brightness scale for all parts (99th percentile of |value| over the whole model). Brightness is linear: a first try used a square root to show small changes, but then every part glowed about the same and the picture said nothing.
- **Glow palette** (near black → purple → red → orange → yellow, brightness always increasing, tested), on dark, like a brain scan. The scan shows size only, not plus or minus; the heatmaps keep the blue/red sign colours.
- **It follows the mode buttons** (Values / Recent change / Push), which are now on the scan and on the heatmaps (same mode, one request). The engine sends every matrix now (25,255 floats per update), not only the visible heatmaps.
- **Under each part:** "4,288 numbers · 17 % of the change". **One sentence** compares the average move of a part's numbers with the whole model's average: "the busiest part is X: its numbers move 1.4 times as much as an average number" only at ≥ 1.3×, "the quietest part is Y … 0.6 times" only at ≤ 0.7×, otherwise "every part is learning, about equally hard". Seen in practice: with Adam it is usually the last one, which is true and worth saying; a part's share of the change mostly reflects its size, so the share is not used for the sentence.
- **Dark spots tell a story:** rows of letters not in the latest examples and positions after the end of the longest word do not move. The "?" says so.
- **Hover** names the part, the matrix and what the number does (the heatmap tooltip text), with its value; the other matrices of the part are dimmed. **Click** opens the heatmaps and scrolls to that part. In plain mode the heatmaps are folded away behind "Show every number up close (small pictures)"; in Expert they are always open, and the old bar diagram with lines over time is kept there as "How fast each part changes".
- Plain page height: ~1 850 px (was ~3 600 before A/E/C). Screenshots: `docs/screenshots/plain-inside/scan-*.png`, `inside-plain-full.png`. New Czech strings `scan.*`, changed `flow.title`, `inside.guide` need the user's review.

### D. What have the detectors learned? (user: "continue with D")
- **The writer's 64 thinking units ("detectors") explained by examples** (`src/train/units.ts`, new `units` message, cached per learning step). The model reads every learning example with its right answer (537 word-form tasks, 3,642 letters including each word's end, no dropout); for each unit it keeps how often it is on (> 0.05, as in "picks one letter"), its 8 strongest moments (each shown once: several animals share e.g. "calf"), and the 3 letters its pattern votes for (its row of "thinking, part 2" through the last volume knobs and the letter table: the direct effect only, so the texts call it a hint).
- **A theme only when it is real:** among the 8 strongest moments, the most common value of the next letter, the letter just written, language + form, form or language (in that order, most specific first) must be shared by at least 6 of 8 **and** be at least twice as common as among all 3,642 moments. Measured (seed 1, 5,000 steps): 51 of 64 units get a theme, and they read well: *next letter is "c"* (jezevči**c**e, bažanti**c**e, klokani**c**e), the Slovak *-**c**a* twin, *next letter is "i"* (prasn**i**ca, slon**i**ca…), *the word is about to end*, *German, female* (… -**k**uh, -hen**n**e). Units without a theme say so ("no clear job: it helps in many different situations") rather than being hidden.
- **Where it shows:** in "How the model picks one letter", the Think stage lists the 3 units most strongly on for this letter, with a bar "on now" against their strongest moment (buvol → buvol**i**ce: the strongest is the "next letter is i" unit, voting for i). A new panel "What have the detectors learned?" below it shows the 12 clearest in plain mode ("Show all 64"), all in Expert (with the vote scores).
- **Cost:** reading all the examples takes ~0.3 s in the worker (Node measurement). It is asked for when the Inside tab is shown and, while learning, at most every 15 s; the engine caches it per step and drops the cache on Reset.
- Only the writer's thinking (dec0.ffn) for now; the reader's thinking units would need moments about input letters, not answer letters.
- Screenshots: `docs/screenshots/plain-inside/units*.png`, `explain-think-units.png`. New Czech strings `units.*`, `unit.*`, `explain.ffn.units*` need the user's review.

