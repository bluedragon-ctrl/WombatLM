# WombatLM handoff — read this first

State of the project after the third working session (M3 in a pull request). **`docs/DESIGN.md` is the current spec** (rewritten for the word-forms design); `docs/DECISIONS.md` has the reasoning and numbers behind it.

## Priority (user)

This is a **teaching demo**. The goal is to show how the model works (token-by-token generation, weights changing, copying vs inventing), not perfect accuracy. Prefer clearer visuals and explanations over accuracy tuning; accuracy targets are guides, not gates.

## How to work (user)

**From M3 on, all work goes through pull requests** (DESIGN §12.1):
- One branch and one PR per milestone, or per coherent step of a milestone. Don't push feature work directly to the base branch.
- Each PR: `npm test` and `npm run build` green, `npm run e2e` for UI changes, screenshots of UI changes in the PR description, `docs/DECISIONS.md` updated for any deviation, this file updated with the new state.
- Questions for the user (data, Czech wording, trade-offs) go into the PR description.
- **Base branch:** `main`, once the M1–M2 work from `ccr-0f86f24a-pa08xs` has been merged into it. If `main` still only has the initial commit, branch from `ccr-0f86f24a-pa08xs` and target it instead (and mention this in the PR).

## Status

| Milestone | State |
|---|---|
| M1 brain library | **Done.** `src/nn`: tensors, tape autograd, ops (all gradient-checked), Adam/AdamW, dropout, translator with KV-cached greedy decoding, target-token start, serialisation f32/f16/int8, vocabulary growth. |
| M2 training in the browser | **Done** (pending the user's data review). Dataset v2 + validator + review doc, `FormTrainer` with copy practice, engine + worker + protocol, minimal UI, Playwright smoke test. ~38 steps/s in the browser; known cs 98% / en 93% at 5k steps (accepted by the user). |
| M3 seeing inside | **Done** (merged into `main`). Tabs (Forms · Inside the model), attention grids linked to the guess bars, letter map (PCA + Procrustes + glide), weight / change / gradient heatmaps, "follow the signal" (expert). |
| Learning vs answering | **Done in a PR** (user feedback after M3): ① Build / ② Ask phases, Practice tab (now part of Model learning) with the latest batch: learning data vs what it answered before the step (`tick.practice`, `watchPractice`), frozen-brain step label, answer waits while training ("keep answering" toggle later removed), Write it slowly, one-step cycle. See DECISIONS. |
| Tab reorganisation (user, in steps) | **Steps 1–3 done in a PR.** Tabs now follow the story: Learning data (opens first, `src/ui/dataTab.ts`) · Model learning (build controls, cycle, batch, loss, letter map) · Ask the model (question, answer, guesses, attention; status line, no training controls) · Inside the model · Test (every animal, tested only while open: `watchTests`). Expert checkbox in the header. All UI text says "model" (not "brain"). Further steps to come from the user. See DECISIONS. |
| Tokens on screen (user) | **Steps 1–2 done** (branch `claude/sweet-einstein-p7md6p`). Ask tab: token strip (reads / writes, numbers, `<unk>` warning, hover linked to the guess bars). Learning data tab: clicking a word shows that learning example as tokens (reads / shown so far / guess next). Helpers in `src/ui/tokens.ts`. Step 3 (vocabulary list, toy subword tokenizer comparison) proposed, not started. See DECISIONS. |
| Build time (user) | **Done in PRs.** Quiet learning switch + real (wall-clock) speed (PR #7); "Learn until" step goal, default 5,000 (training stops by itself). Measured: known forms ~99 % by 7,500 steps, nothing gained after; bigger batches need far fewer steps (possible later WebGPU experiment). See DECISIONS. |
| Mistake per line (user) | **Done in a PR.** Practice table: mistake (loss) per example, letters tinted by their own mistake, an explanation how the step's mistake is the average over all letters. See DECISIONS. |
| Letter map explained (user) | **Done in a PR.** What-to-notice text first (how it is made behind "?"), click or type a letter to see its 4 closest letters on all 32 numbers (centred cosine), rare letters flagged. See DECISIONS. |
| Inside the model explained (user) | **Done** (branch `claude/epic-darwin-rcolin`). "Where is it learning?" diagram (change per step per part, one scale, lines over time), "Look inside one letter" (every number on the way to one answer letter, what each step adds, exact where-it-comes-from sums on click, used weights outlined in the heatmaps; also from a click on the guess bars), heatmaps with letter/position labels, one shared colour scale (toggle), plain-language tooltips, number × number matrices as thumbnails in plain mode. See DECISIONS. |
| Slovak and German (user) | **Done** (branch `claude/gallant-tesla-msdpx7`, first draft of the data). `data/animals.sk.json`, `data/animals.de.json` merged into the base animals by id (`LANGUAGE_PACKS`), 8 new letters, 71 tokens. Measured: d_model 32 is big enough (4 languages 95–99 % known at 30 000 steps, same as d 48), but it needs ~4× the steps. See DECISIONS. |
| Ask form (user) | **Done** (branch `ccr-a4ae8508-s8hr46`). Ask tab: 1. Language · 2. Animal · 3. Which word, then **Answer now** / **✍️ Answer slowly** (letter by letter with a "what the model is doing" box); answers only when asked; no silent language switch. See DECISIONS. |
| One step in full + tokens' way (user) | **Done** (branch `ccr-8bd786fd-ary3n5`). Step ×1 while paused records the step in detail (`src/train/stepDetail.ts`, `FormTrainer.record`, `stepDetail` / `why` messages): batch before → after (no dropout), "What this step changed" panel with each number's whole calculation (gradient split by example, clipping, Adam, result), 5th cycle stage "check", heatmap click → why. Ask tab: "Follow the tokens through the model" stepper (`src/ui/journey.ts`). See DECISIONS and `docs/REVIEW_TRANSPARENCY.md`. |
| Inside the model for non-experts (user) | **Done** (branch `claude/admiring-galileo-qb1rto`). Analysis of the tab + options A–F; built E (plain words and numbers, help behind "?") and A ("what would it write now" after each stage, logit lens, `lensLogits`, `ExplainMsg.lens`). Then C: one "brain scan" picture of all the numbers (`src/viz/brainScan.ts`) replacing the flow diagram and the heatmap grid in plain mode (heatmaps folded away, opened by a click on the scan; flow diagram kept in Expert). Then D: "What have the detectors learned?", the writer's thinking units explained by their strongest moments in the learning data (`src/train/units.ts`, `units` message, `src/ui/unitCard.ts`), also in the Think stage. PR #25. Proposed, not built: B attention as lines, F three levels. See DECISIONS. |
| M4 time travel | **Next.** |
| M5–M7 | Not started. |

## Where things are

- **Trainer:** `src/train/trainer.ts` (`FormTrainer`, `TRAINER_DEFAULTS`, `scoreResults`, `confidence`). Node run: `npx tsx scripts/train-node.ts 5000 1`.
- **Worker:** `src/train/engine.ts` holds the loop and message handling (testable with a fake host, see `tests/train/engine.test.ts`); `worker.ts` only wires it up; messages in `protocol.ts`.
- **UI:** `src/ui/app.ts` (one class, vanilla DOM), visuals in `src/viz/` (Canvas 2D), strings in `src/ui/i18n/{en,cs}.json`. Elements used by tests have `data-test` attributes.
- **Data:** `data/animals.base.json` (151 animals, cs + en, 92 with forms, 11 never-seen); Slovak and German in `data/animals.sk.json`, `data/animals.de.json`. `Dataset.formTasks()`, `missingFormTasks()` (inventions), `nameTasks(…, true)` (copy practice), `formKind()`.

## Notes for M4

- `inspect` already serves any parameter by name; for time travel the engine needs a separate **view model** (DESIGN §7.3) and `inspect`/`ask`/`letterMap` must read from it while a snapshot is shown. `ParamSet.snapshot()`/`load()` exist.
- The change mode keeps per-matrix baselines in the engine (`deltaBase`, `lastDelta`, `beforeStep`, and `summaryBase`/`lastSummary` for the change summary); seeking must clear them like `reset` does. `explain` must also read from the view model.
- Panels request data only while visible: the Model learning tab asks for the letter map and turns on `watchPractice`, the Inside tab asks for the heatmaps (`App.requestInspect`, at most one in flight, every 250 ms), the Test tab turns on testing (`watchTests`).
- The UI is in `src/ui/app.ts` (tabs, Model learning, Ask, Test), `src/ui/dataTab.ts` (Learning data tab) and `src/ui/inside.ts` (Inside tab); visuals in `src/viz/` (`attentionGrid`, `letterMap`, `weightHeatmap`).
- `npm run screenshots` regenerates the PR screenshots in `docs/screenshots/m3/` and `docs/screenshots/learn-vs-answer/` (opt-in, ~3 min).
- Time travel must respect the learning/answering split: a seeked snapshot is a frozen brain too, so the step label under the answer ("Answer from the brain at step N") should show the snapshot's step, and the practice strip should be hidden or say it is history.

## Waiting for the user

- **Inside the model for non-experts** (PR #25): check the new Czech strings (listed in DECISIONS: `explain.*`, `scan.*`, `units.*`, `unit.*`, …) and pick the next step: B or F.

- **Plain-language pass on Inside the model** (branch `claude/youthful-heisenberg-ieybd3`): rewrote `inside.*`, `flow.*`, `explain.*`, `tip.*`, `signal.*` in en and cs (keys unchanged). "Step" now only means a learning step; the 5 parts of "How the model picks one letter" are "stages". Heads, thinking units, volume knobs and softmax are explained where they first appear. Heatmap modes: Values / Recent change / Push (gradient). Czech wording needs a check.
- **Czech UI copy:** step-in-full / tokens PR: `sd.*`, `jr.*`, `cycle.check*`, `practice.check*`, `practice.nowAnswers`, `inside.whyClick`, `signal.memory`, `signal.dec.final`.
- **Transparency review:** `docs/REVIEW_TRANSPARENCY.md` lists what a user cannot yet see in one learning step and in answering, with a suggested order (before → after the nudge, sampling/temperature, the optimiser, the generation loop). Pick what to build.

- **Slovak and German:** review of `data/animals.sk.json` and `data/animals.de.json` by a fluent speaker (`docs/DATA_REVIEW.md` has the tables); decide the "Learn until" default / options for four languages (5 000 steps gives known English ~40–50 %; ~30 000 for 95 %+).

- **Data review:** no flags any more (one right word per answer; fix or delete doubtful words in the JSON); regenerate the table with `npm run validate-data -- --review` → `docs/DATA_REVIEW.md` (deferred by the user: "data tuning later").
- **Czech UI copy:** Ask form PR: `ask.step.*`, `ask.now*`, `ask.slow*`, `ask.prompt`, `ask.thinking`, `ask.word.*`, `att.prompt`, `slow.*` (reworded), `ask.help` (reworded).
- **Czech UI copy:** Ask tab review: `form.name` ("stejné jméno"), `form.tip.*`, reworded `ask.help`, `ask.title` ("Odpovídání"), `build.title` ("Učení"); "same name" everywhere: `practice.copy`, `data.col.name`, `data.tip.copy`, `data.help`, `data.rec.names`, `data.sum.learn`.
- **Czech UI copy:** new in the Inside-the-model PR: `inside.guide`, `inside.heatTitle`, `inside.shared*`, `inside.rate`, `inside.val.*`, `tip.*`, `flow.*`, `explain.*`, `ask.lookInside`, `signal.decRow`.
- **Czech UI copy:** new in the build-time PRs: `quiet.*`, `wallPerSec`, `goal.*`, `practice.col.loss`, `practice.end`, `practice.letterTip`, `practice.lossTip`, `practice.lossExplain`; letter map PR: `map.help`, `map.how`, `map.note`, `map.pick`, `map.near`, `map.rare`, `map.clear`.
- **Czech UI copy (earlier):** `src/ui/i18n/cs.json` (M3 added ~70 strings: tabs, attention, letter map, Inside the model; the learning-vs-answering PR added 42: `tab.practice`, `build.*`, `learning.*`, `practice.*`, `cycle.*`, `ask.*`, `stamp.*`, `slow.*`; the Learning data tab added 26: `tab.data`, `data.*`; the tab reorganisation added `tab.test`, `tab.learn`, `tab.ask`, `test.*`, `askStatus.*` and reworded `data.explain`, `data.help`).
- **M3 review:** the PR's screenshots and questions.

## Hosting

- **Live site:** `https://bluedragon-ctrl.github.io/WombatLM/`, deployed by `.github/workflows/deploy.yml` on every push to `main` (needs Settings → Pages → Source: GitHub Actions, once).

## Useful commands

```
npm test                                   # vitest (113 tests)
npm run build                              # typecheck + vite build
npm run e2e                                # Playwright smoke test (builds + previews on :4173)
npm run screenshots                        # PR screenshots (docs/screenshots/m3, docs/screenshots/learn-vs-answer)
npm run dev                                # the app
npm run validate-data [-- --review]        # data linter (+ review table)
npx tsx scripts/train-node.ts 5000 1       # the app's trainer in Node
npx tsx scripts/experiments/languages.ts cs,en,sk,de 10000 1 32 16
    # langs steps seed dModel batch [lr]: accuracy per language at 4 checkpoints
npx tsx scripts/experiments/forms-full.ts 5000 1 0.1 1 1
    # steps seed dropout copyPracticeWeight targetStart
```

Playwright 1.63.0 is pinned to the pre-installed browser; don't run `playwright install`.
