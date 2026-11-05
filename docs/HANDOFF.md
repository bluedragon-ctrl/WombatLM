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
| M1 brain library | **Done.** `src/nn`: tensors, tape autograd, ops (all gradient-checked), Adam/AdamW, dropout, translator with KV-cached greedy decoding, copy mechanism, target-token start, serialisation f32/f16/int8, vocabulary growth. |
| M2 training in the browser | **Done** (pending the user's data review). Dataset v2 + validator + review doc, `FormTrainer` with copy practice, engine + worker + protocol, minimal UI, Playwright smoke test. ~38 steps/s in the browser; known cs 98% / en 93% at 5k steps (accepted by the user). |
| M3 seeing inside | **Done** (merged into `main`). Tabs (Forms · Inside the brain), attention grids linked to the guess bars, letter map (PCA + Procrustes + glide), weight / change / gradient heatmaps, "follow the signal" (expert). |
| Learning vs answering | **Done in a PR** (user feedback after M3): ① Build / ② Ask phases, practice strip (`tick.practice`), frozen-brain step label, answer waits while training (+ "keep answering" toggle), Write it slowly, one-step cycle. See DECISIONS. |
| M4 time travel | **Next.** |
| M5–M7 | Not started. |

## Where things are

- **Trainer:** `src/train/trainer.ts` (`FormTrainer`, `TRAINER_DEFAULTS`, `scoreResults`, `confidence`). Node run: `npx tsx scripts/train-node.ts 5000 1`.
- **Worker:** `src/train/engine.ts` holds the loop and message handling (testable with a fake host, see `tests/train/engine.test.ts`); `worker.ts` only wires it up; messages in `protocol.ts`.
- **UI:** `src/ui/app.ts` (one class, vanilla DOM), visuals in `src/viz/` (Canvas 2D), strings in `src/ui/i18n/{en,cs}.json`. Elements used by tests have `data-test` attributes.
- **Data:** `data/animals.base.json` (151 animals, 92 with forms, 11 never-seen). `Dataset.formTasks()`, `missingFormTasks()` (inventions), `nameTasks(…, true)` (copy practice), `formKind()`.

## Notes for M4

- `inspect` already serves any parameter by name; for time travel the engine needs a separate **view model** (DESIGN §7.3) and `inspect`/`ask`/`letterMap` must read from it while a snapshot is shown. `ParamSet.snapshot()`/`load()` exist.
- The change mode keeps per-matrix baselines in the engine (`deltaBase`, `lastDelta`, `beforeStep`); seeking must clear them like `reset` does.
- Panels request data only while visible: the Forms tab asks for the letter map, the Inside tab for the heatmaps (`App.requestInspect`, at most one in flight, every 250 ms).
- The UI is in `src/ui/app.ts` (Forms tab + tabs) and `src/ui/inside.ts` (Inside tab); visuals in `src/viz/` (`attentionGrid`, `letterMap`, `weightHeatmap`).
- `npm run screenshots` regenerates the PR screenshots in `docs/screenshots/m3/` and `docs/screenshots/learn-vs-answer/` (opt-in, ~3 min).
- Time travel must respect the learning/answering split: a seeked snapshot is a frozen brain too, so the step label under the answer ("Answer from the brain at step N") should show the snapshot's step, and the practice strip should be hidden or say it is history.

## Waiting for the user

- **Data review:** 19 `unsure` entries; regenerate the table with `npm run validate-data -- --review` → `docs/DATA_REVIEW.md` (deferred by the user: "data tuning later").
- **Czech UI copy:** `src/ui/i18n/cs.json` (M3 added ~70 strings: tabs, attention, letter map, Inside the brain; the learning-vs-answering PR added 38: `build.*`, `learning.*`, `practice.*`, `cycle.*`, `ask.*`, `stamp.*`, `slow.*`).
- **M3 review:** the PR's screenshots and questions.

## Hosting

- **Live site:** `https://bluedragon-ctrl.github.io/WombatLM/`, deployed by `.github/workflows/deploy.yml` on every push to `main` (needs Settings → Pages → Source: GitHub Actions, once).

## Useful commands

```
npm test                                   # vitest (80 tests)
npm run build                              # typecheck + vite build
npm run e2e                                # Playwright smoke test (builds + previews on :4173)
npm run screenshots                        # PR screenshots (docs/screenshots/m3, docs/screenshots/learn-vs-answer)
npm run dev                                # the app
npm run validate-data [-- --review]        # data linter (+ review table)
npx tsx scripts/train-node.ts 5000 1       # the app's trainer in Node
npx tsx scripts/experiments/forms-full.ts 5000 1 0.1 1 1 1
    # steps seed dropout copyPracticeWeight copyMechanism targetStart
```

Playwright 1.63.0 is pinned to the pre-installed browser; don't run `playwright install`.
