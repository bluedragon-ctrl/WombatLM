# Handoff — read this first

State of the project at the end of the first working session, and what changed against `docs/DESIGN.md`.
Where this file and DESIGN.md disagree, **this file wins**. Detailed reasoning and numbers: `docs/DECISIONS.md`.

## The big change: word forms instead of translation

Experiments showed that cs↔en translation on ~150 animals cannot generalise: never-seen animals scored 0%, even identical words (leopard → "kowpa"). Agreed with the user, the **main task is now within-language word forms**:

```
<cs:male>   tygr  → tygr        <en:female> lion → lioness
<cs:female> tygr  → tygřice     <en:young>  cat  → kitten
<cs:young>  kočka → kotě        <cs:female> pes  → fena   (irregular)
```

- Forms: `male`, `female`, `young` (`data/forms.json` also has `name`, used for "copy the name" practice: `<cs> slon → slon`).
- The same drill runs for every data language (cs and en now; more languages later, no code changes).
- **The lesson:** Czech forms mostly follow patterns, so the model partly generalises to never-seen animals (copies the stem, adds an ending, makes believable mistakes such as vlk → *vlkice*). English forms are irregular (cub, calf, foal), so it must memorise and fails on never-seen ones.
- **Missing forms are deliberate** (žirafa, panda, …): the model invents them. This is the former M8 "baby animals / hallucination" idea, now part of the core (badges real / hidden / invention, confidence meter).
- Translation is out of v1. It may come back later as a Teach scenario ("teach the brain a new kind of task"); the model and `Dataset.nameTasks()` still support it.

### Spec sections affected (DESIGN.md not yet rewritten)
- §1 goals, §4.2–4.3 data, §6.1 task format, §8.1 Translate tab (becomes a form picker + language picker), §8.3 Teach packs, §8.7 (merged into core), §12 milestones/acceptance, §14 (language "to:" dropdown → form + language).
- "Kind" within a language is now `pattern / partly / irregular` (`formKind()` in `src/data/dataset.ts`), replacing lookalike/partial/different for scoring and colours.
- Scoreboard lines, per language: known · never-seen pattern · never-seen partly · never-seen irregular. Show the "close" status (edit distance 1) prominently, since believable near-misses are part of the lesson.

## Status

| Milestone | State |
|---|---|
| M1 brain library | **Done.** `src/nn`: tensors, tape autograd, ops (all gradient-checked), Adam/AdamW, dropout, translator with KV-cached greedy decoding, copy mechanism, target-token start, serialisation f32/f16/int8, vocabulary growth. 54 tests. |
| M2 training in the browser | **In progress.** Done: dataset v2 + validator + review doc, `Dataset.formTasks()`, experiment scripts. Not done: trainer switched to form tasks, worker + protocol, UI. |
| M3–M8 | Not started. |

## Data

- `data/animals.base.json` (version 2): 151 animals, 92 with forms (Czech 49 animals / 128 forms, English 87 / 174), 11 never-seen animals (53 forms).
- Fields: `names`, `forms` `{lang: {male, female, young}}`, `formsAlt` (accepted alternatives, never trained), `unsure` (e.g. `"cs.female"`, waiting for native-speaker review), `split`.
- **User review pending** (deferred by the user, "data tuning later"): 19 `unsure` entries; regenerate the review table with `npm run validate-data -- --review` → `docs/DATA_REVIEW.md`.
- Packs `ocean`, `insects` (names only) and `farm-babies` are from the translation design. `farm-babies` is now redundant with `young` forms; rework packs in M5.

## Model defaults (chosen from experiments)

`TRANSLATOR_DEFAULTS` + `{ copy: true, targetStart: true, dropout: 0.1 }`, Adam LR 3e-3, batch 16, **copy practice 1:1** (half of each batch is `<lang> name → name` for train-split animals). Vocabulary 55 tokens; ≈26.7k parameters with the copy head. ~43–45 steps/s in Node.

Expected results after 5 000 steps: known Czech ~95%, known English ~90–95%, never-seen Czech 7–8/26 exact (+ a few close), never-seen English ~2/27. Copy practice is the essential ingredient; the copy mechanism mainly keeps known accuracy up and gives a "copied vs written" visual (`TranslationTrace.writeShare`, `attention.copy`).

## Next steps (M2)

1. `src/train/trainer.ts`: `TranslatorTrainer` still samples translation tasks (`nameTasks`). Switch its mix to `formTasks()` (train split) + identity copy practice (`nameTasks(filter, true)`), use the defaults above, and score with `formKind()` per language (known / pattern / partly / irregular, plus "close").
2. Worker (`src/train/worker.ts`) + typed protocol (`src/train/protocol.ts`) as in DESIGN §7, with form + language instead of "to:".
3. Minimal UI: controls (play/pause/step/speed/reset), input box with form and language pickers, guess bars, loss chart, scoreboard per language, animal list with ✓ / ≈ / ✗ and never-seen badges.
4. Measure steps/s in the browser; acceptance: known ≥ 95% (or a documented decision), some never-seen Czech forms correct.
5. Update DESIGN.md to the forms design once M2 settles.

## Useful commands

```
npm test                                   # vitest (54 tests)
npm run build                              # typecheck + vite build
npm run validate-data [-- --review]        # data linter (+ review table)
npx tsx scripts/experiments/forms-full.ts 5000 1 0.1 1 1 1
    # steps seed dropout copyPracticeWeight copyMechanism targetStart
```

Branch: `ccr-0f86f24a-pa08xs`. Playwright 1.63.0 is pinned to the pre-installed browser; don't run `playwright install`.
