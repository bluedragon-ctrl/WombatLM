# WombatLM handoff — read this first

State of the project at the end of the first working session, and what changed against `docs/DESIGN.md`.
**Project name: WombatLM** (same as the repository; it replaces the working title "Zvířátka AI" everywhere: page title, package name, docs).

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
| M2 training in the browser | **Nearly done.** Dataset v2 + validator + review doc, `FormTrainer` on word forms with copy practice, worker + engine + protocol, minimal UI, Playwright smoke test. ~38 steps/s in the browser. Open: a long browser run to confirm known ≥ 95%, the user's data review, DESIGN.md rewrite. |
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

Done in the second session: steps 1–3 below (see DECISIONS.md "M2 — trainer on word forms", "worker and protocol", "minimal browser UI").

1. ~~Trainer on form tasks + copy practice~~ → `FormTrainer` (`src/train/trainer.ts`).
2. ~~Worker + protocol~~ → `src/train/{engine,worker,protocol}.ts`.
3. ~~Minimal UI~~ → `src/ui/app.ts`, `src/viz/*`.
4. Acceptance: run ~5 000 steps in the browser (≈ 2.5 min at max speed) and record known / never-seen numbers; Node runs give known cs 98% / en 93% at 5k steps, so English may need a decision (longer training or accept ~93%).
5. User reviews: dataset (`npm run validate-data -- --review`) and the Czech UI strings (`src/ui/i18n/cs.json`).
6. Rewrite DESIGN.md for the forms design, then M3 (attention grids, letter map, Inside the brain). The answer message already carries the attention maps.

## Useful commands

```
npm test                                   # vitest (66 tests)
npm run e2e                                # Playwright smoke test (builds + previews)
npm run dev                                # the app
npx tsx scripts/train-node.ts 5000 1       # the app's trainer in Node
npm run build                              # typecheck + vite build
npm run validate-data [-- --review]        # data linter (+ review table)
npx tsx scripts/experiments/forms-full.ts 5000 1 0.1 1 1 1
    # steps seed dropout copyPracticeWeight copyMechanism targetStart
```

Branch: `ccr-0f86f24a-pa08xs`. Playwright 1.63.0 is pinned to the pre-installed browser; don't run `playwright install`.
