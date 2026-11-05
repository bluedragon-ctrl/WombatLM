# WombatLM — design document

A tiny neural network that learns **animal word forms** (male / female / young) in Czech and English, trained live in the browser, with every part of its "brain" visible while it learns.

```
<cs:female> tygr  → tygřice      <en:female> lion → lioness
<cs:young>  kočka → kotě         <en:young>  cat  → kitten
<cs:female> buvol → buvolice  (never seen in training, guessed from the pattern)
<cs:young>  žirafa → žirafě?  (no real word exists: the model invents one)
```

> **For the implementing session:** this is the spec. Build it milestone by milestone (section 12). Each milestone is a pull request (section 12.1). If a number below turns out wrong in practice, change it and note why in `docs/DECISIONS.md`; don't silently drift from the spec. `docs/HANDOFF.md` has the current state and next steps.

---

## 1. Goals and non-goals

**This is a teaching demo.** Its value is in showing *how* a model works: how tokens are generated letter by letter, how weights change while it learns, what it copies and what it invents. **Model accuracy does not need to be perfect.** Believable mistakes are part of the lesson. When choosing what to build or tune, prefer clearer visuals and explanations over accuracy. Accuracy numbers in this document are guides, not gates.

**Goals**
- Show how a model learns: weights changing, mistakes going down, attention forming, letters organising themselves.
- Show **token-by-token generation**: for every letter, the model's top guesses and how sure it is, and (in the attention grid) which input letters it looked at while writing it.
- Make **memorisation vs. generalisation** visible. Czech forms mostly follow patterns (tygr → tygřice, slon → slonice), so the model partly generalises to never-seen animals, with believable over-regularised mistakes (vlk → *vlkice* instead of vlčice). English forms are irregular (cub, calf, foal), so the model must memorise them and fails on never-seen ones. The contrast between the two languages is the lesson.
- Show **hallucination**: a model always answers, even when no real answer exists (žirafa has no Czech young-animal word), and it can be confident about it.
- Show that a trained model can be **taught new things later** (new animals, a new kind of task) and what happens to old knowledge when it is (forgetting).
- Run fully in the browser: static hosting, no server, no GPU.
- Small, readable code: the whole neural network is our own TypeScript that a curious teenager can read.

**Audience:** age 11+ and adults. Plain-language text by default. Each panel has a "?" with a short explanation plus a "More detail" section that uses real terms (weights, gradient, loss, attention, embedding).

**Non-goals:** mobile/tablet layout, Safari support, pictures/emoji of animals, a backend, accounts, pretrained big models, TensorFlow.js, state-of-the-art accuracy.

**Supported:** desktop Chrome, Edge, Firefox (current versions), viewport ≥ 1280×720.

### 1.1 History: why word forms and not translation
The first design taught cs↔en translation of animal names. Experiments (M2, `docs/DECISIONS.md`) showed it cannot generalise on ~150 animals: never-seen animals scored 0%, even identical words (leopard → "kowpa"). The data simply has no learnable rule. Within-language word forms do have rules, so the main task changed to forms. Translation may return as a Teach scenario ("teach the brain a new kind of task"); the model and `Dataset.nameTasks()` still support it.

---

## 2. Tech stack

| Concern | Choice |
|---|---|
| Language | TypeScript (strict) |
| Build | Vite |
| UI | Vanilla TS + DOM for controls, **Canvas 2D** for all visualisations. No UI framework. |
| Maths | Own tensor and autograd library on `Float32Array` (section 5). Zero ML dependencies. |
| Concurrency | Training in a **Web Worker**, rendering on the main thread |
| Tests | Vitest (unit, gradient checks), Playwright (smoke test, screenshots) |
| Hosting | GitHub Pages via GitHub Actions |
| Randomness | Seeded PRNG (mulberry32). Same seed, same run, bit for bit (verified: Node and the browser worker give identical results). |

Runtime dependencies: none. Dev dependencies: vite, typescript, vitest, tsx, @playwright/test (pinned to the browser pre-installed in Claude cloud sessions; never run `playwright install`).

---

## 3. Repository layout

✔ = exists, ◌ = planned.

```
/
├─ index.html                      ✔
├─ data/
│  ├─ languages.json               ✔ data languages (cs, en, sk, de)
│  ├─ vocab.json                   ✔ characters + special tokens
│  ├─ forms.json                   ✔ ["name", "male", "female", "young"]
│  ├─ animals.base.json            ✔ base dataset, 151 animals (cs, en)
│  ├─ animals.sk.json, .de.json    ✔ Slovak / German names and forms for the base animals
│  └─ packs/                       ✔ ocean, insects, farm-babies (rework in M5)
├─ public/replay/                  ◌ baked pretrained runs (section 9)
├─ scripts/
│  ├─ validate-data.ts             ✔ data linter (+ --review table)
│  ├─ train-node.ts                ✔ the app's trainer in Node
│  ├─ bench.ts                     ✔ speed benchmark
│  ├─ experiments/                 ✔ generalisation experiments (M2)
│  └─ bake-replay.ts               ◌
├─ src/
│  ├─ nn/                          ✔ the "brain": pure, no DOM
│  │  ├─ tensor.ts autograd.ts ops.ts optim.ts rng.ts vocab.ts
│  │  ├─ transformer.ts            attention, blocks, KV cache
│  │  ├─ translator.ts             encoder–decoder
│  │  ├─ explain.ts                ✔ "look inside one letter": every number of one position, split into its sources
│  │  ├─ serialize.ts              weights <-> binary (f32/f16/int8), vocab growth
│  │  ├─ pca.ts                    ✔ 2-D PCA + Procrustes (letter map)
│  │  └─ namer.ts                  ◌ decoder-only "invent an animal" model
│  ├─ train/
│  │  ├─ trainer.ts                ✔ FormTrainer: batches, steps, evaluation
│  │  ├─ engine.ts                 ✔ training loop + message handling (testable in Node)
│  │  ├─ worker.ts                 ✔ Web Worker entry
│  │  ├─ protocol.ts               ✔ typed messages main <-> worker
│  │  └─ snapshots.ts              ◌ snapshot store + time travel
│  ├─ data/
│  │  ├─ dataset.ts                ✔ load, merge packs, tasks, form kind, validation
│  │  ├─ bundle.ts                 ✔ the shipped data files
│  │  └─ storage.ts                ◌ localStorage, import/export
│  ├─ viz/                         one file per visual, Canvas 2D
│  │  ├─ canvas.ts colors.ts       ✔ HiDPI setup, shared palettes
│  │  ├─ guessBars.ts lineChart.ts ✔
│  │  ├─ attentionGrid.ts letterMap.ts weightHeatmap.ts   ✔ (M3)
│  │  ├─ vectorStrip.ts            ✔ one list of numbers as squares
│  │  └─ timeline.ts               ◌
│  ├─ ui/
│  │  ├─ app.ts style.css          ✔ the app: tabs, Forms tab
│  │  ├─ inside.ts dom.ts          ✔ Inside the brain tab, DOM helpers
│  │  ├─ explainView.ts            ✔ "look inside one letter"
│  │  ├─ i18n/{cs,en}.json, index.ts ✔
│  │  └─ tabs, tour, help          ◌
│  └─ main.ts                      ✔
├─ tests/                          ✔ vitest
├─ e2e/                            ✔ playwright smoke test
└─ docs/                           DESIGN (this), DECISIONS, HANDOFF, DATA_REVIEW
```

Rule: `src/nn` and `src/data` must not import DOM or worker code, so the same code runs in tests, in Node and in the worker. `src/train/engine.ts` gets its host (post, schedule, clock) passed in for the same reason.

---

## 4. Data

### 4.1 Vocabulary (data-driven, can grow)
Lower-case, Unicode NFC. Defined in `data/vocab.json`, not hard-coded:

- 4 special tokens: `<pad> <bos> <eos> <unk>`
- 51 characters: `a–z`, Czech letters `á č ď é ě í ň ó ř š ť ú ů ý ž`, Slovak `ä ĺ ľ ô ŕ`, German `ö ü ß` (`ä` shared), space, hyphen
- one **target token per (form, language)**: `<cs> <en> <sk> <de>` (form `name`), `<cs:male> … <de:male>`, `<cs:female> … <de:female>`, `<cs:young> … <de:young>`. Built form-major from `forms.json` × `languages.json`, so adding a form only appends tokens.

Total: **71 tokens** (55 with cs and en only). Code never assumes 71 or a fixed set of languages. Max word length 20 characters, max sequence length 22. Input outside the vocabulary maps to `<unk>`.

**Growing the vocabulary** (`growVocab` in `src/nn/serialize.ts`): new tokens are appended, the embedding table gains new rows, all existing weights stay. Models remap embeddings by token text, so the order is not load-bearing. Used for Teach scenarios and new languages.

### 4.2 Forms
`data/forms.json` = `["name", "male", "female", "young"]`.
- `male`, `female`, `young` are the word forms the model learns, within one language.
- `name` means "copy the name" (`<cs> slon → slon`). It is used as **copy practice** during training (section 6.3) and is shown everywhere in the UI as "same name" (cs "stejné jméno"): the Ask tab form button (with a tooltip saying what it does), the Model learning batch and step views, and the Learning data tab.

### 4.3 Animal file format (version 2)
```json
{
  "pack": "base",
  "version": 2,
  "title": { "cs": "Základní zvířata", "en": "Basic animals" },
  "animals": [
    { "id": "tiger", "names": { "cs": "tygr", "en": "tiger" },
      "forms": { "cs": { "male": "tygr", "female": "tygřice", "young": "tygře" },
                 "en": { "male": "tiger", "female": "tigress", "young": "cub" } } },
    { "id": "wolf", "names": { "cs": "vlk", "en": "wolf" },
      "forms": { "cs": { "male": "vlk", "female": "vlčice", "young": "vlče" },
                 "en": { "female": "she-wolf", "young": "pup" } },
      "unseen": true },
    { "id": "giraffe", "names": { "cs": "žirafa", "en": "giraffe" },
      "forms": { "en": { "male": "bull", "female": "cow", "young": "calf" } },
      "unseen": true }
  ]
}
```
- `names`: one entry per language code. Packs **merge by `id`**, so a later pack can add languages or forms to existing animals.
- `forms`: per language, any subset of the word forms. **Missing forms are deliberate**: no real word exists (žirafa has no Czech male/female/young word), and that is what the model invents. Generic phrases (*mládě*, *baby tiger*) are never main entries. Words unsuitable for an 11+ audience are left out.
- **One right word per answer.** No alternative names or forms, no "unsure" flags: a doubtful word is fixed or deleted in the data (the user edits it). The validator rejects any other field.
- `unseen: true` marks a never-seen animal; without it the model learns from the animal. **Per animal** (an unseen animal is unseen in all languages and forms, and in every pack that mentions it, so nothing leaks). Never-seen animals give the "hidden real answers".
- That is the whole format: `id`, `names`, `forms`, `unseen`. Nothing else (no kind overrides; the form kind below is always computed).

### 4.4 Form kind
How a form relates to its name, computed (`formKind()` in `src/data/dataset.ts`), not stored. The share of the name's letters (without diacritics) found in order in the form (longest common subsequence / name length):
- ≥ 0.75 **pattern** (tygr → tygřice, buvol → buvolice)
- ≥ 0.5 **partly** (lev → lvice)
- otherwise **irregular** (pes → štěně, cat → kitten)

Used for colours, the scoreboard lines and the explanations. Thresholds live in one place (`FORM_KIND_THRESHOLDS`).

### 4.5 Base dataset
151 animals: mammals, birds, reptiles, fish, insects, sea animals, with names in all four languages. Forms: Czech 49 animals / 128 forms, English 87 / 174, Slovak 43 / 107, German 52 / 128. 11 never-seen animals (98 forms: cs 26, en 27, sk 19, de 26), chosen so every language has never-seen forms of every kind. Solid Czech male/female/young words exist for only ~50 common animals; the rest are inventions by design.

Slovak and German live in their own files (`animals.sk.json`, `animals.de.json`, packs `base-sk`, `base-de`) that only add names and forms to the base animals by id; never-seen marks come from the base pack. German nouns are lower-cased. German male/female/young words for big animals are the real compounds (*elefantenkuh*, *zebrafohlen*), not *-junges* / *-baby* phrases (those count as generic).

**The user (native Czech speaker) reviews the data.** Slovak and German need a fluent reviewer (section 15). `npm run validate-data -- --review` writes `docs/DATA_REVIEW.md` with every form.

### 4.6 Extension packs (Teach mode)
Separate files, same format. Existing: `ocean` and `insects` (names only, from the translation design) and `farm-babies` (now redundant with `young` forms). **Rework in M5** into packs with forms, e.g. "farm animals" (kůň → klisna / hříbě, horse → mare / foal) and "birds" (páv → pávice). Adding animals later means adding a JSON file, no code changes.

### 4.7 Validation (`scripts/validate-data.ts`, also a test)
All characters in the vocabulary · lengths ≤ 20 · NFC · lower-case · no duplicate ids within a pack · no name used by two animals · forms only for languages with a name · known form names · no unknown fields · `unseen` is `true` or left out · packs that add ≥ 8 new animals have both trained and unseen animals (a pack that only adds languages to earlier animals keeps their split).

### 4.8 User additions
In Teach mode a user can type an animal with forms. Stored in `localStorage` as pack `user`; export/import as JSON in the same format.

---

## 5. The neural-network library (`src/nn`) — done in M1

- **Autograd:** tape-based reverse mode over dense `Float32Array` tensors (row-major, up to 3-D). Plain loops; readable first, fast enough second.
- **Ops:** `matmul`/`bmm`, `add` (broadcast bias), `mul`, `scale`, `gather`, `layerNorm`, `gelu`, `softmax` (with additive mask), `transpose`, `reshape`/heads, `concat`, `crossEntropy` (ignore `<pad>`, fused log-softmax), `dropout`.
- **Correctness:** every op gradient-checked; the full model gradient-checked at a tiny size; overfit test; determinism test; cached decoding checked against full recomputation.
- **Optimiser:** Adam (β1 0.9, β2 0.98, ε 1e-9), optional AdamW weight decay, global gradient-norm clipping at 1.0, linear warm-up over 100 steps, then constant LR. Default LR 3e-3.
- **Serialisation:** f32 / f16 / int8 (per-matrix scale), with vocabulary remapping on load.

---

## 6. Models

### 6.1 The form model: tiny Transformer encoder–decoder (character level)

| Hyperparameter | Value |
|---|---|
| d_model | 32 |
| heads | 2 (head dim 16) |
| FFN hidden | 64, GELU |
| encoder / decoder layers | 1 / 1 (decoder: masked self-attention + cross-attention) |
| positions | learned, 22, separate tables for encoder and decoder |
| norm | pre-LayerNorm, final LayerNorm |
| embeddings | **one shared table** for encoder input, decoder input and output projection (tied), plus an output bias. This keeps a single "letter map". |
| decoder start | the target token (`<cs:female>`), not `<bos>` |
| dropout | 0.1 while training |
| parameters | **24 727** |

**Input:** the target token followed by the name: `<cs:female> t y g r`. The target token says what to write (language + form). **Output:** `t y g ř i c e <eos>`, greedy decoding with a key/value cache, max 21 steps.

Why the decoder starts from the target token: without it the model often mixed up forms (čáp → male *čápice*). Seeing "what to write" at every output step fixes that.

### 6.2 No copy mechanism: copying is learned
The model has **no special copy machinery**. Keeping the stem (tygr → tygr|ice) is learned by ordinary cross-attention, the same way real language models learn to copy. The cross-attention grid shows it: while writing the stem the model looks at the matching input letter. An earlier version had a pointer-style copy mechanism with a write-or-copy gate; it was removed to keep the model simple (see `docs/DECISIONS.md`).

### 6.3 Training
- **Batch mix:** word-form tasks of trained animals (not `unseen`) (all languages, all forms) mixed **1:1 with copy practice** (`<cs> tygr → tygr` for every trained name). Copy practice is what makes never-seen forms work: it teaches "keep the stem".
- Teacher forcing, cross-entropy on target characters, batch 16, no label smoothing (so probability bars stay honest).
- `FormTrainer` in `src/train/trainer.ts`; defaults in `TRAINER_DEFAULTS`.

**Measured behaviour** (5 000 steps, ~2.5 min in the browser): known forms Czech ~98%, English ~93%. Never-seen Czech pattern forms ~30–45% exact (buvol → buvolice), more counting "one letter off"; never-seen English near 0, as expected. What the model reliably generalises is keeping the stem; endings on never-seen stems are hit-and-miss; consonant changes (vlk → vlčice) fail with believable mistakes. Exact numbers move by a few points between seeds.

### 6.4 Evaluation and results
- Every real form of every animal is asked for and judged **correct**, **close** (edit distance 1 from any accepted answer) or **wrong**.
- Forms with no real word are asked for too and marked **invented** (never scored).
- **Scoreboard per language:** known · never-seen pattern · never-seen partly · never-seen irregular, plus never-seen "right or one letter off" shown prominently (believable near-misses are part of the lesson).
- **Badge for a question** (`ask`): **real** (the data has it and the model trained on it) · **hidden** (the data has it, the animal is never-seen; a Reveal button shows it) · **invention** (known animal, no real word) · **free** (not an animal in the data).
- **Confidence:** geometric mean of the chosen letters' probabilities (including the end of the word), 0–100, so long words are not penalised. Help text: a confident answer can still be wrong.

### 6.5 Namer: "Invent an animal" (M6)
- Decoder-only character model: 2 blocks (masked self-attention), d_model 32, 2 heads, its own embeddings, ≈ 25k params.
- Trained on all trained (not `unseen`) names with a language token (`<cs>` + name, `<en>` + name), so the user can pick "Czech-sounding" or "English-sounding".
- Sampling with a **temperature** slider (0.2–1.5) and per-letter probability bars.
- Labels: **"real animal!"** (exists in the data), **"new!"**, or **"looks like X"** (nearest real name by edit distance ≤ 2).
- Trains in the same worker with its own trainer, only when its tab is active.

---

## 7. Training engine and worker protocol

The worker owns the models, optimiser state, dataset and (from M4) the snapshot store. The main thread never holds live weights; it receives data ready to draw. The loop and message handling live in `src/train/engine.ts`; `worker.ts` only wires it to the worker globals.

**Loop:** slices of ~12 ms of training, then yield (through a `MessageChannel`, since nested `setTimeout(0)` is clamped to 4 ms), so `pause` and questions are handled promptly. Speed is steps per second (1 / 5 / 20) or `max`; slow speeds earn steps over time, and the first step happens immediately on Play.

**Evaluation schedule:** real forms every 100 steps (or every 3 s at slow speeds), inventions every 500 steps, plus on init / reset / step / request. Evaluating all ~900 forms every 50 steps would halve training speed.

### 7.1 Messages main → worker (`src/train/protocol.ts`)
Built (M2): `init{seed, speed?}` · `play` · `pause` · `step{n}` · `setSpeed{speed}` · `setHyper{lr?, batchSize?}` · `reset{seed}` · `ask{id, text, lang, form}` · `evaluate{inventions?}`. Built (M3): `inspect{id, matrixIds, mode, letterMap?}`.

Planned: `seekSnapshot{index}` · `returnToLive` (M4) · `addPack{pack}` · `setRehearsal{on}` · `growTask{…}` (M5) · `namer.*` (M6) · `exportWeights` · `importWeights{buffer}`.

### 7.2 Messages worker → main
Built (M2):
- `ready`: seed, parameter count, languages, forms, animal list (names, forms, unseen), vocabulary tokens.
- `tick` (≤ 20 Hz): step, running, speed, loss EMA, LR, batch size, gradient norm, measured steps/s, and **every batch loss since the previous tick** (the loss chart is drawn from these).
- `eval`: step, scores per language, per-form results (`TaskResult`: animal, language, form, output, status, unseen, kind), whether inventions are included, and how long it took.
- `answer`: the output word, whether it finished, the input tokens, **per letter the top-5 candidates with probabilities**, the confidence, **all attention maps** (transferred buffers), the badge and, for real/hidden, the accepted answers and status.
- `error`.

Built (learning vs answering): `tick.practice` — the latest training batch: step, batch size, batch loss and up to 32 examples of **learning data** (source, form, language, right answer) and, when the brain answered them **before** the step, its answers with ✓/≈/✗ and how many were right. Answering first costs one greedy pass, so the engine does it on Step and at slow speeds (≤ 5 steps/s) always, and at higher speeds only while the Practice tab is open (`watchPractice{on}`, main → worker), at most every 500 ms.

Built (M3): `weights` (requested matrices as transferable `Float32Array`s: the weights, their change since the previous request, or the latest gradient), `letterMap` (PCA coordinates of characters, end-of-word and task tokens), `ready.matrices` (shapes and axis meanings), `answer.activations`.

Planned: `snapshotIndex` in M4.

**Learning and answering are separate.** Only `play` / `step` change the weights; `ask`, `evaluate` and `inspect` only read them (tested bit for bit). While training runs, the UI does **not** re-ask its question: the answer stays, marked out of date, until training pauses or the user presses "Ask again" (at most one request in flight). When paused it re-asks after every step. Only visible panels request data.

### 7.3 Snapshots and time travel (M4)
- Save a full weight snapshot at steps 0, 1, 2, 4, 8, … (log-spaced) and every 100 steps after that, capped at 300 (≈ 27k × 4 B ≈ 110 KB each; drop every other old one when full).
- The timeline slider seeks: the worker loads the snapshot into a separate **view model** and all panels show it, while live training is paused. "Back to now" returns to the live model.
- Snapshots also store loss and eval scores, so charts can show a cursor at that time.

---

## 8. Screens and panels

Desktop layout, ≥ 1280 wide. Top bar: logo · tabs **Forms · Inside the brain · Teach · Invent** · CZ/EN switch · "Expert" toggle · "?" tour.

### 8.1 Forms (main tab) — minimal version built in M2
```
┌──────────────────────┬────────────────────────────────────┬──────────────┐
│ CONTROLS             │ [ tygr      ] [Czech][English]     │ LETTER MAP   │
│ ▶ Play  Step×1 ×10   │ [copy name][male][female][young]   │ (M3)         │
│ Speed ▾  Seed  Reset │ looks like Czech                   │              │
│ [Expert: LR, batch]  │ tygr → tygřice   ✓ real word       │              │
│ step · steps/s · loss│ Sure ████████░░ 82                 │              │
│                      │                                    │              │
│ ANIMALS  all|known|  │ GUESS BARS per letter              │              │
│  never seen|wrong    │ t ███ 99% … ř ██ 71% e ▌ 12% …     │              │
│      male female young│                                    │              │
│ tygr ✓tygr ✓tygřice ✓│                                    │              │
│ vlk⌗ ✓vlk ≈vlkice ✗… │ ATTENTION GRID (M3)                │              │
│ žirafa ✨… ✨… ✨…     │                                    │              │
├──────────────────────┴───────────────┬────────────────────┴──────────────┤
│ MISTAKES (loss) chart                │ SCOREBOARD per language            │
│ TIMELINE ◀━━━━━━━●━━━━━▶ (M4)        │ known / pattern / partly / irregular│
└──────────────────────────────────────┴───────────────────────────────────┘
```
- **Two numbered phases** (user feedback: "not clear when it learns and when it answers"): **① Build the brain: learning** (left: controls, a "learning: N numbers nudged" badge, the one-step cycle *pick 16 examples → it answers them → measure the mistake (loss) → nudge the numbers*, animated stage by stage after Step ×1 while paused, and a link to the **Practice** tab) and **② Ask the brain: answering** (centre). Under the answer: "🔒 Answer from the brain at step N. Answering never changes the brain", flashing when the brain changed under the same question; while training: "⏳ learned N more steps since this answer" + Ask again, answer greyed.
- **Practice tab** (between Forms and Inside the brain): **what it's practising now**, the whole latest batch (up to 32) as a simple table: example · **right answer (learning data)** · **what it answered** (before learning from it, ✓ ≈ ✗), plus "it answered 10 of 16 right · mistake 0.36". Its own Play / Step ×1, the learning badge and the cycle; the columns are revealed with the cycle stages. Moved off the Forms tab at the user's request so the animal list stays high.
- **Write it slowly:** replays the current answer letter by letter (400 ms/letter): guess-bar columns and the linked attention rows appear one at a time, caption "reading the animal → writing letter 3 of 7".
- **Question box:** text input, **language picker** and **form picker** (both built from the data files). Typing a name from another language switches the language (forms stay within one language) and the hint says "looks like Czech" or "not an animal from the data".
- **Answer:** the generated word, the badge (section 6.4), the confidence meter. **A hidden answer is shown uncoloured until Reveal**, because a green or red word would give it away.
- **Guess bars:** the main visual. For each output letter, the top-5 candidates (including "end of word") with bars; the chosen one highlighted. M3: hovering a column highlights the matching attention row.
- **Animal list:** per selected language, one row per animal with columns male / female / young showing the model's current answer with ✓ ≈ ✗ ✨, a dashed "never seen" badge, filters all / known / never seen / wrong. Clicking a row or cell puts it in the question box. Tooltips show real words only for known animals.
- **Scoreboard:** lines over time for the selected language: known (solid), never-seen pattern / partly / irregular (dashed and dotted, so colour is never the only cue). A table below shows the latest numbers for every language, with the "right or one letter off" row in bold. This chart *is* the memorise-vs-generalise lesson, and the Czech-vs-English difference is visible side by side.
- **Loss chart:** EMA-smoothed batch loss, labelled "Mistakes"; linear in plain mode, log y-axis in expert mode.
- **Attention grid (M3):** cells shaded 0–1, one grid per head, letters on the axes (cross-attention: output letters × input tokens). Expert mode adds decoder and encoder self-attention.
- **Letter map (M3):** PCA of the shared embedding table, dots labelled with the letter; vowels, consonants, Czech-only letters and target tokens use different markers and colours. PCA axes are sign-aligned and Procrustes-rotated to the previous frame; points lerp between updates.

### 8.2 Inside the brain (M3)
- A grid of **weight heatmaps**, one per matrix (letter table 71×32, Q/K/V/O per attention, FFN in/out), labelled in plain words ("letter table", "what to look for (Q)", …) with the technical name in expert mode.
- Diverging palette: blue negative, white zero, red positive; symmetric scale per matrix (99th percentile of |w|).
- Mode toggle: **Weights** / **Change in the last steps** (default: raw weights look like noise, changes show learning) / **Gradient**.
- Hovering a cell shows its value and its meaning when known ("letter 'ž', dimension 7").
- Header: "This brain has 26 872 numbers. ChatGPT-class models have hundreds of billions."
- Expert: "follow the signal" for the current word: activations through each layer as heatmap strips (the `answer` trace already has them in the worker).
- **"Where is it learning?"** (added after review): the model's parts as a diagram in signal order, each with its change per step (one scale for all) and a line over time; click to jump to its heatmaps.
- **"Look inside one letter"** (added after review): for one letter of the current answer, every number of its position on the way through the decoder (goes in → each step adds → guess), with exact "where it comes from" sums on click and the used weights outlined in the heatmaps. Computed on request (`explain`). See DECISIONS.
- Heatmaps: letters / positions on the axes, a shared colour scale by default, plain-language tooltips, thumbnails for number × number matrices in plain mode.
- **Plain mode for non-experts** (user review): "How the model picks one letter" shows, after every stage, which letter the model **would write now** and how sure it is (logit lens), a "how sure, stage by stage" ladder, attention of both heads together and plain words; no rows of numbers, scores or matrix shapes (those stay in Expert). It opens on the letter where the answer stops copying the animal. See DECISIONS.

### 8.3 Teach (M5)
The point: a trained model can learn new things, and it can forget old ones.
1. Choose a pack (reworked packs with forms, section 4.6) or type your own animals.
2. **Before:** the model tries the new animals cold.
3. Toggle **"Keep practising old animals"** (rehearsal: each batch 50% old, 50% new).
4. Train. The scoreboard gains "new animals" and keeps "old animals". Without rehearsal old knowledge visibly drops (catastrophic forgetting); with it, it doesn't.
5. **After:** a before/after table.

**Scenario "Teach it a new kind of task":** start from a model trained only on some forms (e.g. without `young`), grow the vocabulary with the new target tokens (existing weights kept) and train the new form with rehearsal on/off. A second scenario can teach **cs↔en translation** to a forms model and show why it fails on never-seen animals (section 1.1).

Also: export the user pack as JSON, import a pack.

### 8.4 Invent (M6)
- **Invent a name** (namer): language choice, temperature slider, "Invent 10", labels (real! / new! / looks like X), click a name to see its per-letter guess bars.
- **Inventions gallery** (form model): for every animal form with no real word, the model's invention as a card: name → invented form, ✨ badge, confidence meter. Sorted by confidence or A–Z. Clicking a card opens it in the Forms tab with guess bars and attention (the attention often shows the stem being copied and an ending attached: the pattern made visible). Updates live while training.

### 8.5 Help and tour (M7)
- A first-visit **tour** (5 steps): this is a tiny brain → press Play → watch mistakes go down → look at the guess bars and what it copies → try a never-seen animal, then one with no real answer.
- A "?" on every panel: "In short" (2–3 sentences, no jargon) and "More detail" (correct terms, a formula where helpful). Topics include hallucination (a confident answer can be invented).
- An "About" page: how it works, a link to the `src/nn` code, an honest comparison with real LLMs (same idea of generating one token at a time, tokens there are word pieces, sizes differ by ~10⁷).

### 8.6 Internationalisation
All UI strings live in `src/ui/i18n/{cs,en}.json`; the default comes from `navigator.language`, the choice is kept in `localStorage`. The UI language is independent of the data languages. **The Czech copy must be reviewed by the user.**

---

## 9. Pretrained replay ("show something immediately") — M4
- `scripts/bake-replay.ts` runs the same code in Node with a fixed seed (training is deterministic across Node and the worker), recording the snapshot schedule (thinned to ~60 snapshots).
- Stored as `public/replay/forms.bin` + a JSON index, weights int8 with a per-matrix scale (~27 KB per snapshot, ~1.6 MB total), with the loss/eval history.
- On first load the app opens in **replay mode**: the timeline is populated and the user can scrub from "baby brain" to "trained brain" at once. **"Train your own brain"** resets with a new seed and trains live.
- `npm run bake` regenerates it. The index stores a hash of dataset, vocabulary and model config; on mismatch the app falls back to live training.

---

## 10. Performance

Measured in M2:
- **Node 22, one core:** ~40 training steps/s at batch 16.
- **Headless Chromium worker:** ~38 training steps/s, ~32 steps/s wall-clock including evaluation and UI. 5 000 steps take ~160 s.
- **Evaluation:** ~250 ms for the 355 real forms, ~640 ms with inventions (906 forms), hence the schedule in section 7.

Rules: the speed slider goes from 1 step/s (kids can watch each step) to max. Panels redraw only when their data changes, at most once per animation frame. Canvases are HiDPI-aware. If a later feature makes training noticeably slower, first reuse buffers in the tape, then fuse layernorm/softmax backward.

---

## 11. Visual style
- Clean, light theme (dark mode in M7). Readable type (≥ 15 px body text).
- **One sequential palette** (probabilities, attention: white → blue) and **one diverging palette** (weights: blue / white / red), defined in `src/viz/colors.ts` and handed to CSS as variables.
- Status colours and icons: correct ✓ green, close ≈ amber, wrong ✗ red, invented ✨ purple. Never-seen animals get a dashed badge. Scoreboard lines: known solid, never-seen dashed/dotted.
- Motion: smooth transitions (letter map lerp, bar easing), no flashing.
- Accessible: colour is never the only cue (icons, dash patterns), keyboard focus works on all controls.

---

## 12. Milestones

Each milestone ends with tests green, `npm run build` passing, and a merged pull request. Accuracy targets are guides (section 1).

**M1: The brain library.** ✅ Done. Scaffold, `src/nn` (tensors, autograd, ops, Adam, RNG, vocabulary, translator, serialisation, vocab growth), gradient checks, overfit and determinism tests.

**M2: Training in the browser.** ✅ Done (pending the user's data review). Dataset v2 with forms + validator + review doc, generalisation experiments, copy mechanism (later removed) and target-token start, `FormTrainer` with copy practice, engine + worker + protocol, minimal UI (controls, question box with pickers, badge, confidence, guess bars, loss chart, scoreboard, animal list), Playwright smoke test. Measured: ~38 steps/s; known cs 98% / en 93%; never-seen Czech pattern forms partly right.

**M3: Seeing inside.** ✅ Done (PR, pending review). Attention grids (cross, copy, self in expert mode) linked to the guess bars by hover; letter map (PCA + Procrustes + lerp); "Inside the brain" tab with weight / change / gradient heatmaps; `inspect`/`weights`/`letterMap` messages; tabs in the UI.
✔ Playwright screenshots of each panel at step 0 and after training, reviewed visually.

**M4: Time travel and replay.** Snapshot store, timeline seeking, bake script, int8 replay, replay-first start.
✔ The page opens to a ready replay with no training needed · seeking is instant (< 100 ms) · "Train your own" works.

**M5: Teach mode.** Reworked packs with forms, user additions, localStorage, import/export, rehearsal toggle, before/after table, "new animals" line, the "new kind of task" scenario with vocabulary growth.
✔ Forgetting is visible with rehearsal off and largely prevented with it on (tune defaults so the effect shows in < 1 min).

**M6: Invent.** Namer model and its tab; inventions gallery for the form model.
✔ Plausible names at temperature 0.8, mostly "new!" · the gallery fills with inventions within ~1 min of training · the user reviews a sample of inventions.

**M7: Polish and ship.** CZ/EN strings complete, help popovers, tour, About page, dark mode, GitHub Pages deploy workflow (✔ done early), Playwright smoke test (load, play, ask a word, switch tabs and language).
✔ Live URL works in Chrome, Edge and Firefox · **the user reviews the Czech copy**.

*v1 is complete here.* Then: more languages (section 15).

### 12.1 Workflow
- From M3 on, work happens through **pull requests**: one branch and one PR per milestone (or per coherent step of a milestone), reviewed by the user before merging.
- Each PR: tests and build green, `docs/DECISIONS.md` updated for any deviation, `docs/HANDOFF.md` updated with the new state, screenshots for UI changes.
- Decisions that need the user (data, Czech wording, trade-offs) are asked in the PR, not decided silently.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Our own backprop has subtle bugs | Gradient checks for each op and end-to-end, overfit test, cached-vs-full decoding test |
| Too slow in pure JS | Measured fine (~38 steps/s). If a feature slows it: buffer reuse, op fusion, smaller evaluation set |
| Never-seen forms barely generalise | Accepted: partial generalisation with believable mistakes is the lesson. Copy practice is the lever if it gets worse |
| English looks "broken" next to Czech | Explain it: English forms are irregular words that must be memorised. The scoreboard by kind shows why |
| Kids take inventions as real words | ✨ badge and "its own invention" wording everywhere, confidence meter with "can still be wrong", help on hallucination |
| Letter map is jittery or meaningless | Procrustes alignment and lerp; if structure is weak, colour by vowel/consonant and say so honestly |
| Forgetting demo too weak or too strong | Tune pack size, LR and steps in Teach mode |
| Dataset mistakes (Czech forms) | Validator, user review (fix or delete doubtful words) |
| Memory growth from snapshots | Cap the count and thin old ones out |

---

## 14. Multilingual rules (cheap now, expensive later)
The app ships Czech and English, but the code already follows these rules:
- No `cs`/`en` literals outside `data/`, i18n files and tests. Languages come from `languages.json`.
- Vocabulary size and contents come from `vocab.json`; models are built from those sizes.
- Target tokens come from `forms.json` × `languages.json`. No code assumes a fixed set of forms.
- The UI has a **language picker and a form picker**, both built from the data. Forms are always within one language.
- The scoreboard and animal list work per language.
- The replay index stores the language set and vocabulary hash; on mismatch, fall back to live training.
- The UI language is independent of the data languages.

---

## 15. Next step: more languages

Adding a language = an entry in `languages.json`, its letters in `vocab.json` (if new), a pack with names and forms for existing animal ids (listed in `LANGUAGE_PACKS`, `src/data/bundle.ts`), UI copy for the language name. No other code changes.

**Status:** Slovak and German added (first draft of the data, waiting for review); measurements in `docs/DECISIONS.md` ("Slovak and German"). The same form drill runs for every language.

**Recommended order:**
1. **Slovak (sk):** close to Czech, similar form patterns (tiger → tigrica, mačka → mačiatko). New letters `ä ĺ ľ ŕ ô`. Shows that a related language is learned quickly.
2. **German (de):** female forms mostly *-in* (Löwe → Löwin), young forms mostly *-chen* diminutives or compounds. A clean, different pattern. New letters `ä ö ü ß`.
3. **Polish (pl):** Czech-like patterns, very different spelling. Letters `ą ć ę ł ń ś ź ż`.
4. **Later, a non-Latin script** (Ukrainian, Russian, Greek): the letter map shows a new island of letters.

**Model size:** keep d_model 32 for up to 3 languages; raise to 48 if accuracy drops noticeably with more. Measure and record in `docs/DECISIONS.md`.

**Experiments this unlocks:** compare how fast patterns are learned per language (a scoreboard row per language); teach a new language to a trained brain with vocabulary growth and rehearsal on/off; colour the letter map by language.

**Data review:** each new language needs a native or fluent speaker to review its pack.

---

## 16. Open items for later (not v1)
- Translation as a Teach scenario (section 8.3), including zero-shot directions once there are 3+ languages.
- Comparison mode: the same data on a no-attention model to show why attention helps.
- "Break the brain": zero out a weight or a head and see what fails (ablation).
- Beam search / top-3 decodings for the inventions gallery.
- WebGPU acceleration (not needed at this size).
- Shareable links to a trained model state.
