# Zvířátka AI — design document

> **⚠️ Superseded in part:** the main task changed from cs↔en translation to within-language word forms (animal → male / female / young). Read `docs/HANDOFF.md` first; it lists which sections below are out of date.

A tiny neural network that learns to translate Czech ↔ English animal names, trained live in the browser, with every part of its "brain" visible while it learns. A second mode lets the same machinery invent new animal names.

> **For the implementing session:** this is the spec. Build it milestone by milestone (section 12). Each milestone ends with its acceptance criteria passing and a commit. If a number below (model size, learning rate, steps) turns out wrong in practice, change it and note why in `docs/DECISIONS.md`. Don't silently drift from the spec.

---

## 1. Goals and non-goals

**Goals**
- Show *how* a model learns: weights changing, mistakes going down, attention forming, letters organising themselves.
- Make **memorisation vs. generalisation** visible: the model must get some never-seen animals right (lookalike words such as `tygr → tiger`) and visibly fail on others (`veverka → squirrel`).
- Show that a model can be **taught new things later** (new animals, new packs) and what happens to old knowledge when it is (forgetting).
- Run fully in the browser, static hosting, no server, no GPU.
- Small, readable code: the whole neural network is our own TypeScript that a curious teenager can read.

**Audience:** age 11+ and adults. Plain-language text by default. Each panel has a "?" with a short explanation plus a "More detail" section that uses real terms (weights, gradient, loss, attention, embedding).

**Non-goals:** mobile/tablet layout, Safari support, pictures/emoji of animals, a backend, accounts, pretrained big models, TensorFlow.js.

**Supported:** desktop Chrome, Edge, Firefox (current versions), viewport ≥ 1280×720.

---

## 2. Tech stack

| Concern | Choice |
|---|---|
| Language | TypeScript (strict) |
| Build | Vite |
| UI | Vanilla TS + DOM for controls, **Canvas 2D** for all visualisations. No UI framework (keeps the bundle tiny and code readable). |
| Maths | Own tensor and autograd library on `Float32Array` (section 5). Zero ML dependencies. |
| Concurrency | Training in a **Web Worker**, rendering on the main thread |
| Tests | Vitest (unit, gradient checks), Playwright (smoke test) |
| Hosting | GitHub Pages via GitHub Actions |
| Randomness | Seeded PRNG (mulberry32). Same seed means the same run, bit for bit. |

Runtime dependencies: none. Dev dependencies: vite, typescript, vitest, playwright (pre-installed in Claude cloud sessions; don't reinstall browsers).

---

## 3. Repository layout

```
/
├─ index.html
├─ data/
│  ├─ languages.json           # list of data languages (v1: cs, en)
│  ├─ vocab.json               # character vocabulary + special/target tokens
│  ├─ forms.json               # word forms (v1: ["name"], M8 adds "young")
│  ├─ animals.base.json        # base dataset (~150 animals)
│  └─ packs/
│     ├─ ocean.json            # extension packs for "Teach" mode
│     ├─ insects.json
│     └─ farm-babies.json      # e.g. štěně → puppy (hard: no pattern)
├─ public/
│  └─ replay/                  # baked pretrained runs (section 9)
├─ scripts/
│  ├─ bake-replay.ts           # trains in Node, writes replay files
│  └─ validate-data.ts         # dataset linter (also run in tests)
├─ src/
│  ├─ nn/                      # the "brain": pure, no DOM, runs in Node + worker
│  │  ├─ tensor.ts             # Tensor type, shape helpers
│  │  ├─ autograd.ts           # tape, backward
│  │  ├─ ops.ts                # matmul, add, softmax, layernorm, gelu, gather, crossEntropy, …
│  │  ├─ optim.ts              # Adam, grad clipping, LR schedule
│  │  ├─ rng.ts
│  │  ├─ vocab.ts              # fixed character vocabulary
│  │  ├─ transformer.ts        # attention, blocks
│  │  ├─ translator.ts         # encoder–decoder model
│  │  ├─ namer.ts              # decoder-only "invent an animal" model
│  │  └─ serialize.ts          # weights <-> binary (f32 / f16 / int8)
│  ├─ train/
│  │  ├─ worker.ts             # Web Worker entry: owns models and training loop
│  │  ├─ protocol.ts           # typed messages main <-> worker
│  │  ├─ trainer.ts            # batching, steps, eval, snapshots
│  │  └─ snapshots.ts          # snapshot store + time-travel
│  ├─ data/
│  │  ├─ dataset.ts            # load, merge packs, user additions, splits
│  │  └─ storage.ts            # localStorage, import/export JSON
│  ├─ viz/                     # one file per panel, Canvas 2D
│  │  ├─ guessBars.ts
│  │  ├─ attentionGrid.ts
│  │  ├─ letterMap.ts          # PCA 2D map of character embeddings
│  │  ├─ weightHeatmap.ts
│  │  ├─ lossChart.ts
│  │  ├─ scoreboard.ts         # known vs never-seen accuracy
│  │  ├─ timeline.ts
│  │  └─ colors.ts             # shared diverging palette
│  ├─ ui/
│  │  ├─ app.ts, layout.ts, controls.ts, tabs.ts, tour.ts, help.ts
│  │  └─ i18n/{cs.json,en.json,index.ts}
│  └─ main.ts
├─ tests/                      # vitest
├─ e2e/                        # playwright smoke
└─ docs/DECISIONS.md
```

Rule: `src/nn` must not import from the DOM or worker code, so the same code runs in tests, in Node (baking) and in the worker.

---

## 4. Data

### 4.1 Vocabulary (data-driven, can grow)
Lower-case, Unicode NFC. The vocabulary is **defined in `data/vocab.json`, not hard-coded**, so more languages can be added later (section 15). v1 content:

- `a–z` (26)
- Czech letters: `á č ď é ě í ň ó ř š ť ú ů ý ž` (15)
- space, hyphen `-` (2)
- special tokens: `<pad> <bos> <eos> <unk>` (4)
- one **target token per (language, form)** pair: in v1 the only form is `name`, giving `<cs> <en>` (2). A target token tells the model what to write (and in Invent mode, which language to sound like). Forms are listed in `data/forms.json` (v1: `["name"]`). M8 adds the form `young`, giving the extra tokens `<cs:young> <en:young>` through vocabulary growth (see below).

Total in v1: **49 tokens**. Code must never assume 49 or a fixed set of languages: it reads the sizes from `vocab.json` and the language list from `data/languages.json`. Max word length **20 characters**, so the max sequence length is 22 (including the special tokens). Input outside the vocabulary is shown with a warning and mapped to `<unk>`.

`data/languages.json` (v1):
```json
[
  { "code": "cs", "name": { "cs": "čeština", "en": "Czech" } },
  { "code": "en", "name": { "cs": "angličtina", "en": "English" } }
]
```

**Growing the vocabulary** (`growVocab` in `src/nn/serialize.ts`): new tokens are appended to the end of the vocabulary, and the embedding table gains new rows (small random init) while all existing weights stay unchanged. A trained model therefore keeps what it knows when a language with new letters is added. Implement and test this in M1 even though v1 doesn't need it.

### 4.2 Animal file format
```json
{
  "pack": "base",
  "version": 1,
  "title": { "cs": "Základní zvířata", "en": "Basic animals" },
  "animals": [
    { "id": "tiger",    "names": { "cs": "tygr",    "en": "tiger" },    "split": "train" },
    { "id": "leopard",  "names": { "cs": "leopard", "en": "leopard" },  "split": "unseen" },
    { "id": "dog",      "names": { "cs": "pes",     "en": "dog" },      "split": "train" },
    { "id": "squirrel", "names": { "cs": "veverka", "en": "squirrel" }, "split": "unseen" },
    { "id": "cat",      "names": { "cs": "kočka",   "en": "cat" },      "split": "train",
      "alt": { "cs": ["kocour"] } }
  ]
}
```
- `names`: one entry per language code. An animal may lack some languages (e.g. a pack that only adds German names to existing ids). Packs **merge by `id`**, so a later "German" pack can add `"de"` names to the base animals.
- `kind` is **computed, not stored**, because it depends on the language pair (tygr/tiger are lookalikes, but pes/dog are different, while Czech pes and Slovak pes are identical). `kind(a, b)` uses the normalised edit distance after stripping diacritics: ≤ 0.34 → `lookalike`, ≤ 0.6 → `partial`, otherwise `different`. Thresholds live in one place and can be tuned. For cs↔en this gives e.g. tygr/tiger lookalike, papoušek/parrot partial, pes/dog different. Used for colouring and to explain results. An optional `"kindOverride": {"cs-en": "different"}` handles the rare wrong guesses.
- `split`: `train` or `unseen`, **per animal** (an unseen animal is unseen in all languages, so there is no leakage through a third language). About 15–20% `unseen`, stratified so each kind (for cs↔en) has some unseen examples.
- `alt`: accepted alternative answers for evaluation only. Training uses the main form.
- `young` (optional, used from M8): the name of the animal's young, per language, e.g. `"young": {"cs": "kotě", "en": "kitten"}`, with optional `"youngAlt": {"cs": ["koťátko"]}`. Any language may be missing, and that is expected: missing young names are what the model will invent. v1 code must already accept and validate this field (and ignore it for training) so the base dataset can carry young names from the start. Generic phrases (*mládě*, *tygří mládě*, *baby tiger*, *elephant calf*) are **not** main entries; put them in `youngAlt` only.

### 4.3 Base dataset content
About **150 animals**, roughly 45% lookalike, 20% partial, 35% different. Mammals, birds, reptiles, fish, insects, sea animals. Prefer common, well-known names. **The user is a native Czech speaker and must review the list** before M2 is accepted (spelling, diacritics, natural word choice).

### 4.4 Extension packs (for "Teach" mode)
Separate files, same format, all their animals `split: "train"` plus a few `unseen`. Planned: `ocean` (~20), `insects` (~20), `farm-babies` (~12: štěně→puppy, kotě→kitten, hříbě→foal…: deliberately "different" words, hard to learn). Packs are how the project grows: adding animals later means adding a JSON file, no code changes.

### 4.5 Validation (`scripts/validate-data.ts`, also run as a test)
All characters in vocabulary · lengths ≤ 20 · no duplicate ids · no duplicate cs or en across packs · NFC normalised · lower-case · every pack has both splits (except tiny user packs).

### 4.6 User additions
In Teach mode a user can type a pair. It is stored in `localStorage` as a pack `user`. Export/import as JSON in the same format (so good additions can be sent to the maintainer and merged into the repo).

---

## 5. The neural-network library (`src/nn`)

### 5.1 Approach
A small **tape-based reverse-mode autograd** over dense `Float32Array` tensors (row-major, up to 3-D). Each op records its backward closure on a tape, and `backward(loss)` walks the tape in reverse. Parameters are `Tensor`s with `.grad`. Use plain loops, no SIMD tricks. Make it readable first and fast enough second (see the performance budget in section 10).

### 5.2 Ops needed
`matmul` (with batched variant), `add` (with broadcasting of bias), `mul`, `scale`, `gather` (embedding lookup), `layerNorm`, `gelu`, `softmax` (row-wise, with additive mask), `transpose` (last two dims), `reshape/splitHeads/mergeHeads`, `concat`, `crossEntropy` (with ignore-index for `<pad>`, fused log-softmax for stability), `dropout` (not used in v1; leave it out).

### 5.3 Correctness
- **Gradient check** every op with finite differences (float64 reference path or eps=1e-3 with relative tolerance 1e-2 in f32).
- Gradient check the full translator at a tiny config (d=8, 1 head).
- **Overfit test:** a model must reach loss < 0.05 on 4 pairs within N steps.
- **Determinism test:** two runs with the same seed give identical weights after 50 steps.

### 5.4 Optimiser
Adam (β1 0.9, β2 0.98, ε 1e-9), global gradient-norm clipping at 1.0, linear warmup over 100 steps then constant LR. Default LR 3e-3. The expert panel exposes LR (log slider 1e-4…3e-2) and batch size.

---

## 6. Models

### 6.1 Translator: tiny Transformer encoder–decoder (character level)

| Hyperparameter | Value |
|---|---|
| d_model | 32 |
| heads | 2 (head dim 16) |
| FFN hidden | 64, GELU |
| encoder layers | 1 |
| decoder layers | 1 (masked self-attention + cross-attention) |
| positions | learned, 22 |
| norm | pre-LayerNorm, final LayerNorm |
| embeddings | **one shared character embedding table** for encoder input, decoder input **and** output projection (tied). This keeps the single "letter map" meaningful. |
| parameters | ≈ 25k (log the exact count at startup) |

**One model, all directions.** The encoder input is the target-language token followed by the source word: `<en> t y g r` means "write this in English", `<cs> t i g e r` means "write this in Czech". The source language is **not** given; the model works it out from the letters. This is how real multilingual translators work, and it scales to N languages without changing anything. The decoder generates the target from `<bos>` until `<eos>`. Each animal with names in languages A and B gives the examples A→B and B→A. With N languages, the trainer samples random (source, target) pairs among the languages an animal has. From M8 the target can also be a young form (`<cs:young> tiger` → the model's guess, e.g. `tygře`); the source is always an adult name in any language (section 8.7).

**Training:** teacher forcing, cross-entropy on target characters, batch of 16 examples sampled from the train split (both directions mixed), no label smoothing (so probability bars stay honest).

**Inference:** greedy decoding, max 22 steps. For visualisation the worker returns, per output step: the top-5 characters with probabilities, the cross-attention weights per head (output step × input position), and the decoder self-attention.

**Expected behaviour** (verify in M2, adjust size/LR if needed): train accuracy close to 100% after a few thousand steps. Unseen lookalike words often correct or nearly correct; unseen different words wrong. That contrast is the key teaching moment.

### 6.2 Namer: "Invent an animal" (decoder-only character language model)
- Same building blocks: 2 decoder-only blocks (masked self-attention only), d_model 32, 2 heads, its own embeddings, ≈ 25k params.
- Trained on all train-split names from all loaded packs, with a language token: `<cs>` + name for Czech names and `<en>` + name for English names, so the user can pick "Czech-sounding" or "English-sounding" (and later any loaded language).
- Sampling with a **temperature** slider (0.2 to 1.5) and the probability bars shown per generated letter.
- Each generated name gets a label: **"real animal!"** (exists in the dataset), **"new!"**, or **"looks like X"** (nearest real name by edit distance ≤ 2).
- It trains in the same worker, with its own trainer instance, only when its tab is active (or on demand).

---

## 7. Training engine and worker protocol

The worker owns the models, the optimiser state, the dataset, and the snapshot store. The main thread never holds live weights; it receives rendering-ready data.

### 7.1 Messages (main → worker)
`init{seed, packs}` · `play` · `pause` · `step{n}` · `setSpeed{stepsPerTick}` · `setHyper{lr, batch}` · `reset{seed}` · `translate{text, to}` · `inspect{matrixIds}` · `seekSnapshot{index}` · `returnToLive` · `addAnimals{pack}` · `setRehearsal{on}` · `namer.*` (equivalents) · `exportWeights` · `importWeights{buffer}`.

### 7.2 Messages (worker → main)
- `tick` (≤ 20 Hz, throttled): step, train loss (EMA), last LR, gradient norm.
- `eval` (every 50 steps): accuracy on train known / unseen lookalike / unseen different, plus a per-animal status (correct, close (edit distance 1), wrong).
- `translation`: output string, per-step top-5, attention matrices.
- `weights`: requested matrices as `Float32Array` (transferable), plus their **delta since the previous request**.
- `letterMap`: vocab-size × 2 PCA coordinates.
- `snapshotIndex`: list of (step, loss) for the timeline.

The UI re-requests `translation`/`weights`/`letterMap` for the currently shown word after each tick while visible. Only the visible panels request data.

### 7.3 Snapshots and time travel
- Save a full weight snapshot at steps 0, 1, 2, 4, 8, … (log-spaced) and every 100 steps after that, capped at 300 snapshots (≈ 25k × 4 B ≈ 100 KB each, max ~30 MB; drop every other old one when full).
- The timeline slider seeks: the worker loads the snapshot into a separate **view model** and all panels show it, while live training is paused. "Back to now" returns to the live view model.
- Snapshots also store loss and eval scores, so the scoreboard and loss chart can show a cursor at that time.

---

## 8. Screens and panels

Desktop layout, ≥ 1280 wide. Top bar: logo · tabs **Translate · Inside the brain · Teach · Invent** · CZ/EN switch · "Expert" toggle · "?" tour.

### 8.1 Translate (main tab)
```
┌───────────────┬──────────────────────────────┬──────────────────┐
│ CONTROLS      │  [ tygr          ] (CZ→EN ⇄) │  LETTER MAP      │
│ ▶ Play ⏸      │  → t i g e r                 │  (2D dots, vowels│
│ Step ×1 ×10   │                              │   vs consonants, │
│ Speed ▭▭▭     │  GUESS BARS per letter       │   Czech letters  │
│ Reset (seed)  │  t ███████ 81%  d ██ 9% …    │   highlighted)   │
│ [Expert: LR,  │                              │                  │
│  batch]       │  ATTENTION GRID              │                  │
│               │  input letters × output      │                  │
│ ANIMAL LIST   │  letters, head 1 | head 2    │                  │
│ ✓ tygr→tiger  │                              │                  │
│ ✗ pes→dag     │                              │                  │
│ ◌ leopard (never seen)                       │                  │
├───────────────┴──────────────────────────────┴──────────────────┤
│ MISTAKES (loss) chart  │  SCOREBOARD: known ▬ / unseen-lookalike ▬ / unseen-different ▬ │
│ TIMELINE ◀━━━━━━━━━●━━━━━━━━▶  step 1840   [back to now]                               │
└─────────────────────────────────────────────────────────────────┘
```
- **Animal list:** every animal with its current status, coloured by `kind`, unseen ones marked with a "never seen" badge. Clicking one puts it in the input box. Filters: all / known / never seen / wrong.
- **Guess bars:** the main visual. For each output position, the top-5 characters with bars. The chosen character is highlighted. Hovering a column highlights the matching attention row.
- **Attention grid:** cells shaded 0 to 1 (sequential palette), one grid per head, letters on the axes. Hovering shows the value. Expert mode also shows decoder self-attention and encoder self-attention.
- **Letter map:** PCA of the shared embedding table. Dots labelled with the letter; vowels, consonants, Czech-only letters and special tokens use different marker shapes and colours. **Stability:** the PCA axes are sign-aligned and Procrustes-rotated to the previous frame so the map moves smoothly instead of flipping. Points animate (lerp) between updates.
- **Scoreboard:** three lines over time: accuracy on known animals, on unseen lookalikes, on unseen different words. This chart *is* the memorise-vs-generalise lesson.
- **Loss chart:** EMA-smoothed train loss with a log y-axis in expert mode and linear in plain mode, labelled "Mistakes" in plain mode.

### 8.2 Inside the brain
- A grid of **weight heatmaps**, one per matrix (embeddings 49×32, Q/K/V/O per attention, FFN in/out), labelled in plain words ("letter table", "what to look for (Q)", …) with the technical name in expert mode.
- Diverging palette: blue negative, white zero, red positive. Symmetric scale per matrix (robust max = 99th percentile of |w|).
- Mode toggle: **Weights** / **Change in the last steps** (delta, own scale) / **Gradient**. The change view is the default: raw weights look like noise, changes show learning.
- Hovering a cell shows its exact value and its row/column meaning when known (e.g. "embedding of letter 'ž', dimension 7").
- Header shows the total parameter count ("This brain has 24 816 numbers. ChatGPT-class models have hundreds of billions.").
- One "follow the signal" view in expert mode: for the current word, the activations through each layer as heatmap strips.

### 8.3 Teach (new things later)
The point: a trained model can learn new animals, and it can also forget old ones.
1. Choose a pack (ocean, insects, farm babies) or type your own pairs.
2. **Before:** the model tries the new animals cold, and the results are shown.
3. Toggle **"Keep practising old animals"** (rehearsal: each batch is 50% old, 50% new) **on/off**.
4. Train. The scoreboard gains a fourth line, "new animals", and you watch old-animal accuracy too. With rehearsal off, old knowledge visibly drops (catastrophic forgetting). With it on, it doesn't.
5. **After:** the comparison table, before vs after.

Also: export the user pack as JSON, and import a pack.

### 8.4 Invent
- Language choice (Czech-sounding / English-sounding), temperature slider, "Invent 10" button.
- The list of generated names with labels (real! / new! / looks like X).
- Click a name to see its letter-by-letter guess bars ("why did it pick 'ž' here?").
- Mini loss chart and Train/Pause controls for the namer.

### 8.5 Help and tour
- A first-visit **tour** (5 steps): this is a tiny brain → press Play → watch mistakes go down → look at the attention → try a never-seen animal.
- A "?" on every panel opens a popover with "In short" (2–3 sentences, no jargon) and "More detail" (correct terminology, a formula where helpful).
- An "About" page: how it works, a link to the `src/nn` code, honest comparison with real LLMs.

### 8.6 Internationalisation
All UI strings are in `cs.json` / `en.json`; the default comes from `navigator.language`, and the choice is stored in `localStorage`. Animal data is inherently bilingual. **The Czech copy must be reviewed by the user.**

---

### 8.7 Baby animals (M8)
The lesson: **a model always answers, even when it doesn't know**, and sometimes it's confident about it (hallucination). Czech young names follow a learnable pattern (kočka → kotě, kachna → kachně, slon → slůně, vlk → vlče, lev → lvíče), with irregular exceptions (pes → štěně, kůň → hříbě). English is mostly irregular (puppy, kitten, foal, lamb, joey), with weak patterns (*-ling*, *-let*, "cub" for big cats and bears, "calf" for big animals). Expect believable Czech inventions (*žirafě*) and funny English ones.

**Translate tab:** the "to:" dropdown gains "Czech baby" and "English baby". Each result gets a badge:
- ✓ **real word**: the dataset has it and the model trained on it.
- 🔒 **hidden real answer**: the dataset has it but it's in the `unseen` split. A "reveal" button shows the real word and whether the model got it.
- ✨ **the model's own invention**: no real answer exists in the data.

Next to the badge, a **confidence meter**: the probability of the whole generated word (product of per-letter probabilities, shown as a 0–100 scale using the geometric mean per letter so long words aren't penalised). The help text says plainly that a confident answer can still be wrong.

**"Baby animals" gallery (new tab, or a section inside Invent):** for every animal without a young name, the model's inventions in Czech and English, as cards: adult name → invented young name, ✨ badge, confidence meter. Sorted by confidence, with a toggle to sort A–Z. Clicking a card opens it in Translate with guess bars and attention (the attention often shows the model copying the stem and attaching an ending, which is the pattern made visible). The gallery updates live while training, so you can watch inventions change.

**Scoreboard:** a line "young names (hidden real answers)", split Czech / English, so the different learnability of the two languages is visible.

**Data:** about 60–100 animals with at least one young name, the rest without. About 20% of the young names go to `unseen` (following the animal's split). A native speaker reviews all young names; rare or debated Czech forms are better left out than taught.

**Training mix:** young-name examples make up about 30% of each batch once the form is enabled, so the smaller dataset isn't drowned out. Tune and record in `docs/DECISIONS.md`.

**Teach mode tie-in:** a Teach scenario "Teach it baby names" starts from a model trained on names only, grows the vocabulary with the `young` target tokens (existing weights kept), and trains on young names with rehearsal on/off. It shows that a model can learn a new *kind* of task, not only new words.

---

## 9. Pretrained replay ("show something immediately")
- `scripts/bake-replay.ts` runs the same `src/nn` code in Node with a fixed seed, training translator and namer on the base dataset, and records the snapshot schedule from section 7.3 (thinned to ~60 snapshots).
- Stored as `public/replay/translator.bin` plus a JSON index, weights quantised to **int8 with a per-matrix scale** (~25 KB per snapshot, ~1.5 MB total), along with the loss/eval history.
- On first load the app opens in **replay mode**: the timeline is fully populated, and the user can scrub from "baby brain" to "trained brain" immediately. The button **"Train your own brain"** resets with a new seed and trains live.
- `npm run bake` regenerates it. CI checks that the replay was baked with the current dataset and model version (hash stored in the index); if they don't match, the app falls back to live training only.

---

## 10. Performance budget
- Target: **≥ 30 training steps/s** (batch 16) for the translator in the worker on a mid-range laptop, so a run converges in ~1–3 minutes at full speed. Measure in M2. If too slow: avoid allocating per op (reuse buffers in a tape arena), fuse layernorm and softmax backward, then reduce d_model to 24.
- Speed slider: 1 step per tick (slow, kids can watch) to "as fast as possible" (time-boxed loop, ~12 ms of work per slice, so `pause` messages are handled promptly).
- Rendering: each panel redraws only when its data changes, and at most 30 fps. Canvases are HiDPI-aware (devicePixelRatio).

---

## 11. Visual style
- Clean, light theme with a dark-mode toggle. Large readable type (≥ 15 px body text).
- **One diverging palette** (weights) and **one sequential palette** (attention, probabilities) used everywhere. Category colours for `kind`: lookalike, partial, different. Unseen items use a dashed outline.
- Motion: smooth transitions (letter map lerp, bar easing), no flashing.
- Accessible: colour is never the only cue (✓/✗ icons, patterns), keyboard focus works for all controls.

---

## 12. Milestones

Each milestone ends with tests green, `npm run build` passing, and a commit and push.

**M1: The brain library.** Project scaffold (Vite, TS strict, Vitest, CI). `src/nn`: tensors, autograd, ops, Adam, RNG, vocabulary, translator model.
✔ Gradient checks for every op and for the full tiny model · overfit test · determinism test · parameter count logged.

**M2: Training in the browser.** Dataset files (base list for user review) and validator. Worker plus protocol, trainer, eval. Minimal UI: controls, translate box, guess bars, loss chart, scoreboard, animal list.
✔ Training converges in the browser · ≥ 30 steps/s measured (or a documented decision) · train accuracy ≥ 95% · at least some unseen lookalikes correct · **user reviews the dataset**.

**M3: Seeing inside.** Attention grids, letter map (PCA with Procrustes alignment), "Inside the brain" tab with weight/delta/gradient heatmaps.
✔ Visual check with Playwright screenshots of each panel at step 0 and after training.

**M4: Time travel and replay.** Snapshot store, timeline seeking, bake script, int8 replay, replay-first start.
✔ The page opens to a ready replay with no training needed · seeking is instant (< 100 ms) · "Train your own" works.

**M5: Teach mode.** Packs, user additions, localStorage, import/export, rehearsal toggle, before/after table, "new animals" line.
✔ Forgetting is visible with rehearsal off and largely prevented with it on (verify, and tune the defaults if not).

**M6: Invent mode.** Namer model, training, sampling with temperature, labels, per-letter bars.
✔ Generates plausible names at temperature 0.8, with mostly "new!" labels.

**M7: Polish and ship.** CZ/EN strings complete, help popovers, tour, About page, dark mode, GitHub Pages deploy workflow, Playwright smoke test (load, play 5 s, translate a word, switch tabs and language).
✔ Live URL works in Chrome, Edge and Firefox · **user reviews the Czech copy**.

*v1 is complete here. What follows is the first step after launch.*

**M8: Baby animals.** Add the form `young` to `forms.json`, the target tokens `<cs:young> <en:young>`, young names in the base dataset, the training mix, result badges (real / hidden / invention), the confidence meter, the Baby animals gallery, the scoreboard line, the Teach scenario, and a re-baked replay that includes young names. Raise d_model to 40–48 if name-translation accuracy drops noticeably (measure first and record the decision).
✔ Name translation accuracy within 3 percentage points of v1 · Czech hidden young names noticeably better than English (expected; if not, document what happened) · gallery fills with inventions within ~1 min of training · **user reviews the young-name list and a sample of inventions** · Teach scenario loads a names-only model, grows the vocabulary, and learns young names without breaking it.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Our own backprop has subtle bugs | Gradient checks for each op and end-to-end, the overfit test, and reference values from a tiny hand-computed example |
| Too slow in pure JS | Buffer reuse, op fusion, smaller d_model. The worker keeps the UI smooth regardless. |
| Unseen lookalikes don't generalise | Tune data balance (more lookalike pairs), model size, steps. Adjust the dataset `kind`/`split` labels after observing, and document it. |
| Letter map is jittery or meaningless | Procrustes alignment and lerp. If structure is weak, also offer a "colour by vowel/consonant" view and explain honestly. |
| Forgetting demo too weak or too strong | Tune pack size, LR and steps in Teach mode. Choose defaults that make the effect clear in < 1 min. |
| Dataset mistakes (Czech spelling) | Validator plus mandatory user review |
| Memory growth from snapshots | Cap the count and thin old ones out |
| (M8) Young-name task hurts name translation | Batch mix tuning, larger d_model, measure before/after |
| (M8) Inventions are boring (always "mládě", "cub") | Generic phrases kept out of main entries. If still dull, show the top-3 decodings (beam search) as alternatives in the gallery. |
| (M8) Kids take inventions as real words | ✨ badge and "invented by the model" wording on every invention, plus a help popover on hallucination |

---

## 14. Multilingual rules for v1 code (cheap now, expensive later)
v1 ships only Czech and English, but the code must already follow these rules:
- No `cs`/`en` literals outside `data/` and i18n files. Languages come from `languages.json`.
- Vocabulary size and contents come from `vocab.json`. Models are built from those sizes.
- Target tokens are generated from languages × forms (`languages.json` × `forms.json`). No code assumes the only form is `name`.
- The language picker in the Translate tab is a "to:" dropdown (only two entries in v1), and the source language is auto-detected (shown as "looks like Czech" from the dataset, or "unknown").
- The scoreboard and animal list work per selected language pair.
- The replay index stores the language set and vocab hash; if they don't match, fall back to live training.
- The UI language (cs/en strings) is independent of the data languages.

---

## 15. Next step: more languages

Adding a language = add an entry to `languages.json`, add its letters to `vocab.json` (if any are new), add a pack with `"names": {"de": …}` for existing animal ids, add UI copy for the language name. No code changes.

**Recommended order:**
1. **Slovak (sk):** very close to Czech (many identical words: pes, vlk, medveď/medvěd). New letters: `ä ĺ ľ ŕ ô`. Shows that similar languages are learned quickly and that the model leans on Czech knowledge.
2. **German (de):** new letters `ä ö ü ß`. Several animals are lookalikes of English (Hund/hound, Maus/mouse), others of Czech borrowing (Tiger, Krokodil).
3. **Polish (pl):** letters `ą ć ę ł ń ś ź ż`. Close to Czech and Slovak, very different spelling. Good for "same sound, different letters".
4. **Later, a non-Latin script:** Ukrainian or Russian (Cyrillic, ~33 new letters) or Greek. The letter map then shows a whole new island of letters that slowly drifts to sit near their Latin counterparts. A strong visual.

**Model size:** keep d_model 32 for up to 3 languages. From 4 languages, or with Cyrillic, raise d_model to 48 (≈ 50k params) if accuracy drops. Measure and record the change in `docs/DECISIONS.md`.

**New experiments this unlocks (add as panels or Teach scenarios):**
- **Zero-shot translation:** train on cs↔en and en↔de only, then test cs→de, a direction the model never practised. Show whether it works (partially, for lookalikes) and explain that big translators do the same thing.
- **Language detector:** the scoreboard per pair, plus a "which language is this word?" readout derived from the model's outputs.
- **Teach a new language to a trained brain:** load the German pack into a cs/en model, with vocabulary growth for new letters, rehearsal on/off, and watch old languages keep or lose accuracy.
- **Letter map by language:** colour letters by which languages use them.

**UI changes when there are more than 2 languages:** the "to:" dropdown gets more entries, the scoreboard gets a pair selector (or a small accuracy matrix: source × target language, colour = accuracy), and the animal list shows a column per language.

**Data review:** each new language needs a native or fluent speaker to review the pack, the same as Czech in M2.

**Young names in new languages:** each language pack may include `young` names. Slovak and Polish have Czech-like patterns (sk *mačiatko*, pl *kocię*), so the model can compare patterns across related languages. German mostly uses *-chen*/*-lein* diminutives or compounds (*Kätzchen*, *Elefantenbaby*), a different pattern again.

---

## 16. Open items for later (not v1)
- WebGPU acceleration (not needed at this size).
- Comparison mode: the same data on a no-attention model (plain RNN or bag-of-letters) to show *why* attention helps.
- "Break the brain": zero out a weight or head and see what fails (ablation).
- Shareable links to a trained model state.
