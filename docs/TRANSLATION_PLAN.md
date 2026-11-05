# Translation mode — implementation plan

Translation (cs ↔ en animal names) comes back as a **second task mode**, for its teaching value. It fails on never-seen animals, and that failure is the lesson: the same model that partly generalises word forms can only memorise translations, because the data has no rule for most pairs. Accuracy is not a goal (DESIGN §1).

Evidence: `docs/DECISIONS.md` → "Translation as a teaching mode", script `scripts/experiments/translate-check.ts`.

## 1. What it should teach, and what shows it

| Lesson | Measured (8 000 steps) | Where it shows |
|---|---|---|
| **Memorising ≠ understanding** | plain: known 90–94%, held-out 0% from the first step to the last | Test tab: the "known" line rises, the held-out lines stay flat |
| **The data decides, not the model** | the same model, forms: never-seen pattern words partly right; translation: different pairs 0% in every run | scoreboard split by similarity; an intro text that compares with forms |
| **Look-alike words can be read, other words cannot** | copy practice: lookalikes 7–18% exact, 25–32% right or one letter off; partial and different 0% | scoreboard lines lookalike / partial / different |
| **Confident nonsense** | 97–100% of wrong held-out answers above 0.5 confidence (kočka → *gango* 0.88) | confidence meter next to every answer, on the Ask tab and in the Test table |
| **It answers with a word it already knows** | treska → *tiger*, losos → *moose*, piraňa → *prase* (1–4 per run) | a "this is the word for another animal (tygr)" note on the answer |
| **It learns the look, not the meaning** | plain: "English" answers with Czech letters (zajíc → *čakidd*) | the same note: "Czech letters in an English word" |
| **Practising copying changes how it reads** | attention reads left to right: plain 0.25–0.5, copy practice 0.8–0.95, for every kind of word | the copy-practice switch; attention grids of the two models (step 5) |
| **Forgetting** (later) | translation mixed into forms hurts both | Teach scenario in M5 (step 6) |

## 2. Design decisions

1. **A task mode with its own model, not a mix.** A switch *Word forms / Translation* starts a new model (like Reset). The forms model is not kept; switching back trains forms from scratch (or, after M4, loads the forms replay). Mixing both tasks into one model hurt both in the experiment.
2. **The translation dataset is its own `Dataset` view.**
   - **Data:** every pack (205 animals), not only the base pack: translation needs only names, and every animal has both.
   - **Forms:** only `name`. The vocabulary then has just `<cs>` and `<en>` as task tokens (49 tokens). Unused form tokens would sit untrained on the letter map.
   - **`unseen`** is the **translation hold-out**, not the forms flag. Badges, "never seen" tags, the Learning data tab's hidden answers and the Test filters then work without changes.
3. **The hold-out is computed, not stored in the data:** every 5th animal of each similarity kind, sorted by id (the experiment's rule). That gives 14 lookalike, 6 partial and 21 different animals, and it includes the words worth asking about (kočka, treska, losos, leopard, alpaka, delfín, serval). The data format stays `id / names / forms / unseen`. If the data changes, the split changes with it; `npm run validate-data -- --review` lists it so it can be checked. *(Open question 1.)*
4. **Similarity kind = the existing `classify()`** (`lookalike / partial / different`, `src/data/dataset.ts`), left over from the first translation design. It plays the same part as `formKind()` does for forms.
5. **Copy practice is a switch, off by default.** Off: memorises fast and nothing generalises, which is the cleanest first lesson. On: slower, reads letters, gets look-alike words. Dropout follows the switch (off: 0, on: 0.1), as in the experiment. Changing it starts a new model. *(Open question 2.)*
6. **Direction = target language.** `ask{text, lang, form: 'name'}` already means "write `text` in `lang`". No source-language token: the model tells Czech from English by the spelling (identical names give the same answer either way).
7. **Confidence for every tested word.** `confidence()` needs per-step probabilities, and today only `inspect` has them. `Translator.greedy` gets an optional output array with the log-probability of each chosen sequence (cheap, the probabilities are already computed), so `evaluate` can return a confidence for every word.

## 3. Steps (one PR each, DESIGN §12.1)

### Step 1 — core: data view, trainer, scores (no UI)
- `src/data/dataset.ts`
  - `translationBundle()` (or `Dataset.forTask('translate')`): all packs, forms `['name']`, `unseen` replaced by the computed hold-out (`translationHoldOut(animals)`, exported so the UI and the review table can name it).
  - Keep `nameTasks()` as it is (both directions, and `identity` for copy practice).
- `src/train/trainer.ts`
  - `TrainerOptions.task: 'forms' | 'translate'` (default `forms`).
  - `defaultMix()` for `translate`: `nameTasks(trained)`, plus `nameTasks(trained, true) × copyWeight` when copy practice is on.
  - `evaluate()` for `translate`: `nameTasks()` of every animal, `kind = classify(cs, en)`, plus `TaskResult.from` (the source language) and `TaskResult.confidence`.
  - `Scores`: add `lookalike / partial / different` (null in forms mode); `scoreResults` counts `r.kind` generically.
  - Scores stay per *target* language: "into English" / "into Czech".
- `src/nn/translator.ts`: `greedy(…, logProbs?: Float32Array)`, which fills the summed log-probability and the length per sequence. Test: matches `confidence(inspect(…))`.
- `scripts/train-node.ts`: `--task translate` and `--copy 0|1`. Re-measure with the app's own trainer (seed 1–2, 5 000 and 8 000 steps), record in DECISIONS, and choose the default step goal (the experiment suggests 5 000 for plain; copy practice keeps improving past 8 000).
- Tests:
  - the hold-out is deterministic and has every kind;
  - in translate mode no held-out name reaches a batch (copy practice included);
  - the scores per kind;
  - the `greedy` log-probabilities.
- **Acceptance:** with seed 1, known ≥ 85% and held-out ≈ 0 for plain at 5 000 steps; held-out lookalike > 0 with copy practice. These are guides, not gates.

### Step 2 — worker and protocol
- `protocol.ts`:
  - `init` / `reset` get `task` and `copyPractice`;
  - `ready` gets `task`, and `AnimalInfo` gets `kind` (similarity, translate mode only);
  - `TaskResult` gets `from` and `confidence`;
  - `AnswerMsg` gets `confidence` (it is already computable from the trace) and `sameAs?: { animalId, word }` when the output is the real word of another animal.
- `engine.ts`:
  - builds the dataset for the task;
  - `badge()` in translate mode looks the word up in the *other* language. Today it looks for `names[lang] === word`, which is wrong for translation: `kočka` asked into English must find the cat by its Czech name;
  - `sameAs` checks the output against every animal's name in the target language;
  - `sendLetterMap` needs no change: the task tokens come from the vocabulary.
- `tests/train/engine.test.ts`: reset into translate mode, ask `kočka` → badge `hidden`, the right answer `cat`; ask a trained animal → `real`.

### Step 3 — UI
- **Header:** a *Word forms / Translation* switch next to Expert. If the model has trained, it asks first ("starts a new model"). The subtitle follows the mode.
- **Model learning tab:**
  - a "Practise copying names" checkbox (translate mode, starts a new model);
  - the practice table shows `kočka → cat` rows; copy rows keep the existing "same name" label.
- **Ask tab:**
  - **Form buttons** are replaced by "into English / into Czech".
  - **Example chips:** one or two held-out words of each kind (kočka, treska, leopard, alpaka, sova), plus a trained one (tygr).
  - **Confidence meter** shown large under the answer, with a line for held-out answers: "It never saw this animal, but it is 88% sure."
  - **Notes:** the "word for another animal" note (`sameAs`), and the "Czech letters in an English word" note (simple check against the target language's letters seen in the data).
  - **Token strip:** `<en>` shows as "into English" (`tokens.ts` already names `<cs>` / `<en>` via `tok.form.name`; check the wording).
- **Test tab:**
  - **Scoreboard lines:** known, lookalike, partial, different (`SCORE_LINES` per task), with new colours in `viz/colors.ts`.
  - **Table:** the language picker becomes the direction; rows show name → answer, the right answer (hidden for never-seen animals until revealed), the kind and the confidence. The filters stay (all / known / never seen / wrong).
- **Learning data tab** (`dataTab.ts`):
  - **Overview:** one row per animal with cs and en names, kind and never-seen mark; the questions are the two directions plus copy practice when it is on.
  - **Counts line:** "Learning: N questions (…translations + …copy the name) from 164 animals; testing: 82 never-seen questions".
- **Letter map:** nothing to do (only `<cs>` / `<en>` exist in this vocabulary).
- **i18n:** every new string in `en.json` and `cs.json`. Czech wording goes to the user for review (HANDOFF, "Waiting for the user").
- **E2E:** smoke test switches to Translation, trains a few steps, asks `kočka` into English and sees the hidden badge and a confidence. `npm run screenshots` gets a `translation/` set.

### Step 4 — explanations ("what to notice")
Short texts in the places where each lesson shows, behind "?" where they are long:
- **Test tab intro:**
  - "Known animals go up, never-seen ones stay at zero: it memorised, it did not learn a rule.";
  - compare with Word forms: "There, never-seen words partly work, because forms have a rule (keep the stem, change the ending)."
- **Similarity lines:** "kočka and cat have no letters in common: nothing in the letters says what the word means."
- **Ask tab:**
  - the confidence line, the "another animal's word" note and the Czech-letters note, each with one sentence on why it happens;
  - a link from the confidence line to how big language models "hallucinate".
- **Copy-practice switch help:** what changes (slower to learn, reads letters, gets look-alike words), and a pointer to the attention grid.

### Step 5 — compare the attention of two models (optional)
The attention lesson needs two models side by side.
- **How:** when the user restarts with the other copy-practice setting, keep a frozen copy of the previous model in the worker (`ParamSet.snapshot()`), and let the Ask tab show "the other model's attention" for the same question.
- **When:** after M4, which builds the frozen "view model" this needs.
- **Until then:** the help text and a screenshot pair in the explanation.

### Step 6 — Teach scenario "teach a new kind of task" (M5)
As in DESIGN §8.3: start from a trained forms model and add translation, with rehearsal on and off. The forms scores visibly drop without rehearsal. It needs M5's vocabulary-growth UI and the mixed trainer; the experiment's `mixed` setup is the reference.

## 4. Open questions for the user
1. **Hold-out:** computed by the rule above (recommended, no data change), or chosen by hand (a list in code, or a new data field)?
2. **Copy practice off by default?** Recommended: off, so the first run shows pure memorising; the help suggests trying it on.
3. **Where the switch lives:** in the header (recommended), or as a choice on the Model learning tab next to Reset?
4. **Packs:** use all 205 animals for translation (recommended: more look-alike words, which makes a better hold-out), or the 151 base animals only, like forms?

## 5. Not planned
- A source-language token (`<cs→en>`): not needed with two languages; revisit with 3+ languages and zero-shot directions (DESIGN §16).
- Better translation accuracy (bigger model, more data): not the point, and the experiment shows the limit is the data.
