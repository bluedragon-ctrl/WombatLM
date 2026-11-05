# Translation mode — implementation plan

Translation (cs ↔ en animal names) comes back as a **second task mode**, for its teaching value. It fails on never-seen animals, and that failure is the lesson: the same model that partly generalises word forms can only memorise translations, because the data has no rule for most pairs. Accuracy is not a goal (DESIGN §1).

Evidence: `docs/DECISIONS.md` → "Translation as a teaching mode", script `scripts/experiments/translate-check.ts`. The mode trains like the experiment's **`plain`** setup: translation only, **no copy practice** (user decision), no dropout.

## 1. What it should teach, and what shows it

| Lesson | Measured (`plain`, 8 000 steps) | Where it shows |
|---|---|---|
| **Memorising ≠ understanding** | known 66% after 1 000 steps, 90–94% at the end; never-seen 0% from the first step to the last | Test tab: the "known" line rises, the "never seen" line stays at zero |
| **The data decides, not the model** | the same model, forms: never-seen pattern words partly right; translation: 0% | an intro text that compares with Word forms |
| **Even identical words fail** | leopard → *coce*, serval → *l*, okapi → *pidar*: it never learned to read the letters, only to recognise whole words | a "same word" / "looks alike" tag on names in the Test table and the Learning data tab; example chips |
| **Confident nonsense** | all wrong never-seen answers above 0.5 confidence, mean 0.85 (kočka → *gango* 0.88, shark → *ele* 0.96) | confidence meter next to every answer, on the Ask tab and in the Test table |
| **It answers with a word it already knows** | treska → *tiger*, losos → *moose*, piraňa → *prase*, dolphin → *pes* | a "this is the word for another animal (tygr)" note on the answer |
| **It learns the look, not the meaning** | 4–6 "English" answers with Czech letters (zajíc → *čakidd*, žralok → *mowrdřid*) | a "Czech letters in an English word" note on the answer |
| **It recognises, it does not read** | the cross-attention barely goes left to right (0.25–0.5; the forms model reads the stem letter by letter) | help text by the attention grid, comparing with Word forms |
| **Forgetting** (later) | translation mixed into forms hurts both | Teach scenario in M5 (step 5) |

## 2. Design decisions

1. **A task mode with its own model, not a mix.** A switch *Word forms / Translation* starts a new model (like Reset). The forms model is not kept; switching back trains forms from scratch (or, after M4, loads the forms replay). Mixing both tasks into one model hurt both in the experiment.
2. **No copy practice, no dropout** in translation mode (user decision). The batches are only translations of trained animals, in both directions. This is the cleanest picture of memorising: it learns fast and nothing carries over.
3. **The translation dataset is its own `Dataset` view.**
   - **Data:** every pack (205 animals), not only the base pack: translation needs only names, and every animal has both.
   - **Forms:** only `name`. The vocabulary then has just `<cs>` and `<en>` as task tokens (49 tokens). Unused form tokens would sit untrained on the letter map.
   - **`unseen`** is the **translation hold-out**, not the forms flag. Badges, "never seen" tags, the Learning data tab's hidden answers and the Test filters then work without changes.
4. **The hold-out is computed, not stored in the data:** every 5th animal of each similarity kind, sorted by id (the experiment's rule). That gives 14 lookalike, 6 partial and 21 different animals (41 animals, 82 questions). It includes identical names (leopard, serval, okapi) for "even identical words fail", and the words worth asking about (kočka, treska, losos, delfín). The data format stays `id / names / forms / unseen`. If the data changes, the split changes with it; `npm run validate-data -- --review` lists it so it can be checked. *(Open question 1.)*
5. **Similarity kind is a label, not a scoreboard line.** The existing `classify()` (`lookalike / partial / different`, `src/data/dataset.ts`) balances the hold-out and tags names as "same word" (identical), "looks alike" or "different". Without copy practice every kind scores 0% on never-seen animals, so separate lines would only draw three flat zeros. One "never seen" line plus the tags tells it better: even the identical ones fail.
6. **Direction = target language.** `ask{text, lang, form: 'name'}` already means "write `text` in `lang`". No source-language token: the model tells Czech from English by the spelling (identical names give the same answer either way).
7. **Confidence for every tested word.** `confidence()` needs per-step probabilities, and today only `inspect` has them. `Translator.greedy` gets an optional output array with the log-probability of each chosen sequence (cheap, the probabilities are already computed), so `evaluate` can return a confidence for every word.

## 3. Steps (one PR each, DESIGN §12.1)

### Step 1 — core: data view, trainer, scores (no UI)
- `src/data/dataset.ts`
  - `translationBundle()` (or `Dataset.forTask('translate')`): all packs, forms `['name']`, `unseen` replaced by the computed hold-out (`translationHoldOut(animals)`, exported so the UI and the review table can name it).
  - `nameTasks()` stays as it is (both directions).
- `src/train/trainer.ts`
  - `TrainerOptions.task: 'forms' | 'translate'` (default `forms`).
  - `defaultMix()` for `translate`: only `nameTasks(trained)`, with dropout 0.
  - `evaluate()` for `translate`: `nameTasks()` of every animal, plus `TaskResult.from` (the source language), `TaskResult.similarity` (`classify(cs, en)`) and `TaskResult.confidence`.
  - `Scores` stay as they are: `known`, `unseen`, `close`. The form kinds are null in this mode.
  - Scores stay per *target* language: "into English" / "into Czech".
- `src/nn/translator.ts`: `greedy(…, logProbs?: Float32Array)`, which fills the summed log-probability and the length per sequence. Test: matches `confidence(inspect(…))`.
- `scripts/train-node.ts`: `--task translate`. Re-measure with the app's own trainer (seeds 1–2, 5 000 and 8 000 steps), record in DECISIONS, and choose the default step goal. The experiment suggests 5 000: about 85% known by 4 000 steps, little gain after.
- Tests:
  - the hold-out is deterministic and has every kind;
  - in translate mode no held-out name reaches a batch, and there are no copy tasks;
  - the scores;
  - the `greedy` log-probabilities.
- **Acceptance:** with seed 1, known ≥ 85% and never-seen ≈ 0 at 5 000 steps. This is a guide, not a gate.

### Step 2 — worker and protocol
- `protocol.ts`:
  - `init` / `reset` get `task`;
  - `ready` gets `task`, and `AnimalInfo` gets `similarity` (translate mode only);
  - `TaskResult` gets `from`, `similarity` and `confidence`;
  - `AnswerMsg` gets `confidence` (it is already computable from the trace) and `sameAs?: { animalId, word }` when the output is the real word of another animal.
- `engine.ts`:
  - builds the dataset for the task;
  - `badge()` in translate mode looks the word up in the *other* language. Today it looks for `names[lang] === word`, which is wrong for translation: `kočka` asked into English must find the cat by its Czech name;
  - `sameAs` checks the output against every animal's name in the target language;
  - `sendLetterMap` needs no change: the task tokens come from the vocabulary.
- `tests/train/engine.test.ts`: reset into translate mode, ask `kočka` → badge `hidden`, the right answer `cat`; ask a trained animal → `real`.

### Step 3 — UI
- **Header:** a *Word forms / Translation* switch next to Expert. If the model has trained, it asks first ("starts a new model"). The subtitle follows the mode.
- **Model learning tab:** the practice table shows `kočka → cat` rows. There is no copy-practice control in this mode.
- **Ask tab:**
  - **Form buttons** are replaced by "into English / into Czech".
  - **Example chips:** never-seen words (kočka, treska, losos, leopard, serval), plus a trained one (tygr).
  - **Confidence meter** shown large under the answer, with a line for never-seen answers: "It never saw this animal, but it is 88% sure."
  - **Notes:** the "word for another animal" note (`sameAs`), and the "Czech letters in an English word" note (simple check against the target language's letters seen in the data).
  - **Token strip:** `<en>` shows as "into English" (`tokens.ts` already names `<cs>` / `<en>` via `tok.form.name`; check the wording).
- **Test tab:**
  - **Scoreboard lines:** known and never seen (+ one letter off in the table).
  - **Table:** the language picker becomes the direction; rows show name → answer, the right answer (hidden for never-seen animals until revealed), the similarity tag and the confidence. The filters stay (all / known / never seen / wrong).
- **Learning data tab** (`dataTab.ts`):
  - **Overview:** one row per animal with cs and en names, similarity tag and never-seen mark; the questions are the two directions.
  - **Counts line:** "Learning: 328 translations from 164 animals; testing: 82 never-seen questions".
- **Letter map:** nothing to do (only `<cs>` / `<en>` exist in this vocabulary).
- **i18n:** every new string in `en.json` and `cs.json`. Czech wording goes to the user for review (HANDOFF, "Waiting for the user").
- **E2E:** smoke test switches to Translation, trains a few steps, asks `kočka` into English and sees the hidden badge and a confidence. `npm run screenshots` gets a `translation/` set.

### Step 4 — explanations ("what to notice")
Short texts in the places where each lesson shows, behind "?" where they are long:
- **Test tab intro:**
  - "Known animals go up, never-seen ones stay at zero: it memorised, it did not learn a rule.";
  - compare with Word forms: "There, never-seen words partly work, because forms have a rule (keep the stem, change the ending)."
- **"Same word" tag:** "leopard is leopard in both languages, and it still gets it wrong: it learned whole words, not how to read letters."
- **"Different" tag:** "kočka and cat have no letters in common: nothing in the letters says what the word means."
- **Ask tab:**
  - the confidence line, the "another animal's word" note and the Czech-letters note, each with one sentence on why it happens;
  - a link from the confidence line to how big language models "hallucinate".
- **Attention grid help (translate mode):** "In Word forms it looks at the stem letter by letter. Here it barely does: it recognises the word as a whole."

### Step 5 — Teach scenario "teach a new kind of task" (M5)
As in DESIGN §8.3: start from a trained forms model and add translation, with rehearsal on and off. The forms scores visibly drop without rehearsal. It needs M5's vocabulary-growth UI and the mixed trainer; the experiment's `mixed` setup is the reference. Note that the forms half there includes copy practice, as in the forms mode.

## 4. Open questions for the user
1. **Hold-out:** computed by the rule above (recommended, no data change), or chosen by hand (a list in code, or a new data field)?
2. **Where the switch lives:** in the header (recommended), or as a choice on the Model learning tab next to Reset?
3. **Packs:** use all 205 animals for translation (recommended: more identical and look-alike names in the hold-out), or the 151 base animals only, like forms?

## 5. Not planned
- **Copy practice in translation mode** (user decision). With it, look-alike words partly work (7–18% exact) and the attention reads left to right, but memorising slows down (~56% known at 8 000 steps) and the lesson gets less clear. The experiment's `copy` setup keeps the numbers.
- A source-language token (`<cs→en>`): not needed with two languages; revisit with 3+ languages and zero-shot directions (DESIGN §16).
- Better translation accuracy (bigger model, more data): not the point, and the experiment shows the limit is the data.
