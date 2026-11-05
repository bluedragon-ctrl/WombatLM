# Review: what you can and can't see in one learning step and one answer

The project's goal is to **show how a language model works**. This review checks the app
(the `main` branch after PR #21) against two questions:

1. *What exactly happens in one learning step?*
2. *What exactly happens while the model writes an answer?*

For each one: what the app already shows, then what is **missing**, ordered by how much it
would help someone understand the model. Each gap names the code that already has (or
nearly has) the data, so the work needed can be estimated.

---

## 1. One learning step

### What it shows already

| Part of the step | Where | 
|---|---|
| The batch: 16 random examples (word forms + "copy the name") with their right answers | Model learning → *What it's practising now* |
| What the model answered for each example before the step (greedy, weights only read) | same table, *What it answered* |
| The mistake per letter (probability of the right letter, −ln p), per example and for the whole batch, with an explanation | same table, letters tinted, *Mistake (loss)* column |
| The 4-stage cycle: pick → answer → measure → nudge | *One learning step* |
| Loss over time | *Mistakes (loss)* chart |
| Which parts of the model changed, how much per step | Inside the model → *Where is it learning?* |
| Every number: weights, change in the last steps, gradient | Inside the model → *All the numbers* |
| Teacher forcing for one example (shown so far / guess next, shifted by one) | Learning data → *One learning example, as tokens* |
| The letter table organising itself (vowels group together) | *Letter map* |

So the **forward pass + loss** part of the step is covered well. The step stops being
visible at the most important moment: **the cycle's last stage, "nudge the numbers",
is a black box.** The user sees *that* the numbers changed (heatmaps) but not *why* each
one changed or *what the change achieved*.

### What's missing

**L1. Before → after the nudge, on the same batch (most important).**
The step's whole purpose is "next time the right answers come out more likely", but the
app never measures that. After `opt.step()`, run the same batch forward again (no dropout,
no gradient) and show, per letter, *p before → p after* (e.g. `ř 12 % → 19 %`) and the
batch loss *before → after*. This is the single most convincing picture of learning: one
small step pushes every right letter up a little. A second, optional row could show one
**never-seen** example before → after, so the user sees that learning on *tygr* also
moves *lev* (generalisation), or sometimes makes another answer worse (interference).
*Data:* `FormTrainer.trainStep` already keeps the batch (`last.tasks`) and `tokenP`; one
extra `model.loss(batch, pad, tokenPAfter)` call with `training = false` gives the after
values. Cheap enough to do only when the user presses Step ×1 / when the table is visible.

**L2. Where the push comes from: the error signal of one letter.**
The gradient heatmap shows millions of tiny arrows with no story. The story is simple and
exact for the last layer: for one letter, the gradient on the guess scores is
`p − 1` for the right letter and `p` for every wrong one ("push the right letter up by how
much it was missed, push each wrong guess down by how much it was guessed"). Show this
for one letter of one example (bars like the guess bars, right letter in green going up,
wrong guesses in red going down), then say: *this push is passed backwards through the
model (backpropagation), and every number gets its share*. Link it to the existing
*Look inside one letter* view: the same path, walked backwards.
*Data:* the softmax probabilities are already computed in `rightTokenProbs`; the full
vector would need to be kept for the shown letter.

**L3. How one number is nudged (the optimiser is invisible).**
The app says "nudge all numbers a little" but the rule is not shown anywhere, and the
rule has three things that change what the user sees:
- the **learning rate**, which only appears in Expert mode, and the **warm-up** (the first
  100 steps use a smaller rate, `Adam.lrAt`) — this is why the first steps barely change
  anything;
- **gradient clipping** at norm 1.0 — `gradNorm` is computed every step and sent in every
  tick (`protocol.ts` `gradNorm`) but **never displayed**. Measured (seed 1, base data):
  **991 of the first 1 000 steps are clipped** (gradient norm 1.4–2.3), so clipping is not an
  edge case but part of almost every step;
- **Adam**: the change is *not* proportional to the gradient (it uses momentum and a
  per-number step size), which is why the *Gradient* and *Change* heatmaps look different.
  Without this, the two heatmaps contradict the "nudge against the gradient" explanation.

Suggested: click any square in the *Gradient* or *Change* heatmap → a small panel:
`value before 0.412 · gradient +0.031 (clipped ×0.6) · momentum … · step −0.0030 ·
value after 0.409`. Plus a one-line "learning rate 0.003 (warming up: 47 of 100)" and
"push size (gradient norm) 2.4 → clipped to 1" under the cycle.
*Data:* `Adam.step` has every value; it needs to record them for one requested index.

**L4. Training mode is not answering mode: dropout.**
While learning, 10 % of the numbers are switched off at random (`TRAINER_DEFAULTS.config.dropout`).
This is never mentioned, but it explains things the user can see and may find puzzling:
a word answered *right* can still have a large mistake, and the mistake of the same
example differs from step to step. One sentence in the cycle's "measure" stage and in the
mistake explanation would fix it; Expert mode could show which numbers were dropped for
one letter.

**L5. The difference between "what it answered" and "what it was graded on".**
The table shows the free answer (the model writes on its own) next to letters graded with
the right answer shown (teacher forcing). These are two different things and it is not
obvious: e.g. *turkey male → "bull"*, yet the letter `t` of *tom* may be graded fairly well.
Add, per letter, the model's own **top guess under teacher forcing** (shown "to", it
guessed `n` instead of `m`). This also prepares L1 and makes teacher forcing concrete.
*Data:* needs the argmax per position from the same logits (one more array next to `tokenP`).

**L6. What a step starts from: random numbers and the first loss.**
At step 0 the loss is about `ln(71) ≈ 4.3` (all 71 tokens equally likely) plus noise; the
chart starts at 5.5. One sentence ("at the start every letter is a guess among 71, so the
mistake is about 4.3") plus an "answer at step 0" example would ground the loss scale.

**L7. Follow one example over time.**
Pick one example (e.g. *tygr → tygřice*) and plot the probability of each of its right
letters over the steps. This shows memorisation happening letter by letter (the regular
`-ice` ending is learned long before the irregular `ř`), and pairs naturally with M4 time
travel (snapshots).

**L8. Smaller gaps.**
- The batch is a weighted random mix (forms : copy practice = 1 : 1, `copyWeight`); the
  user sees the result but not the rule.
- Padding: examples of different length are padded to a rectangle and the padding is not
  graded. Only matters for the "113 letters" count in the explanation.
- The loss chart has no reference line for the never-seen animals (test loss). Training
  loss alone can't show memorising vs generalising; the Test tab shows accuracy only, and
  only while it is open.

---

## 2. Writing an answer

### What it shows already

| Part of answering | Where |
|---|---|
| Tokens in (task token + letters) and out, with their numbers, `<unk>` warning | Ask → *Tokens* |
| The top-5 guesses for every letter, with % | Ask → *Guesses, letter by letter* |
| Cross-attention per head (which letter of the animal it looks at) | Ask → *Where it looks* (self-attention in Expert) |
| Letter by letter, with a "what the model is doing" box | Ask → *Answer slowly* |
| Every number on the way to one letter: embedding + position, self-attention, cross-attention, feed-forward, final scores, with exact sums on click | Inside → *Look inside one letter* |
| "Sure" (geometric mean of the chosen letters' probabilities) | Ask |
| Answering never changes the model; the answer is stamped with its step | Ask |

The decoder side of one letter is covered in great depth. The gaps are around the
**loop** that turns one letter into a word, the **choice** of the letter, and the
**encoder**.

### What's missing

**A1. How the next token is chosen: greedy vs sampling (most important for "how LLMs work").**
The model always takes the most likely token (`argmax` in `Translator.greedy`). This is
never said, and it is the opposite of how chat models are usually run (they *sample*,
which is why the same question gives different answers). Suggested:
- say it in the slow answer ("it picks the highest guess");
- an optional "🎲 pick by chance" with a **temperature** slider (`rowSoftmax` already takes
  a temperature; DESIGN plans a slider only for the M6 namer). Show the dice roll in the
  guess bars and let the user answer the same question several times to see different
  words, and a too-high temperature turning words into nonsense.
This is the best single addition for linking the toy to ChatGPT.

**A2. The loop: each written token goes back in.**
Autoregression is the heart of generation, but the app shows input row and output row
side by side; the *Answer slowly* box mentions looking back, but there is no picture of
the chosen token **becoming the next input** (in *Look inside one letter* the "goes in"
token is the previous output, but that link is not drawn). Suggested: in Answer slowly,
animate the chosen token moving from the guess column into the "answer so far" row, with
the decoder input row growing `✱ → ✱t → ✱ty → …`.

**A3. "What if it wrote another letter?" (forcing a token).**
Let the user click a different guess in a column (e.g. `s` instead of `ř`) and continue
the answer from there. This shows that every later letter depends on the earlier ones,
that one bad choice can derail the rest, and why the model is sometimes "sure" of a wrong
word. It is also the clearest way to show the training/answering difference (teacher
forcing vs its own letters, see L5): a button "show it the right letters instead" for
questions that have a real answer.
*Data:* `greedy` would need an optional forced prefix; `inspect` already re-runs with a
given target (`makeBatch` with `tgt`).

**A4. Reading the animal happens once; writing repeats.**
The encoder (letters of the animal looking at each other, then thinking) only appears as
heatmaps in Expert *Follow the signal*; encoder self-attention only in Expert. The user
never sees that reading is done **once**, and that each letter of the answer then reuses
those results (and the decoder's own earlier keys/values — the KV cache, which
`greedy` uses). A short "1. read (once) → 2. write, write, write, … end" strip in Answer
slowly, with a count of computations per step, would show it. Encoder self-attention could
be shown outside Expert for one input letter.

**A5. How attention decides where to look.**
The view shows the attention *weights* and what each position adds, but not the step
that produces the weights: *query · key* scores, divided by √(d/heads), then softmax.
Showing the raw scores for one row ("`t` offers 3.1, `y` 0.4, … → softmax → 95 %, 2 %, …")
would make "what to look for (Q) / what I offer (K)" — names already used in the heatmaps
— concrete. *Data:* the scores before softmax are in the trace path of `MultiHeadAttention`
(`transformer.ts`, before `softmax(scores, …)`) but not recorded.

**A6. The full guess, not just the top 5.**
The guess bars show 5 tokens; the probability left for the other 66 is not shown, and
neither are the scores (logits) before softmax outside the inside view. A "rest: 3 %" row
and, in Expert, raw scores would show that softmax always gives *every* token some chance —
which is what sampling (A1) uses.

**A7. Smaller gaps.**
- *Volume knobs* (layer norm) appear as names, but what they do (scale the 32 numbers
  to a standard size) is not shown in numbers.
- Stopping: the model stops only by writing `end`, or at the length limit of 22 positions
  (`maxLen`). The limit is not mentioned except as "did not finish".
- "Sure" is computed from the chosen letters only; with sampling (A1) it would also be
  worth showing the probability of the *whole word* (product, not geometric mean), since
  that is what "how likely is this answer" means.

---

## 3. Bridging to big language models

The app makes one comparison (number of parameters, number of tokens). A short, fixed
**"this model vs a chat model"** table (e.g. in the planned M7 About page) would answer
questions the demo raises:

| | WombatLM | A chat model |
|---|---|---|
| Tokens | 1 letter = 1 token, 71 tokens | pieces of words, ~100 000 tokens |
| Shape | encoder (reads) + decoder (writes) | usually decoder only: the question is the start of the text it continues |
| The task | a task token (`✱ Czech, female`) | the prompt, in words |
| Numbers | 25 255 | hundreds of billions |
| Learning data | ~1 000 examples | trillions of tokens of text |
| Choosing the next token | always the top guess | usually sampled (temperature) |
| Learning steps | thousands, in your browser | hundreds of thousands, on thousands of chips |

---

## 4. Suggested order

| # | Item | Why first | Effort |
|---|---|---|---|
| 1 | **L1** before → after the nudge | Completes the learning cycle; the core message | small (one extra forward pass) |
| 2 | **A1** greedy vs sampling + temperature | The biggest difference to real LLMs | small–medium |
| 3 | **L3** optimiser: lr, warm-up, clipping, one number's update; show `gradNorm` | Explains the "nudge" and why gradient ≠ change | medium |
| 4 | **A2 + A3** the loop; force a different letter | Makes autoregression visible and interactive | medium |
| 5 | **L2** error signal of one letter | Starts the backprop story | medium |
| 6 | **A5** attention scores, **A4** encoder once / KV cache | Deeper "how" for curious users | medium |
| 7 | **L4–L8, A6–A7** | Sentences and small additions | small each |
| 8 | **L7** one example over time | Fits M4 time travel | with M4 |

Items 1, 3 and 5 together turn the cycle's opaque last stage into three visible ones:
*measure → push back (gradient) → nudge (optimiser) → check (before/after)*.
