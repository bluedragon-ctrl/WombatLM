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
