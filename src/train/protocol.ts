/**
 * Typed messages between the main thread (UI) and the training worker.
 *
 * The worker owns the model, the optimiser and the dataset. The main thread
 * never holds live weights; it receives data ready to draw.
 */
import type { ExplainStage } from '../nn/explain';
import type { AxisKind } from '../nn/transformer';
import type { PracticeBatch, Scores, TaskResult } from './trainer';

/** Training speed: steps per second, or 'max' (as fast as possible). */
export type Speed = number | 'max';

// ---------------------------------------------------------------------------
// Main -> worker
// ---------------------------------------------------------------------------

export type ToWorker =
  | { type: 'init'; seed: number; speed?: Speed }
  | { type: 'play' }
  | { type: 'pause' }
  /** Train exactly `n` steps now (also while paused), then report. */
  | { type: 'step'; n: number }
  | { type: 'setSpeed'; speed: Speed }
  | { type: 'setHyper'; lr?: number; batchSize?: number }
  /** New untrained model with this seed. Pauses training. */
  | { type: 'reset'; seed: number }
  /**
   * Ask for a word form of `text` in language `lang` (form 'name' = copy the
   * name). `id` is echoed back so the UI can drop stale answers.
   */
  | { type: 'ask'; id: number; text: string; lang: string; form: string }
  /** Evaluate now; `inventions` also asks for the forms the data lacks. */
  | { type: 'evaluate'; inventions?: boolean }
  /**
   * The Test tab is open (on) or closed (off). While on, every animal is
   * tested on a schedule (and right away if this model was not tested yet);
   * while off, only on `evaluate`.
   */
  | { type: 'watchTests'; on: boolean }
  /**
   * The Model learning tab is open (on) or closed (off). While on, the model also
   * answers the shown examples before learning from them at high speeds (at
   * most every 500 ms; at slow speeds and on Step it always does).
   */
  | { type: 'watchPractice'; on: boolean }
  /**
   * Quiet learning: while training runs, skip everything that is only for
   * the pictures (testing every animal, answering the batch first). Ticks
   * still come, so the speed can be compared with and without the pictures.
   */
  | { type: 'quiet'; on: boolean }
  /**
   * Learn until this step, then stop by itself (null = no limit). Play does
   * nothing once the goal is reached; Step still works.
   */
  | { type: 'setGoal'; steps: number | null }
  /**
   * Look inside: the listed weight matrices (answered with `weights`, empty
   * list = none) and, if `letterMap`, the letter map (answered with `letterMap`).
   * `id` is echoed back.
   */
  | { type: 'inspect'; id: number; matrixIds: string[]; mode: InspectMode; letterMap?: boolean }
  /**
   * "Look inside one letter": every number on the way to output letter
   * `position` (a guess-bar column) of the answer to this question. Answered
   * with `explain`; `id` is echoed back.
   */
  | { type: 'explain'; id: number; text: string; lang: string; form: string; position: number }
  /**
   * Why did number `index` of parameter `name` change in the latest step?
   * Only for a step recorded in detail (Step ×1 while paused). Answered with
   * `why`; `id` is echoed back.
   */
  | { type: 'why'; id: number; name: string; index: number }
  /**
   * What have the writer's thinking units learned? Reads every learning
   * example (see train/units.ts). Answered with `units`; `id` is echoed back.
   */
  | { type: 'units'; id: number };

/**
 * What a weight heatmap shows:
 * - weights: the numbers themselves;
 * - delta: how much they changed since the previous `inspect` that came
 *   after at least one training step (the "change in the last steps");
 * - grad: the gradient of the latest training batch.
 */
export type InspectMode = 'weights' | 'delta' | 'grad';

// ---------------------------------------------------------------------------
// Worker -> main
// ---------------------------------------------------------------------------

/** What the UI needs to know about one animal (no training data beyond this). */
export interface AnimalInfo {
  id: string;
  /** Never trained on: kept aside to test the patterns. */
  unseen: boolean;
  pack: string;
  names: Record<string, string>;
  /**
   * Known forms per language. Never-seen forms are included: the UI hides
   * them behind a "reveal" button.
   */
  forms: Record<string, Record<string, string>>;
}

export interface ReadyMsg {
  type: 'ready';
  seed: number;
  params: number;
  /** Data languages and forms (form 'name' first). */
  languages: { code: string; name: Record<string, string> }[];
  forms: string[];
  animals: AnimalInfo[];
  tokens: string[];
  /** Every trainable matrix/vector of the model, in model order. */
  matrices: MatrixInfo[];
  /** The model's sizes and learning settings (the expert "numbers" tab lists them). */
  config: ModelConfig;
}

/** Sizes of the model and the fixed settings of its learning rule. */
export interface ModelConfig {
  vocabSize: number;
  dModel: number;
  heads: number;
  ffn: number;
  encLayers: number;
  decLayers: number;
  maxLen: number;
  dropout: number;
  /** Adam: memory of the push (beta1) and of its size (beta2), the tiny number that avoids dividing by 0. */
  beta1: number;
  beta2: number;
  eps: number;
  /** Gradient clipping: all gradients together are shrunk to at most this length. */
  clipNorm: number;
  /** The learning rate grows from 0 to its full value over this many steps. */
  warmupSteps: number;
  weightDecay: number;
  /** Share of "same name" practice relative to word forms. */
  copyWeight: number;
}

/** Describes one weight matrix (vectors have rows = 1). */
export interface MatrixInfo {
  /** Technical name, e.g. "dec0.cross.q.W". */
  name: string;
  rows: number;
  cols: number;
  /** What rows / columns stand for (letters, positions, dimensions, …). */
  rowKind: AxisKind;
  colKind: AxisKind;
  /** Kind of weight ("Wq", "embed", "ln.gamma", …), for a plain-language label. */
  role: string;
}

export interface LossPoint {
  step: number;
  loss: number;
}

/** Sent at most ~20 times a second while training, and after every change. */
export interface TickMsg {
  type: 'tick';
  step: number;
  running: boolean;
  speed: Speed;
  lossEma: number | null;
  lr: number;
  batchSize: number;
  gradNorm: number;
  /** Measured training speed (steps per second of training time, excluding evaluation). */
  stepsPerSec: number;
  /**
   * Real speed while running: steps per second of wall-clock time over the
   * last second or so, including testing, answering and looking inside
   * (0 until measured).
   */
  wallStepsPerSec: number;
  /** Quiet learning is on (see `quiet`). */
  quiet: boolean;
  /** Learning stops by itself at this step (null = no limit). */
  goal: number | null;
  /** Raw batch losses of every step since the previous tick. */
  losses: LossPoint[];
  /**
   * What the latest training step practised on (up to 32 examples of learning data and, when it answered them before the step, its answers) with the
   * model's guesses). Only present when there was a new step since the
   * previous tick.
   */
  practice?: PracticeBatch;
}

export interface EvalMsg {
  type: 'eval';
  step: number;
  scores: Record<string, Scores>;
  results: TaskResult[];
  /** True if `results` include the inventions (forms the data lacks). */
  inventions: boolean;
  /** How long the evaluation took, in ms. */
  ms: number;
}

/**
 * What the dataset says about the asked word:
 * - real: the data has the answer and the model trained on it;
 * - hidden: the data has it, but the animal is never shown in training;
 * - invention: the animal is in the data, but this form has no real word;
 * - free: the word is not an animal in the data.
 */
export type Badge = 'real' | 'hidden' | 'invention' | 'free';

export interface StepGuess {
  /** The chosen character ('' for the end of the word). */
  chosen: string;
  /** Top 5 candidates, most likely first ('' = end of word). */
  top: { char: string; p: number }[];
}

export interface AttentionMap {
  name: string;
  heads: number;
  Tq: number;
  Tk: number;
  data: Float32Array;
}

export interface Activation {
  /** "enc.in", "enc0.out", "dec.in", "dec0.out", … in the order the signal flows. */
  name: string;
  /** Number of positions (tokens). */
  T: number;
  d: number;
  data: Float32Array;
}

export interface AnswerMsg {
  type: 'answer';
  id: number;
  step: number;
  text: string;
  lang: string;
  form: string;
  output: string;
  /** False if the model never finished the word (hit the length limit). */
  finished: boolean;
  /** Encoder input tokens (target token + letters), for attention axes. */
  input: string[];
  steps: StepGuess[];
  /** Geometric mean of the chosen letters' probabilities, 0..1. */
  confidence: number;
  attention: AttentionMap[];
  /**
   * Hidden states along the way, for "follow the signal" and "follow the
   * tokens" (transferred buffers): enc.in, enc0.out, memory, dec.in, dec0.out, dec.final.
   */
  activations: Activation[];
  /** Token ids of the input (task token + letters). */
  inputIds: number[];
  /**
   * The rows looked up for every token, [T × d] each: its row of the letter
   * table and the row of its position, for the input (`encTok`, `encPos`)
   * and for every token going into the writer (`decTok`, `decPos`: the task
   * token, then the answer so far). Their sums are `enc.in` and `dec.in`.
   */
  lookup: { encTok: Float32Array; encPos: Float32Array; decTok: Float32Array; decPos: Float32Array };
  /**
   * What each part of a block adds, [T × d] each (matrix view):
   * enc0.self.add, enc0.ffn.add, dec0.self.add, dec0.cross.add, dec0.ffn.add.
   */
  adds: Activation[];
  /** The whole letter table, [vocabSize × d] (matrix view). */
  letterTable: Float32Array;
  /** Every pass's score for every token, [steps × vocabSize], before they become chances. */
  logits: Float32Array;
  /**
   * The learned numbers each station uses (matrix view): the attention grids
   * (`enc0.self.q.W`, … `.o.W`, `dec0.cross.*`), the thinking grids
   * (`*.ffn.up.W`, `*.ffn.down.W`) and the final volume knobs
   * (`encNorm.gamma`/`beta`, `decNorm.*`). Vectors have rows = 1.
   */
  weights: { name: string; rows: number; cols: number; data: Float32Array }[];
  /** How many tokens the model can choose from (every guess is a score for each of them). */
  vocabSize: number;
  badge: Badge;
  /** For real / hidden: the real word and how the output scored. */
  answer?: { word: string; status: 'correct' | 'close' | 'wrong' };
  animalId?: string;
}

export interface ErrorMsg {
  type: 'error';
  message: string;
}

export interface WeightsMsg {
  type: 'weights';
  id: number;
  step: number;
  mode: InspectMode;
  /**
   * For `delta`: how many training steps the change covers (0 = no change
   * yet). For `grad`: 0 if the model has not trained yet (all zeros).
   */
  steps: number;
  /** Requested matrices, same shapes as in `ready.matrices` (transferred buffers). */
  matrices: { name: string; data: Float32Array }[];
  /** How much every matrix of the model changed lately (all of them, not only the requested ones). */
  summary: MatrixChange[];
}

/**
 * Size of a matrix and of its latest change: `change` / `norm` / `steps` is
 * "the share of it that changes per step", comparable between matrices.
 */
export interface MatrixChange {
  name: string;
  /** Size of the numbers: √(Σ w²). */
  norm: number;
  /** Size of their change over the last `steps` training steps: √(Σ Δw²). */
  change: number;
  /** Steps the change covers (0 = no change yet). */
  steps: number;
}

/** One step of "look inside one letter" (see nn/explain.ts). */
export type { ExplainStage };

/** One of the top guesses at the end of "look inside one letter". */
export interface ExplainGuess {
  /** The letter ('' = end of word) and its token id (= its row in the letter table). */
  char: string;
  token: number;
  p: number;
  /** Its score before softmax: Σ contrib + bias. */
  score: number;
  /** [d] its row of the letter table. */
  row: Float32Array;
  /** [d] final[i] · row[i]: what each number adds to the score. */
  contrib: Float32Array;
  /** "Favourite letters": outBias[token]. */
  bias: number;
}

/** "What would it write now?" after one stage of "look inside one letter" (logit lens). */
export interface ExplainLens {
  /** Its 3 most likely letters at this point ('' = end of word). */
  top: { char: string; p: number }[];
  /** The chance of the letter it finally writes, at this point. */
  chosen: number;
}

export interface ExplainMsg {
  type: 'explain';
  id: number;
  step: number;
  text: string;
  lang: string;
  form: string;
  /** The whole answer. */
  output: string;
  /** Number of guess columns (letters + end of word if it finished). */
  columns: number;
  /** Which column is explained. */
  position: number;
  /** The letter written there ('' = end of word). */
  chosen: string;
  /** Encoder input tokens (task token + letters) and their ids. */
  input: string[];
  inputIds: number[];
  /** Decoder input tokens 0..position (task token, then the answer so far) and their ids. */
  decIn: string[];
  decInIds: number[];
  /** The letter going in: its letter-table row, its position row and their sum. */
  start: { token: Float32Array; pos: Float32Array; x: Float32Array };
  stages: ExplainStage[];
  /** After the last volume knobs: the numbers that make the guess. */
  final: Float32Array;
  /** Top 5 guesses, most likely first. */
  guesses: ExplainGuess[];
  /** "What would it write now?" after the start and after each stage (1 + stages.length). */
  lens: ExplainLens[];
  /**
   * For the last thinking stage: the chance of the written letter if thinking
   * unit k were switched off for this letter (exact: it only removes the
   * unit's part of what thinking adds). Null if thinking is not the last stage.
   */
  unitsOff: Float32Array | null;
}

/** One of a thinking unit's strongest moments in the learning data. */
export interface UnitMoment {
  /** The question (animal name, language, form) and its right answer. */
  source: string;
  answer: string;
  lang: string;
  form: string;
  /** Letters of the answer already written; the unit is measured while writing letter `at` (= answer.length: the end). */
  at: number;
  /** The letter written next ('' = end of word). */
  next: string;
  value: number;
}

/**
 * What a unit's strongest moments share, far more often than by chance:
 * the next letter, the letter just written ('<start>' = nothing yet), the
 * language and form ("cs:female"), the form, or the language.
 */
export interface UnitTheme {
  kind: 'next' | 'prev' | 'task' | 'form' | 'lang';
  value: string;
  count: number;
  of: number;
}

export interface UnitInfo {
  unit: number;
  /** Share of all moments (letters written) where it is on. */
  on: number;
  /** Its strongest moments, strongest first. */
  top: UnitMoment[];
  theme: UnitTheme | null;
  /** The 3 letters its pattern pushes towards most when it is on (direct effect, a hint). */
  pushes: { char: string; score: number }[];
  /**
   * Measured at its strongest moments by switching it off (null if it is not
   * in the last writing layer, where this is exact): the right letter's
   * average chance with and without it, and the letters whose chance falls
   * the most without it (average fall, 0..1).
   */
  off: { withP: number; withoutP: number; helps: { char: string; drop: number }[] } | null;
}

export interface UnitsMsg {
  type: 'units';
  id: number;
  step: number;
  /** Which feed-forward step, e.g. "dec0". */
  layer: string;
  /** How many moments (letters of learning examples) were read. */
  moments: number;
  units: UnitInfo[];
}

export interface LetterMapMsg {
  type: 'letterMap';
  id: number;
  step: number;
  /** Token ids on the map (characters, end of word and the task tokens). */
  ids: number[];
  /** [ids.length, 2] PCA coordinates of their rows in the letter table. */
  coords: Float32Array;
  /** Share of all variation each axis shows. */
  explained: [number, number];
  /**
   * For each id, the positions (in `ids`) of its closest letters in the full
   * letter table, closest first. Empty for task tokens.
   */
  near: number[][];
  /** For each id, how often it appears in the learning data (-1 for the end of word and task tokens). */
  counts: number[];
}

/** One number of the model, and how the latest step changed it. */
export interface NumberRef {
  /** Parameter name and the number's place in it (row-major), with its row and column. */
  name: string;
  index: number;
  row: number;
  col: number;
  before: number;
  after: number;
  /** Its push (gradient) in the step. */
  grad: number;
}

/**
 * Sent after a step recorded in detail (Step ×1 while paused): what the
 * step changed in the model. The batch before / after is in the tick's
 * `practice.check`.
 */
export interface StepDetailMsg {
  type: 'stepDetail';
  step: number;
  /** Learning rate used, the one set, Adam's update count and the warm-up length. */
  lr: number;
  baseLr: number;
  t: number;
  warmupSteps: number;
  /** Length of all pushes together, the limit, and the factor every push was multiplied by (1 = not limited). */
  gradNorm: number;
  clipNorm: number;
  clip: number;
  /** How many numbers the model has, how many moved, the biggest and the median move. */
  numbers: { total: number; moved: number; max: number; median: number };
  /** The numbers that moved the most, and those pushed the hardest. */
  top: { change: NumberRef[]; push: NumberRef[] };
  /** Each example's share of a push is available (the batch was not too big). */
  split: boolean;
}

/** One example of the batch and its share of a number's push. */
export interface PushShare {
  source: string;
  lang: string;
  form: string;
  target: string;
  share: number;
}

/** The whole calculation of one number's change in the latest step (see stepDetail.ts). */
export interface WhyMsg {
  type: 'why';
  id: number;
  /** Training step it explains; `missing` if that step was not recorded in detail (or is no longer the latest). */
  step: number;
  missing?: boolean;
  ref?: NumberRef;
  /** Its value now (the real result of the step). */
  now?: number;
  calc?: {
    grad: number; clip: number; gradUsed: number;
    mPrev: number; vPrev: number; m: number; v: number;
    beta1: number; beta2: number; eps: number; t: number;
    bc1: number; bc2: number; mHat: number; vHat: number;
    lr: number; baseLr: number; warmupSteps: number; decay: number; move: number; after: number;
  };
  /** Each example's share of the push, in batch order (absent if not split). */
  shares?: PushShare[];
}

export type FromWorker = ReadyMsg | TickMsg | EvalMsg | AnswerMsg | WeightsMsg | ExplainMsg | LetterMapMsg | StepDetailMsg | WhyMsg | UnitsMsg | ErrorMsg;
