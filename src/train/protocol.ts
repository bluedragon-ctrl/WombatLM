/**
 * Typed messages between the main thread (UI) and the training worker.
 *
 * The worker owns the model, the optimiser and the dataset. The main thread
 * never holds live weights; it receives data ready to draw.
 */
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
  | { type: 'inspect'; id: number; matrixIds: string[]; mode: InspectMode; letterMap?: boolean };

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
  split: 'train' | 'unseen';
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
  /** Hidden states along the way, for "follow the signal" (transferred buffers). */
  activations: Activation[];
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

export type FromWorker = ReadyMsg | TickMsg | EvalMsg | AnswerMsg | WeightsMsg | LetterMapMsg | ErrorMsg;
