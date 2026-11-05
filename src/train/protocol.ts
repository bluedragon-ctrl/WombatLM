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
   * The Practice tab is open (on) or closed (off). While on, the practice
   * batch in ticks also carries the brain's own answers to the shown examples
   * (sent at most every 500 ms while training fast).
   */
  | { type: 'watchPractice'; on: boolean }
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
  /** Raw batch losses of every step since the previous tick. */
  losses: LossPoint[];
  /**
   * What the latest training step practised on (up to 32 examples; with `watchPractice` also the brain's own answers) with the
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
  /** Copy mechanism: how much the model wrote this letter itself (1) vs copied it (0). */
  write?: number;
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
  /** For real / hidden: the real word(s) and how the output scored. */
  answer?: { accepted: string[]; status: 'correct' | 'close' | 'wrong' };
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
}

export type FromWorker = ReadyMsg | TickMsg | EvalMsg | AnswerMsg | WeightsMsg | LetterMapMsg | ErrorMsg;
