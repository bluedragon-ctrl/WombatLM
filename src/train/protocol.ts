/**
 * Typed messages between the main thread (UI) and the training worker.
 *
 * The worker owns the model, the optimiser and the dataset. The main thread
 * never holds live weights; it receives data ready to draw.
 */
import type { Scores, TaskResult } from './trainer';

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
  | { type: 'evaluate'; inventions?: boolean };

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
  badge: Badge;
  /** For real / hidden: the real word(s) and how the output scored. */
  answer?: { accepted: string[]; status: 'correct' | 'close' | 'wrong' };
  animalId?: string;
}

export interface ErrorMsg {
  type: 'error';
  message: string;
}

export type FromWorker = ReadyMsg | TickMsg | EvalMsg | AnswerMsg | ErrorMsg;
