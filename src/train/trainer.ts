/**
 * Training and evaluation of the word-form model. Pure (no DOM, no worker
 * APIs), so the worker, the tests and Node scripts all use it.
 *
 * The task: within one language, turn an animal's name into a word form,
 *   <cs:female> tygr -> tygřice     <en:young> cat -> kitten
 * mixed 1:1 with "copy the name" practice (<cs> tygr -> tygr), which is what
 * lets the model keep the stem of animals it has never seen.
 */
import { Tape } from '../nn/autograd';
import { Adam } from '../nn/optim';
import { Rng } from '../nn/rng';
import { Example, SpecialIds, TRANSLATOR_DEFAULTS, TranslationTrace, Translator, TranslatorConfig, makeBatch } from '../nn/translator';
import { Vocab } from '../nn/vocab';
import { Animal, Dataset, FormKind, Split, Task, editDistance, formKind } from '../data/dataset';

/**
 * correct / close (edit distance 1) / wrong for forms that exist in the data;
 * `invented` when the data has no such word (the model made it up).
 */
export type Status = 'correct' | 'close' | 'wrong' | 'invented';

export interface TaskResult {
  id: string;
  lang: string;
  form: string;
  source: string;
  /** The real word ('' for inventions). */
  answer: string;
  output: string;
  status: Status;
  split: Split;
  /** How the real word relates to the name (undefined for inventions). */
  kind?: FormKind;
}

/**
 * Scoreboard lines for one language (fractions 0..1, or null when there is
 * nothing to score).
 */
export interface Scores {
  /** Forms of animals the model trains on. */
  known: number | null;
  /** Never-seen forms, by kind. */
  pattern: number | null;
  partly: number | null;
  irregular: number | null;
  /** All never-seen forms: exactly right. */
  unseen: number | null;
  /** All never-seen forms: right or off by one letter. */
  close: number | null;
  /** Animals from packs added in Teach mode ("new animals"). */
  fresh: number | null;
  /** Animals from the original packs (for watching forgetting). */
  old: number | null;
}

export const SCORE_KEYS: readonly (keyof Scores)[] = ['known', 'pattern', 'partly', 'irregular', 'unseen', 'close', 'fresh', 'old'];

export interface EvalResult {
  step: number;
  /** Scores per language code. */
  scores: Record<string, Scores>;
  results: TaskResult[];
}

/** Chosen from the experiments in docs/DECISIONS.md. */
export const TRAINER_DEFAULTS = {
  lr: 3e-3,
  batchSize: 16,
  copyWeight: 1,
  config: { copy: true, targetStart: true, dropout: 0.1 } as Partial<Omit<TranslatorConfig, 'vocabSize'>>,
};

export interface TrainerOptions {
  seed: number;
  lr?: number;
  batchSize?: number;
  config?: Partial<Omit<TranslatorConfig, 'vocabSize'>>;
  /** Share of "copy the name" practice (<cs> tygr -> tygr) relative to word forms (1). */
  copyWeight?: number;
  weightDecay?: number;
}

/** A source of training tasks and its share of each batch. */
export interface MixPart {
  tasks: Task[];
  weight: number;
}

export class FormTrainer {
  model: Translator;
  opt: Adam;
  rng: Rng;
  step = 0;
  lossEma: number | null = null;
  lastLoss = 0;
  lastLr = 0;
  lastGradNorm = 0;
  batchSize: number;
  copyWeight: number;
  /** Packs whose animals count as "new" on the scoreboard (Teach mode). */
  freshPacks = new Set<string>();
  private mix: MixPart[] = [];

  constructor(public dataset: Dataset, opts: TrainerOptions) {
    this.rng = new Rng(opts.seed ^ 0x5eed);
    const config = { ...TRANSLATOR_DEFAULTS, ...TRAINER_DEFAULTS.config, ...opts.config, vocabSize: dataset.vocab.size };
    this.model = new Translator(config, opts.seed);
    this.opt = new Adam(this.model.params.list, { lr: opts.lr ?? TRAINER_DEFAULTS.lr, weightDecay: opts.weightDecay ?? 0 });
    this.batchSize = opts.batchSize ?? TRAINER_DEFAULTS.batchSize;
    this.copyWeight = opts.copyWeight ?? TRAINER_DEFAULTS.copyWeight;
    this.setMix(this.defaultMix());
  }

  get vocab(): Vocab {
    return this.dataset.vocab;
  }

  get ids(): SpecialIds {
    return { pad: this.vocab.pad, bos: this.vocab.bos, eos: this.vocab.eos };
  }

  /** Word-form tasks of train-split animals. */
  trainTasks(filter?: (a: Animal) => boolean): Task[] {
    return this.dataset.formTasks((a) => a.split === 'train' && (!filter || filter(a)));
  }

  /** "Copy the name" tasks of train-split animals. */
  copyTasks(filter?: (a: Animal) => boolean): Task[] {
    return this.dataset.nameTasks((a) => a.split === 'train' && (!filter || filter(a)), true);
  }

  /** Word forms plus copy practice, for the given animals. */
  defaultMix(filter?: (a: Animal) => boolean, weight = 1): MixPart[] {
    return [
      { tasks: this.trainTasks(filter), weight },
      { tasks: this.copyTasks(filter), weight: weight * this.copyWeight },
    ];
  }

  /** Choose what batches are made of, e.g. 50% old animals + 50% new ones. */
  setMix(parts: MixPart[]): void {
    this.mix = parts.filter((p) => p.tasks.length > 0 && p.weight > 0);
  }

  toExample(task: Task): Example {
    return {
      target: this.dataset.targetId(task.to, task.form),
      src: this.vocab.encodeWord(task.source),
      tgt: this.vocab.encodeWord(task.answer),
    };
  }

  private sampleTask(): Task {
    const total = this.mix.reduce((s, p) => s + p.weight, 0);
    let r = this.rng.next() * total;
    for (const part of this.mix) {
      r -= part.weight;
      if (r < 0) return part.tasks[this.rng.int(part.tasks.length)];
    }
    const last = this.mix[this.mix.length - 1];
    return last.tasks[this.rng.int(last.tasks.length)];
  }

  /** One optimisation step on a random batch. Returns the batch loss. */
  trainStep(): number {
    if (this.mix.length === 0) throw new Error('Nothing to train on');
    const examples = Array.from({ length: this.batchSize }, () => this.toExample(this.sampleTask()));
    const batch = makeBatch(examples, this.ids);
    this.opt.zeroGrad();
    const tape = new Tape();
    const loss = tape.run(() => this.model.loss(batch, this.vocab.pad));
    tape.backward(loss);
    const info = this.opt.step();
    this.step++;
    this.lastLoss = loss.item();
    this.lastLr = info.lr;
    this.lastGradNorm = info.gradNorm;
    this.lossEma = this.lossEma === null ? this.lastLoss : 0.95 * this.lossEma + 0.05 * this.lastLoss;
    return this.lastLoss;
  }

  setLearningRate(lr: number): void {
    this.opt.opts.lr = lr;
  }

  /** Greedy answers for many tasks at once. */
  runTasks(tasks: readonly Task[], model: Translator = this.model): string[] {
    const out: string[] = [];
    const chunk = 128;
    for (let i = 0; i < tasks.length; i += chunk) {
      const part = tasks.slice(i, i + chunk);
      const ids = model.greedy(part.map((t) => ({ target: this.dataset.targetId(t.to, t.form), src: this.vocab.encodeWord(t.source) })), this.ids);
      for (const seq of ids) out.push(this.vocab.decode(seq));
    }
    return out;
  }

  /**
   * Ask for every word form of every animal and score the answers. With
   * `inventions`, also ask for the forms the data does not have.
   */
  evaluate(opts: { inventions?: boolean; model?: Translator } = {}): EvalResult {
    const tasks = this.dataset.formTasks();
    if (opts.inventions) tasks.push(...this.dataset.missingFormTasks());
    const outputs = this.runTasks(tasks, opts.model);
    const results: TaskResult[] = tasks.map((t, i) => ({
      id: t.animal.id,
      lang: t.to,
      form: t.form,
      source: t.source,
      answer: t.answer,
      output: outputs[i],
      status: t.answer ? judge(outputs[i], t.accepted) : 'invented',
      split: t.animal.split,
      kind: t.answer ? formKind(t.source, t.answer) : undefined,
    }));
    const packs = new Map(this.dataset.animals.map((a) => [a.id, a.pack]));
    return { step: this.step, scores: scoreResults(results, (id) => this.freshPacks.has(packs.get(id) ?? '')), results };
  }

  /** Ask for one word form of free text, with full visualisation data. */
  inspect(text: string, lang: string, form: string, model: Translator = this.model): TranslationTrace {
    const src = this.vocab.encodeWord(text).slice(0, model.config.maxLen - 2);
    return model.inspect(this.dataset.targetId(lang, form), src, this.ids);
  }
}

export function judge(output: string, accepted: readonly string[]): Status {
  if (accepted.includes(output)) return 'correct';
  return Math.min(...accepted.map((a) => editDistance(output, a))) <= 1 ? 'close' : 'wrong';
}

/**
 * How sure the model was of the whole word: the geometric mean of the
 * probabilities of the letters it chose (and of the final <eos>), so long
 * words are not penalised. 0..1.
 */
export function confidence(trace: TranslationTrace, eos: number): number {
  const chosen = [...trace.output, ...(trace.finished ? [eos] : [])];
  if (chosen.length === 0) return 0;
  let logSum = 0;
  chosen.forEach((id, t) => { logSum += Math.log(Math.max(1e-9, trace.probs[t][id])); });
  return Math.exp(logSum / chosen.length);
}

/** Group results into scoreboard lines per language. */
export function scoreResults(results: readonly TaskResult[], isFresh: (id: string) => boolean = () => false): Record<string, Scores> {
  const acc: Record<string, Record<keyof Scores, [number, number]>> = {};
  for (const r of results) {
    if (r.status === 'invented') continue;
    const s = (acc[r.lang] ??= Object.fromEntries(SCORE_KEYS.map((k) => [k, [0, 0]])) as Record<keyof Scores, [number, number]>);
    const ok = r.status === 'correct' ? 1 : 0;
    const add = (k: keyof Scores, hit = ok) => { s[k][0] += hit; s[k][1] += 1; };
    if (r.split === 'train') add('known');
    else {
      if (r.kind) add(r.kind);
      add('unseen');
      add('close', r.status === 'wrong' ? 0 : 1);
    }
    if (isFresh(r.id)) add('fresh');
    else if (r.split === 'train') add('old');
  }
  const out: Record<string, Scores> = {};
  for (const [lang, s] of Object.entries(acc)) {
    out[lang] = Object.fromEntries(Object.entries(s).map(([k, [c, n]]) => [k, n > 0 ? c / n : null])) as unknown as Scores;
  }
  return out;
}
