/**
 * Training and evaluation of the translator. Pure (no DOM, no worker APIs),
 * so the worker, the tests and the bake script all use it.
 */
import { Tape } from '../nn/autograd';
import { Adam } from '../nn/optim';
import { Rng } from '../nn/rng';
import { Example, SpecialIds, TRANSLATOR_DEFAULTS, TranslationTrace, Translator, TranslatorConfig, makeBatch } from '../nn/translator';
import { Vocab } from '../nn/vocab';
import { Animal, Dataset, Kind, Task, animalKind, editDistance, pairKey } from '../data/dataset';

export type Status = 'correct' | 'close' | 'wrong';

export interface TaskResult {
  id: string;
  from: string;
  to: string;
  form: string;
  output: string;
  status: Status;
}

/** Score lines of the scoreboard (fractions 0..1, or null when there is nothing to score). */
export interface Scores {
  known: number | null;
  lookalike: number | null;
  partial: number | null;
  different: number | null;
  /** Animals from packs added in Teach mode ("new animals"). */
  fresh: number | null;
  /** Animals from the original packs (for watching forgetting). */
  old: number | null;
}

export interface EvalResult {
  step: number;
  /** Scores per language pair key ("cs-en"), averaged over both directions. */
  scores: Record<string, Scores>;
  results: TaskResult[];
}

export interface TrainerOptions {
  seed: number;
  lr: number;
  batchSize: number;
  config?: Partial<Omit<TranslatorConfig, 'vocabSize'>>;
  /** Share of "keep it as it is" practice (<cs> tygr -> tygr) relative to translation (1). */
  identityWeight?: number;
  weightDecay?: number;
  /** Steps over which identity practice fades from 3× its weight down to its weight. */
  identityWarmup?: number;
}

/** A source of training tasks and its share of each batch. */
export interface MixPart {
  tasks: Task[];
  weight: number;
}

export class TranslatorTrainer {
  model: Translator;
  opt: Adam;
  rng: Rng;
  step = 0;
  lossEma: number | null = null;
  lastLoss = 0;
  lastLr = 0;
  lastGradNorm = 0;
  batchSize: number;
  /** Packs whose animals count as "new" on the scoreboard (Teach mode). */
  freshPacks = new Set<string>();
  private mix: MixPart[] = [];

  constructor(public dataset: Dataset, opts: TrainerOptions) {
    this.rng = new Rng(opts.seed ^ 0x5eed);
    this.model = new Translator({ ...TRANSLATOR_DEFAULTS, ...opts.config, vocabSize: dataset.vocab.size }, opts.seed);
    this.opt = new Adam(this.model.params.list, { lr: opts.lr, weightDecay: opts.weightDecay ?? 0 });
    this.batchSize = opts.batchSize;
    this.identityWeight = opts.identityWeight ?? 0;
    this.setMix(this.defaultMix());
  }

  get vocab(): Vocab {
    return this.dataset.vocab;
  }

  get ids(): SpecialIds {
    return { pad: this.vocab.pad, bos: this.vocab.bos, eos: this.vocab.eos };
  }

  identityWeight: number;

  /** All name-translation tasks of train-split animals. */
  trainTasks(filter?: (a: Animal) => boolean): Task[] {
    return this.dataset.nameTasks((a) => a.split === 'train' && (!filter || filter(a)));
  }

  /** "Keep it as it is" tasks of train-split animals. */
  identityTasks(filter?: (a: Animal) => boolean): Task[] {
    return this.dataset.nameTasks((a) => a.split === 'train' && (!filter || filter(a)), true);
  }

  /** Translation practice plus (optionally) identity practice, for the given animals. */
  defaultMix(filter?: (a: Animal) => boolean, weight = 1): MixPart[] {
    return [
      { tasks: this.trainTasks(filter), weight },
      { tasks: this.identityTasks(filter), weight: weight * this.identityWeight },
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

  /** Greedy translations for many tasks at once. */
  translateTasks(tasks: readonly Task[], model: Translator = this.model): string[] {
    const out: string[] = [];
    const chunk = 128;
    for (let i = 0; i < tasks.length; i += chunk) {
      const part = tasks.slice(i, i + chunk);
      const ids = model.greedy(part.map((t) => ({ target: this.dataset.targetId(t.to, t.form), src: this.vocab.encodeWord(t.source) })), this.ids);
      for (const seq of ids) out.push(this.vocab.decode(seq));
    }
    return out;
  }

  /** Translate every animal in every direction and score the results. */
  evaluate(model: Translator = this.model): EvalResult {
    const tasks = this.dataset.nameTasks();
    const outputs = this.translateTasks(tasks, model);
    const results: TaskResult[] = tasks.map((t, i) => ({
      id: t.animal.id, from: t.from, to: t.to, form: t.form, output: outputs[i], status: judge(outputs[i], t.accepted),
    }));
    return { step: this.step, scores: scoreResults(tasks, results, this.freshPacks), results };
  }

  /** Translate free text into language `to` (form `form`) with full visualisation data. */
  inspect(text: string, to: string, form = 'name', model: Translator = this.model): TranslationTrace {
    const src = this.vocab.encodeWord(text).slice(0, model.config.maxLen - 2);
    return model.inspect(this.dataset.targetId(to, form), src, this.ids);
  }
}

export function judge(output: string, accepted: readonly string[]): Status {
  if (accepted.includes(output)) return 'correct';
  return Math.min(...accepted.map((a) => editDistance(output, a))) <= 1 ? 'close' : 'wrong';
}

/** Group results into scoreboard lines per language pair. */
export function scoreResults(tasks: readonly Task[], results: readonly TaskResult[], freshPacks: ReadonlySet<string>): Record<string, Scores> {
  const acc: Record<string, Record<keyof Scores, [number, number]>> = {};
  tasks.forEach((t, i) => {
    const key = pairKey(t.from, t.to);
    const s = (acc[key] ??= { known: [0, 0], lookalike: [0, 0], partial: [0, 0], different: [0, 0], fresh: [0, 0], old: [0, 0] });
    const ok = results[i].status === 'correct' ? 1 : 0;
    const add = (k: keyof Scores) => { s[k][0] += ok; s[k][1] += 1; };
    if (t.animal.split === 'train') add('known');
    else {
      const kind: Kind | undefined = animalKind(t.animal, t.from, t.to);
      if (kind) add(kind);
    }
    if (freshPacks.has(t.animal.pack)) add('fresh');
    else if (t.animal.split === 'train') add('old');
  });
  const out: Record<string, Scores> = {};
  for (const [key, s] of Object.entries(acc)) {
    out[key] = Object.fromEntries(
      Object.entries(s).map(([k, [c, n]]) => [k, n > 0 ? c / n : null]),
    ) as unknown as Scores;
  }
  return out;
}
