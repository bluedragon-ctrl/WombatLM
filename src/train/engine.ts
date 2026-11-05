/**
 * The training engine behind the worker: handles protocol messages, runs the
 * training loop in short time slices (so messages such as `pause` are handled
 * promptly), evaluates on a schedule and answers questions.
 *
 * It does not touch worker globals: `worker.ts` passes in how to post
 * messages and how to schedule the next slice, so tests can drive it in Node.
 */
import { Dataset, Animal } from '../data/dataset';
import { baseBundle } from '../data/bundle';
import { Vocab } from '../nn/vocab';
import { FormTrainer, TRAINER_DEFAULTS, confidence, judge } from './trainer';
import type { AnimalInfo, AnswerMsg, AttentionMap, Badge, FromWorker, LossPoint, Speed, StepGuess, ToWorker } from './protocol';

export interface EngineHost {
  post(msg: FromWorker, transfer?: Transferable[]): void;
  /** Run `fn` after `ms` milliseconds (0 = as soon as pending messages are handled). */
  schedule(fn: () => void, ms: number): void;
  now(): number;
}

/** Evaluation schedule (see docs/DECISIONS.md). */
export const EVAL = {
  /** Evaluate the real forms every this many steps… */
  everySteps: 100,
  /** …or after this long, if at least one step was trained (slow speeds). */
  everyMs: 3000,
  /** Include the inventions every this many steps. */
  inventionsEverySteps: 500,
};

/** Work per slice before yielding to the message queue. */
const SLICE_MS = 12;
const TICK_MS = 50;

export class Engine {
  dataset: Dataset;
  trainer!: FormTrainer;
  seed = 1;
  running = false;
  speed: Speed = 'max';
  private generation = 0;
  private budget = 0;
  private lastSlice = 0;
  private lastTick = -Infinity;
  private lastEvalStep = -1;
  private lastEvalTime = 0;
  private lastInventionStep = -1;
  private losses: LossPoint[] = [];
  private stepsPerSec = 0;
  private lr: number = TRAINER_DEFAULTS.lr;
  private batchSize: number = TRAINER_DEFAULTS.batchSize;

  constructor(private host: EngineHost, dataset?: Dataset) {
    this.dataset = dataset ?? new Dataset(baseBundle());
  }

  handle(msg: ToWorker): void {
    try {
      this.dispatch(msg);
    } catch (e) {
      this.host.post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
    }
  }

  private dispatch(msg: ToWorker): void {
    switch (msg.type) {
      case 'init':
        if (msg.speed !== undefined) this.speed = msg.speed;
        this.reset(msg.seed);
        this.host.post({
          type: 'ready',
          seed: this.seed,
          params: this.trainer.model.params.count(),
          languages: this.dataset.languages.map((l) => ({ code: l.code, name: l.name })),
          forms: this.dataset.forms.slice(),
          animals: this.dataset.animals.map(animalInfo),
          tokens: this.dataset.vocab.tokens.slice(),
        });
        this.evaluate(true);
        this.tick(true);
        break;
      case 'play':
        if (!this.running) {
          this.running = true;
          this.budget = 1; // the first step happens right away, even at slow speeds
          this.lastSlice = this.host.now();
          const gen = ++this.generation;
          this.host.schedule(() => this.slice(gen), 0);
        }
        this.tick(true);
        break;
      case 'pause':
        this.stop();
        this.tick(true);
        break;
      case 'step':
        for (let i = 0; i < msg.n; i++) this.trainOne();
        this.evaluate(this.trainer.step - this.lastInventionStep >= EVAL.inventionsEverySteps);
        this.tick(true);
        break;
      case 'setSpeed':
        this.speed = msg.speed;
        this.budget = 0;
        this.tick(true);
        break;
      case 'setHyper':
        if (msg.lr !== undefined) { this.lr = msg.lr; this.trainer.setLearningRate(msg.lr); }
        if (msg.batchSize !== undefined) { this.batchSize = Math.max(1, Math.round(msg.batchSize)); this.trainer.batchSize = this.batchSize; }
        this.tick(true);
        break;
      case 'reset':
        this.reset(msg.seed);
        this.evaluate(true);
        this.tick(true);
        break;
      case 'ask':
        this.answer(msg.id, msg.text, msg.lang, msg.form);
        break;
      case 'evaluate':
        this.evaluate(msg.inventions ?? false);
        break;
    }
  }

  private reset(seed: number): void {
    this.stop();
    this.seed = seed;
    this.trainer = new FormTrainer(this.dataset, { seed, lr: this.lr, batchSize: this.batchSize });
    this.losses = [];
    this.stepsPerSec = 0;
    this.lastEvalStep = -1;
    this.lastInventionStep = -1;
  }

  private stop(): void {
    this.running = false;
    this.generation++; // any scheduled slice of the old loop sees a stale generation and stops
  }

  private trainOne(): void {
    const t0 = this.host.now();
    const loss = this.trainer.trainStep();
    const dt = this.host.now() - t0;
    if (dt > 0) {
      const sps = 1000 / dt;
      this.stepsPerSec = this.stepsPerSec === 0 ? sps : 0.98 * this.stepsPerSec + 0.02 * sps;
    }
    this.losses.push({ step: this.trainer.step, loss });
  }

  /** One time slice of the training loop. */
  private slice(gen: number): void {
    if (gen !== this.generation || !this.running) return;
    const start = this.host.now();
    if (this.speed === 'max') {
      do this.trainOne(); while (this.host.now() - start < SLICE_MS);
    } else {
      // Steps are "earned" over time; never bank more than a quarter second's worth.
      this.budget = Math.min(this.budget + ((start - this.lastSlice) * this.speed) / 1000, Math.max(1, this.speed / 4));
      while (this.budget >= 1 && this.host.now() - start < SLICE_MS) { this.trainOne(); this.budget -= 1; }
    }
    this.lastSlice = start;
    const sinceEval = this.trainer.step - this.lastEvalStep;
    if (sinceEval >= EVAL.everySteps || (sinceEval > 0 && start - this.lastEvalTime >= EVAL.everyMs)) {
      this.evaluate(this.trainer.step - this.lastInventionStep >= EVAL.inventionsEverySteps);
    }
    this.tick(false);
    const wait = this.speed === 'max' ? 0 : Math.max(4, Math.min(100, 1000 / this.speed));
    this.host.schedule(() => this.slice(gen), wait);
  }

  private tick(force: boolean): void {
    const now = this.host.now();
    if (!force && now - this.lastTick < TICK_MS) return;
    this.lastTick = now;
    const tr = this.trainer;
    this.host.post({
      type: 'tick', step: tr.step, running: this.running, speed: this.speed, lossEma: tr.lossEma,
      lr: this.lr, batchSize: tr.batchSize, gradNorm: tr.lastGradNorm, stepsPerSec: this.stepsPerSec, losses: this.losses,
    });
    this.losses = [];
  }

  private evaluate(inventions: boolean): void {
    const t0 = this.host.now();
    const ev = this.trainer.evaluate({ inventions });
    this.lastEvalStep = this.trainer.step;
    this.lastEvalTime = this.host.now();
    if (inventions) this.lastInventionStep = this.trainer.step;
    this.host.post({ type: 'eval', step: ev.step, scores: ev.scores, results: ev.results, inventions, ms: this.lastEvalTime - t0 });
  }

  private answer(id: number, text: string, lang: string, form: string): void {
    const vocab = this.dataset.vocab;
    const word = Vocab.normalize(text);
    const trace = this.trainer.inspect(word, lang, form);
    const tokens = vocab.tokens;
    const charOf = (i: number) => (i === vocab.eos ? '' : tokens[i]);
    const steps: StepGuess[] = trace.probs.map((p, t) => {
      const chosen = t < trace.output.length ? trace.output[t] : vocab.eos;
      const cands: number[] = [];
      for (let i = 0; i < p.length; i++) if (i === vocab.eos || vocab.isChar(i)) cands.push(i);
      cands.sort((a, b) => p[b] - p[a]);
      return {
        chosen: charOf(chosen),
        top: cands.slice(0, 5).map((i) => ({ char: charOf(i), p: p[i] })),
        write: trace.writeShare?.[t],
      };
    });
    const attention: AttentionMap[] = Object.entries(trace.attention).map(([name, a]) => ({ name, ...a }));
    const output = vocab.decode(trace.output);
    const msg: AnswerMsg = {
      type: 'answer', id, step: this.trainer.step, text: word, lang, form, output, finished: trace.finished,
      input: trace.input.map((i) => tokens[i]), steps, confidence: confidence(trace, vocab.eos), attention,
      ...this.badge(word, lang, form, output),
    };
    this.host.post(msg, attention.map((a) => a.data.buffer as ArrayBuffer));
  }

  /** What the data says about this question (see `Badge`). */
  private badge(word: string, lang: string, form: string, output: string): Pick<AnswerMsg, 'badge' | 'answer' | 'animalId'> {
    const animal = this.dataset.animals.find((a) => a.names[lang] === word || (a.alt[lang] ?? []).includes(word));
    if (!animal) return { badge: 'free' };
    const main = form === 'name' ? animal.names[lang] : animal.forms[lang]?.[form];
    if (!main) return { badge: 'invention', animalId: animal.id };
    const alts = form === 'name' ? animal.alt[lang] ?? [] : animal.formsAlt[lang]?.[form] ?? [];
    const accepted = [main, ...alts];
    const badge: Badge = animal.split === 'train' ? 'real' : 'hidden';
    return { badge, animalId: animal.id, answer: { accepted, status: judge(output, accepted) as 'correct' | 'close' | 'wrong' } };
  }
}

function animalInfo(a: Animal): AnimalInfo {
  return { id: a.id, split: a.split, pack: a.pack, names: { ...a.names }, forms: structuredClone(a.forms) };
}
