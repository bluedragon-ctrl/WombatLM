/**
 * The training engine behind the worker: handles protocol messages, runs the
 * training loop in short time slices (so messages such as `pause` are handled
 * promptly), tests every animal on a schedule while the Test tab is open and answers questions.
 *
 * A test of every animal takes up to about a second, so it runs as a
 * background job: it answers with a frozen copy of the weights (taken when the
 * test starts) a few questions at a time, taking turns with training slices
 * and messages.
 *
 * It does not touch worker globals: `worker.ts` passes in how to post
 * messages and how to schedule the next slice, so tests can drive it in Node.
 */
import { Dataset, Animal, type Task } from '../data/dataset';
import { baseBundle } from '../data/bundle';
import { Vocab } from '../nn/vocab';
import { nearestRows, pca2 } from '../nn/pca';
import { FormTrainer, PracticeBatch, TRAINER_DEFAULTS, confidence, judge } from './trainer';
import { Translator } from '../nn/translator';
import type {
  Activation, AnimalInfo, AnswerMsg, AttentionMap, Badge, ExplainGuess, ExplainMsg, FromWorker, InspectMode, LossPoint, MatrixChange, MatrixInfo, Speed,
  StepGuess, ToWorker,
} from './protocol';
import { rowSoftmax } from '../nn/transformer';

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
/** Test questions answered at a time by the background test (a few ms each). */
const EVAL_CHUNK = 32;
const TICK_MS = 50;
/** Wall-clock speed is measured over windows of at least this long. */
const WALL_WINDOW_MS = 1000;
/** How many closest letters the letter map lists for a clicked letter. */
const NEAR_LETTERS = 4;
/** How many examples of the latest batch a tick carries. */
export const PRACTICE_SAMPLE = 32;
/** While the Model learning tab is open and training is fast: answers before a step at most this often. */
const OWN_EVERY_MS = 500;

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
  /** For `inspect` in delta mode: weights at the previous request, and the change found then. */
  private deltaBase = new Map<string, { step: number; data: Float32Array }>();
  private lastDelta = new Map<string, { steps: number; data: Float32Array }>();
  /** For the change summary (`weights.summary`): all weights at the previous request, and the change found then. */
  private summaryBase: { step: number; data: Float32Array[] } | null = null;
  private lastSummary: MatrixChange[] | null = null;
  /** Weights before the latest training step (reused buffers), so the first delta request has something to show. */
  private beforeStep: Float32Array[] | null = null;
  /** Step of the latest practice batch sent in a tick (sent once per new step). */
  private lastPracticeStep = -1;
  /** The Model learning tab is open: practice batches carry the model's own answers (see `watchPractice`). */
  private watchPractice = false;
  private lastOwnAt = -Infinity;
  /** The latest batch the model answered before learning from it. */
  private answered: PracticeBatch | undefined;
  /**
   * The Test tab is open: test every animal on the EVAL schedule. Off by
   * default (beginners only learn and ask); `evaluate` still tests on request.
   */
  private watchTests = false;
  /** Quiet learning: skip work that is only for the pictures while training runs. */
  private quiet = false;
  /** Learn until this step, then stop by itself (null = no limit). */
  goal: number | null = null;
  /** The test in progress (see `startEval`), answered a chunk at a time. */
  private evalJob: { step: number; inventions: boolean; tasks: Task[]; outputs: string[]; ms: number } | null = null;
  /** Frozen copy of the weights the test in progress answers with (reused between tests). */
  private evalModel: Translator | null = null;
  /** A chunk of the test is scheduled. */
  private evalScheduled = false;
  /** Wall-clock speed: steps and time at the start of the current measuring window. */
  private wallMark: { t: number; step: number } | null = null;
  private wallStepsPerSec = 0;

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
          matrices: this.matrices(),
        });
        if (this.watchTests) this.startEval(true);
        this.tick(true);
        break;
      case 'play':
        if (!this.running && !this.goalReached()) {
          this.running = true;
          this.budget = 1; // the first step happens right away, even at slow speeds
          this.lastSlice = this.host.now();
          this.wallMark = null;
          this.wallStepsPerSec = 0;
          const gen = ++this.generation;
          this.host.schedule(() => this.slice(gen), 0);
        }
        this.tick(true);
        break;
      case 'pause':
        this.finish();
        break;
      case 'step':
        for (let i = 0; i < msg.n; i++) this.trainOne();
        if (this.watchTests) this.startEval(this.trainer.step - this.lastInventionStep >= EVAL.inventionsEverySteps);
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
        if (this.watchTests) this.startEval(true);
        this.tick(true);
        break;
      case 'ask':
        this.answer(msg.id, msg.text, msg.lang, msg.form);
        break;
      case 'evaluate':
        this.startEval(msg.inventions ?? false);
        break;
      case 'watchTests':
        this.watchTests = msg.on;
        // Test now if this model has not been tested yet (with the inventions if they are out of date).
        if (msg.on && this.lastEvalStep !== this.trainer.step) {
          this.startEval(this.trainer.step - this.lastInventionStep >= EVAL.inventionsEverySteps || this.lastInventionStep < 0);
        }
        break;
      case 'watchPractice':
        this.watchPractice = msg.on;
        if (msg.on) { this.lastPracticeStep = -1; this.tick(true); } // send the current batch with own answers now
        break;
      case 'quiet':
        this.quiet = msg.on;
        this.wallMark = null; // measure the new setting from scratch
        this.wallStepsPerSec = 0;
        this.tick(true);
        break;
      case 'setGoal':
        this.goal = msg.steps === null ? null : Math.max(1, Math.round(msg.steps));
        if (this.running && this.goalReached()) this.finish(true);
        else this.tick(true);
        break;
      case 'inspect':
        if (msg.matrixIds.length) this.sendWeights(msg.id, msg.matrixIds, msg.mode);
        if (msg.letterMap) this.sendLetterMap(msg.id);
        break;
      case 'explain':
        this.explain(msg.id, msg.text, msg.lang, msg.form, msg.position);
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
    this.evalJob = null; // a test of the old model is no longer wanted
    this.deltaBase.clear();
    this.lastDelta.clear();
    this.summaryBase = null;
    this.lastSummary = null;
    this.beforeStep = null;
    this.lastPracticeStep = -1;
    this.answered = undefined;
  }

  /** The step goal is reached (Play does nothing until the goal is raised). */
  goalReached(): boolean {
    return this.goal !== null && this.trainer.step >= this.goal;
  }

  /**
   * Stop training (pause or goal reached) and report. Tests once more if the
   * Test tab is open and `test` (quiet learning skipped the tests while running;
   * a reached goal is the final model).
   */
  private finish(test = this.quiet): void {
    this.stop();
    if (test && this.watchTests && this.lastEvalStep !== this.trainer.step) this.startEval(true);
    this.tick(true);
  }

  private stop(): void {
    this.running = false;
    this.generation++; // any scheduled slice of the old loop sees a stale generation and stops
  }

  /** Training too fast to follow single steps. */
  private fastRun(): boolean {
    return this.running && (this.speed === 'max' || this.speed > 5);
  }

  /**
   * How many of the next batch's questions the model should answer before
   * learning from them (for the Model learning tab and the one-step cycle). Always
   * for Step and slow speeds (cheap); at high speeds only while the Practice
   * tab is open, and at most every OWN_EVERY_MS (one greedy pass each).
   */
  private answerFirst(): number {
    if (!this.fastRun()) return PRACTICE_SAMPLE;
    if (!this.watchPractice || this.quiet) return 0;
    const now = this.host.now();
    if (now - this.lastOwnAt < OWN_EVERY_MS) return 0;
    this.lastOwnAt = now;
    return PRACTICE_SAMPLE;
  }

  private trainOne(): void {
    const params = this.trainer.model.params.list;
    if (!this.beforeStep || this.beforeStep.length !== params.length) this.beforeStep = params.map((p) => new Float32Array(p.size));
    params.forEach((p, i) => {
      if (this.beforeStep![i].length !== p.size) this.beforeStep![i] = new Float32Array(p.size);
      this.beforeStep![i].set(p.data);
    });
    const t0 = this.host.now();
    const answerFirst = this.answerFirst();
    const loss = this.trainer.trainStep(answerFirst);
    if (answerFirst > 0) this.answered = this.trainer.practice(PRACTICE_SAMPLE) ?? undefined;
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
      do this.trainOne(); while (this.host.now() - start < SLICE_MS && !this.goalReached());
    } else {
      // Steps are "earned" over time; never bank more than a quarter second's worth.
      this.budget = Math.min(this.budget + ((start - this.lastSlice) * this.speed) / 1000, Math.max(1, this.speed / 4));
      while (this.budget >= 1 && this.host.now() - start < SLICE_MS && !this.goalReached()) { this.trainOne(); this.budget -= 1; }
    }
    if (this.goalReached()) { this.finish(true); return; } // stops by itself
    this.lastSlice = start;
    const sinceEval = this.trainer.step - this.lastEvalStep;
    if (this.watchTests && !this.quiet && !this.evalJob && (sinceEval >= EVAL.everySteps || (sinceEval > 0 && start - this.lastEvalTime >= EVAL.everyMs))) {
      this.startEval(this.trainer.step - this.lastInventionStep >= EVAL.inventionsEverySteps);
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
    this.measureWall(now);
    // The latest batch it practised on (only when there is a new one).
    let practice: PracticeBatch | undefined;
    if (this.watchPractice && this.fastRun()) {
      // Model learning tab open at high speed: show the latest batch it answered (kept
      // until sent, since later steps of the same tick would replace it).
      if (this.answered && this.answered.step !== this.lastPracticeStep) practice = this.answered;
    } else if (tr.step !== this.lastPracticeStep && !(this.quiet && this.running)) {
      practice = tr.practice(PRACTICE_SAMPLE) ?? undefined;
    }
    if (practice) this.lastPracticeStep = practice.step;
    this.host.post({
      type: 'tick', step: tr.step, running: this.running, speed: this.speed, lossEma: tr.lossEma,
      lr: this.lr, batchSize: tr.batchSize, gradNorm: tr.lastGradNorm, stepsPerSec: this.stepsPerSec,
      wallStepsPerSec: this.wallStepsPerSec, quiet: this.quiet, goal: this.goal, losses: this.losses, practice,
    });
    this.losses = [];
  }

  /** Real speed: steps per wall-clock second, measured over windows of about a second while running. */
  private measureWall(now: number): void {
    const step = this.trainer.step;
    if (!this.running) { this.wallMark = null; return; }
    if (!this.wallMark || step < this.wallMark.step) { this.wallMark = { t: now, step }; return; }
    const dt = now - this.wallMark.t;
    if (dt < WALL_WINDOW_MS) return;
    const sps = ((step - this.wallMark.step) * 1000) / dt;
    this.wallStepsPerSec = this.wallStepsPerSec === 0 ? sps : 0.5 * this.wallStepsPerSec + 0.5 * sps;
    this.wallMark = { t: now, step };
  }

  /**
   * Start testing the current model (in the background, see `evalWork`).
   * A test still in progress is replaced (keeping its inventions), unless it
   * already tests this step with everything asked for.
   */
  private startEval(inventions: boolean): void {
    const tr = this.trainer;
    const prev = this.evalJob;
    if (prev && prev.step === tr.step && (prev.inventions || !inventions)) return;
    const inv = inventions || (prev?.inventions ?? false);
    this.evalJob = { step: tr.step, inventions: inv, tasks: tr.evalTasks(inv), outputs: [], ms: 0 };
    this.freezeWeights();
    this.lastEvalStep = tr.step;
    if (inv) this.lastInventionStep = tr.step;
    this.scheduleEval();
  }

  /** Copy the current weights into the model the test answers with. */
  private freezeWeights(): void {
    const live = this.trainer.model;
    const list = live.params.list;
    const frozen = this.evalModel?.params.list;
    if (!frozen || frozen.length !== list.length || frozen.some((p, i) => p.size !== list[i].size)) {
      this.evalModel = new Translator(live.config, 0);
    }
    this.evalModel!.params.load(list.map((p) => p.data));
  }

  private scheduleEval(): void {
    if (this.evalScheduled || !this.evalJob) return;
    this.evalScheduled = true;
    this.host.schedule(() => {
      this.evalScheduled = false;
      try {
        this.evalWork();
      } catch (e) {
        this.evalJob = null;
        this.host.post({ type: 'error', message: e instanceof Error ? e.message : String(e) });
      }
      this.scheduleEval();
    }, 0);
  }

  /**
   * Answer test questions for about one slice, then yield. Sends the result
   * once every question is answered.
   */
  private evalWork(): void {
    const job = this.evalJob;
    if (!job) return;
    const start = this.host.now();
    do {
      const part = job.tasks.slice(job.outputs.length, job.outputs.length + EVAL_CHUNK);
      job.outputs.push(...this.trainer.runTasks(part, this.evalModel!));
    } while (job.outputs.length < job.tasks.length && this.host.now() - start < SLICE_MS);
    job.ms += this.host.now() - start;
    if (job.outputs.length < job.tasks.length) return;
    this.evalJob = null;
    this.lastEvalTime = this.host.now();
    const ev = this.trainer.scoreTasks(job.tasks, job.outputs, job.step);
    this.host.post({ type: 'eval', step: ev.step, scores: ev.scores, results: ev.results, inventions: job.inventions, ms: job.ms });
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
      };
    });
    const attention: AttentionMap[] = Object.entries(trace.attention).map(([name, a]) => ({ name, ...a }));
    const activations: Activation[] = Object.entries(trace.activations).map(([name, a]) => ({ name, ...a }));
    const output = vocab.decode(trace.output);
    const msg: AnswerMsg = {
      type: 'answer', id, step: this.trainer.step, text: word, lang, form, output, finished: trace.finished,
      input: trace.input.map((i) => tokens[i]), steps, confidence: confidence(trace, vocab.eos), attention, activations,
      ...this.badge(word, lang, form, output),
    };
    this.host.post(msg, [...attention, ...activations].map((a) => a.data.buffer as ArrayBuffer));
  }

  /** Every parameter as a matrix (vectors are one row). */
  private matrices(): MatrixInfo[] {
    const ps = this.trainer.model.params;
    return ps.list.map((t) => {
      const meta = ps.meta.get(t.name)!;
      const [rows, cols] = t.shape.length === 2 ? t.shape : [1, t.shape[0]];
      return { name: t.name, rows, cols, rowKind: t.shape.length === 2 ? meta.rows : 'none', colKind: meta.cols, role: meta.role };
    });
  }

  private sendWeights(id: number, names: string[], mode: InspectMode): void {
    const ps = this.trainer.model.params;
    const step = this.trainer.step;
    let steps = 0;
    const matrices = names.map((name) => {
      const p = ps.get(name);
      if (mode === 'weights') return { name, data: p.data.slice() };
      if (mode === 'grad') {
        steps = step > 0 && p.grad ? 1 : 0;
        return { name, data: p.grad && step > 0 ? p.grad.slice() : new Float32Array(p.size) };
      }
      // delta: change since the previous request. If nothing was trained in
      // between, show the previous change again rather than a blank picture.
      const base = this.deltaBase.get(name);
      let delta = this.lastDelta.get(name);
      if (!base || base.data.length !== p.size) {
        // First request: show the change of the latest step, if there was one.
        const before = this.beforeStep?.[ps.list.indexOf(p)];
        const d = new Float32Array(p.size);
        if (step > 0 && before && before.length === p.size) for (let i = 0; i < d.length; i++) d[i] = p.data[i] - before[i];
        delta = { steps: step > 0 && before ? 1 : 0, data: d };
        this.lastDelta.set(name, delta);
        this.deltaBase.set(name, { step, data: p.data.slice() });
      } else if (base.step !== step) {
        const d = new Float32Array(p.size);
        for (let i = 0; i < d.length; i++) d[i] = p.data[i] - base.data[i];
        delta = { steps: step - base.step, data: d };
        this.lastDelta.set(name, delta);
        this.deltaBase.set(name, { step, data: p.data.slice() });
      } else if (!delta) {
        delta = { steps: 0, data: new Float32Array(p.size) };
      }
      steps = Math.max(steps, delta.steps);
      return { name, data: delta.data.slice() };
    });
    this.host.post({ type: 'weights', id, step, mode, steps, matrices, summary: this.summary() }, matrices.map((m) => m.data.buffer as ArrayBuffer));
  }

  /**
   * How much every matrix changed since the previous summary (the first one:
   * in the latest step). If nothing was trained in between, the previous
   * change again.
   */
  private summary(): MatrixChange[] {
    const list = this.trainer.model.params.list;
    const step = this.trainer.step;
    const norm2 = (a: Float32Array) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s); };
    const change = (a: Float32Array, b: Float32Array | undefined) => {
      if (!b || b.length !== a.length) return 0;
      let s = 0;
      for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
      return Math.sqrt(s);
    };
    const base = this.summaryBase;
    if (base && base.data.length === list.length && base.step === step && this.lastSummary) return this.lastSummary;
    let from: Float32Array[] | null, steps: number;
    if (base && base.data.length === list.length && base.step < step) { from = base.data; steps = step - base.step; }
    else { from = step > 0 ? this.beforeStep : null; steps = from ? 1 : 0; }
    const out = list.map((p, i) => ({ name: p.name, norm: norm2(p.data), change: from ? change(p.data, from[i]) : 0, steps }));
    this.summaryBase = { step, data: list.map((p) => p.data.slice()) };
    this.lastSummary = out;
    return out;
  }

  /** "Look inside one letter" (see nn/explain.ts). */
  private explain(id: number, text: string, lang: string, form: string, position: number): void {
    const vocab = this.dataset.vocab;
    const word = Vocab.normalize(text);
    const ex = this.trainer.explain(word, lang, form, position);
    const model = this.trainer.model;
    const tokens = vocab.tokens;
    const charOf = (i: number) => (i === vocab.eos ? '' : tokens[i]);
    const finished = ex.output.length < model.config.maxLen - 1;
    const probs = rowSoftmax(ex.logits, 0, ex.logits.length);
    const cands: number[] = [];
    for (let i = 0; i < probs.length; i++) if (i === vocab.eos || vocab.isChar(i)) cands.push(i);
    cands.sort((a, b) => probs[b] - probs[a]);
    const d = ex.final.length;
    const guesses: ExplainGuess[] = cands.slice(0, 5).map((tok) => ({
      char: charOf(tok), token: tok, p: probs[tok], score: ex.logits[tok],
      row: model.tokEmb.data.slice(tok * d, (tok + 1) * d),
      contrib: ex.final.map((v, i) => v * model.tokEmb.data[tok * d + i]), bias: model.outBias.data[tok],
    }));
    const p = ex.position;
    const input = [this.dataset.targetId(lang, form), ...vocab.encodeWord(word).slice(0, model.config.maxLen - 2)];
    const msg: ExplainMsg = {
      type: 'explain', id, step: this.trainer.step, text: word, lang, form, output: vocab.decode(ex.output),
      columns: finished ? ex.output.length + 1 : ex.output.length, position: p,
      chosen: p < ex.output.length ? charOf(ex.output[p]) : '',
      input: input.map((i) => tokens[i]), inputIds: input,
      decIn: ex.decIn.map((i) => tokens[i]), decInIds: ex.decIn,
      start: ex.start, stages: ex.stages, final: ex.final, guesses,
    };
    this.host.post(msg);
  }

  /** PCA of the letter table: characters, end of word and the task tokens. */
  private sendLetterMap(id: number): void {
    const vocab = this.dataset.vocab;
    const emb = this.trainer.model.tokEmb;
    const ids: number[] = [];
    for (let i = 0; i < vocab.size; i++) if (vocab.isChar(i) || i === vocab.eos) ids.push(i);
    const letters = ids.length;
    ids.push(...vocab.targetIds());
    const { coords, explained } = pca2(emb.data, emb.shape[1], ids);
    const near = nearestRows(emb.data, emb.shape[1], ids, ids.map((_, i) => i < letters), NEAR_LETTERS);
    const seen = this.trainer.letterCounts();
    const counts = ids.map((t, i) => (i < letters && vocab.isChar(t) ? seen.get(vocab.tokens[t]) ?? 0 : -1));
    this.host.post({ type: 'letterMap', id, step: this.trainer.step, ids, coords, explained, near, counts }, [coords.buffer as ArrayBuffer]);
  }

  /** What the data says about this question (see `Badge`). */
  private badge(word: string, lang: string, form: string, output: string): Pick<AnswerMsg, 'badge' | 'answer' | 'animalId'> {
    const animal = this.dataset.animals.find((a) => a.names[lang] === word);
    if (!animal) return { badge: 'free' };
    const main = form === 'name' ? animal.names[lang] : animal.forms[lang]?.[form];
    if (!main) return { badge: 'invention', animalId: animal.id };
    const badge: Badge = !animal.unseen ? 'real' : 'hidden';
    return { badge, animalId: animal.id, answer: { word: main, status: judge(output, main) as 'correct' | 'close' | 'wrong' } };
  }
}

function animalInfo(a: Animal): AnimalInfo {
  return { id: a.id, unseen: a.unseen, pack: a.pack, names: { ...a.names }, forms: structuredClone(a.forms) };
}
