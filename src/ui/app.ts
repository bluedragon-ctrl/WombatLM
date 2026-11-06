/**
 * The app: eight tabs, in the order of the story (two of them expert only).
 * - Learning data: everything the model learns from (see dataTab.ts).
 * - Learn: everything that changes the model: Play / Step / speed /
 *   seed, the learning badge, the one-step cycle, the latest batch (learning
 *   data and what it answered before learning from it), the loss chart and
 *   the letter map.
 * - Ask: everything that only reads the model: the question box
 *   with form and language pickers, the answer with its Insights block (the word, kind of
 *   answer, real answer, how sure, which model), guess bars, attention grids.
 * - Test: every animal tested, scoreboard. The model is tested only while
 *   this tab is open (or on request).
 * - Letter selection: one letter of the answer in detail, and the
 *   detectors (see letterTab.ts).
 * - Where it learns (expert only): the brain scan of all the model's numbers (see inside.ts).
 * - The matrix: the answer as the matrix of numbers it is inside the model,
 *   station by station (see matrixTab.ts).
 * - Numbers (expert only): the real numbers behind the other tabs:
 *   sizes, every weight matrix, gradients, the hidden numbers of an answer
 *   and the full calculation of one learning step (see numbersTab.ts).
 *
 * All training happens in the worker; this file only sends messages and
 * draws what comes back.
 */
import type { AnimalInfo, AnswerMsg, EvalMsg, FromWorker, LetterMapMsg, LossPoint, ReadyMsg, Speed, TickMsg, ToWorker } from '../train/protocol';
import { StepView } from './stepView';
import { JourneyView } from './journey';
import { MatrixTab } from './matrixTab';
import type { PracticeBatch, PracticeExample, Scores, TaskResult } from '../train/trainer';
import { LETTER_KINDS, LetterKind, STATUS_COLORS, STATUS_ICONS, SCORE_STYLE } from '../viz/colors';
import { attentionCellAt, drawAttentionGrid, GridSpec } from '../viz/attentionGrid';
import { drawGuessBars, guessColumnAt } from '../viz/guessBars';
import { LetterMapView, letterKind, letterLabel, marker, type LetterPoint } from '../viz/letterMap';
import { drawLineChart } from '../viz/lineChart';
import { parentWidth, prepare } from '../viz/canvas';
import { setUiLang, t, uiLang, UI_LANGUAGES } from './i18n';
import { h } from './dom';
import { ScanTab } from './inside';
import { LetterTab } from './letterTab';
import { NumbersTab } from './numbersTab';
import { DataTab } from './dataTab';
import { shortLabel, tokenChip, tokenIndex, tokenName, toTok } from './tokens';
import { EOS, Vocab } from '../nn/vocab';

type Filter = 'all' | 'known' | 'unseen' | 'wrong';
type Tab = 'data' | 'learn' | 'scan' | 'ask' | 'matrix' | 'letter' | 'test' | 'numbers';
const TABS: Tab[] = ['data', 'learn', 'ask', 'test', 'letter', 'scan', 'matrix', 'numbers'];
/** Tabs shown only in expert mode. */
const EXPERT_TABS: Tab[] = ['scan', 'numbers'];
/** Look-inside requests: at most one in flight, at most this often. */
const INSPECT_MS = 250;
/** "Write it slowly": time per letter, and the "reading the animal" pause before the first one. */
const SLOW_MS = 1100;
const SLOW_READ_MS = 1400;
/** One-step learning cycle: time each stage stays lit before the next one. */
const CYCLE_MS = 700;
/** At high speeds the practice strip changes at most this often (so it can be read at all). */
const PRACTICE_FAST_MS = 500;
/** After a step while paused, the learning badge says "learning" for this long. */
const LEARN_GLOW_MS = 1500;
const CYCLE_STAGES = ['pick', 'answer', 'measure', 'nudge', 'check'] as const;
/** The cycle stage in which the numbers change (the learning badge lights up then). */
const NUDGE_PHASE = 4;

/** A "?" that opens a short explanation (keyboard accessible). */
const help = (text: string) => h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, text));
/** One row of the Insights block under an answer: a label and its value. */
const insightRow = (label: string, ...value: (Node | string)[]) =>
  h('div', { class: 'insight' }, h('span', { class: 'insight-label' }, label), h('span', { class: 'insight-value' }, ...value));
/** Restart a CSS animation class on an element. */
function retrigger(el: HTMLElement, cls: string): void {
  el.classList.remove(cls);
  void el.offsetWidth;
  el.classList.add(cls);
}

/** One attention grid on screen. `linked`: its rows are the output letters (hover links to the guess bars). */
interface AttGrid {
  key: string;
  title: string;
  spec: GridSpec;
  /** Full token names for the tooltip (the grid itself shows short labels). */
  rowNames: string[];
  colNames: string[];
  linked: boolean;
  canvas?: HTMLCanvasElement;
}

const SPEEDS: Speed[] = [1, 5, 20, 'max'];
/**
 * "Learn until" choices (steps, null = no limit). Measured with seed 1 at
 * batch 16: ~2,500 steps knows most known animals (cs 85 %, en 74 %), ~5,000
 * almost all (98 % / 93 %), ~7,500 all; more steps barely change the answers.
 */
const GOALS: (number | null)[] = [1000, 2500, 5000, 10000, null];
const DEFAULT_GOAL = 5000;
const SCORE_LINES = ['known', 'pattern', 'partly', 'irregular'] as const;
/** Examples of the latest batch a tick carries at most (engine PRACTICE_SAMPLE). */
const PRACTICE_SHOWN = 32;

/** Training runs with quiet learning on: the pictures wait. */
const quietRun = (tk: TickMsg | undefined) => !!tk?.quiet && !!tk.running;

/** A rough duration for "about … left": seconds below 90 s, else minutes. */
const duration = (sec: number) => (sec < 90 ? t('goal.sec', { n: Math.max(1, Math.round(sec)) }) : t('goal.min', { n: Math.round(sec / 60) }));

/** Mistake (loss) of one letter: −ln of the chance the model gave the right one. */
const letterLoss = (p: number) => -Math.log(p + 1e-9);
/** Mistakes above this count as "very surprised" (full colour); −ln 5 % ≈ 3. */
const LOSS_FULL = 3;

/** One letter of the right answer, tinted by its mistake, with the numbers on hover. */
function letterSpan(l: { char: string; p: number }): HTMLElement {
  const m = letterLoss(l.p);
  const k = Math.min(1, m / LOSS_FULL);
  return h('span', {
    class: `let${l.char === '' ? ' end' : ''}`,
    style: `background: rgba(220, 38, 38, ${(0.06 + 0.6 * k).toFixed(2)})`,
    title: t('practice.letterTip', { char: l.char === '' ? t('practice.end') : l.char, p: Math.round(l.p * 100), m: m.toFixed(2) }),
  }, l.char === '' ? '⏎' : l.char === ' ' ? '␣' : l.char);
}

/** Letters seen fewer times than this in the learning data get a "rare" note on the letter map. */
const RARE_LETTER = 30;

/** A chance as a percent: one decimal near 0 % and near 100 %, else whole. */
function surePct(p: number): string {
  const v = p * 100;
  if (v < 0.05) return '< 0.1';
  const digits = v < 1 || (v > 99 && v < 99.95) ? 1 : 0;
  return v.toLocaleString(uiLang(), { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

/**
 * A line's mistake: the number, a short bar and, below, how sure the model was
 * of the right answer: e^(−mistake), the average (geometric mean) of the
 * chances it gave each right letter. It always agrees with the mistake.
 */
function lossBar(loss: number): HTMLElement {
  const w = Math.round(Math.min(1, loss / LOSS_FULL) * 100);
  return h('span', { class: 'loss-cell', title: t('practice.lossTip') },
    h('span', { class: 'loss-val' },
      h('span', { class: 'num' }, loss.toFixed(2)),
      h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${w}%` }))),
    h('span', { class: 'sure', 'data-test': 'sure' }, t('practice.sure', { p: surePct(Math.exp(-loss)) })));
}

/**
 * After a step recorded in detail: the same example measured again after the
 * nudge (letters tinted by their new mistake, before → after on hover), its
 * mistake before → after, and its answer now if it changed.
 */
function checkCell(ex: PracticeExample): HTMLElement {
  const c = ex.check!;
  const letters = ex.letters.map((l, i) => {
    const span = letterSpan({ char: l.char, p: c.after[i] });
    const ch = l.char === '' ? t('practice.end') : l.char;
    span.title = t('practice.checkLetterTip', { char: ch, before: (c.before[i] * 100).toFixed(1), after: (c.after[i] * 100).toFixed(1) });
    span.classList.add(c.after[i] > c.before[i] ? 'up' : c.after[i] < c.before[i] ? 'down' : 'same');
    return span;
  });
  const better = c.lossAfter < c.lossBefore;
  return h('span', { class: 'check-cell' },
    h('span', { class: 'check-letters' }, ...letters),
    h('span', { class: `check-loss ${better ? 'better' : 'worse'}`, title: t('practice.checkLossTip') },
      `${c.lossBefore.toFixed(2)} → ${c.lossAfter.toFixed(2)} ${better ? '▼' : '▲'}`),
    ex.answer !== undefined && c.answer !== ex.answer
      ? h('span', { class: `word-own ${c.status}`, title: t('practice.checkAnswerTip') }, `${t('practice.nowAnswers')} ${STATUS_ICONS[c.status]} ${c.answer || '∅'}`) : '');
}

const resultKey = (id: string, lang: string, form: string) => `${id}|${lang}|${form}`;
const pct = (v: number | null | undefined) => (v === null || v === undefined ? '–' : `${Math.round(v * 100)} %`);

export class App {
  private worker: Worker;
  private ready?: ReadyMsg;
  private tick?: TickMsg;
  private losses: LossPoint[] = [];
  private evals: { step: number; scores: Record<string, Scores> }[] = [];
  private results = new Map<string, TaskResult>();
  private answer?: AnswerMsg;
  private askId = 0;
  private askInFlight = false;
  /** Id of the last request sent; while it equals askId, the reply on its way is for the current question. */
  private sentId = 0;
  private askDirty = false;
  private revealed = false;
  private expert = false;
  private filter: Filter = 'all';
  /** Data language shown on the Test tab. */
  private testLang = '';
  private question = { text: '', lang: '', form: 'female' };
  private dirty = { controls: true, answer: true, loss: true, scores: true, animals: true, practice: true, stamp: true };
  /** The latest batch the model practised on (learning), and when the strip last changed. */
  private practice?: PracticeBatch;
  private practiceShownAt = -Infinity;
  private practiceTimer = 0;
  private practiceNew = false;
  private learnedAt = -Infinity;
  /** Token strip on the Ask tab: only the numbers, as the model sees them. */
  private tokenNumbers = false;
  private tokenIdx = new Map<string, number>();
  /** The answer changed because the model changed (not the question): flash its step label. */
  private stampFlash = false;
  /** "Write it slowly" replay of one answer: how many letters are shown so far. */
  private slow: { answer: AnswerMsg; shown: number; done: boolean; timer: number } | null = null;
  /** "Answer slowly" was pressed: write the next answer letter by letter. */
  private slowWanted = false;
  /** One-step learning cycle animation (stage 1..4), started by Step ×1 while paused. */
  private cycle: { phase: number; timer: number } | null = null;
  private cycleWanted = false;
  private frame = 0;
  private els!: ReturnType<App['build']>;
  private tab: Tab = 'data';
  /** Output step linked between the guess bars and the attention grids (hover). */
  private hoverStep: number | null = null;
  private attGrids: AttGrid[] = [];
  private attKeys = '';
  /** Where it learns: the brain scan. */
  private inside: ScanTab;
  /** How it picks a letter: one letter in detail, and the detectors. */
  private letter: LetterTab;
  /** The real numbers (expert only). */
  private numbers: NumbersTab;
  private dataTab: DataTab;
  private letterMap!: LetterMapView;
  private inspectId = 0;
  private inspectPending: number | null = null;
  private inspectDirty = false;
  private lastInspect = -Infinity;
  private inspectTimer = 0;
  /** Latest "look inside one letter" request (older replies are dropped). */
  private explainId = 0;
  private unitsId = 0;
  /** What the word hint shows now (it is rebuilt only when this changes). */
  private hintKey = '';
  /** "What this step changed" (Model learning tab) and its latest "why" request. */
  private stepView: StepView;
  private whyId = 0;
  /** "Follow the tokens through the model" (Ask tab). */
  private journey: JourneyView;
  /** The answer as a matrix, station by station (Matrix tab). */
  private matrix: MatrixTab;

  constructor(private root: HTMLElement) {
    this.worker = new Worker(new URL('../train/worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
    this.worker.onerror = (e) => this.showError(e.message);
    this.inside = new ScanTab(() => this.requestInspect(true));
    this.letter = new LetterTab((q, position) => {
      this.send({ type: 'explain', id: ++this.explainId, ...q, position });
    });
    this.dataTab = new DataTab((text, lang, form) => this.askOnAskTab({ text, lang, form }));
    this.stepView = new StepView((name, index) => this.send({ type: 'why', id: ++this.whyId, name, index }), (c) => this.langName(c));
    this.journey = new JourneyView((c) => this.langName(c), (position) => {
      this.lookAtLetter(position);
    });
    this.matrix = new MatrixTab((c) => this.langName(c), () => this.setTab('ask'));
    this.letter.requestUnits = () => this.send({ type: 'units', id: ++this.unitsId });
    this.numbers = new NumbersTab(() => this.requestInspect(true), this.stepView);
    this.letter.onMarks = (marks) => this.numbers.setMarks(marks);
    // Expert: a part of the brain scan opens its heatmaps on "The numbers".
    this.inside.onOpenPart = (selector) => {
      if (!this.expert) return;
      this.setTab('numbers');
      this.numbers.scrollTo(selector);
    };
    // A click on a number in the heatmaps (after Step ×1): why did it change?
    this.numbers.onPickNumber = (name, index) => {
      if (!this.stepView.current()) return;
      this.stepView.pick(name, index);
      this.numbers.scrollToStep();
    };
    this.els = this.build();
    this.letterMap = new LetterMapView(this.els.mapCanvas, () => Math.min(360, parentWidth(this.els.mapCanvas, 320)));
    this.letterMap.onSelect = (sel) => this.renderMapPick(sel);
    this.renderMapPick(null);
    window.addEventListener('resize', () => { this.invalidate('loss', 'scores'); this.letterMap.draw(); });
    const seed = 1 + Math.floor(Math.random() * 9999);
    this.send({ type: 'init', seed, speed: 'max' });
    this.send({ type: 'setGoal', steps: DEFAULT_GOAL });
  }

  private send(msg: ToWorker): void {
    this.worker.postMessage(msg);
  }

  // -------------------------------------------------------------------------
  // Messages
  // -------------------------------------------------------------------------

  private onMessage(msg: FromWorker): void {
    switch (msg.type) {
      case 'ready':
        this.ready = msg;
        this.hintKey = '';
        this.tokenIdx = tokenIndex(msg.tokens);
        this.losses = [];
        this.evals = [];
        this.results.clear();
        if (!this.question.lang) this.question.lang = msg.languages[0].code;
        if (!this.testLang) this.testLang = msg.languages[0].code;
        if (!this.question.text) this.question.text = msg.animals.find((a) => a.forms[this.question.lang]?.female)?.names[this.question.lang] ?? '';
        this.els.seed.value = String(msg.seed);
        this.letterMap.reset();
        this.inside.setReady(msg);
        this.letter.setReady(msg);
        this.numbers.setReady(msg);
        this.dataTab.setReady(msg);
        this.stepView.setReady(msg);
        this.journey.setTokens(msg.tokens);
        this.matrix.setTokens(msg.tokens);
        this.renderPickers();
        this.ask();
        this.requestInspect(true);
        this.invalidate('controls', 'loss', 'scores', 'animals');
        break;
      case 'tick': {
        const prevTick = this.tick;
        if (this.tick && msg.step < this.tick.step) {
          this.losses = []; this.evals = []; this.results.clear();
          this.letterMap.reset();
          this.practice = undefined;
          this.stopCycle();
          this.stopSlow();
          this.requestInspect(true);
          this.invalidate('practice');
        }
        this.tick = msg;
        this.losses.push(...msg.losses);
        if (this.losses.length > 40000) this.losses = this.losses.filter((_, i) => i % 2 === 0);
        if (msg.practice) this.onPractice(msg.practice, msg);
        this.stepView.setStep(msg.step, msg.running, this.practice);
        if (this.tab === 'letter') this.letter.onStep(msg.step);
        this.numbers.canWhy = this.stepView.current();
        if (this.tab === 'numbers') this.numbers.setTick(msg);
        {
          // Quiet learning: no pictures while it learns; catch up when it pauses.
          const quietNow = msg.quiet && msg.running;
          const justPaused = !msg.running && !!prevTick?.running;
          if (!quietNow && (msg.losses.length || justPaused)) this.requestInspect(justPaused);
          // Answering is separate from learning: while the model learns, the answer
          // waits (marked as out of date, with "Ask again") until it pauses.
          if (this.answer && this.answer.step !== msg.step && !msg.running) this.ask();
        }
        this.els.learnMain.classList.toggle('quiet', quietRun(msg));
        this.invalidate('controls', 'loss', 'stamp');
        if (!msg.running && this.practice) this.invalidate('practice');
        break;
      }
      case 'eval':
        this.onEval(msg);
        break;
      case 'answer':
        this.askInFlight = false;
        // "Answer slowly" writes the newest reply: when another request is queued, wait for it.
        if (msg.id === this.askId && !(this.slowWanted && this.askDirty)) {
          const prev = this.answer;
          // Same question, new model: the answer refreshed because the model changed.
          if (prev && prev.step !== msg.step && prev.text === msg.text && prev.lang === msg.lang && prev.form === msg.form) this.stampFlash = true;
          this.answer = msg;
          if (this.slowWanted) { this.slowWanted = false; this.startSlow(msg); }
          this.invalidate('answer', 'stamp');
          this.letter.setAnswer(msg);
          this.numbers.setAnswer(msg);
          this.journey.setAnswer(msg);
          this.matrix.setAnswer(msg);
          if (this.tab === 'letter') this.letter.shown();
        }
        if (this.askDirty) this.ask();
        break;
      case 'weights':
        this.inspectDone(msg.id);
        this.inside.onWeights(msg);
        this.numbers.onWeights(msg);
        break;
      case 'explain':
        if (msg.id === this.explainId) this.letter.onExplain(msg);
        break;
      case 'units':
        if (msg.id === this.unitsId) this.letter.onUnits(msg);
        break;
      case 'letterMap':
        this.inspectDone(msg.id);
        this.onLetterMap(msg);
        break;
      case 'stepDetail':
        this.stepView.onDetail(msg);
        this.numbers.canWhy = this.stepView.current();
        break;
      case 'why':
        if (msg.id === this.whyId) this.stepView.onWhy(msg);
        break;
      case 'error':
        this.showError(msg.message);
        break;
    }
  }

  private onEval(msg: EvalMsg): void {
    if (msg.step === 0 || (this.evals.length && msg.step < this.evals[this.evals.length - 1].step)) this.evals = [];
    if (!this.evals.length || this.evals[this.evals.length - 1].step !== msg.step) this.evals.push({ step: msg.step, scores: msg.scores });
    for (const r of msg.results) this.results.set(resultKey(r.id, r.lang, r.form), r);
    this.invalidate('scores', 'animals');
  }

  private onPractice(p: PracticeBatch, tk: TickMsg): void {
    this.practice = p;
    this.practiceNew = true;
    this.learnedAt = performance.now();
    if (this.cycleWanted && !tk.running) { this.cycleWanted = false; this.startCycle(); }
    if (this.fast() && performance.now() - this.practiceShownAt < PRACTICE_FAST_MS) {
      if (!this.practiceTimer) this.practiceTimer = window.setTimeout(() => { this.practiceTimer = 0; this.invalidate('practice'); }, PRACTICE_FAST_MS);
      return;
    }
    this.invalidate('practice');
    if (!tk.running) window.setTimeout(() => this.invalidate('practice'), LEARN_GLOW_MS + 50);
  }

  /** Training too fast to follow single steps. */
  private fast(): boolean {
    const tk = this.tick;
    return !!tk?.running && (tk.speed === 'max' || tk.speed > 5);
  }

  /** One-step learning cycle: light the four stages one after another. */
  private startCycle(): void {
    this.stopCycle();
    const c = { phase: 1, timer: 0 };
    this.cycle = c;
    const next = () => {
      if (this.cycle !== c) return;
      if (c.phase < CYCLE_STAGES.length) {
        c.phase++;
        c.timer = window.setTimeout(next, CYCLE_MS);
        if (c.phase === NUDGE_PHASE) {
          // "Nudge the numbers": the learning badge lights up now.
          this.learnedAt = performance.now();
          this.practiceNew = true;
          window.setTimeout(() => this.invalidate('practice'), LEARN_GLOW_MS + 50);
        }
      } else this.cycle = null;
      this.invalidate('practice');
    };
    c.timer = window.setTimeout(next, CYCLE_MS);
  }

  private stopCycle(): void {
    if (this.cycle) window.clearTimeout(this.cycle.timer);
    this.cycle = null;
  }

  /** The two answer buttons: the whole answer at once, or written letter by letter. */
  private answerNow(slowly: boolean): void {
    if (slowly && this.slow) { this.stopSlow(); return; }
    if (!this.question.text.trim()) { this.els.input.focus(); return; }
    this.stopSlow();
    this.slowWanted = slowly;
    this.revealed = false;
    // A reply already on its way was asked before this press: drop it and ask anew.
    this.askId++;
    this.askDirty = false;
    this.ask();
    this.invalidate('answer');
  }

  /** "Answer slowly": reveal an answer one letter at a time. */
  private startSlow(a: AnswerMsg): void {
    this.stopSlow();
    const s = { answer: a, shown: 0, done: false, timer: 0 };
    this.slow = s;
    const next = () => {
      if (this.slow !== s) return;
      if (s.shown < a.steps.length) {
        s.shown++;
        this.hoverStep = s.shown - 1;
        s.timer = window.setTimeout(next, SLOW_MS);
      } else {
        s.done = true;
        this.hoverStep = null;
        s.timer = window.setTimeout(() => this.stopSlow(), 2 * SLOW_MS + 600);
      }
      this.invalidate('answer', 'stamp');
    };
    this.hoverStep = null;
    s.timer = window.setTimeout(next, SLOW_READ_MS);
    this.invalidate('answer', 'stamp');
  }

  private stopSlow(): void {
    if (!this.slow) return;
    window.clearTimeout(this.slow.timer);
    this.slow = null;
    this.hoverStep = null;
    this.invalidate('answer', 'stamp');
  }

  /** The answer on screen: the one being written slowly, else the latest. */
  private shown(): AnswerMsg | undefined {
    return this.slow?.answer ?? this.answer;
  }

  /** How many letters of the answer on screen are visible (all, unless written slowly). */
  private shownSteps(): number | undefined {
    return this.slow && !this.slow.done ? this.slow.shown : undefined;
  }

  /** Ask the worker about the current question (at most one request in flight). */
  private ask(): void {
    if (!this.ready || !this.question.text.trim()) return;
    if (this.askInFlight) { this.askDirty = true; return; }
    this.askDirty = false;
    this.askInFlight = true;
    this.sentId = ++this.askId;
    this.send({ type: 'ask', id: this.sentId, ...this.question });
  }

  /**
   * Ask the worker for what the visible tab shows inside the model: the
   * weight heatmaps (Inside tab) or the letter map (Model learning tab). At most one
   * request in flight and one every INSPECT_MS; `now` skips the wait.
   */
  private requestInspect(now = false): void {
    if (!this.ready) return;
    if (this.inspectPending !== null) { this.inspectDirty = true; return; }
    const wait = this.lastInspect + INSPECT_MS - performance.now();
    if (wait > 0 && !now) {
      if (!this.inspectTimer) this.inspectTimer = window.setTimeout(() => { this.inspectTimer = 0; this.requestInspect(); }, wait);
      return;
    }
    this.inspectDirty = false;
    this.lastInspect = performance.now();
    if (this.tab !== 'learn' && this.tab !== 'scan' && this.tab !== 'numbers') return; // nothing inside the model to show there
    const view = this.tab === 'scan' ? this.inside : this.tab === 'numbers' ? this.numbers : null;
    const matrixIds = view ? view.wanted() : [];
    if (view && !matrixIds.length) return;
    this.inspectPending = ++this.inspectId;
    this.send({ type: 'inspect', id: this.inspectId, matrixIds, mode: view?.mode ?? this.inside.mode, letterMap: !view });
  }

  private inspectDone(id: number): void {
    if (id !== this.inspectPending) return;
    this.inspectPending = null;
    if (this.inspectDirty) this.requestInspect();
  }

  private onLetterMap(msg: LetterMapMsg): void {
    if (!this.ready) return;
    const tokens = this.ready.tokens;
    const points = msg.ids.map((id, i) => ({ token: tokens[id], label: letterLabel(tokens[id]), kind: letterKind(tokens[id]), count: msg.counts[i] }));
    this.letterMap.update(points, msg.coords, msg.near);
    this.els.mapNote.textContent = t('map.note', { p: Math.round((msg.explained[0] + msg.explained[1]) * 100) });
  }

  /** The line under the letter map: a hint, or the clicked letter's closest letters (clickable). */
  private renderMapPick(sel: { point: LetterPoint; near: LetterPoint[] } | null): void {
    const el = this.els.mapPick;
    if (!sel) { el.replaceChildren(h('span', { class: 'hint' }, t('map.pick'))); return; }
    const chip = (p: LetterPoint) => {
      const b = h('button', { class: 'map-chip', title: p.token }, p.label);
      b.onclick = () => this.letterMap.select(p.token);
      return b;
    };
    const clear = h('button', { class: 'map-clear', title: t('map.clear'), 'aria-label': t('map.clear') }, '×');
    clear.onclick = () => this.letterMap.select(null);
    const count = sel.point.count ?? -1;
    el.replaceChildren(
      h('span', {}, ...t('map.near').split('{letter}').flatMap((part, i) => (i ? [h('b', { class: 'map-chip on' }, sel.point.label), part] : [part]))),
      ' ', ...sel.near.map(chip), clear,
      ...(count >= 0 && count < RARE_LETTER ? [h('p', { class: 'map-rare', 'data-test': 'map-rare' }, t('map.rare', { letter: sel.point.label, n: count }))] : []));
  }

  /** Look inside letter `position` of the answer, on the "How it picks a letter" tab. */
  private lookAtLetter(position: number): void {
    this.letter.explainAt(position);
    this.setTab('letter');
    this.letter.scrollToExplain();
  }

  private setTab(tab: Tab): void {
    if (EXPERT_TABS.includes(tab) && !this.expert) tab = 'learn';
    this.tab = tab;
    for (const b of this.els.tabs.querySelectorAll('button')) {
      b.classList.toggle('on', b.dataset.tab === tab);
      b.setAttribute('aria-selected', String(b.dataset.tab === tab));
    }
    this.els.dataMain.hidden = tab !== 'data';
    this.els.learnMain.hidden = tab !== 'learn';
    this.els.askMain.hidden = tab !== 'ask';
    this.els.matrixMain.hidden = tab !== 'matrix';
    this.els.insideMain.hidden = tab !== 'scan';
    this.els.letterMain.hidden = tab !== 'letter';
    this.matrix.setVisible(tab === 'matrix');
    this.els.testMain.hidden = tab !== 'test';
    this.els.numbersMain.hidden = tab !== 'numbers';
    if (tab === 'numbers' && this.tick) this.numbers.setTick(this.tick);
    this.send({ type: 'watchTests', on: tab === 'test' });
    this.send({ type: 'watchPractice', on: tab === 'learn' });
    if (tab === 'test') this.invalidate('scores', 'animals');
    if (tab === 'learn') { this.invalidate('practice', 'loss'); this.letterMap.draw(); }
    if (tab === 'ask') this.invalidate('answer', 'stamp');
    if (tab === 'letter') this.letter.shown();
    this.requestInspect(true);
  }

  private showError(message: string): void {
    this.els.error.textContent = t('workerError', { message });
    this.els.error.hidden = false;
  }

  // -------------------------------------------------------------------------
  // Layout
  // -------------------------------------------------------------------------

  private build() {
    const play = h('button', { class: 'primary', 'data-test': 'play' });
    play.onclick = () => this.send({ type: this.tick?.running ? 'pause' : 'play' });
    const step1 = h('button', {}, t('step1'));
    const stepOnce = () => {
      // Paused: show this one step slowly (the four-stage cycle).
      if (!this.tick?.running) this.cycleWanted = true;
      this.send({ type: 'step', n: 1 });
    };
    step1.onclick = stepOnce;
    const step10 = h('button', {}, t('step10'));
    step10.onclick = () => this.send({ type: 'step', n: 10 });
    const speed = h('select', { 'aria-label': t('speed') });
    for (const s of SPEEDS) speed.append(h('option', { value: String(s) }, s === 'max' ? t('speedMax') : t('speedSteps', { n: s })));
    speed.value = 'max';
    speed.onchange = () => this.send({ type: 'setSpeed', speed: speed.value === 'max' ? 'max' : Number(speed.value) });
    const goal = h('select', { 'aria-label': t('goal.label'), 'data-test': 'goal' });
    for (const g of GOALS) goal.append(h('option', { value: String(g) }, g === null ? t('goal.none') : t('goal.option', { n: g.toLocaleString(uiLang()) })));
    goal.value = String(DEFAULT_GOAL);
    goal.onchange = () => this.send({ type: 'setGoal', steps: goal.value === 'null' ? null : Number(goal.value) });
    const goalBar = h('progress', { class: 'goal-bar', max: 1, value: 0, 'data-test': 'goal-bar' });
    const goalNote = h('div', { class: 'goal-note small', 'data-test': 'goal-note', 'aria-live': 'polite' });
    const seed = h('input', { type: 'number', min: 1, step: 1, class: 'seed', 'aria-label': t('seed') });
    const reset = h('button', {}, t('reset'));
    reset.onclick = () => this.send({ type: 'reset', seed: Number(seed.value) || 1 });
    const quietBox = h('input', { type: 'checkbox', 'data-test': 'quiet' });
    quietBox.onchange = () => this.send({ type: 'quiet', on: quietBox.checked });
    const quietLabel = h('label', { class: 'row small quiet-toggle', title: t('quiet.help') }, quietBox, ' ', t('quiet.label'));
    const quietNote = h('p', { class: 'quiet-note small', 'data-test': 'quiet-note' }, t('quiet.note'));
    const expertBox = h('input', { type: 'checkbox', 'data-test': 'expert' });
    expertBox.onchange = () => {
      this.expert = expertBox.checked;
      expertPanel.hidden = !this.expert;
      stepLink.hidden = !this.expert;
      for (const b of learnMain.querySelectorAll<HTMLElement>('.see-change')) b.hidden = !this.expert;
      for (const b of this.els.tabs.querySelectorAll<HTMLElement>('button')) {
        if (EXPERT_TABS.includes(b.dataset.tab as Tab)) b.hidden = !this.expert;
      }
      // Leaving expert mode on an expert tab: go to the nearest plain one.
      if (!this.expert && EXPERT_TABS.includes(this.tab)) this.setTab('learn');
      this.inside.setExpert(this.expert);
      this.letter.setExpert(this.expert);
      this.dataTab.setExpert(this.expert);
      this.journey.setExpert(this.expert);
      this.matrix.setExpert(this.expert);
      this.invalidate('loss', 'answer');
      this.requestInspect(true);
    };
    const lr = h('input', { type: 'number', step: 0.0005, min: 0.0001, max: 0.05, value: 0.003 });
    lr.onchange = () => this.send({ type: 'setHyper', lr: Number(lr.value) });
    const batch = h('input', { type: 'number', step: 1, min: 1, max: 128, value: 16 });
    batch.onchange = () => this.send({ type: 'setHyper', batchSize: Number(batch.value) });
    // Expert: the whole calculation of one step is on "The numbers" tab.
    const stepLinkBtn = h('button', { class: 'link', 'data-test': 'step-link' }, t('nums.stepLink'));
    stepLinkBtn.onclick = () => { this.setTab('numbers'); this.numbers.scrollToStep(); };
    const stepLink = h('section', { class: 'panel phase build step-link', hidden: true },
      h('p', { class: 'small' }, t('nums.stepLinkHelp'), ' ', stepLinkBtn));
    const expertPanel = h('div', { class: 'expert', hidden: true },
      h('label', {}, t('lr'), ' ', lr), h('label', {}, t('batch'), ' ', batch));
    const status = h('div', { class: 'status', 'data-test': 'status' });
    // On the Ask tab: which model answers, and whether it is learning (no training controls there).
    // Shown in the Insights block until there is an answer (then the stamp says which model answered).
    const askStatusText = h('span', { class: 'ask-status', 'data-test': 'ask-status' });
    const goLearn = h('button', { class: 'link small', 'data-test': 'go-learn' }, t('askStatus.go'));
    goLearn.onclick = () => this.setTab('learn');

    const input = h('input', { type: 'text', id: 'ask-word', class: 'ask', placeholder: t('askPlaceholder'), spellcheck: 'false', autocomplete: 'off', 'data-test': 'ask' });
    input.oninput = () => this.setQuestion({ text: input.value }, true);
    input.onkeydown = (e) => { if (e.key === 'Enter' && !e.isComposing && !e.repeat) this.answerNow(false); };
    const langPick = h('div', { class: 'picker', role: 'group', 'aria-label': t('language') });
    const formPick = h('div', { class: 'picker', role: 'group', 'aria-label': t('form') });
    // In the Insights block: is the word an animal from the data, in this language? (before answering)
    const hint = h('span', { class: 'hint word-hint', 'data-test': 'word-hint', 'aria-live': 'polite' });
    const answerNowBtn = h('button', { class: 'primary big', 'data-test': 'answer-now', title: t('ask.nowTip') }, t('ask.now'));
    answerNowBtn.onclick = () => this.answerNow(false);
    const slowBtn = h('button', { class: 'big', 'data-test': 'answer-slow', title: t('ask.slowTip') }, t('ask.slow'));
    slowBtn.onclick = () => this.answerNow(true);
    const field = (label: string, control: HTMLElement, ...rest: HTMLElement[]) =>
      h('div', { class: 'ask-field' }, h('span', { class: 'ask-label' }, label), h('div', { class: 'ask-control' }, control, ...rest));
    const askForm = h('div', { class: 'ask-form' },
      field(t('ask.step.lang'), langPick),
      field(t('ask.step.word'), input),
      field(t('ask.step.form'), formPick),
      h('div', { class: 'ask-buttons' }, answerNowBtn, slowBtn));
    const prompt = h('p', { class: 'ask-prompt', 'data-test': 'ask-prompt' }, t('ask.prompt'));
    const output = h('div', { class: 'output', 'data-test': 'output' });
    // Built once: it updates on every tick, and a rebuilt button would swallow clicks.
    const stampText = h('span');
    const askAgain = h('button', { class: 'small-btn', 'data-test': 'ask-again', hidden: true }, t('stamp.askAgain'));
    askAgain.onclick = () => this.ask();
    const stamp = h('div', { class: 'stamp', 'data-test': 'stamp', 'aria-live': 'polite' }, stampText, ' ', askAgain);
    // Insights: everything said about the question and its answer (the word,
    // what kind of answer, the real answer, how sure, which model), in one block.
    const wordRow = insightRow(t('insights.word'), hint);
    wordRow.hidden = true;
    const insightRows = h('div', { class: 'insight-rows' });
    const insights = h('section', { class: 'insights', 'data-test': 'insights' },
      h('h3', {}, t('insights.title')),
      wordRow,
      insightRows,
      insightRow(t('insights.model'), askStatusText, stamp, ' ', goLearn));
    const tokenRows = h('div', { class: 'token-rows', 'data-test': 'tokens' });
    const tokenNumbers = h('input', { type: 'checkbox', 'data-test': 'token-numbers' });
    tokenNumbers.onchange = () => { this.tokenNumbers = tokenNumbers.checked; this.invalidate('answer'); };
    const tokenNote = h('p', { class: 'small' });
    const tokensPanel = h('section', { class: 'panel tokens-panel' },
      h('div', { class: 'row guess-head' }, h('h2', {}, t('tokens.title')), help(t('tokens.help'))),
      tokenRows,
      h('div', { class: 'row small' }, h('label', {}, tokenNumbers, ' ', t('tokens.numbers'))),
      tokenNote);
    const bars = h('canvas', { class: 'bars', 'data-test': 'bars' });
    bars.onmousemove = (e) => {
      const steps = this.shown()?.steps ?? [];
      const col = guessColumnAt(e.offsetX, steps.length);
      if (!this.slow) this.setHover(col);
      // Short labels (✱, ⏎, ?) get their full name in the tooltip.
      const specials = col === null ? [] : [...new Set([steps[col].chosen, ...steps[col].top.map((g) => g.char)])].filter((c) => shortLabel(c) !== c);
      bars.title = specials.map((c) => `${shortLabel(c)} = ${this.fullName(c)}`).join('\n');
    };
    bars.onmouseleave = () => { if (!this.slow) this.setHover(null); };
    // A click on a letter's column looks inside the model while it writes that letter.
    bars.onclick = (e) => {
      const steps = this.shown()?.steps ?? [];
      const col = guessColumnAt(e.offsetX, steps.length);
      if (col === null || this.slow) return;
      this.lookAtLetter(col);
    };
    const barsWrap = h('div', { class: 'bars-wrap' }, bars);
    // Guesses by letter, in its own block under the tokens: hidden until there is an answer.
    const guessPanel = h('section', { class: 'panel guess-panel' },
      h('div', { class: 'row guess-head' }, h('h2', {}, t('guessBars'))),
      barsWrap,
      h('p', { class: 'small', 'data-test': 'bars-hint' }, t('ask.lookInside')));
    // Everything about one answer (tokens, guesses): hidden until there is an answer on screen.
    const results = [tokensPanel, guessPanel];
    const slowCaption = h('div', { class: 'slow-caption', 'data-test': 'slow-caption', 'aria-live': 'polite' });
    const learning = h('div', { class: 'learning off', 'data-test': 'learning' });
    const makeCycle = (test: string) => {
      const stages = CYCLE_STAGES.map((k, i) => h('li', { class: 'stage', 'data-stage': k },
        h('span', { class: 'num' }, String(i + 1)),
        h('span', { class: 'what' }),
        h('span', { class: 'val' })));
      // Expert only: the scan ("Where it learns") is an expert tab.
      const seeChange = h('button', { class: 'link see-change', hidden: !this.expert }, t('cycle.seeChange'));
      seeChange.onclick = () => { this.inside.setMode('delta'); this.setTab('scan'); };
      stages[3].append(seeChange);
      return { el: h('ol', { class: 'cycle', 'data-test': test, 'aria-label': t('cycle.title') }, ...stages), stages, note: h('p', { class: 'small cycle-note' }) };
    };
    const cycleA = makeCycle('cycle');
    const practiceList = h('div', { class: 'practice-table', 'data-test': 'practice' });
    const practiceHelp = help(t('practice.help', { shown: PRACTICE_SHOWN, batch: 16 }));
    const attention = h('div', { class: 'att-grids', 'data-test': 'attention' });
    const attPrompt = h('p', { class: 'ask-prompt small', hidden: true }, t('att.prompt'));
    const mapCanvas = h('canvas', { class: 'letter-map', 'data-test': 'letter-map' });
    const mapNote = h('p', { class: 'small map-note' });
    const mapPick = h('div', { class: 'map-pick', 'data-test': 'map-pick', 'aria-live': 'polite' });
    const mapLegend = h('div', { class: 'map-legend' }, ...(Object.keys(LETTER_KINDS) as LetterKind[]).map((k) => {
      const c = h('canvas');
      const ctx = prepare(c, 16, 16);
      ctx.fillStyle = LETTER_KINDS[k].color + '22';
      ctx.strokeStyle = LETTER_KINDS[k].color;
      ctx.lineWidth = 1.5;
      marker(ctx, LETTER_KINDS[k].shape, 8, 8, k === 'task' ? 6 : 5.5);
      return h('span', {}, c, t(`map.kind.${k}`));
    }));
    const tabs = h('nav', { class: 'tabs', role: 'tablist' }, ...TABS.map((tab) => {
      const expertTab = EXPERT_TABS.includes(tab);
      const b = h('button', {
        role: 'tab', 'data-tab': tab, 'data-test': `tab-${tab}`, hidden: expertTab && !this.expert,
        class: [tab === this.tab ? 'on' : '', expertTab ? 'expert-tab' : ''].filter(Boolean).join(' '),
        title: expertTab ? t(`tab.${tab}Tip`) : '',
      }, t(`tab.${tab}`));
      b.onclick = () => this.setTab(tab);
      return b;
    }));
    const lossCanvas = h('canvas');
    const scoreCanvas = h('canvas');
    const scoreTable = h('div', { class: 'score-table' });
    const filters = h('div', { class: 'filters' });
    for (const f of ['all', 'known', 'unseen', 'wrong'] as Filter[]) {
      const b = h('button', { 'data-filter': f }, t(`filter.${f}`));
      b.onclick = () => { this.filter = f; this.invalidate('animals'); };
      filters.append(b);
    }
    const animals = h('div', { class: 'animals', 'data-test': 'animals' });
    const testLangPick = h('div', { class: 'picker', role: 'group', 'aria-label': t('language') });
    const testWhen = h('span', { class: 'small', 'data-test': 'test-when' });
    const testAgain = h('button', { 'data-test': 'test-again' }, t('test.again'));
    testAgain.onclick = () => this.send({ type: 'evaluate', inventions: true });
    const testLegend = h('div', { class: 'test-legend small' },
      ...(['correct', 'close', 'wrong', 'invented'] as const).map((k) => h('span', { class: `cell ${k}` }, `${STATUS_ICONS[k]} ${t(`status.${k}`)}`)));
    const params = h('div', { class: 'params' });
    const uiSwitch = h('button', { class: 'link' }, t('uiLanguage'));
    uiSwitch.onclick = () => {
      setUiLang(UI_LANGUAGES.find((l) => l !== uiLang()) ?? 'en');
      location.reload();
    };
    const error = h('div', { class: 'error', hidden: true });
    const learnMain = h('main', { class: 'learn-main', hidden: this.tab !== 'learn' },
      h('aside', { class: 'left' },
        h('section', { class: 'panel controls phase build' },
          h('div', { class: 'phase-head' }, h('h2', {}, t('build.title')), help(t('build.help'))),
          h('div', { class: 'row' }, play, step1, step10),
          h('label', { class: 'row' }, t('speed'), ' ', speed),
          h('div', { class: 'row' }, h('label', {}, t('goal.label'), ' ', goal), help(t('goal.help'))),
          quietLabel,
          h('div', { class: 'row' }, h('label', {}, t('seed'), ' ', seed), reset),
          expertPanel,
          status,
          goalBar,
          goalNote,
          learning,
          h('div', { class: 'phase-head sub' }, h('h3', {}, t('cycle.title'))),
          cycleA.el,
          cycleA.note),
        h('section', { class: 'panel' }, h('h2', {}, t('loss')), h('div', { class: 'chart' }, lossCanvas))),
      h('section', { class: 'centre' },
        h('section', { class: 'panel phase build' },
          h('div', { class: 'phase-head' }, h('h2', {}, t('practice.title')), practiceHelp),
          h('p', { class: 'practice-explain' }, t('practice.explain')),
          quietNote,
          practiceList),
        stepLink),
      h('aside', { class: 'right' },
        h('section', { class: 'panel map' },
          h('div', { class: 'phase-head' }, h('h2', {}, t('map.title')), help(t('map.how'))),
          h('p', { class: 'map-help', 'data-test': 'map-help' }, t('map.help')),
          h('div', { class: 'map-wrap' }, mapCanvas),
          mapPick,
          mapLegend,
          mapNote)));
    const askMain = h('main', { class: 'ask-main', hidden: this.tab !== 'ask' },
      h('section', { class: 'panel askbox phase ask' },
        h('div', { class: 'phase-head' }, h('h2', {}, t('ask.title')), help(t('ask.help'))),
        askForm,
        slowCaption,
        prompt,
        output),
      // Row 1: asking and Insights. Row 2: tokens, and "Where it looks" (going on
      // down beside row 3). Row 3: guesses by letter.
      insights,
      tokensPanel,
      h('section', { class: 'panel attention' },
        h('h2', {}, t('att.title')),
        h('p', { class: 'small' }, t('att.help')),
        attPrompt,
        attention),
      guessPanel,
      this.journey.el);
    const matrixMain = h('main', { class: 'matrix-main', hidden: this.tab !== 'matrix' }, this.matrix.el);
    const insideMain = h('main', { class: 'inside-main', hidden: this.tab !== 'scan' }, this.inside.el);
    const letterMain = h('main', { class: 'inside-main', hidden: this.tab !== 'letter' }, this.letter.el);
    const dataMain = h('main', { class: 'data-main', hidden: this.tab !== 'data' }, this.dataTab.el);
    const numbersMain = h('main', { class: 'inside-main numbers-main', hidden: this.tab !== 'numbers' }, this.numbers.el);
    const testMain = h('main', { class: 'test-main', hidden: this.tab !== 'test' },
      h('section', { class: 'panel test-intro' },
        h('div', { class: 'phase-head' }, h('h2', {}, t('test.title')), help(t('test.help'))),
        h('p', { class: 'practice-explain' }, t('test.explain')),
        h('div', { class: 'row' }, testLangPick, testAgain, testWhen)),
      h('div', { class: 'test-grid' },
        h('section', { class: 'panel list' }, h('h2', {}, t('animals')), testLegend, filters, animals),
        h('section', { class: 'panel' }, h('h2', {}, t('scoreboard')), h('p', { class: 'small' }, t('test.scoreHelp')),
          h('div', { class: 'chart' }, scoreCanvas), scoreTable)));

    this.root.replaceChildren(
      h('header', {}, h('h1', {}, t('title')), h('span', { class: 'sub' }, t('subtitle')), tabs, params,
        h('label', { class: 'small expert-toggle' }, expertBox, ' ', t('expert')), uiSwitch),
      error,
      dataMain,
      learnMain,
      askMain,
      matrixMain,
      insideMain,
      letterMain,
      testMain,
      numbersMain,
    );
    return {
      play, speed, seed, status, goalBar, goalNote, askStatusText, input, langPick, formPick, hint, output, bars, barsWrap, tokenRows, tokenNote, attention, mapCanvas, mapNote, mapPick, tabs,
      dataMain, learnMain, askMain, matrixMain, insideMain, letterMain, testMain, numbersMain, testLangPick, testWhen,
      stamp, stampText, askAgain, insights, insightRows, wordRow, slowBtn, answerNowBtn, slowCaption, prompt, results, attPrompt, practiceList, practiceHelp,
      playButtons: [play], learnings: [learning], cycles: [cycleA],
      lossCanvas, scoreCanvas, scoreTable, filters, animals, params, error,
    };
  }

  /** Language and form pickers (built from the data, never hard-coded). */
  private renderPickers(): void {
    if (!this.ready) return;
    const { langPick, formPick } = this.els;
    langPick.replaceChildren(...this.ready.languages.map((l) => {
      const b = h('button', { 'data-lang': l.code }, l.name[uiLang()] ?? l.code);
      b.onclick = () => this.setQuestion({ lang: l.code });
      return b;
    }));
    formPick.replaceChildren(...this.ready.forms.map((f) => {
      const b = h('button', { 'data-form': f, title: t(`form.tip.${f}`) }, t(`form.${f}`));
      b.onclick = () => this.setQuestion({ form: f });
      return b;
    }));
    this.els.params.textContent = t('params', { n: this.ready.params.toLocaleString(uiLang()) });
    this.invalidate('answer');
  }

  /** A word clicked on another tab: ask it on the Ask tab. */
  private askOnAskTab(q: Partial<App['question']>): void {
    this.setTab('ask');
    this.setQuestion(q, false, true);
  }

  /**
   * A new question (language, word or form). It is answered only when asked:
   * by the Answer buttons, or right away for a word clicked on another tab.
   */
  private setQuestion(q: Partial<App['question']>, typed = false, answer = false): void {
    Object.assign(this.question, q);
    if (!typed) this.els.input.value = this.question.text;
    this.revealed = false;
    this.stopSlow();
    this.slowWanted = false;
    // Any reply on its way was for another question: drop it.
    this.askId++;
    this.askDirty = false;
    if (answer) this.ask();
    else if (this.answer) {
      // The answer on screen was for another question: take it away (on the letter tab too).
      this.answer = undefined;
      this.letter.setAnswer(undefined);
      this.journey.setAnswer(undefined);
      this.matrix.setAnswer(undefined);
    }
    this.invalidate('answer', 'stamp', 'animals');
  }

  /** The typed word: an animal of the chosen language, of another language, or not from the data. */
  private wordKind(): { kind: 'empty' | 'known' | 'free' | 'other'; lang?: string } {
    const w = Vocab.normalize(this.question.text);
    if (!w || !this.ready) return { kind: 'empty' };
    const animals = this.ready.animals;
    if (animals.some((a) => a.names[this.question.lang] === w)) return { kind: 'known' };
    const other = this.ready.languages.find((l) => animals.some((a) => a.names[l.code] === w));
    return other ? { kind: 'other', lang: other.code } : { kind: 'free' };
  }

  // -------------------------------------------------------------------------
  // Rendering (at most once per animation frame, only what changed)
  // -------------------------------------------------------------------------

  private invalidate(...parts: (keyof App['dirty'])[]): void {
    for (const p of parts) this.dirty[p] = true;
    if (!this.frame) this.frame = requestAnimationFrame(() => this.render());
  }

  private render(): void {
    this.frame = 0;
    const d = this.dirty;
    if (d.controls) this.renderControls();
    if (d.answer) this.renderAnswer();
    if (d.loss) this.renderLoss();
    if (d.scores) this.renderScores();
    if (d.animals) this.renderAnimals();
    if (d.practice) this.renderPractice();
    if (d.stamp) this.renderStamp();
    this.dirty = { controls: false, answer: false, loss: false, scores: false, animals: false, practice: false, stamp: false };
  }

  private renderControls(): void {
    const tk = this.tick;
    for (const b of this.els.playButtons) b.textContent = tk?.running ? t('pause') : t('play');
    if (!tk) return;
    const parts = [t('stepLabel', { step: tk.step.toLocaleString(uiLang()) })];
    if (tk.stepsPerSec > 0) parts.push(t('stepsPerSec', { n: tk.stepsPerSec.toFixed(0) }));
    if (tk.running && tk.wallStepsPerSec > 0) parts.push(t('wallPerSec', { n: tk.wallStepsPerSec.toFixed(0) }));
    if (tk.lossEma !== null) parts.push(`loss ${tk.lossEma.toFixed(3)}`);
    this.els.status.textContent = parts.join(' · ');
    this.renderGoal(tk);
    this.els.askStatusText.textContent = t(tk.running ? 'askStatus.learning' : 'askStatus.paused', { step: tk.step.toLocaleString(uiLang()) });
  }

  /** "Learn until": progress towards the step goal, time left, and Play off once it is reached. */
  private renderGoal(tk: TickMsg): void {
    const { goalBar, goalNote, play } = this.els;
    const reached = tk.goal !== null && tk.step >= tk.goal && !tk.running;
    play.disabled = reached;
    play.title = reached ? t('goal.reached', { n: tk.goal!.toLocaleString(uiLang()) }) : '';
    goalBar.hidden = tk.goal === null;
    if (tk.goal === null) { goalNote.textContent = ''; return; }
    goalBar.value = Math.min(1, tk.step / tk.goal);
    if (reached) { goalNote.textContent = t('goal.reached', { n: tk.goal.toLocaleString(uiLang()) }); return; }
    const parts = [t('goal.of', { step: tk.step.toLocaleString(uiLang()), goal: tk.goal.toLocaleString(uiLang()) })];
    if (tk.running && tk.wallStepsPerSec > 0) parts.push(t('goal.left', { t: duration((tk.goal - tk.step) / tk.wallStepsPerSec) }));
    goalNote.textContent = parts.join(' · ');
  }

  private renderAnswer(): void {
    const { input, langPick, formPick, hint, output, bars } = this.els;
    if (document.activeElement !== input) input.value = this.question.text;
    for (const b of langPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.lang === this.question.lang);
    for (const b of formPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.form === this.question.form);
    this.renderWordHint(hint);
    const hasText = !!this.question.text.trim();
    this.els.answerNowBtn.disabled = !hasText;
    this.els.slowBtn.disabled = !hasText && !this.slow;
    this.els.slowBtn.textContent = this.slow ? t('slow.stop') : t('ask.slow');
    this.els.slowBtn.classList.toggle('on', !!this.slow);
    const a = this.shown();
    // Only a request for the current question counts: a dropped one will not answer.
    const waiting = !a && ((this.askInFlight && this.sentId === this.askId) || this.askDirty);
    this.els.prompt.hidden = !!a;
    this.els.prompt.textContent = waiting ? t('ask.thinking') : t('ask.prompt');
    for (const el of this.els.results) el.hidden = !a;
    this.els.attPrompt.hidden = !!a;
    this.els.attention.hidden = !a;
    if (!a || !this.ready) {
      output.replaceChildren();
      this.els.insightRows.replaceChildren();
      this.els.slowCaption.replaceChildren();
      this.els.slowCaption.classList.remove('active');
      return;
    }
    const writing = this.shownSteps();

    // A hidden answer is not judged on screen until revealed: the colour would give it away.
    const wordClass = (a.badge === 'hidden' && !this.revealed) || writing !== undefined ? 'unrevealed' : a.answer ? a.answer.status : 'invented';
    const word = writing !== undefined
      ? h('span', { class: `word ${wordClass}` }, a.steps.slice(0, writing).map((s) => s.chosen).join(''), h('span', { class: 'cursor' }, '▌'))
      : h('span', { class: `word ${wordClass}` }, a.output || '∅');
    output.replaceChildren(h('div', { class: 'answer-line' }, h('span', { class: 'from' }, `${a.text} →`), word));
    this.renderInsights(a, writing);
    if (this.hoverStep !== null && this.hoverStep >= a.steps.length) this.hoverStep = null;
    drawGuessBars(bars, a.steps, this.hoverStep, writing, shortLabel);
    this.journey.follow(this.slow && a === this.slow.answer ? (writing ?? null) : null);
    this.renderTokens(a, writing);
    this.renderSlowCaption(a);
    this.renderAttention();
  }

  /**
   * Insights under the answer: what kind of word it is, the real answer and
   * whether the model got it, how sure it is and whether it finished. The
   * model row (the stamp) is built once and stays below these rows.
   */
  private renderInsights(a: AnswerMsg, writing: number | undefined): void {
    const rows: HTMLElement[] = [insightRow(t('insights.kind'), h('span', { class: `kind ${a.badge}` }, t(`badge.${a.badge}`)))];
    if (a.answer) {
      if (a.badge === 'hidden' && !this.revealed) {
        const reveal = h('button', { class: 'small-btn', 'data-test': 'reveal' }, t('reveal'));
        reveal.onclick = () => { this.revealed = true; this.invalidate('answer'); };
        rows.push(insightRow(t('insights.answer'), h('span', { class: 'muted' }, t('insights.hiddenAnswer')), ' ', reveal));
      } else {
        // While the model is still writing, the verdict would give the end away.
        const { status, word } = a.answer;
        const verdict = writing !== undefined ? [] : [' ', h('span', { class: `verdict ${status}` }, `${STATUS_ICONS[status]} ${t(`status.${status}`)}`)];
        rows.push(insightRow(t('insights.answer'), h('strong', { class: 'real-word' }, word), ...verdict));
      }
    }
    const sure = Math.round(a.confidence * 100);
    rows.push(insightRow(t('confidence'), h('span', { class: 'meter', title: t('confidenceHelp') },
      h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${sure}%` })),
      h('span', {}, `${sure}`))));
    if (!a.finished && writing === undefined) rows.push(insightRow(t('insights.end'), t('notFinished')));
    this.els.insightRows.replaceChildren(...rows);
  }

  /**
   * Token strip: the question as the model reads it (task token + letters)
   * and the answer as it writes it (letters + end), each with its number.
   * Answer tokens line up with the guess-bar columns (hover links them).
   */
  private renderTokens(a: AnswerMsg, writing: number | undefined): void {
    const idx = this.tokenIdx;
    const opts = { expert: this.expert, numbers: this.tokenNumbers, langName: (c: string) => this.langName(c) };
    const reads = a.input.map((tok) => tokenChip(toTok(tok, idx), opts));
    const steps = writing !== undefined ? a.steps.slice(0, writing) : a.steps;
    const writes = steps.map((s, i) => {
      const chip = tokenChip(toTok(s.chosen || EOS, idx), opts, i === this.hoverStep ? 'hover' : '');
      chip.dataset.step = String(i);
      chip.onmouseenter = () => { if (!this.slow) this.setHover(i); };
      chip.onmouseleave = () => { if (!this.slow) this.setHover(null); };
      return chip;
    });
    const row = (label: string, chips: HTMLElement[], test: string) =>
      h('div', { class: 'token-row', 'data-test': test }, h('span', { class: 'token-label small' }, label), h('span', { class: 'chips' }, ...chips));
    const unknown = [...new Set([...a.text].filter((c) => !idx.has(c)))];
    this.els.tokenRows.replaceChildren(
      row(t('tokens.reads'), reads, 'tokens-in'),
      row(t('tokens.writes'), writes, 'tokens-out'),
      ...(unknown.length ? [h('p', { class: 'tok-warn', 'data-test': 'tokens-unk' }, t('tokens.unk', { chars: unknown.map((c) => `“${c}”`).join(', ') }))] : []));
    this.els.tokenNote.textContent = t('tokens.note', { n: idx.size });
  }

  /** Link a guess-bar column, the answer token and the matching attention rows (null = none). */
  private setHover(step: number | null): void {
    if (step === this.hoverStep) return;
    this.hoverStep = step;
    for (const c of this.els.tokenRows.querySelectorAll<HTMLElement>('[data-step]')) c.classList.toggle('hover', c.dataset.step === String(step));
    const a = this.shown();
    if (a) drawGuessBars(this.els.bars, a.steps, step, this.shownSteps(), shortLabel);
    this.drawAttention();
  }

  /** Language code -> its name in the UI language. */
  private langName(code: string): string {
    return this.ready?.languages.find((l) => l.code === code)?.name[uiLang()] ?? code;
  }

  /** A token's full name for tooltips: the letter, or e.g. "Czech, female" ('' = end). */
  private fullName(token: string): string {
    return tokenName(token || EOS, { expert: this.expert, langName: (c) => this.langName(c) });
  }

  /** Which attention grids to show for this answer (expert mode adds the self-attention). */
  private attentionGrids(a: AnswerMsg): AttGrid[] {
    // Short labels fit the cells; the full names go in the tooltip.
    const input = a.input.map(shortLabel);
    const out = a.steps.map((s) => shortLabel(s.chosen));
    const decIn = ['✱', ...out];
    const inputNames = a.input.map((tok) => this.fullName(tok));
    const outNames = a.steps.map((s) => this.fullName(s.chosen));
    const decInNames = [this.fullName(a.input[0] ?? ''), ...outNames];
    const grids: AttGrid[] = [];
    const add = (name: string, title: (head: number) => string, rows: string[], cols: string[], linked: boolean, rowNames: string[], colNames: string[]) => {
      const m = a.attention.find((x) => x.name === name);
      if (!m) return;
      for (let head = 0; head < m.heads; head++) {
        grids.push({
          key: `${name}#${head}`, title: title(head), linked,
          rowNames: rowNames.slice(0, m.Tq), colNames: colNames.slice(0, m.Tk),
          spec: { rows: rows.slice(0, m.Tq), cols: cols.slice(0, m.Tk), data: m.data, offset: head * m.Tq * m.Tk, stride: m.Tk },
        });
      }
    };
    const layers = [...new Set(a.attention.map((x) => x.name.match(/^dec(\d+)\./)?.[1]).filter((x): x is string => x !== undefined))];
    const layerNote = (l: string) => (layers.length > 1 ? ` · ${t('att.layer', { n: Number(l) + 1 })}` : '');
    for (const l of layers) add(`dec${l}.cross`, (hd) => t('att.cross', { n: hd + 1 }) + layerNote(l), out, input, true, outNames, inputNames);
    if (this.expert) {
      for (const l of layers) add(`dec${l}.self`, (hd) => t('att.decSelf', { n: hd + 1 }) + layerNote(l), out, decIn, true, outNames, decInNames);
      for (const name of a.attention.map((x) => x.name).filter((n) => /^enc\d+\.self$/.test(n))) {
        add(name, (hd) => t('att.encSelf', { n: hd + 1 }), input, input, false, inputNames, inputNames);
      }
    }
    return grids;
  }

  private renderAttention(): void {
    const a = this.shown();
    if (!a) return;
    const grids = this.attentionGrids(a);
    const keys = grids.map((g) => g.key).join();
    if (keys !== this.attKeys) {
      this.attKeys = keys;
      this.els.attention.replaceChildren(...grids.map((g, i) => {
        const canvas = h('canvas', { 'data-grid': g.key });
        g.canvas = canvas;
        canvas.onmousemove = (e) => {
          const grid = this.attGrids[i];
          const cell = attentionCellAt(grid.spec, e.offsetX, e.offsetY);
          if (grid.linked && !this.slow) this.setHover(cell ? cell.row : null);
          canvas.title = !cell ? ''
            : cell.col === null ? grid.rowNames[cell.row]
            : t('att.cell', { row: grid.rowNames[cell.row], col: grid.colNames[cell.col], p: Math.round(grid.spec.data[grid.spec.offset + cell.row * grid.spec.stride + cell.col] * 100) });
        };
        canvas.onmouseleave = () => { if (this.attGrids[i].linked && !this.slow) this.setHover(null); };
        return h('figure', { class: 'att' }, h('figcaption', {}, g.title), canvas);
      }));
    } else {
      grids.forEach((g, i) => { g.canvas = this.attGrids[i].canvas; });
    }
    this.attGrids = grids;
    this.drawAttention();
  }

  private drawAttention(): void {
    for (const g of this.attGrids) {
      if (!g.canvas) continue;
      drawAttentionGrid(g.canvas, g.spec, g.linked ? this.hoverStep : null, g.linked ? this.shownSteps() : undefined);
    }
  }

  /** Under the word box: what kind of word it is, before the model answers. */
  private renderWordHint(hint: HTMLElement): void {
    const w = this.wordKind();
    // Rebuilt only when it changes: the hint is read aloud (aria-live), and a rebuilt button swallows clicks.
    const key = `${w.kind}|${w.lang ?? ''}|${this.question.lang}|${uiLang()}`;
    if (key === this.hintKey) return;
    this.hintKey = key;
    hint.className = `hint word-hint ${w.kind}`;
    this.els.wordRow.hidden = w.kind === 'empty';
    if (w.kind === 'empty') { hint.replaceChildren(); return; }
    if (w.kind === 'known') { hint.replaceChildren(t('ask.word.known', { lang: this.langName(this.question.lang) })); return; }
    if (w.kind === 'free') { hint.replaceChildren(t('ask.word.free')); return; }
    const lang = w.lang!;
    const sw = h('button', { class: 'link small', 'data-test': 'switch-lang' }, t('ask.word.switch', { lang: this.langName(lang) }));
    sw.onclick = () => this.setQuestion({ lang });
    hint.replaceChildren(t('ask.word.other', { lang: this.langName(lang) }), ' ', sw);
  }

  /**
   * "Answer slowly": what the model is doing right now, in words
   * (reading the animal → weighing letters for letter 3 of 7 → picks one).
   */
  private renderSlowCaption(a: AnswerMsg): void {
    const s = this.slow;
    const el = this.els.slowCaption;
    el.classList.toggle('active', !!s);
    if (!s) { el.replaceChildren(); return; }
    const pctOf = (p: number) => `${Math.round(p * 100)} %`;
    let head: string;
    let detail = '';
    if (s.done) {
      head = t('slow.done', { word: a.output || '∅' });
      detail = t('slow.doneDetail', { n: a.steps.length });
    } else if (s.shown === 0) {
      head = t('slow.reading', { word: a.text });
      detail = t('slow.readingDetail', { n: a.input.length });
    } else {
      const st = a.steps[s.shown - 1];
      // The last guess of a finished word is "end" (⏎); every other guess is one letter.
      const letters = a.steps.length - (a.finished ? 1 : 0);
      head = s.shown > letters ? t('slow.ending') : t('slow.writing', { k: s.shown, n: letters });
      const guesses = st.top.slice(0, 3).map((g) => `${shortLabel(g.char)} ${pctOf(g.p)}`).join(' · ');
      detail = t('slow.picks', { guesses, chosen: shortLabel(st.chosen) });
    }
    el.replaceChildren(h('strong', {}, head), h('span', { class: 'detail' }, detail));
  }

  /**
   * Under the answer: which model (step) answered, and that answering never
   * changes it. While the model learns, the answer is marked as out of date.
   */
  private renderStamp(): void {
    const { stamp, stampText, askAgain, output, insights, barsWrap, attention } = this.els;
    const a = this.shown();
    const tk = this.tick;
    stamp.hidden = !a;
    this.els.askStatusText.hidden = !!a;
    if (!a) return;
    const behind = tk ? tk.step - a.step : 0;
    const stale = !this.slow && behind > 0 && !!tk?.running;
    stampText.textContent = stale
      ? t('stamp.stale', { n: behind.toLocaleString(uiLang()), step: a.step.toLocaleString(uiLang()) })
      : t('stamp.fresh', { step: a.step.toLocaleString(uiLang()) });
    askAgain.hidden = !stale;
    stamp.classList.toggle('stale', stale);
    for (const el of [output, insights, barsWrap, attention]) el.classList.toggle('stale', stale);
    if (this.stampFlash) {
      this.stampFlash = false;
      retrigger(stamp, 'flash');
    }
  }

  /** Learning: the badge, the one-step cycle and the examples of the latest batch. */
  private renderPractice(): void {
    if (!this.ready) return;
    this.practiceShownAt = performance.now();
    const { learnings, cycles, practiceList, practiceHelp } = this.els;
    const p = this.practice;
    const tk = this.tick;
    const n = this.ready.params.toLocaleString(uiLang());
    const phase = this.cycle?.phase ?? CYCLE_STAGES.length;
    const fast = this.fast();

    // Badge: is it learning right now?
    const glowing = !tk?.running && !!p && performance.now() - this.learnedAt < LEARN_GLOW_MS && phase >= NUDGE_PHASE;
    const on = !!tk?.running || glowing || !!this.cycle;
    for (const learning of learnings) {
      learning.textContent = !on ? t('learning.off')
        : fast ? t('learning.fast', { n, sps: Math.round(tk!.stepsPerSec) }) : t('learning.on', { n });
      learning.classList.toggle('on', on);
      learning.classList.toggle('off', !on);
      learning.classList.toggle('busy', on && fast);
      if (this.practiceNew && on && !fast && phase >= NUDGE_PHASE) retrigger(learning, 'pulse');
      if (fast) learning.classList.remove('pulse');
    }
    this.practiceNew = false;

    // The four stages of one step.
    const values = p ? [
      t('cycle.pickVal'),
      p.rightCount === undefined ? '' : t('cycle.answerVal', { right: p.rightCount, n: p.examples.length }),
      t('cycle.measureVal', { loss: p.loss.toFixed(2) }),
      t('cycle.nudgeVal', { n }),
      p.check ? t('cycle.checkVal', { before: p.check.lossBefore.toFixed(2), after: p.check.lossAfter.toFixed(2) }) : t('cycle.checkOff'),
    ] : ['', '', '', '', ''];
    for (const { stages, note } of cycles) {
      stages.forEach((el, i) => {
        // "Check" happens only for a step recorded in detail (Step ×1 while paused).
        const lit = !!p && phase > i && (CYCLE_STAGES[i] !== 'check' || !!p.check);
        el.classList.toggle('off', CYCLE_STAGES[i] === 'check' && !!p && !p.check);
        el.classList.toggle('lit', lit);
        el.classList.toggle('current', !!this.cycle && phase === i + 1);
        (el.querySelector('.what') as HTMLElement).textContent = t(`cycle.${CYCLE_STAGES[i]}`, { n: p?.batchSize ?? tk?.batchSize ?? 16 });
        (el.querySelector('.num') as HTMLElement).textContent = this.cycle && phase === i + 1 ? '▶' : String(i + 1);
        (el.querySelector('.val') as HTMLElement).textContent = lit || (CYCLE_STAGES[i] === 'check' && p && !p.check && phase >= NUDGE_PHASE) ? values[i] : '';
      });
      note.textContent = fast ? t('cycle.running', { n: Math.round(tk!.stepsPerSec) }) : t('cycle.hint');
    }
    if (this.tab !== 'learn' || quietRun(tk)) return; // drawn only while visible, and not during quiet learning
    practiceHelp.querySelector('p')!.textContent = t('practice.help', { shown: Math.min(PRACTICE_SHOWN, p?.batchSize ?? 16), batch: p?.batchSize ?? 16 });

    // One row per question, read left to right in the order of the stages:
    // question → its answer → right answer → mistake → (after the step).
    if (!p) {
      practiceList.replaceChildren(h('p', { class: 'empty small' }, t('practice.empty')));
      return;
    }
    const answered = p.rightCount !== undefined;
    const checked = !!p.check;
    practiceList.classList.toggle('checked', checked);
    const summary = [t('practice.step', { step: p.step.toLocaleString(uiLang()) })];
    if (answered && phase >= 3) summary.push(t('practice.sumRight', { right: p.rightCount!, n: p.examples.length }));
    if (phase >= 3) summary.push(t('cycle.measureVal', { loss: p.loss.toFixed(2) }));
    const rows = p.examples.map((ex) => {
      const task = ex.form === 'name' ? t('practice.copy') : `${t(`form.${ex.form}`)} · ${this.langName(ex.lang)}`;
      const answerCell = h('td', { class: 'answer' });
      if (phase >= 2 && ex.answer !== undefined) {
        const judged = phase >= 3;
        answerCell.append(h('span', { class: `word-own ${judged ? ex.status : ''}`, title: judged ? t(`status.${ex.status}`) : '' },
          `${judged ? STATUS_ICONS[ex.status!] + ' ' : ''}${ex.answer || '∅'}`));
      }
      // Once the mistake is measured: each letter of the right answer tinted by how surprised the model was.
      const measured = phase >= 3;
      const tgtCell = h('td', { class: 'tgt' }, ...(measured ? ex.letters.map((l) => letterSpan(l)) : [ex.target]));
      const lossCell = h('td', { class: 'loss' }, measured ? lossBar(ex.loss) : '');
      const cells = [
        h('td', { class: 'example' }, h('span', { class: 'src' }, ex.source), h('span', { class: 'task' }, task)),
        answerCell,
        tgtCell,
        lossCell];
      if (checked) cells.push(h('td', { class: 'check' }, phase >= 5 && ex.check ? checkCell(ex) : ''));
      return h('tr', { class: 'ex' }, ...cells);
    });
    practiceList.replaceChildren(
      h('p', { class: 'small step-of' }, summary.join(' · ')),
      answered ? '' : h('p', { class: 'small' }, t('practice.noAnswers')),
      h('table', {},
        h('thead', {}, h('tr', {},
          h('th', {}, t('practice.col.example')),
          h('th', { class: 'answer' }, t('practice.col.answer')),
          h('th', { class: 'data' }, t('practice.col.right')),
          h('th', { class: 'loss' }, t('practice.col.loss')),
          checked ? h('th', { class: 'check', title: t('practice.checkTip') }, t('practice.col.check')) : '')),
        h('tbody', {}, ...rows)),
      phase >= 3 ? h('p', { class: 'small loss-explain', 'data-test': 'loss-explain' },
        t('practice.lossExplain', { loss: p.loss.toFixed(2), n: p.letterCount.toLocaleString(uiLang()), b: p.batchSize })) : '',
      checked && phase >= 5 ? h('p', { class: 'small loss-explain', 'data-test': 'check-explain' },
        t('practice.checkExplain', { b: p.batchSize, before: p.check!.lossBefore.toFixed(3), after: p.check!.lossAfter.toFixed(3), up: p.check!.lettersUp, down: p.check!.lettersDown, step: p.loss.toFixed(2) })) : '');
  }

  private renderLoss(): void {
    const c = this.els.lossCanvas;
    const width = parentWidth(c);
    // Exponential moving average of the raw batch losses, thinned to ~400 points.
    const pts: [number, number][] = [];
    let ema: number | null = null;
    const every = Math.max(1, Math.floor(this.losses.length / 400));
    this.losses.forEach((p, i) => {
      ema = ema === null ? p.loss : 0.95 * ema + 0.05 * p.loss;
      if (i % every === 0 || i === this.losses.length - 1) pts.push([p.step, ema]);
    });
    const max = Math.max(0.5, ...pts.slice(Math.min(10, pts.length - 1)).map((p) => p[1]));
    drawLineChart(c, [{ points: pts, color: '#2563eb' }], this.expert
      ? { width, height: 170, yMin: 0.01, yMax: Math.max(4, max), log: true, yTicks: [0.01, 0.1, 1], yLabel: (v) => String(v) }
      : { width, height: 170, yMin: 0, yMax: max, yLabel: (v) => v.toFixed(1) });
  }

  private renderScores(): void {
    if (!this.ready || this.tab !== 'test') return; // drawn only while the Test tab is visible
    const { testLangPick, testWhen } = this.els;
    if (testLangPick.childElementCount !== this.ready.languages.length) {
      testLangPick.replaceChildren(...this.ready.languages.map((l) => {
        const b = h('button', { 'data-lang': l.code }, l.name[uiLang()] ?? l.code);
        b.onclick = () => { this.testLang = l.code; this.invalidate('scores', 'animals'); };
        return b;
      }));
    }
    for (const b of testLangPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.lang === this.testLang);
    const tested = this.evals[this.evals.length - 1];
    testWhen.textContent = tested ? t('test.when', { step: tested.step.toLocaleString(uiLang()) }) : t('test.none');
    const c = this.els.scoreCanvas;
    const width = parentWidth(c);
    const langs = this.ready.languages.map((l) => l.code);
    // Lines for the selected language; the table below has every language.
    const lang = this.testLang;
    const series = SCORE_LINES.map((k) => ({
      points: this.evals.flatMap((e) => (e.scores[lang]?.[k] != null ? [[e.step, e.scores[lang][k]!] as [number, number]] : [])),
      color: SCORE_STYLE[k].color,
      dash: SCORE_STYLE[k].dash,
    }));
    drawLineChart(c, series, { width, height: 170, yMin: 0, yMax: 1, yTicks: [0, 0.5, 1], yLabel: (v) => `${v * 100}%` });

    const last = this.evals[this.evals.length - 1];
    const table = h('table', {},
      h('tr', {}, h('th', {}), ...this.ready.languages.map((l) => h('th', {}, l.name[uiLang()] ?? l.code))),
      ...[...SCORE_LINES, 'close' as const].map((k) => h('tr', { class: k === 'close' ? 'close-row' : '' },
        h('td', {}, k === 'close' ? '' : h('span', { class: `swatch ${k}` }), ' ', t(`score.${k}`)),
        ...langs.map((lang) => h('td', { class: 'num' }, pct(last?.scores[lang]?.[k]))))));
    this.els.scoreTable.replaceChildren(table);
  }

  private renderAnimals(): void {
    if (!this.ready || this.tab !== 'test') return; // drawn only while the Test tab is visible
    const lang = this.testLang;
    const forms = this.ready.forms.filter((f) => f !== 'name');
    for (const b of this.els.filters.querySelectorAll('button')) b.classList.toggle('on', b.dataset.filter === this.filter);
    const rows: HTMLElement[] = [];
    const list = this.ready.animals.filter((a) => a.names[lang]).sort((a, b) => a.names[lang].localeCompare(b.names[lang], lang));
    for (const a of list) {
      const rs = forms.map((f) => this.results.get(resultKey(a.id, lang, f)));
      if (!this.keep(a, rs)) continue;
      const cells = forms.map((f, i) => {
        const r = rs[i];
        if (!r) return h('td', { class: 'cell none' }, t('noForm'));
        const title = `${t(`form.${f}`)}: ${r.output} — ${t(`status.${r.status}`)}${r.answer && !a.unseen ? ` (${r.answer})` : ''}`;
        const cell = h('td', { class: `cell ${r.status}`, title }, `${STATUS_ICONS[r.status]} ${r.output || '∅'}`);
        cell.onclick = (e) => { e.stopPropagation(); this.askOnAskTab({ text: a.names[lang], lang, form: f }); };
        return cell;
      });
      const name = h('td', { class: 'name' }, a.names[lang], a.unseen ? h('span', { class: 'never' }, t('neverSeen')) : null);
      const row = h('tr', { class: a.names[lang] === this.question.text ? 'sel' : '' }, name, ...cells);
      row.onclick = () => this.askOnAskTab({ text: a.names[lang], lang });
      rows.push(row);
    }
    this.els.animals.replaceChildren(h('table', {},
      h('tr', {}, h('th', {}), ...forms.map((f) => h('th', {}, t(`form.${f}`)))), ...rows));
  }

  private keep(a: AnimalInfo, rs: (TaskResult | undefined)[]): boolean {
    switch (this.filter) {
      case 'all': return true;
      case 'known': return !a.unseen;
      case 'unseen': return a.unseen;
      case 'wrong': return rs.some((r) => r && (r.status === 'wrong' || r.status === 'close'));
    }
  }
}

/** Colours are defined once in colors.ts and handed to CSS as variables. */
export function colorVars(): string {
  return [
    ...Object.entries(STATUS_COLORS).map(([k, v]) => `--${k}: ${v};`),
    ...Object.entries(SCORE_STYLE).map(([k, v]) => `--line-${k}: ${v.color};`),
  ].join(' ');
}
