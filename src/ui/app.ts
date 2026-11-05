/**
 * The app: two tabs.
 * - Forms: controls, the question box with form and language pickers, guess
 *   bars, attention grids, letter map, loss chart, scoreboard, animal list.
 * - Inside the brain: weight heatmaps (see inside.ts).
 *
 * All training happens in the worker; this file only sends messages and
 * draws what comes back.
 */
import type { AnimalInfo, AnswerMsg, EvalMsg, FromWorker, LetterMapMsg, LossPoint, ReadyMsg, Speed, TickMsg, ToWorker } from '../train/protocol';
import type { Scores, TaskResult } from '../train/trainer';
import { LETTER_KINDS, LetterKind, STATUS_COLORS, STATUS_ICONS, SCORE_STYLE } from '../viz/colors';
import { attentionCellAt, drawAttentionGrid, GridSpec } from '../viz/attentionGrid';
import { drawGuessBars, guessColumnAt } from '../viz/guessBars';
import { LetterMapView, letterKind, letterLabel, marker } from '../viz/letterMap';
import { drawLineChart } from '../viz/lineChart';
import { parentWidth, prepare } from '../viz/canvas';
import { setUiLang, t, uiLang, UI_LANGUAGES } from './i18n';
import { h } from './dom';
import { InsideTab } from './inside';

type Filter = 'all' | 'known' | 'unseen' | 'wrong';
type Tab = 'forms' | 'inside';
const TABS: Tab[] = ['forms', 'inside'];
/** Look-inside requests: at most one in flight, at most this often. */
const INSPECT_MS = 250;

/** One attention grid on screen. `linked`: its rows are the output letters (hover links to the guess bars). */
interface AttGrid {
  key: string;
  title: string;
  spec: GridSpec;
  linked: boolean;
  canvas?: HTMLCanvasElement;
}

const tokenLabel = (tok: string, i: number) => (i === 0 ? '✱' : tok === ' ' ? '␣' : tok);
const SPEEDS: Speed[] = [1, 5, 20, 'max'];
const SCORE_LINES = ['known', 'pattern', 'partly', 'irregular'] as const;

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
  private askDirty = false;
  private revealed = false;
  private expert = false;
  private filter: Filter = 'all';
  private question = { text: '', lang: '', form: 'female' };
  private dirty = { controls: true, answer: true, loss: true, scores: true, animals: true };
  private frame = 0;
  private els!: ReturnType<App['build']>;
  private tab: Tab = 'forms';
  /** Output step linked between the guess bars and the attention grids (hover). */
  private hoverStep: number | null = null;
  private attGrids: AttGrid[] = [];
  private attKeys = '';
  private inside: InsideTab;
  private letterMap!: LetterMapView;
  private inspectId = 0;
  private inspectPending: number | null = null;
  private inspectDirty = false;
  private lastInspect = -Infinity;
  private inspectTimer = 0;

  constructor(private root: HTMLElement) {
    this.worker = new Worker(new URL('../train/worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
    this.worker.onerror = (e) => this.showError(e.message);
    this.inside = new InsideTab(() => this.requestInspect(true));
    this.els = this.build();
    this.letterMap = new LetterMapView(this.els.mapCanvas, () => Math.min(360, parentWidth(this.els.mapCanvas, 320)));
    window.addEventListener('resize', () => { this.invalidate('loss', 'scores'); this.letterMap.draw(); });
    const seed = 1 + Math.floor(Math.random() * 9999);
    this.send({ type: 'init', seed, speed: 'max' });
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
        this.losses = [];
        this.evals = [];
        this.results.clear();
        if (!this.question.lang) this.question.lang = msg.languages[0].code;
        if (!this.question.text) this.question.text = msg.animals.find((a) => a.forms[this.question.lang]?.female)?.names[this.question.lang] ?? '';
        this.els.seed.value = String(msg.seed);
        this.letterMap.reset();
        this.inside.setReady(msg);
        this.renderPickers();
        this.ask();
        this.requestInspect(true);
        this.invalidate('controls', 'loss', 'scores', 'animals');
        break;
      case 'tick':
        if (this.tick && msg.step < this.tick.step) {
          this.losses = []; this.evals = []; this.results.clear();
          this.letterMap.reset();
          this.requestInspect(true);
        }
        this.tick = msg;
        this.losses.push(...msg.losses);
        if (this.losses.length > 40000) this.losses = this.losses.filter((_, i) => i % 2 === 0);
        if (msg.losses.length) { this.ask(); this.requestInspect(); }
        this.invalidate('controls', 'loss');
        break;
      case 'eval':
        this.onEval(msg);
        break;
      case 'answer':
        this.askInFlight = false;
        if (msg.id === this.askId) { this.answer = msg; this.invalidate('answer'); this.inside.setAnswer(msg); }
        if (this.askDirty) this.ask();
        break;
      case 'weights':
        this.inspectDone(msg.id);
        this.inside.onWeights(msg);
        break;
      case 'letterMap':
        this.inspectDone(msg.id);
        this.onLetterMap(msg);
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

  /** Ask the worker about the current question (at most one request in flight). */
  private ask(): void {
    if (!this.ready || !this.question.text.trim()) return;
    if (this.askInFlight) { this.askDirty = true; return; }
    this.askDirty = false;
    this.askInFlight = true;
    this.send({ type: 'ask', id: ++this.askId, ...this.question });
  }

  /**
   * Ask the worker for what the visible tab shows inside the brain: the
   * weight heatmaps (Inside tab) or the letter map (Forms tab). At most one
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
    const inside = this.tab === 'inside';
    const matrixIds = inside ? this.inside.wanted() : [];
    if (inside && !matrixIds.length) return;
    this.inspectPending = ++this.inspectId;
    this.send({ type: 'inspect', id: this.inspectId, matrixIds, mode: this.inside.mode, letterMap: !inside });
  }

  private inspectDone(id: number): void {
    if (id !== this.inspectPending) return;
    this.inspectPending = null;
    if (this.inspectDirty) this.requestInspect();
  }

  private onLetterMap(msg: LetterMapMsg): void {
    if (!this.ready) return;
    const tokens = this.ready.tokens;
    const points = msg.ids.map((id) => ({ token: tokens[id], label: letterLabel(tokens[id]), kind: letterKind(tokens[id]) }));
    this.letterMap.update(points, msg.coords);
    const p = (v: number) => Math.round(v * 100);
    this.els.mapNote.textContent = t('map.note', { a: p(msg.explained[0]), b: p(msg.explained[1]) });
  }

  private setTab(tab: Tab): void {
    this.tab = tab;
    for (const b of this.els.tabs.querySelectorAll('button')) {
      b.classList.toggle('on', b.dataset.tab === tab);
      b.setAttribute('aria-selected', String(b.dataset.tab === tab));
    }
    this.els.formsMain.hidden = tab !== 'forms';
    this.els.insideMain.hidden = tab !== 'inside';
    if (tab === 'forms') { this.invalidate('loss', 'scores', 'answer'); this.letterMap.draw(); }
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
    step1.onclick = () => this.send({ type: 'step', n: 1 });
    const step10 = h('button', {}, t('step10'));
    step10.onclick = () => this.send({ type: 'step', n: 10 });
    const speed = h('select', { 'aria-label': t('speed') });
    for (const s of SPEEDS) speed.append(h('option', { value: String(s) }, s === 'max' ? t('speedMax') : t('speedSteps', { n: s })));
    speed.value = 'max';
    speed.onchange = () => this.send({ type: 'setSpeed', speed: speed.value === 'max' ? 'max' : Number(speed.value) });
    const seed = h('input', { type: 'number', min: 1, step: 1, class: 'seed', 'aria-label': t('seed') });
    const reset = h('button', {}, t('reset'));
    reset.onclick = () => this.send({ type: 'reset', seed: Number(seed.value) || 1 });
    const expertBox = h('input', { type: 'checkbox' });
    expertBox.onchange = () => {
      this.expert = expertBox.checked;
      expertPanel.hidden = !this.expert;
      this.inside.setExpert(this.expert);
      this.invalidate('loss', 'answer');
      this.requestInspect(true);
    };
    const lr = h('input', { type: 'number', step: 0.0005, min: 0.0001, max: 0.05, value: 0.003 });
    lr.onchange = () => this.send({ type: 'setHyper', lr: Number(lr.value) });
    const batch = h('input', { type: 'number', step: 1, min: 1, max: 128, value: 16 });
    batch.onchange = () => this.send({ type: 'setHyper', batchSize: Number(batch.value) });
    const expertPanel = h('div', { class: 'expert', hidden: true },
      h('label', {}, t('lr'), ' ', lr), h('label', {}, t('batch'), ' ', batch));
    const status = h('div', { class: 'status', 'data-test': 'status' });

    const input = h('input', { type: 'text', class: 'ask', placeholder: t('askPlaceholder'), spellcheck: 'false', 'data-test': 'ask' });
    input.oninput = () => this.setQuestion({ text: input.value }, true);
    const langPick = h('div', { class: 'picker', role: 'group', 'aria-label': t('language') });
    const formPick = h('div', { class: 'picker', role: 'group', 'aria-label': t('form') });
    const hint = h('div', { class: 'hint' });
    const output = h('div', { class: 'output', 'data-test': 'output' });
    const bars = h('canvas', { class: 'bars', 'data-test': 'bars' });
    bars.onmousemove = (e) => this.setHover(guessColumnAt(e.offsetX, this.answer?.steps.length ?? 0));
    bars.onmouseleave = () => this.setHover(null);
    const attention = h('div', { class: 'att-grids', 'data-test': 'attention' });
    const mapCanvas = h('canvas', { class: 'letter-map', 'data-test': 'letter-map' });
    const mapNote = h('p', { class: 'small' });
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
      const b = h('button', { role: 'tab', 'data-tab': tab, 'data-test': `tab-${tab}`, class: tab === this.tab ? 'on' : '' }, t(`tab.${tab}`));
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
    const params = h('div', { class: 'params' });
    const uiSwitch = h('button', { class: 'link' }, t('uiLanguage'));
    uiSwitch.onclick = () => {
      setUiLang(UI_LANGUAGES.find((l) => l !== uiLang()) ?? 'en');
      location.reload();
    };
    const error = h('div', { class: 'error', hidden: true });
    const formsMain = this.buildForms({
      play, step1, step10, speed, seed, reset, expertBox, expertPanel, status, input, langPick, formPick, hint, output, bars,
      attention, mapCanvas, mapNote, mapLegend, lossCanvas, scoreCanvas, scoreTable, filters, animals,
    });
    const insideMain = h('main', { class: 'inside-main', hidden: true }, this.inside.el);

    this.root.replaceChildren(
      h('header', {}, h('h1', {}, t('title')), h('span', { class: 'sub' }, t('subtitle')), tabs, params, uiSwitch),
      error,
      formsMain,
      insideMain,
    );
    return {
      play, speed, seed, status, input, langPick, formPick, hint, output, bars, attention, mapCanvas, mapNote, tabs, formsMain, insideMain,
      lossCanvas, scoreCanvas, scoreTable, filters, animals, params, error,
    };
  }

  private buildForms(parts: Record<string, HTMLElement>) {
    const { play, step1, step10, speed, seed, reset, expertBox, expertPanel, status, input, langPick, formPick, hint, output, bars,
      attention, mapCanvas, mapNote, mapLegend, lossCanvas, scoreCanvas, scoreTable, filters, animals } = parts;
    return h('main', { class: 'forms' },
        h('aside', { class: 'left' },
          h('section', { class: 'panel controls' },
            h('div', { class: 'row' }, play, step1, step10),
            h('label', { class: 'row' }, t('speed'), ' ', speed),
            h('div', { class: 'row' }, h('label', {}, t('seed'), ' ', seed), reset),
            h('label', { class: 'row small' }, expertBox, ' ', t('expert')),
            expertPanel,
            status),
          h('section', { class: 'panel list' }, h('h2', {}, t('animals')), filters, animals)),
        h('section', { class: 'centre' },
          h('section', { class: 'panel askbox' },
            h('div', { class: 'row' }, input, langPick),
            formPick, hint, output,
            h('h2', {}, t('guessBars')), h('div', { class: 'bars-wrap' }, bars)),
          h('section', { class: 'panel attention' },
            h('h2', {}, t('att.title')),
            h('p', { class: 'small' }, t('att.help')),
            attention),
          h('div', { class: 'charts' },
            h('section', { class: 'panel' }, h('h2', {}, t('loss')), h('div', { class: 'chart' }, lossCanvas)),
            h('section', { class: 'panel' }, h('h2', {}, t('scoreboard')), h('div', { class: 'chart' }, scoreCanvas), scoreTable))),
        h('aside', { class: 'right' },
          h('section', { class: 'panel map' },
            h('h2', {}, t('map.title')),
            h('div', { class: 'map-wrap' }, mapCanvas),
            mapLegend,
            mapNote,
            h('p', { class: 'small' }, t('map.help')))));
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
      const b = h('button', { 'data-form': f }, t(`form.${f}`));
      b.onclick = () => this.setQuestion({ form: f });
      return b;
    }));
    this.els.params.textContent = t('params', { n: this.ready.params.toLocaleString(uiLang()) });
    this.invalidate('answer');
  }

  private setQuestion(q: Partial<App['question']>, typed = false): void {
    Object.assign(this.question, q);
    // Forms stay within one language: if the typed word is an animal of
    // another language, switch to it.
    if (typed && this.ready) {
      const w = this.question.text.trim().toLowerCase().normalize('NFC');
      const here = this.ready.animals.some((a) => a.names[this.question.lang] === w);
      const other = here ? undefined : this.ready.languages.find((l) => this.ready!.animals.some((a) => a.names[l.code] === w));
      if (other) this.question.lang = other.code;
    }
    if (!typed) this.els.input.value = this.question.text;
    this.revealed = false;
    this.ask();
    this.invalidate('answer', 'animals');
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
    this.dirty = { controls: false, answer: false, loss: false, scores: false, animals: false };
  }

  private renderControls(): void {
    const tk = this.tick;
    this.els.play.textContent = tk?.running ? t('pause') : t('play');
    if (!tk) return;
    const parts = [t('stepLabel', { step: tk.step.toLocaleString(uiLang()) })];
    if (tk.stepsPerSec > 0) parts.push(t('stepsPerSec', { n: tk.stepsPerSec.toFixed(0) }));
    if (tk.lossEma !== null) parts.push(`loss ${tk.lossEma.toFixed(3)}`);
    this.els.status.textContent = parts.join(' · ');
  }

  private renderAnswer(): void {
    const { input, langPick, formPick, hint, output, bars } = this.els;
    if (document.activeElement !== input) input.value = this.question.text;
    for (const b of langPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.lang === this.question.lang);
    for (const b of formPick.querySelectorAll('button')) b.classList.toggle('on', b.dataset.form === this.question.form);
    const a = this.answer;
    if (!a || !this.ready) { output.replaceChildren(); return; }
    const langName = (code: string) => this.ready!.languages.find((l) => l.code === code)?.name[uiLang()] ?? code;
    hint.textContent = a.badge === 'free' ? t('unknownWord') : t('looksLike', { lang: langName(a.lang) });

    // A hidden answer is not judged on screen until revealed: the colour would give it away.
    const wordClass = a.badge === 'hidden' && !this.revealed ? 'unrevealed' : a.answer ? a.answer.status : 'invented';
    const word = h('span', { class: `word ${wordClass}` }, a.output || '∅');
    const parts: (Node | string)[] = [h('span', { class: 'from' }, `${a.text} →`), word];
    if (!a.finished) parts.push(h('span', { class: 'small' }, t('notFinished')));
    const badge = h('div', { class: `badge ${a.badge}` }, t(`badge.${a.badge}`));
    if (a.answer && a.badge === 'real') {
      badge.append(' ', h('span', { class: 'small' }, `${STATUS_ICONS[a.answer.status]} ${t('realAnswer', { word: a.answer.accepted.join(' / ') })}`));
    }
    if (a.answer && a.badge === 'hidden') {
      if (this.revealed) {
        badge.append(' ', h('span', { class: 'small' }, `${STATUS_ICONS[a.answer.status]} ${t('realAnswer', { word: a.answer.accepted.join(' / ') })}`));
      } else {
        const reveal = h('button', { class: 'small' }, t('reveal'));
        reveal.onclick = () => { this.revealed = true; this.invalidate('answer'); };
        badge.append(' ', reveal);
      }
    }
    const meter = h('div', { class: 'meter', title: t('confidenceHelp') },
      h('span', {}, t('confidence')),
      h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.round(a.confidence * 100)}%` })),
      h('span', {}, `${Math.round(a.confidence * 100)}`));
    output.replaceChildren(h('div', { class: 'answer-line' }, ...parts), badge, meter);
    if (this.hoverStep !== null && this.hoverStep >= a.steps.length) this.hoverStep = null;
    drawGuessBars(bars, a.steps, { copied: t('copied'), wrote: t('wrote') }, this.hoverStep);
    this.renderAttention();
  }

  /** Link a guess-bar column and the matching attention rows (null = none). */
  private setHover(step: number | null): void {
    if (step === this.hoverStep) return;
    this.hoverStep = step;
    if (this.answer) drawGuessBars(this.els.bars, this.answer.steps, { copied: t('copied'), wrote: t('wrote') }, step);
    this.drawAttention();
  }

  /** Which attention grids to show for this answer (expert mode adds the self-attention). */
  private attentionGrids(a: AnswerMsg): AttGrid[] {
    const input = a.input.map(tokenLabel);
    const out = a.steps.map((s) => (s.chosen === ' ' ? '␣' : s.chosen || '⏎'));
    const decIn = ['✱', ...out];
    const grids: AttGrid[] = [];
    const add = (name: string, title: (head: number) => string, rows: string[], cols: string[], linked: boolean) => {
      const m = a.attention.find((x) => x.name === name);
      if (!m) return;
      for (let head = 0; head < m.heads; head++) {
        grids.push({
          key: `${name}#${head}`, title: title(head), linked,
          spec: { rows: rows.slice(0, m.Tq), cols: cols.slice(0, m.Tk), data: m.data, offset: head * m.Tq * m.Tk, stride: m.Tk },
        });
      }
    };
    const layers = [...new Set(a.attention.map((x) => x.name.match(/^dec(\d+)\./)?.[1]).filter((x): x is string => x !== undefined))];
    const layerNote = (l: string) => (layers.length > 1 ? ` · ${t('att.layer', { n: Number(l) + 1 })}` : '');
    for (const l of layers) add(`dec${l}.cross`, (hd) => t('att.cross', { n: hd + 1 }) + layerNote(l), out, input, true);
    add('copy', () => t('att.copy'), out, input, true);
    if (this.expert) {
      for (const l of layers) add(`dec${l}.self`, (hd) => t('att.decSelf', { n: hd + 1 }) + layerNote(l), out, decIn, true);
      for (const name of a.attention.map((x) => x.name).filter((n) => /^enc\d+\.self$/.test(n))) {
        add(name, (hd) => t('att.encSelf', { n: hd + 1 }), input, input, false);
      }
    }
    return grids;
  }

  private renderAttention(): void {
    const a = this.answer;
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
          if (grid.linked) this.setHover(cell ? cell.row : null);
          canvas.title = cell && cell.col !== null
            ? t('att.cell', { row: grid.spec.rows[cell.row], col: grid.spec.cols[cell.col], p: Math.round(grid.spec.data[grid.spec.offset + cell.row * grid.spec.stride + cell.col] * 100) })
            : '';
        };
        canvas.onmouseleave = () => { if (this.attGrids[i].linked) this.setHover(null); };
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
      drawAttentionGrid(g.canvas, g.spec, g.linked ? this.hoverStep : null);
    }
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
    if (!this.ready) return;
    const c = this.els.scoreCanvas;
    const width = parentWidth(c);
    const langs = this.ready.languages.map((l) => l.code);
    // Lines for the selected language; the table below has every language.
    const lang = this.question.lang;
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
    if (!this.ready) return;
    const lang = this.question.lang;
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
        const title = `${t(`form.${f}`)}: ${r.output} — ${t(`status.${r.status}`)}${r.answer && a.split === 'train' ? ` (${r.answer})` : ''}`;
        const cell = h('td', { class: `cell ${r.status}`, title }, `${STATUS_ICONS[r.status]} ${r.output || '∅'}`);
        cell.onclick = (e) => { e.stopPropagation(); this.setQuestion({ text: a.names[lang], form: f }); };
        return cell;
      });
      const name = h('td', { class: 'name' }, a.names[lang], a.split === 'unseen' ? h('span', { class: 'never' }, t('neverSeen')) : null);
      const row = h('tr', { class: a.names[lang] === this.question.text ? 'sel' : '' }, name, ...cells);
      row.onclick = () => this.setQuestion({ text: a.names[lang] });
      rows.push(row);
    }
    this.els.animals.replaceChildren(h('table', {},
      h('tr', {}, h('th', {}), ...forms.map((f) => h('th', {}, t(`form.${f}`)))), ...rows));
  }

  private keep(a: AnimalInfo, rs: (TaskResult | undefined)[]): boolean {
    switch (this.filter) {
      case 'all': return true;
      case 'known': return a.split === 'train';
      case 'unseen': return a.split === 'unseen';
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
