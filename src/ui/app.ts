/**
 * The minimal M2 app: controls, the question box with form and language
 * pickers, guess bars, loss chart, scoreboard and the animal list.
 *
 * All training happens in the worker; this file only sends messages and
 * draws what comes back.
 */
import type { AnimalInfo, AnswerMsg, EvalMsg, FromWorker, LossPoint, ReadyMsg, Speed, TickMsg, ToWorker } from '../train/protocol';
import type { Scores, TaskResult } from '../train/trainer';
import { STATUS_COLORS, STATUS_ICONS, SCORE_STYLE } from '../viz/colors';
import { drawGuessBars } from '../viz/guessBars';
import { drawLineChart } from '../viz/lineChart';
import { parentWidth } from '../viz/canvas';
import { setUiLang, t, uiLang, UI_LANGUAGES } from './i18n';

type Filter = 'all' | 'known' | 'unseen' | 'wrong';
const SPEEDS: Speed[] = [1, 5, 20, 'max'];
const SCORE_LINES = ['known', 'pattern', 'partly', 'irregular'] as const;

/** Tiny DOM helper. */
function h<K extends keyof HTMLElementTagNameMap>(
  tag: K, attrs: Record<string, string | number | boolean> = {}, ...children: (Node | string | null | undefined)[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === false) continue;
    if (k === 'class') el.className = String(v);
    else el.setAttribute(k, v === true ? '' : String(v));
  }
  for (const c of children) if (c !== null && c !== undefined) el.append(c);
  return el;
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
  private askDirty = false;
  private revealed = false;
  private expert = false;
  private filter: Filter = 'all';
  private question = { text: '', lang: '', form: 'female' };
  private dirty = { controls: true, answer: true, loss: true, scores: true, animals: true };
  private frame = 0;
  private els!: ReturnType<App['build']>;

  constructor(private root: HTMLElement) {
    this.worker = new Worker(new URL('../train/worker.ts', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e: MessageEvent<FromWorker>) => this.onMessage(e.data);
    this.worker.onerror = (e) => this.showError(e.message);
    this.els = this.build();
    window.addEventListener('resize', () => this.invalidate('loss', 'scores'));
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
        this.renderPickers();
        this.ask();
        this.invalidate('controls', 'loss', 'scores', 'animals');
        break;
      case 'tick':
        if (this.tick && msg.step < this.tick.step) { this.losses = []; this.evals = []; this.results.clear(); }
        this.tick = msg;
        this.losses.push(...msg.losses);
        if (this.losses.length > 40000) this.losses = this.losses.filter((_, i) => i % 2 === 0);
        if (msg.losses.length) this.ask();
        this.invalidate('controls', 'loss');
        break;
      case 'eval':
        this.onEval(msg);
        break;
      case 'answer':
        this.askInFlight = false;
        if (msg.id === this.askId) { this.answer = msg; this.invalidate('answer'); }
        if (this.askDirty) this.ask();
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
    expertBox.onchange = () => { this.expert = expertBox.checked; expertPanel.hidden = !this.expert; this.invalidate('loss'); };
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
    const bars = h('canvas', { class: 'bars' });
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

    this.root.replaceChildren(
      h('header', {}, h('h1', {}, t('title')), h('span', { class: 'sub' }, t('subtitle')), params, uiSwitch),
      error,
      h('main', {},
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
          h('div', { class: 'charts' },
            h('section', { class: 'panel' }, h('h2', {}, t('loss')), h('div', { class: 'chart' }, lossCanvas)),
            h('section', { class: 'panel' }, h('h2', {}, t('scoreboard')), h('div', { class: 'chart' }, scoreCanvas), scoreTable)))),
    );
    return { play, speed, seed, status, input, langPick, formPick, hint, output, bars, lossCanvas, scoreCanvas, scoreTable, filters, animals, params, error };
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
    drawGuessBars(bars, a.steps, { copied: t('copied'), wrote: t('wrote') });
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
