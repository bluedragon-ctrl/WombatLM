/**
 * "How it picks a letter" tab: what one answer does inside the model.
 * - "How the model picks one letter" (see explainView.ts): one letter of the
 *   answer on the Ask tab, stage by stage;
 * - "What have the detectors learned?": the writer's thinking units, each
 *   with the moments of the learning data that switch it on most (see
 *   train/units.ts and unitCard.ts). The letter uses the same detectors in
 *   its "think" stage.
 * Where the model is learning (its numbers) is on the "Where it learns" tab
 * (inside.ts).
 */
import type { AnswerMsg, ExplainMsg, ReadyMsg, UnitsMsg } from '../train/protocol';
import type { HeatmapMarks } from '../viz/weightHeatmap';
import { clearestFirst, matters, unitCard } from './unitCard';
import { boardLegend, detectorBoard, detectorFlow } from './detectorBoard';
import { h } from './dom';
import { ExplainView, Question } from './explainView';
import { t, uiLang } from './i18n';
import { help } from './inside';

/** "What have the detectors learned?": at most one request per this many ms while learning; clear ones shown in plain mode. */
const UNITS_EVERY = 15000;
const UNITS_CLEAR = 12;

export class LetterTab {
  readonly el: HTMLElement;
  private ready?: ReadyMsg;
  private expert = false;
  private explain: ExplainView;
  private unitsEl: HTMLElement;
  private unitsBody: HTMLElement;
  private unitsNote: HTMLElement;
  private unitsTeam: HTMLElement;
  private unitsButton: HTMLButtonElement;
  private boardEl: HTMLElement;
  private pickedEl: HTMLElement;
  /** The detector clicked on the board (its card is shown first). */
  private picked: number | null = null;
  private units?: UnitsMsg;
  private unitsAll = false;
  private unitsAsked = 0;
  private unitsInFlight = false;
  /** Ask the worker what the thinking units have learned (set by the app). */
  requestUnits?: () => void;
  /** The numbers used by the letter, for the heatmaps on "The numbers" tab (set by the app). */
  onMarks?: (marks: Map<string, HeatmapMarks>) => void;

  constructor(requestExplain: (q: Question, position: number) => void) {
    this.explain = new ExplainView(requestExplain, (marks) => this.onMarks?.(marks), (code) => this.langName(code));
    this.unitsBody = h('div', { class: 'ucards', 'data-test': 'unit-cards' });
    this.unitsNote = h('p', { class: 'small', 'data-test': 'units-note' }, t('units.wait'));
    this.unitsTeam = h('p', { class: 'units-team small', 'data-test': 'units-team' });
    this.unitsButton = h('button', { class: 'units-toggle', 'data-test': 'units-toggle' });
    this.unitsButton.onclick = () => { this.unitsAll = !this.unitsAll; this.renderUnits(); };
    this.boardEl = h('div', { class: 'dboard-wrap' });
    this.pickedEl = h('div', { class: 'upicked', 'data-test': 'unit-picked' });
    this.unitsEl = h('section', { class: 'panel units', 'data-test': 'units' },
      h('h2', {}, t('units.title')),
      h('p', { class: 'small' }, h('span', { class: 'units-help' }), ' ', help(t('units.help'))),
      detectorFlow(),
      h('p', { class: 'small' }, t('units.how')),
      h('h3', { class: 'units-sub' }, t('units.boardTitle')),
      this.unitsNote,
      this.boardEl,
      this.unitsTeam,
      this.pickedEl,
      h('h3', { class: 'units-sub' }, t('units.cardsTitle')),
      this.unitsBody,
      this.unitsButton);
    this.el = h('div', { class: 'inside letter-tab plain' },
      h('section', { class: 'panel inside-head' },
        h('p', { class: 'intro', 'data-test': 'letter-intro' }, t('letter.intro')),
        h('ol', { class: 'small letter-guide' }, h('li', {}, t('letter.guide1')), h('li', {}, t('letter.guide2')))),
      this.explain.el,
      this.unitsEl);
  }

  setReady(ready: ReadyMsg): void {
    this.ready = ready;
    this.units = undefined;
    this.picked = null;
    this.explain.setUnits(undefined);
    this.renderUnits();
  }

  setExpert(on: boolean): void {
    this.expert = on;
    // Plain mode hides the engineering details: see style.css.
    this.el.classList.toggle('plain', !on);
    this.explain.setExpert(on);
    this.renderUnits();
  }

  /** The tab became visible (or its answer changed while visible): bring the letter and the detectors up to date. */
  shown(): void {
    this.explain.refresh();
    this.askUnits(true);
  }

  /** The model learned (a new step): the detectors may have changed. Asked again every few seconds at most. */
  onStep(step: number): void {
    if (this.units && step !== this.units.step) this.askUnits();
  }

  private askUnits(now = false): void {
    if (!this.requestUnits || this.unitsInFlight) return;
    if (!now && Date.now() - this.unitsAsked < UNITS_EVERY) return;
    this.unitsInFlight = true;
    this.unitsAsked = Date.now();
    this.requestUnits();
  }

  onUnits(msg: UnitsMsg): void {
    this.unitsInFlight = false;
    this.units = msg;
    this.explain.setUnits(msg);
    this.renderUnits();
  }

  private renderUnits(): void {
    const u = this.units;
    this.el.querySelector('.units-help')!.textContent = t('units.helpShort', { n: u?.units.length ?? 64 });
    if (!u) {
      this.unitsNote.textContent = t('units.wait');
      this.unitsBody.replaceChildren(); this.boardEl.replaceChildren(); this.pickedEl.replaceChildren(); this.unitsTeam.textContent = '';
      this.unitsButton.hidden = true;
      return;
    }
    const sorted = clearestFirst(u.units);
    const clear = sorted.filter((x) => x.theme);
    const all = this.expert || this.unitsAll || clear.length === 0;
    const shown = all ? sorted : clear.slice(0, UNITS_CLEAR);
    const alone = u.units.filter(matters).length;
    const card = (x: (typeof u.units)[number], picked = false) => unitCard(x, (c) => this.langName(c), { expert: this.expert, picked });
    this.unitsNote.textContent = `${t('units.note', { n: u.units.length, m: u.moments.toLocaleString(uiLang()) })} ${t('units.step', { step: u.step })}`;
    this.boardEl.replaceChildren(
      detectorBoard(u.units, { selected: this.picked, langName: (c) => this.langName(c), onPick: (k) => this.pick(k) }),
      boardLegend(u.units));
    this.unitsTeam.textContent = u.units.some((x) => x.off) ? t('units.team', { k: alone, n: u.units.length }) : '';
    const picked = this.picked !== null ? u.units[this.picked] : undefined;
    this.pickedEl.replaceChildren(...(picked ? [h('div', { class: 'ucards' }, card(picked, true))] : [h('p', { class: 'small muted' }, t('units.pickHint'))]));
    this.unitsBody.replaceChildren(...shown.map((x) => card(x)));
    this.unitsButton.hidden = this.expert || clear.length === 0;
    this.unitsButton.textContent = this.unitsAll ? t('units.showClear') : t('units.showAll', { n: u.units.length });
  }

  /** A square on the board was clicked: show that detector's card (click again to close it). */
  private pick(unit: number): void {
    this.picked = this.picked === unit ? null : unit;
    this.renderUnits();
  }

  /** Look inside letter `position` of the current answer (from a click on the guess bars). */
  explainAt(position: number): void {
    this.explain.select(position);
  }

  scrollToExplain(): void {
    this.explain.el.scrollIntoView?.({ block: 'start' });
  }

  onExplain(msg: ExplainMsg): void {
    this.explain.onExplain(msg);
  }

  setAnswer(a: AnswerMsg | undefined): void {
    this.explain.setAnswer(a);
  }

  private langName(code: string): string {
    return this.ready?.languages.find((l) => l.code === code)?.name[uiLang()] ?? code;
  }
}
