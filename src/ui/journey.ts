/**
 * "Follow the tokens through the model" (Ask tab): one answer, stage by
 * stage, with the real numbers:
 *   1. tokens: the question becomes token numbers (task token first);
 *   2. letter table: every number picks its row (32 numbers) + a position row;
 *   3. reading (once): the input tokens look at each other and think; the
 *      result ("memory") is kept for the whole answer;
 *   4…: writing, one token per pass: the token going in (the task token,
 *      then the token written last) picks its rows, looks back at the answer
 *      so far, looks at the memory, thinks, and its numbers are compared with
 *      every row of the letter table: scores → chances → the top one is
 *      written and goes back in for the next pass. Writing "end" stops.
 *
 * All data comes with the answer (`AnswerMsg`): activations, attention and
 * the looked-up rows. Nothing is computed here.
 */
import type { Activation, AnswerMsg, AttentionMap } from '../train/protocol';
import { sequential, symmetricScale } from '../viz/colors';
import { drawStrip, StripLayout } from '../viz/vectorStrip';
import { h, showTip } from './dom';
import { t } from './i18n';
import { shortLabel, tokenChip, tokenName, toTok } from './tokens';
import { EOS } from '../nn/vocab';

/** Mini strips: 32 numbers in one row of thin squares. */
const CELL_W = 5;
const CELL_H = 12;

export class JourneyView {
  readonly el: HTMLElement;
  private a?: AnswerMsg;
  private tokens: string[] = [];
  private idx = new Map<string, number>();
  private expert = false;
  /** 0 tokens, 1 letter table, 2 reading, 3+k writing pass k. */
  private stage = 0;
  private key = '';
  private body: HTMLElement;
  private pills: HTMLElement;
  private back: HTMLButtonElement;
  private next: HTMLButtonElement;

  constructor(private langName: (code: string) => string, private lookInside: (position: number) => void) {
    this.back = h('button', { 'data-test': 'journey-back' }, t('jr.back'));
    this.next = h('button', { class: 'primary', 'data-test': 'journey-next' }, t('jr.next'));
    this.back.onclick = () => this.go(this.stage - 1);
    this.next.onclick = () => this.go(this.stage + 1);
    this.pills = h('div', { class: 'jr-pills', role: 'group', 'aria-label': t('jr.title') });
    this.body = h('div', { class: 'jr-body', 'aria-live': 'polite' });
    this.el = h('section', { class: 'panel journey', 'data-test': 'journey', hidden: true },
      h('div', { class: 'phase-head' }, h('h2', {}, t('jr.title')),
        h('details', { class: 'help' }, h('summary', { 'aria-label': '?' }, '?'), h('p', {}, t('jr.help')))),
      h('p', { class: 'small' }, t('jr.intro')),
      h('div', { class: 'row jr-controls' }, this.back, this.next, this.pills),
      this.body);
  }

  setTokens(tokens: string[]): void {
    this.tokens = tokens;
    this.idx = new Map(tokens.map((tok, i) => [tok, i]));
  }

  setExpert(on: boolean): void {
    this.expert = on;
    this.el.hidden = !this.a || !on;
    this.render();
  }

  /** Shown only in expert mode, once there is an answer to follow. */
  setAnswer(a: AnswerMsg | undefined): void {
    this.a = a;
    this.el.hidden = !a || !this.expert;
    const key = a ? `${a.text}|${a.lang}|${a.form}` : '';
    if (key !== this.key) { this.key = key; this.stage = 0; }
    this.stage = Math.min(this.stage, this.last());
    this.render();
  }

  /**
   * "Answer slowly" drives the stages: reading while it reads, then the pass
   * of the letter it is writing (`shown` letters written so far; null = not writing).
   */
  follow(shown: number | null): void {
    if (shown === null) return;
    const stage = shown === 0 ? 2 : 2 + shown;
    if (stage !== this.stage) { this.stage = Math.min(stage, this.last()); this.render(); }
  }

  private last(): number {
    return this.a ? 2 + this.a.steps.length : 0;
  }

  private go(stage: number): void {
    this.stage = Math.max(0, Math.min(this.last(), stage));
    this.render();
  }

  // -------------------------------------------------------------------------

  private act(name: string): Activation | undefined {
    return this.a?.activations.find((x) => x.name === name);
  }

  private att(re: RegExp): AttentionMap[] {
    return this.a?.attention.filter((x) => re.test(x.name)) ?? [];
  }

  private chip(token: string, cls = ''): HTMLElement {
    return tokenChip(toTok(token || EOS, this.idx), { expert: this.expert, langName: this.langName }, cls);
  }

  /** Writing pass k: the token that goes in (the task token, then the one written last). */
  private decInToken(k: number): string {
    const a = this.a!;
    return k === 0 ? a.input[0] : a.steps[k - 1].chosen || EOS;
  }

  private render(): void {
    const a = this.a;
    if (!a || !this.tokens.length) { this.body.replaceChildren(); this.pills.replaceChildren(); return; }
    const s = this.stage;
    this.back.disabled = s === 0;
    this.next.disabled = s === this.last();
    const names = [t('jr.p.tokens'), t('jr.p.lookup'), t('jr.p.read'),
      ...a.steps.map((st, k) => t('jr.p.write', { k: k + 1, tok: shortLabel(st.chosen) }))];
    this.pills.replaceChildren(...names.map((n, i) => {
      const b = h('button', { class: `jr-pill${i === s ? ' on' : ''}${i < s ? ' done' : ''}${i >= 3 ? ' write' : ''}`, 'data-stage': String(i) }, n);
      b.onclick = () => this.go(i);
      return b;
    }));
    this.body.replaceChildren(
      h('div', { class: 'jr-caption', 'data-test': 'journey-caption' }, ...this.caption(s)),
      this.diagram(s),
      this.detail(s));
  }

  /** What happens in this stage, in words, with the real values. */
  private caption(s: number): (HTMLElement | string)[] {
    const a = this.a!;
    const d = this.act('enc.in')?.d ?? 32;
    if (s === 0) {
      return [h('strong', {}, t('jr.c.tokensHead')), h('span', {}, t('jr.c.tokens', { word: a.text, n: a.input.length, task: this.full(a.input[0]) }))];
    }
    if (s === 1) {
      const first = a.input[1] ?? a.input[0];
      return [h('strong', {}, t('jr.c.lookupHead')), h('span', {}, t('jr.c.lookup', { tok: shortLabel(first), id: a.inputIds[1] ?? a.inputIds[0], d }))];
    }
    if (s === 2) return [h('strong', {}, t('jr.c.readHead')), h('span', {}, t('jr.c.read', { n: a.input.length, d }))];
    const k = s - 3;
    const st = a.steps[k];
    const tokIn = this.decInToken(k);
    const top = st.top[0];
    const end = st.chosen === '';
    const head = t('jr.c.writeHead', { k: k + 1, n: a.steps.length });
    const body = t(k === 0 ? 'jr.c.writeFirst' : 'jr.c.write', {
      tok: this.full(tokIn), pos: k + 1, v: a.vocabSize, chosen: this.full(st.chosen), p: Math.round((top?.p ?? 0) * 100),
    });
    const after = end ? t('jr.c.end', { word: a.output || '∅', n: a.steps.length }) : t('jr.c.back', { chosen: this.full(st.chosen), pos: k + 2 });
    return [h('strong', {}, head), h('span', {}, body), h('span', { class: end ? 'jr-end' : 'jr-loop' }, after)];
  }

  private full(token: string): string {
    return tokenName(token || EOS, { expert: this.expert, langName: this.langName });
  }

  /**
   * Two lanes: reading (one row per input token, done once) and writing (one
   * row per pass, revealed as the stages go on). Every row shows the real
   * numbers as strips.
   */
  private diagram(s: number): HTMLElement {
    const a = this.a!;
    const encIn = this.act('enc.in'), memory = this.act('memory'), decIn = this.act('dec.in'), final = this.act('dec.final');
    const d = encIn?.d ?? 32;
    const lk = a.lookup;
    // One colour scale per column, so rows can be compared.
    const sc = (x: Float32Array | undefined) => (x ? symmetricScale(x) : 1);
    const scTok = sc(new Float32Array([...lk.encTok, ...lk.decTok])), scPos = sc(new Float32Array([...lk.encPos, ...lk.decPos]));
    const scIn = sc(new Float32Array([...(encIn?.data ?? []), ...(decIn?.data ?? [])]));
    const scMem = sc(memory?.data), scFinal = sc(final?.data);
    const row = (data: Float32Array | undefined, r: number, scale: number, label: string) =>
      data ? this.strip(data.subarray(r * d, (r + 1) * d), scale, label) : h('span');

    // Reading lane.
    const writing = s >= 3 ? s - 3 : -1;
    const cross = writing >= 0 ? this.att(/^dec\d+\.cross$/)[0] : undefined;
    const readRows = a.input.map((tok, i) => {
      const cells: (HTMLElement | string)[] = [this.chip(tok)];
      if (s >= 1) {
        cells.push(row(lk.encTok, i, scTok, t('jr.s.tok', { tok: shortLabel(tok), id: a.inputIds[i] })), h('span', { class: 'op' }, '+'),
          row(lk.encPos, i, scPos, t('jr.s.pos', { n: i + 1 })), h('span', { class: 'op' }, '='),
          row(encIn?.data, i, scIn, t('jr.s.in')));
      }
      if (s >= 2) cells.push(h('span', { class: 'op arrow' }, '→'), row(memory?.data, i, scMem, t('jr.s.memory')));
      if (cross) cells.push(this.attCells(cross, writing, i));
      return h('div', { class: 'jr-row' }, ...cells);
    });
    const readHead = h('div', { class: 'jr-row jr-head small' },
      h('span', { class: 'jr-hcell chipcol' }, t('jr.h.token')),
      ...(s >= 1 ? [h('span', { class: 'jr-hcell' }, t('jr.h.table')), h('span', { class: 'op' }), h('span', { class: 'jr-hcell' }, t('jr.h.pos')), h('span', { class: 'op' }), h('span', { class: 'jr-hcell' }, t('jr.h.in'))] : []),
      ...(s >= 2 ? [h('span', { class: 'op' }), h('span', { class: 'jr-hcell' }, t('jr.h.memory'))] : []),
      ...(cross ? [h('span', { class: 'jr-hcell att' }, t('jr.h.looks', { k: writing + 1 }))] : []));
    const readLane = h('div', { class: `jr-lane read${s === 2 ? ' now' : ''}` },
      h('h3', {}, t('jr.lane.read')), readHead, ...readRows);

    // Writing lane: one row per pass so far.
    const lanes: HTMLElement[] = [readLane];
    if (s >= 3) {
      const writeRows = Array.from({ length: writing + 1 }, (_, k) => {
        const st = a.steps[k];
        const tokIn = this.decInToken(k);
        const now = k === writing;
        const top = st.top.find((g) => g.char === st.chosen) ?? st.top[0];
        const inChip = this.chip(tokIn, k > 0 ? 'jr-fed' : '');
        return h('div', { class: `jr-row${now ? ' now' : ' past'}`, 'data-pass': String(k) },
          k > 0 ? h('span', { class: 'jr-loopmark', title: t('jr.loopTip') }, '↩') : h('span', { class: 'jr-loopmark' }),
          inChip,
          row(lk.decTok, k, scTok, t('jr.s.tok', { tok: shortLabel(tokIn), id: this.idx.get(tokIn) ?? -1 })), h('span', { class: 'op' }, '+'),
          row(lk.decPos, k, scPos, t('jr.s.pos', { n: k + 1 })), h('span', { class: 'op' }, '='),
          row(decIn?.data, k, scIn, t('jr.s.in')),
          h('span', { class: 'op arrow jr-model', title: t('jr.modelTip') }, t('jr.model')),
          row(final?.data, k, scFinal, t('jr.s.final')),
          h('span', { class: 'op arrow' }, '→'),
          h('span', { class: 'jr-out' }, this.chip(st.chosen, 'jr-chosen'), h('small', {}, `${Math.round((top?.p ?? 0) * 100)} %`)));
      });
      const writeHead = h('div', { class: 'jr-row jr-head small' },
        h('span', { class: 'jr-loopmark' }), h('span', { class: 'jr-hcell chipcol' }, t('jr.h.goesIn')),
        h('span', { class: 'jr-hcell' }, t('jr.h.table')), h('span', { class: 'op' }), h('span', { class: 'jr-hcell' }, t('jr.h.pos')), h('span', { class: 'op' }),
        h('span', { class: 'jr-hcell' }, t('jr.h.in')), h('span', { class: 'op arrow jr-model' }), h('span', { class: 'jr-hcell' }, t('jr.h.final')),
        h('span', { class: 'op' }), h('span', { class: 'jr-hcell' }, t('jr.h.writes')));
      lanes.push(h('div', { class: 'jr-lane write now' }, h('h3', {}, t('jr.lane.write')), writeHead, ...writeRows));
    }
    return h('div', { class: 'jr-diagram', 'data-test': 'journey-diagram' }, ...lanes);
  }

  /** How much writing pass `k` looks at input token `i`, per head. */
  private attCells(m: AttentionMap, k: number, i: number): HTMLElement {
    const cells = Array.from({ length: m.heads }, (_, hd) => {
      const w = m.data[hd * m.Tq * m.Tk + k * m.Tk + i] ?? 0;
      return h('span', { class: 'xatt jr-att', style: `background:${sequential(w)};color:${w > 0.6 ? '#fff' : 'inherit'}`, title: t('jr.attTip', { n: hd + 1, p: Math.round(w * 100) }) },
        h('small', {}, `${Math.round(w * 100)}`));
    });
    return h('span', { class: 'jr-atts' }, ...cells);
  }

  /** Below the diagram: the details of the current stage. */
  private detail(s: number): HTMLElement {
    const a = this.a!;
    const box = h('div', { class: 'jr-detail' });
    if (s === 0) {
      const letters = [...a.text];
      box.append(
        h('div', { class: 'jr-split' },
          h('span', { class: 'small' }, t('jr.d.word')), h('b', { class: 'jr-word' }, a.text), h('span', { class: 'op' }, '→'),
          ...letters.map((c) => h('span', { class: 'jr-letter' }, c === ' ' ? '␣' : c))),
        h('div', { class: 'jr-split' },
          h('span', { class: 'small' }, t('jr.d.tokens')), ...a.input.map((tok) => this.chip(tok))),
        h('div', { class: 'jr-split' },
          h('span', { class: 'small' }, t('jr.d.numbers')), h('code', {}, `[${a.inputIds.join(', ')}]`)),
        h('p', { class: 'small' }, t('jr.d.tokensNote', { v: a.vocabSize })));
      return box;
    }
    if (s === 1) {
      box.append(h('p', { class: 'small' }, t('jr.d.lookup', { v: a.vocabSize, d: this.act('enc.in')?.d ?? 32 })));
      return box;
    }
    if (s === 2) {
      const self = this.att(/^enc\d+\.self$/);
      box.append(h('p', { class: 'small' }, t('jr.d.read')));
      for (const m of self) box.append(this.attTable(m, a.input, a.input, t('jr.d.readAtt')));
      box.append(h('p', { class: 'small' }, t('jr.d.readOnce')));
      return box;
    }
    const k = s - 3;
    const st = a.steps[k];
    const decIn = Array.from({ length: k + 1 }, (_, i) => this.decInToken(i));
    const self = this.att(/^dec\d+\.self$/), cross = this.att(/^dec\d+\.cross$/);
    const parts: HTMLElement[] = [];
    parts.push(h('div', { class: 'jr-sub' }, h('h4', {}, `1. ${t('jr.d.inTitle')}`),
      h('p', { class: 'small' }, t('jr.d.in', { tok: this.full(decIn[k]), id: this.idx.get(decIn[k]) ?? -1, pos: k + 1 }))));
    parts.push(h('div', { class: 'jr-sub' }, h('h4', {}, `2. ${t('jr.d.selfTitle')}`),
      h('p', { class: 'small' }, t(k === 0 ? 'jr.d.selfFirst' : 'jr.d.self')),
      ...self.map((m) => this.attRow(m, k, decIn))));
    parts.push(h('div', { class: 'jr-sub' }, h('h4', {}, `3. ${t('jr.d.crossTitle')}`),
      h('p', { class: 'small' }, t('jr.d.cross')),
      ...cross.map((m) => this.attRow(m, k, a.input))));
    parts.push(h('div', { class: 'jr-sub' }, h('h4', {}, `4. ${t('jr.d.thinkTitle')}`), h('p', { class: 'small' }, t('jr.d.think'))));
    const guesses = h('div', { class: 'jr-guesses' }, ...st.top.map((g) => h('div', { class: `jr-guess${g.char === st.chosen ? ' chosen' : ''}` },
      h('b', {}, shortLabel(g.char)),
      h('span', { class: 'track' }, h('span', { class: 'fill', style: `width:${Math.max(1, Math.round(g.p * 100))}%` })),
      h('span', { class: 'small' }, `${(g.p * 100).toFixed(1)} %`))));
    const rest = Math.max(0, 1 - st.top.reduce((x, g) => x + g.p, 0));
    parts.push(h('div', { class: 'jr-sub' }, h('h4', {}, `5. ${t('jr.d.guessTitle')}`),
      h('p', { class: 'small' }, t('jr.d.guess', { v: a.vocabSize })),
      guesses,
      h('p', { class: 'small' }, t('jr.d.rest', { n: a.vocabSize - st.top.length, p: (rest * 100).toFixed(1) })),
      h('p', { class: 'small' }, t('jr.d.pick', { chosen: this.full(st.chosen) }))));
    const look = h('button', { class: 'link', 'data-test': 'journey-inside' }, t('jr.d.inside'));
    look.onclick = () => this.lookInside(k);
    box.append(h('div', { class: 'jr-subs' }, ...parts), look);
    return box;
  }

  /** Attention of writing pass `k` over the given keys, one line per head. */
  private attRow(m: AttentionMap, k: number, keys: string[]): HTMLElement {
    const heads = Array.from({ length: m.heads }, (_, hd) => h('div', { class: 'xatt-head' },
      h('span', { class: 'xlabel small' }, t('explain.head', { n: hd + 1 })),
      ...keys.slice(0, m.Tk).map((key, i) => {
        const w = m.data[hd * m.Tq * m.Tk + k * m.Tk + i] ?? 0;
        return h('span', { class: 'xatt', style: `background:${sequential(w)};color:${w > 0.6 ? '#fff' : 'inherit'}`, title: `${this.full(key)}: ${Math.round(w * 100)} %` },
          h('b', {}, shortLabel(key)), h('small', {}, `${Math.round(w * 100)}`));
      })));
    return h('div', { class: 'xatt-wrap' }, ...heads);
  }

  /** A full attention map (rows look at columns), one table per head. */
  private attTable(m: AttentionMap, rows: string[], cols: string[], caption: string): HTMLElement {
    const tables = Array.from({ length: m.heads }, (_, hd) => h('table', { class: 'jr-atttable' },
      h('caption', { class: 'small' }, `${caption} · ${t('explain.head', { n: hd + 1 })}`),
      h('tr', {}, h('th', {}), ...cols.slice(0, m.Tk).map((c) => h('th', {}, shortLabel(c)))),
      ...rows.slice(0, m.Tq).map((r, i) => h('tr', {}, h('th', {}, shortLabel(r)),
        ...cols.slice(0, m.Tk).map((_, j) => {
          const w = m.data[hd * m.Tq * m.Tk + i * m.Tk + j];
          return h('td', { style: `background:${sequential(w)};color:${w > 0.6 ? '#fff' : 'inherit'}`, title: `${this.full(r)} → ${this.full(cols[j])}: ${Math.round(w * 100)} %` }, String(Math.round(w * 100)));
        })))));
    return h('div', { class: 'jr-atttables' }, ...tables);
  }

  /** One list of numbers as a thin strip; hover shows the values. */
  private strip(values: Float32Array, scale: number, label: string): HTMLElement {
    const layout: StripLayout = { n: values.length, perRow: values.length, cellW: CELL_W, cellH: CELL_H, numbers: false };
    const canvas = h('canvas', { class: 'jr-strip' });
    drawStrip(canvas, values, layout, scale);
    canvas.addEventListener('mousemove', (e) => {
      const i = Math.floor((e.offsetX - 1) / CELL_W);
      showTip(e, i >= 0 && i < values.length ? `${label}\n${t('axis.dim', { n: i + 1 })}: ${values[i].toFixed(3)}` : null);
    });
    canvas.addEventListener('mouseleave', (e) => showTip(e, null));
    return canvas;
  }
}
