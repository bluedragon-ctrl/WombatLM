/**
 * The start-up window: a short introduction and one line about each tab, in
 * the order of the story. It opens at start until "Don't show this at start"
 * is ticked (kept in localStorage); the "About" link in the header opens it
 * again at any time.
 */
import { t } from './i18n';
import { h } from './dom';

const STORE_KEY = 'wombatlm.introHidden';

/** Tabs described in the window, in tab order; `expert`: shown only in expert mode. */
const STEPS: { tab: string; expert?: boolean }[] = [
  { tab: 'data' }, { tab: 'learn' }, { tab: 'ask' }, { tab: 'test' }, { tab: 'letter' },
  { tab: 'scan', expert: true }, { tab: 'matrix' }, { tab: 'numbers', expert: true },
];

function hiddenAtStart(): boolean {
  try { return localStorage.getItem(STORE_KEY) === '1'; } catch { return false; }
}

function setHiddenAtStart(on: boolean): void {
  try {
    if (on) localStorage.setItem(STORE_KEY, '1');
    else localStorage.removeItem(STORE_KEY);
  } catch { /* storage may be blocked */ }
}

export class IntroDialog {
  readonly el: HTMLDialogElement;

  /** `onPick`: a tab name in the list was clicked (the window closes and that tab opens). */
  constructor(onPick: (tab: string) => void) {
    const skip = h('input', { type: 'checkbox', 'data-test': 'intro-skip' });
    skip.checked = hiddenAtStart();
    skip.onchange = () => setHiddenAtStart(skip.checked);
    const start = h('button', { class: 'primary', 'data-test': 'intro-start' }, t('intro.start'));
    start.onclick = () => this.el.close();
    const steps = h('ol', { class: 'intro-steps' }, ...STEPS.map(({ tab, expert }) => {
      const name = h('button', { class: 'link intro-tab', 'data-test': `intro-${tab}` }, t(`tab.${tab}`));
      name.onclick = () => { this.el.close(); onPick(tab); };
      return h('li', {},
        name,
        expert ? h('span', { class: 'intro-expert' }, t('intro.expertOnly')) : '',
        h('span', { class: 'intro-text' }, t(`intro.${tab}`)));
    }));
    this.el = h('dialog', { class: 'intro', 'data-test': 'intro', 'aria-labelledby': 'intro-title' },
      h('h2', { id: 'intro-title' }, t('intro.title')),
      h('p', {}, t('intro.text')),
      h('p', {}, t('intro.stepsLead')),
      steps,
      h('p', { class: 'small' }, t('intro.expertNote')),
      h('div', { class: 'intro-foot' },
        h('label', { class: 'small' }, skip, ' ', t('intro.skip')),
        start));
    // A click on the dark backdrop (outside the window) closes it too.
    this.el.addEventListener('click', (e) => { if (e.target === this.el) this.el.close(); });
  }

  open(): void {
    if (!this.el.isConnected) document.body.append(this.el);
    if (!this.el.open) this.el.showModal();
    (this.el.querySelector('[data-test=intro-start]') as HTMLElement).focus();
  }

  /** Open at start unless the user asked not to. */
  openAtStart(): void {
    if (!hiddenAtStart()) this.open();
  }
}
