/**
 * Screenshots for reviewing the visuals in a pull request (seed 1). Opt-in,
 * because it trains for a while: `npm run screenshots`.
 * - M3: every panel at step 0 and after training (docs/screenshots/m3/).
 * - Learning vs answering: the Learning data tab, the Learn tab
 *   (one-step cycle, the batch), the out-of-date answer on the Ask tab while
 *   training, "write it slowly" and the Test tab
 *   (docs/screenshots/learn-vs-answer/).
 */
import { expect, test, Page } from '@playwright/test';

const DIR = 'docs/screenshots/m3';
const STEPS = Number(process.env.SCREENSHOT_STEPS ?? 2000);

test.skip(!process.env.SCREENSHOTS, 'set SCREENSHOTS=1 (npm run screenshots)');
test.use({ deviceScaleFactor: 2, viewport: { width: 1400, height: 900 } });

const stepOf = async (page: Page) => Number((await page.textContent('[data-test=status]'))!.match(/[\d,  ]+/)![0].replace(/\D/g, ''));

async function shoot(page: Page, when: string) {
  await page.click('[data-test=tab-ask]');
  await page.fill('[data-test=ask]', 'buvol');
  await page.locator('[data-form=female]').click();
  await page.click('[data-test=answer-now]');
  await expect(page.locator('[data-test=output] .from')).toContainText('buvol');
  await page.waitForTimeout(300);
  await page.locator('.panel.askbox').screenshot({ path: `${DIR}/${when}-guesses.png` });
  // Link a guess column to the attention rows by hovering it.
  const bars = (await page.locator('[data-test=bars]').boundingBox())!;
  await page.mouse.move(bars.x + 78 * 5 + 30, bars.y + 60);
  await page.locator('.panel.attention').screenshot({ path: `${DIR}/${when}-attention.png` });
  await page.click('[data-test=tab-learn]');
  await page.waitForTimeout(1200); // letter map glides into place
  await page.locator('.panel.map').screenshot({ path: `${DIR}/${when}-letter-map.png` });
  await page.click('[data-test=tab-letter]');
  await page.waitForTimeout(800);
  await page.locator('[data-test=explain]').screenshot({ path: `${DIR}/${when}-inside-explain.png` });
  // The scan and the heatmaps are on expert-only tabs.
  const expert = page.locator('[data-test=expert]');
  const wasExpert = await expert.isChecked();
  await expert.check();
  await page.click('[data-test=tab-scan]');
  await page.waitForTimeout(800);
  await page.locator('[data-test=scan]').screenshot({ path: `${DIR}/${when}-inside-scan.png` });
  await page.click('[data-test=tab-numbers]');
  await page.click('.numbers [data-mode=delta]');
  await page.waitForTimeout(800);
  await page.locator('.numbers .heat-head').screenshot({ path: `${DIR}/${when}-inside-head.png` });
  await page.locator('.numbers .group.letters').screenshot({ path: `${DIR}/${when}-inside-letters.png` });
  await page.locator('.numbers .group.decoder').screenshot({ path: `${DIR}/${when}-inside-decoder.png` });
  await page.click('.numbers [data-mode=weights]');
  await page.waitForTimeout(600);
  await page.locator('.numbers .group.letters').screenshot({ path: `${DIR}/${when}-inside-letters-weights.png` });
  if (!wasExpert) await expert.uncheck();
  await page.click('[data-test=tab-learn]');
}

test('M3 panels at step 0 and after training', async ({ page }) => {
  test.setTimeout(STEPS * 60 + 120_000);
  await page.addInitScript(() => { localStorage.setItem('wombatlm.uiLang', 'en'); localStorage.setItem('wombatlm.introHidden', '1'); });
  await page.goto('/');
  await page.click('[data-test=tab-learn]');
  await page.fill('input.seed', '1');
  await page.click('text=Reset');
  await expect(page.locator('[data-test=status]')).toContainText(/Step 0/);
  await shoot(page, 'step0');
  await page.click('[data-test=play]');
  await expect.poll(() => stepOf(page), { timeout: STEPS * 60 }).toBeGreaterThanOrEqual(STEPS);
  await page.click('[data-test=play]');
  await shoot(page, `step${STEPS}`);
  await page.locator('[data-test=expert]').check();
  await page.click('[data-test=tab-ask]');
  await page.locator('.panel.attention').screenshot({ path: `${DIR}/expert-attention.png` });
  await page.click('[data-test=tab-numbers]');
  await page.locator('[data-test=signal]').screenshot({ path: `${DIR}/expert-signal.png` });
});

const NDIR = 'docs/screenshots/numbers';

test('the numbers tab (expert only)', async ({ page }) => {
  test.setTimeout(120_000);
  await page.setViewportSize({ width: 1400, height: 900 });
  await page.addInitScript(() => { localStorage.setItem('wombatlm.uiLang', 'en'); localStorage.setItem('wombatlm.introHidden', '1'); });
  await page.goto('/');
  await page.click('[data-test=tab-learn]');
  await page.fill('input.seed', '1');
  await page.click('text=Reset');
  await page.click('[data-test=play]');
  await expect.poll(() => stepOf(page), { timeout: 90_000 }).toBeGreaterThanOrEqual(1500);
  await page.click('[data-test=play]');
  await page.screenshot({ path: `${NDIR}/1-plain-tabs.png`, clip: { x: 0, y: 0, width: 1400, height: 60 } });
  await page.locator('[data-test=expert]').check();
  await page.click('text=Step ×1');
  await expect(page.locator('[data-test=cycle] .stage.lit')).toHaveCount(5, { timeout: 5_000 });
  await page.screenshot({ path: `${NDIR}/2-expert-tabs.png`, clip: { x: 0, y: 0, width: 1400, height: 60 } });
  await page.locator('.step-link').screenshot({ path: `${NDIR}/3-learning-link.png` });
  await page.click('[data-test=tab-ask]');
  await page.fill('[data-test=ask]', 'buvol');
  await page.click('[data-test=answer-now]');
  await expect(page.locator('[data-test=output] .from')).toContainText('buvol');
  await page.click('[data-test=tab-numbers]');
  await page.waitForTimeout(800);
  await page.locator('.nums-intro').screenshot({ path: `${NDIR}/4-intro.png` });
  await page.locator('[data-test=nums-matrices] summary').click();
  await page.locator('[data-test=nums-model]').screenshot({ path: `${NDIR}/5-model-in-numbers.png` });
  await page.locator('[data-test=overview]').screenshot({ path: `${NDIR}/6-how-fast.png` });
  await page.locator('.numbers .group.decoder').screenshot({ path: `${NDIR}/7-heatmaps-decoder.png` });
  await page.locator('[data-test=step-detail]').screenshot({ path: `${NDIR}/8-step-detail.png` });
  await page.screenshot({ path: `${NDIR}/9-numbers-tab.png`, fullPage: true });
});

const LDIR = 'docs/screenshots/learn-vs-answer';

test('learning vs answering', async ({ page }) => {
  test.setTimeout(180_000);
  await page.addInitScript(() => { localStorage.setItem('wombatlm.uiLang', 'en'); localStorage.setItem('wombatlm.introHidden', '1'); });
  await page.goto('/');
  // The Learning data tab: everything it learns from (the app opens on it).
  await expect(page.locator('[data-test=data-summary]')).toContainText(/Learning examples/, { timeout: 15_000 });
  await page.locator('main.data-main').screenshot({ path: `${LDIR}/12-learning-data-tab.png` });
  // One learning example as tokens, a guess pointed at.
  await page.locator('[data-test=example-grid] td.guess').nth(2).hover();
  await page.locator('[data-test=example]').screenshot({ path: `${LDIR}/17-learning-example.png` });
  await page.click('[data-test=tab-learn]');
  await page.fill('input.seed', '1');
  await page.click('text=Reset');
  await expect(page.locator('[data-test=status]')).toContainText(/Step 0/);
  await page.click('[data-test=tab-ask]');
  await page.fill('[data-test=ask]', 'tygr');
  await page.locator('[data-form=female]').click();
  await page.click('[data-test=answer-now]');
  await expect(page.locator('[data-test=output] .from')).toContainText('tygr');
  await page.waitForTimeout(300);
  await page.click('[data-test=tab-learn]');
  await page.locator('.panel.controls').screenshot({ path: `${LDIR}/1-build-step0.png` });
  // Train a while at max speed: the answer waits, marked out of date.
  await page.click('[data-test=play]');
  await expect.poll(() => stepOf(page), { timeout: 60_000 }).toBeGreaterThanOrEqual(600);
  await page.screenshot({ path: `${LDIR}/2-training-overview.png` });
  await page.click('[data-test=tab-ask]');
  await page.waitForTimeout(300);
  await page.locator('.panel.askbox').screenshot({ path: `${LDIR}/3-answer-out-of-date.png` });
  await page.click('[data-test=go-learn]');
  await page.click('[data-test=play]');
  await page.click('[data-test=tab-ask]');
  await expect(page.locator('[data-test=ask-again]')).toBeHidden();
  await page.locator('.panel.askbox').screenshot({ path: `${LDIR}/4-answer-fresh.png` });
  // Tokens: what it reads and writes, as letters with numbers and as numbers only.
  await page.locator('[data-test=tokens-out] .tok').nth(2).hover();
  await page.locator('.tokens-panel').screenshot({ path: `${LDIR}/15-tokens-ask.png` });
  await page.locator('[data-test=token-numbers]').check();
  await page.locator('.tokens-panel').screenshot({ path: `${LDIR}/16-tokens-numbers.png` });
  await page.locator('[data-test=token-numbers]').uncheck();
  // One step slowly: the four stages light up one after another, with the batch.
  await page.click('[data-test=tab-learn]');
  await page.click('text=Step ×1');
  await expect(page.locator('[data-test=cycle] .stage.current[data-stage=answer]')).toBeVisible();
  await page.locator('.panel.controls').screenshot({ path: `${LDIR}/5-cycle-answer.png` });
  await page.locator('main.learn-main').screenshot({ path: `${LDIR}/11-practice-tab-answer.png` });
  await expect(page.locator('[data-test=cycle] .stage.current[data-stage=measure]')).toBeVisible();
  await page.locator('.panel.controls').screenshot({ path: `${LDIR}/6-cycle-measure.png` });
  await expect(page.locator('[data-test=cycle] .stage.current')).toHaveCount(0, { timeout: 5_000 });
  await page.locator('.panel.controls').screenshot({ path: `${LDIR}/7-cycle-done.png` });
  await page.locator('main.learn-main').screenshot({ path: `${LDIR}/10-practice-tab.png` });
  await page.click('[data-test=tab-ask]');
  await page.locator('[data-test=stamp]').screenshot({ path: `${LDIR}/8-stamp-after-step.png` });
  // Answer slowly.
  await page.click('[data-test=answer-slow]');
  await expect(page.locator('[data-test=slow-caption]')).toContainText(/letter 3 of/, { timeout: 8_000 });
  await page.locator('main.ask-main').screenshot({ path: `${LDIR}/9-write-slowly.png` });
  // The Test tab: every animal tested.
  await page.click('[data-test=tab-test]');
  await expect(page.locator('[data-test=test-when]')).toContainText(/step \d/, { timeout: 10_000 });
  await page.locator('main.test-main').screenshot({ path: `${LDIR}/14-test-tab.png` });
});
