/**
 * Screenshots of every M3 panel at step 0 and after training (seed 1), for
 * reviewing the visuals in a pull request. Opt-in, because it trains for a
 * while: `npm run screenshots` (writes docs/screenshots/m3/).
 */
import { expect, test, Page } from '@playwright/test';

const DIR = 'docs/screenshots/m3';
const STEPS = Number(process.env.SCREENSHOT_STEPS ?? 2000);

test.skip(!process.env.SCREENSHOTS, 'set SCREENSHOTS=1 (npm run screenshots)');
test.use({ deviceScaleFactor: 2, viewport: { width: 1400, height: 900 } });

const stepOf = async (page: Page) => Number((await page.textContent('[data-test=status]'))!.match(/[\d,  ]+/)![0].replace(/\D/g, ''));

async function shoot(page: Page, when: string) {
  await page.click('[data-test=tab-forms]');
  await page.fill('[data-test=ask]', 'buvol');
  await page.locator('[data-form=female]').click();
  await page.waitForTimeout(1200); // letter map glides into place
  await page.locator('.panel.askbox').screenshot({ path: `${DIR}/${when}-guesses.png` });
  // Link a guess column to the attention rows by hovering it.
  const bars = (await page.locator('[data-test=bars]').boundingBox())!;
  await page.mouse.move(bars.x + 78 * 5 + 30, bars.y + 60);
  await page.locator('.panel.attention').screenshot({ path: `${DIR}/${when}-attention.png` });
  await page.locator('.panel.map').screenshot({ path: `${DIR}/${when}-letter-map.png` });
  await page.click('[data-test=tab-inside]');
  await page.waitForTimeout(800);
  await page.locator('.inside-head').screenshot({ path: `${DIR}/${when}-inside-head.png` });
  await page.locator('.group.letters').screenshot({ path: `${DIR}/${when}-inside-letters.png` });
  await page.locator('.group.decoder').screenshot({ path: `${DIR}/${when}-inside-decoder.png` });
  await page.click('[data-mode=weights]');
  await page.waitForTimeout(600);
  await page.locator('.group.letters').screenshot({ path: `${DIR}/${when}-inside-letters-weights.png` });
  await page.click('[data-mode=delta]');
  await page.click('[data-test=tab-forms]');
}

test('M3 panels at step 0 and after training', async ({ page }) => {
  test.setTimeout(STEPS * 60 + 120_000);
  await page.goto('/');
  await expect(page.locator('[data-test=output] .word')).toBeVisible({ timeout: 15_000 });
  await page.fill('input.seed', '1');
  await page.click('text=Reset');
  await expect(page.locator('[data-test=status]')).toContainText(/Step 0/);
  await shoot(page, 'step0');
  await page.click('[data-test=play]');
  await expect.poll(() => stepOf(page), { timeout: STEPS * 60 }).toBeGreaterThanOrEqual(STEPS);
  await page.click('[data-test=play]');
  await shoot(page, `step${STEPS}`);
  await page.locator('input[type=checkbox]').check();
  await page.locator('.panel.attention').screenshot({ path: `${DIR}/expert-attention.png` });
  await page.click('[data-test=tab-inside]');
  await page.locator('[data-test=signal]').screenshot({ path: `${DIR}/expert-signal.png` });
});
