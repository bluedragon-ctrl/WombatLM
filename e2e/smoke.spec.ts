import { expect, test } from '@playwright/test';

test('loads, trains in the worker and answers a question', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');

  // The story starts with the Learning data tab: everything it learns from.
  await expect(page.locator('[data-test=tab-data]')).toHaveClass(/on/);
  await expect(page.locator('[data-test=data-summary]')).toContainText(/learning examples/, { timeout: 15_000 });
  await expect(page.locator('[data-test=data-table] tr.kept td.hidden').first()).toBeVisible();
  await page.click('[data-test=data-reveal]');
  await expect(page.locator('[data-test=data-table] tr.kept td.hidden')).toHaveCount(0);

  // Model learning: nothing is practised before a step.
  await page.click('[data-test=tab-learn]');
  await expect(page.locator('[data-test=learning]')).toContainText(/not learning/);
  await page.click('text=Step ×1');
  await expect(page.locator('[data-test=cycle] .stage.lit')).toHaveCount(4, { timeout: 5_000 });
  // The whole batch: learning data with the right answers, and what it answered before learning from them.
  await expect(page.locator('[data-test=practice] tr.ex')).toHaveCount(16);
  await expect(page.locator('[data-test=practice] td.answer .word-own')).toHaveCount(16);
  await expect(page.locator('[data-test=letter-map]')).toBeVisible();

  // Ask the model: the answer comes from the brain at step 1; no training controls here.
  await page.click('[data-test=tab-ask]');
  await expect(page.locator('[data-test=output] .word')).toBeVisible();
  await expect(page.locator('[data-test=stamp]')).toContainText(/step 1\b/);
  await expect(page.locator('[data-test=ask-status]')).toContainText(/step 1\b.*paused/);
  await expect(page.locator('[data-test=play]')).toBeHidden();

  await page.click('[data-test=go-learn]');
  await page.click('[data-test=play]');
  await expect(page.locator('[data-test=status]')).toContainText(/steps\/s/, { timeout: 10_000 });
  await expect(page.locator('[data-test=learning]')).toContainText(/numbers nudged/);
  // While it learns, the answer waits and says it is out of date.
  await page.click('[data-test=tab-ask]');
  await expect(page.locator('[data-test=ask-status]')).toContainText(/learning right now/);
  await expect(page.locator('[data-test=ask-again]')).toBeVisible({ timeout: 5_000 });
  await page.waitForTimeout(4_000);
  await page.click('[data-test=tab-learn]');
  await page.click('[data-test=play]');
  const step = Number((await page.textContent('[data-test=status]'))!.match(/[\d,  ]+/)![0].replace(/\D/g, ''));
  expect(step).toBeGreaterThan(50);
  await page.click('[data-test=tab-ask]');
  await expect(page.locator('[data-test=ask-again]')).toBeHidden();

  await page.fill('[data-test=ask]', 'kočka');
  await page.locator('[data-form=young]').click();
  await expect(page.locator('[data-test=output] .from')).toContainText('kočka');
  await expect(page.locator('canvas.bars')).toBeVisible();
  await page.click('[data-test=slow]');
  await expect(page.locator('[data-test=slow-caption]')).toContainText(/writing letter 1 of/, { timeout: 3_000 });
  await page.click('[data-test=slow]');
  await expect(page.locator('[data-test=slow-caption]')).toHaveText('');
  await expect(page.locator('[data-test=attention] canvas')).toHaveCount(3);

  // Inside the brain: heatmaps.
  await page.click('[data-test=tab-inside]');
  await expect(page.locator('[data-test=heatmaps] canvas[data-matrix=tokEmb]')).toBeVisible();
  await expect(page.locator('[data-test=mode-note]')).toContainText(/changed/);
  await page.click('[data-mode=grad]');
  await expect(page.locator('[data-test=mode-note]')).toContainText(/batch/);

  // A word clicked on the Learning data tab is asked on the Ask tab.
  await page.click('[data-test=tab-data]');
  await page.locator('[data-test=data-table] td.learned:not(.name)').first().click();
  await expect(page.locator('[data-test=tab-ask]')).toHaveClass(/on/);
  await expect(page.locator('[data-test=bars]')).toBeVisible();

  // Test: every animal tested, only while the tab is open; a click asks on the Ask tab.
  await page.click('[data-test=tab-test]');
  await expect(page.locator('[data-test=animals] tr').nth(50)).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('[data-test=test-when]')).toContainText(/step \d/);
  await page.locator('[data-test=animals] td.cell').first().click();
  await expect(page.locator('[data-test=bars]')).toBeVisible();

  // Expert mode (in the header) adds settings and the self-attention grids.
  await page.locator('[data-test=expert]').check();
  await expect(page.locator('[data-test=attention] canvas').nth(3)).toBeVisible();
  expect(errors).toEqual([]);
});
