import { expect, test } from '@playwright/test';

test('loads, trains in the worker and answers a question', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');
  await expect(page.locator('[data-test=output] .word')).toBeVisible({ timeout: 15_000 });
  // The full test is an expert extra: no Test tab, no animal table at first.
  await expect(page.locator('[data-test=tab-test]')).toBeHidden();
  await expect(page.locator('[data-test=animals]')).toBeHidden();

  // Learning and answering are separate: nothing is practised before a step.
  await expect(page.locator('[data-test=learning]')).toContainText(/not learning/);
  await expect(page.locator('[data-test=stamp]')).toContainText(/step 0/);
  await page.click('text=Step ×1');
  await expect(page.locator('[data-test=cycle] .stage.lit')).toHaveCount(4, { timeout: 5_000 });
  // The Practice tab shows the whole batch.
  await page.click('[data-test=open-practice]');
  await expect(page.locator('[data-test=practice] tr.ex')).toHaveCount(16);
  // Learning data with the right answers, and what it answered before learning from them.
  await expect(page.locator('[data-test=practice] td.answer .word-own')).toHaveCount(16);
  await expect(page.locator('[data-test=cycle-practice] .stage.lit')).toHaveCount(4);
  await page.click('[data-test=tab-forms]');
  await expect(page.locator('[data-test=stamp]')).toContainText(/step 1\b/);

  await page.click('[data-test=play]');
  await expect(page.locator('[data-test=status]')).toContainText(/steps\/s/, { timeout: 10_000 });
  await expect(page.locator('[data-test=learning]')).toContainText(/numbers nudged/);
  // While it learns, the answer waits and says it is out of date.
  await expect(page.locator('[data-test=ask-again]')).toBeVisible({ timeout: 5_000 });
  await page.waitForTimeout(4_000);
  await page.click('[data-test=play]');
  await expect(page.locator('[data-test=ask-again]')).toBeHidden();
  const step = Number((await page.textContent('[data-test=status]'))!.match(/[\d,  ]+/)![0].replace(/\D/g, ''));
  expect(step).toBeGreaterThan(50);

  await page.fill('[data-test=ask]', 'kočka');
  await page.locator('[data-form=young]').click();
  await expect(page.locator('[data-test=output] .from')).toContainText('kočka');
  await expect(page.locator('canvas.bars')).toBeVisible();
  await page.click('[data-test=slow]');
  await expect(page.locator('[data-test=slow-caption]')).toContainText(/writing letter 1 of/, { timeout: 3_000 });
  await page.click('[data-test=slow]');
  await expect(page.locator('[data-test=slow-caption]')).toHaveText('');

  // M3: attention grids, letter map, the Inside tab with heatmaps.
  await expect(page.locator('[data-test=attention] canvas')).toHaveCount(3);
  await expect(page.locator('[data-test=letter-map]')).toBeVisible();
  await page.click('[data-test=tab-inside]');
  await expect(page.locator('[data-test=heatmaps] canvas[data-matrix=tokEmb]')).toBeVisible();
  await expect(page.locator('[data-test=mode-note]')).toContainText(/changed/);
  await page.click('[data-mode=grad]');
  await expect(page.locator('[data-test=mode-note]')).toContainText(/batch/);
  await page.click('[data-test=tab-forms]');
  await expect(page.locator('[data-test=bars]')).toBeVisible();

  // The Learning data tab: everything it learns from; a click asks on the Forms tab.
  await page.click('[data-test=tab-data]');
  await expect(page.locator('[data-test=data-summary]')).toContainText(/learning examples/);
  await expect(page.locator('[data-test=data-table] tr.kept td.hidden').first()).toBeVisible();
  await page.click('[data-test=data-reveal]');
  await expect(page.locator('[data-test=data-table] tr.kept td.hidden')).toHaveCount(0);
  await page.locator('[data-test=data-table] td.learned:not(.name)').first().click();
  await expect(page.locator('[data-test=bars]')).toBeVisible();
  await expect(page.locator('[data-test=output] .badge')).toContainText(/.+/);

  // Expert mode adds the Test tab: every animal tested, on request.
  await page.locator('[data-test=expert]').check();
  await page.click('[data-test=tab-test]');
  await expect(page.locator('[data-test=animals] tr').nth(50)).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('[data-test=test-when]')).toContainText(/step \d/);
  await page.locator('[data-test=animals] td.cell').first().click(); // asks it on the Forms tab
  await expect(page.locator('[data-test=bars]')).toBeVisible();
  await page.locator('[data-test=expert]').uncheck();
  await expect(page.locator('[data-test=tab-test]')).toBeHidden();
  expect(errors).toEqual([]);
});
