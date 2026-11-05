import { expect, test } from '@playwright/test';

test('loads, trains in the worker and answers a question', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.goto('/');
  await expect(page.locator('[data-test=output] .word')).toBeVisible({ timeout: 15_000 });
  await expect(page.locator('[data-test=animals] tr').nth(50)).toBeVisible();

  await page.click('[data-test=play]');
  await expect(page.locator('[data-test=status]')).toContainText(/steps\/s/, { timeout: 10_000 });
  await page.waitForTimeout(5_000);
  await page.click('[data-test=play]');
  const step = Number((await page.textContent('[data-test=status]'))!.match(/[\d,  ]+/)![0].replace(/\D/g, ''));
  expect(step).toBeGreaterThan(50);

  await page.fill('[data-test=ask]', 'kočka');
  await page.locator('[data-form=young]').click();
  await expect(page.locator('[data-test=output] .from')).toContainText('kočka');
  await expect(page.locator('canvas.bars')).toBeVisible();
  expect(errors).toEqual([]);
});
