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
  // The same data exactly as it is: every question, and the data records with every field.
  await page.click('[data-view=questions]');
  await expect(page.locator('[data-test=data-questions] tbody tr').first()).toBeVisible();
  await page.click('[data-view=records]');
  await expect(page.locator('[data-test=data-records]')).toContainText('"unseen": true');
  await page.click('[data-view=table]');
  // A learning example as tokens: shown so far and guess next, shifted by one, ending with "end".
  await expect(page.locator('[data-test=example-grid]')).toBeVisible();
  await page.locator('[data-test=data-table] td.learned:not(.name)').first().click();
  await expect(page.locator('[data-test=data-table] td.sel')).toHaveCount(1);
  const shown = page.locator('[data-test=example-grid] td.shown');
  const guess = page.locator('[data-test=example-grid] td.guess');
  await expect(guess.last()).toContainText('end');
  expect(await guess.count()).toBe(await shown.count());
  await guess.nth(1).hover();
  await expect(page.locator('[data-test=example-caption]')).toContainText(/must guess/);

  // Model learning: nothing is practised before a step.
  await page.click('[data-test=tab-learn]');
  await expect(page.locator('[data-test=learning]')).toContainText(/not learning/);
  await page.click('text=Step ×1');
  // Step ×1 while paused is recorded in detail: the fifth stage checks the same examples again.
  await expect(page.locator('[data-test=cycle] .stage.lit')).toHaveCount(5, { timeout: 5_000 });
  await expect(page.locator('[data-test=practice] td.check .check-loss')).toHaveCount(16);
  await expect(page.locator('[data-test=check-explain]')).toContainText(/After the step/);
  // What this step changed: the numbers that moved, and why the first one moved, to the last digit.
  await expect(page.locator('[data-test=sd-numbers] tbody tr')).toHaveCount(10);
  await expect(page.locator('[data-test=sd-why]')).toContainText(/exactly the model's number now/);
  await expect(page.locator('[data-test=sd-shares] tr')).not.toHaveCount(0);
  await page.click('[data-list=push]');
  await page.locator('[data-test=sd-numbers] tbody tr').nth(1).click();
  await expect(page.locator('[data-test=sd-numbers] tbody tr.on')).toHaveCount(1);
  await expect(page.locator('[data-test=sd-why]')).toContainText(/exactly the model's number now/);
  // The whole batch: learning data with the right answers, and what it answered before learning from them.
  await expect(page.locator('[data-test=practice] tr.ex')).toHaveCount(16);
  await expect(page.locator('[data-test=practice] td.answer .word-own')).toHaveCount(16);
  await expect(page.locator('[data-test=letter-map]')).toBeVisible();
  // Letter map: pick a letter (here by typing it) to see its closest letters.
  await expect(page.locator('[data-test=map-pick]')).toContainText(/Click a letter/);
  await page.focus('[data-test=letter-map]');
  await page.keyboard.press('a');
  await expect(page.locator('[data-test=map-pick] .map-chip.on')).toHaveText('a');
  await expect(page.locator('[data-test=map-pick] button.map-chip')).toHaveCount(4);
  await page.locator('[data-test=map-pick] button.map-chip').first().click();
  await expect(page.locator('[data-test=map-pick] .map-chip.on')).not.toHaveText('a');
  await page.focus('[data-test=letter-map]');
  await page.keyboard.press('Escape');
  await expect(page.locator('[data-test=map-pick]')).toContainText(/Click a letter/);

  // Ask the model: the answer comes from the model at step 1; no training controls here.
  await page.click('[data-test=tab-ask]');
  await expect(page.locator('[data-test=output] .word')).toBeVisible();
  await expect(page.locator('[data-test=stamp]')).toContainText(/step 1\b/);
  await expect(page.locator('[data-test=ask-status]')).toContainText(/step 1\b.*paused/);
  await expect(page.locator('[data-test=play]')).toBeHidden();

  await page.click('[data-test=go-learn]');
  await page.click('[data-test=play]');
  await expect(page.locator('[data-test=status]')).toContainText(/steps\/s/, { timeout: 10_000 });
  await expect(page.locator('[data-test=step-detail]')).toContainText(/Not shown while it learns/);
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

  // Tokens: what it reads (task token first) and what it writes (ending with "end" if finished).
  await expect(page.locator('[data-test=tokens-in] .tok').first()).toHaveClass(/tok-task/);
  await page.locator('[data-test=token-numbers]').check();
  await expect(page.locator('[data-test=tokens-in] .tok.num').first()).toHaveText(/^\d+$/);
  await page.locator('[data-test=token-numbers]').uncheck();
  // A new question is answered only when asked: language, word, which word, then a button.
  await page.fill('[data-test=ask]', 'zø');
  await expect(page.locator('[data-test=ask-prompt]')).toBeVisible();
  await expect(page.locator('[data-test=output] .word')).toHaveCount(0);
  await expect(page.locator('[data-test=word-hint]')).toContainText(/Not an animal/);
  await page.locator('[data-test=ask]').press('Enter');
  await expect(page.locator('[data-test=tokens-unk]')).toContainText('ø');
  await expect(page.locator('[data-test=tokens-in] .tok-unk')).toHaveCount(1);
  await page.fill('[data-test=ask]', '');
  await expect(page.locator('[data-test=answer-now]')).toBeDisabled();
  // A word of another language: offered, not switched behind your back.
  await page.fill('[data-test=ask]', 'cat');
  await expect(page.locator('[data-test=word-hint]')).toContainText(/in the English learning data/);
  await page.click('[data-test=switch-lang]');
  await expect(page.locator('.askbox [data-lang=en]')).toHaveClass(/on/);
  await expect(page.locator('[data-test=word-hint]')).toContainText(/from the learning data/);
  await page.locator('.askbox [data-lang=cs]').click();
  await page.fill('[data-test=ask]', 'kočka');
  await page.locator('[data-form=young]').click();
  await page.click('[data-test=answer-now]');
  await expect(page.locator('[data-test=output] .from')).toContainText('kočka');
  await expect(page.locator('canvas.bars')).toBeVisible();
  // Follow the tokens: tokens, letter table, reading once, then one writing pass per token.
  await expect(page.locator('[data-test=journey-caption]')).toContainText(/becomes tokens/);
  await page.click('[data-test=journey-next]');
  await page.click('[data-test=journey-next]');
  await expect(page.locator('[data-test=journey-caption]')).toContainText(/only once/);
  await page.click('[data-test=journey-next]');
  await expect(page.locator('[data-test=journey-caption]')).toContainText(/pass 1 of/);
  await page.click('[data-test=journey-next]');
  await expect(page.locator('[data-test=journey-diagram] [data-pass]')).toHaveCount(2);
  await expect(page.locator('[data-test=journey-caption]')).toContainText(/goes in at position 2/);
  // Answer slowly: what the model is doing, letter by letter; the same button stops it.
  await page.click('[data-test=answer-slow]');
  await expect(page.locator('[data-test=slow-caption]')).toContainText(/Reading the animal/, { timeout: 3_000 });
  await expect(page.locator('[data-test=slow-caption]')).toContainText(/Writing letter 1 of.*it writes/s, { timeout: 4_000 });
  await expect(page.locator('[data-test=answer-slow]')).toContainText('Stop');
  await page.click('[data-test=answer-slow]');
  await expect(page.locator('[data-test=slow-caption]')).toHaveText('');
  await expect(page.locator('[data-test=attention] canvas')).toHaveCount(2);

  // Inside the model: the brain scan, one tile per part of the model.
  await page.click('[data-test=tab-inside]');
  await expect(page.locator('[data-test=scan] .scan-tile')).toHaveCount(7);
  await expect(page.locator('[data-test=scan] .scan-tile').first()).toContainText(/2,343 numbers/);
  await expect(page.locator('[data-test=scan-verdict]')).not.toBeEmpty();
  // Plain mode: the heatmaps are folded away until a part of the scan is clicked.
  await expect(page.locator('[data-test=heatmaps]')).toBeHidden();
  await page.locator('[data-test=scan] .scan-tile[data-box=letters]').click();
  await expect(page.locator('[data-test=heatmaps] canvas[data-matrix=tokEmb]')).toBeVisible();
  await expect(page.locator('[data-test=mode-note]')).toContainText(/moved/);
  await page.click('[data-mode=grad]');
  await expect(page.locator('[data-test=mode-note]')).toContainText(/batch/);
  await expect(page.locator('[data-test=scan] .scan-tile').first()).toContainText(/of the push/);
  await expect(page.locator('[data-test=overview]')).toBeHidden();

  // Look inside one letter: from a click on a guess-bar column.
  await page.click('[data-test=tab-ask]');
  const bars = (await page.locator('[data-test=bars]').boundingBox())!;
  await page.mouse.click(bars.x + 78 + 30, bars.y + 60);
  await expect(page.locator('[data-test=tab-inside]')).toHaveClass(/on/);
  await expect(page.locator('[data-test=explain-caption]')).toContainText(/Letter 2 of/);
  await expect(page.locator('.xletter[data-pos="1"]')).toHaveClass(/on/);
  await expect(page.locator('[data-test=explain-guesses] button')).toHaveCount(5);
  // Plain mode: "what would it write now" after each stage, no rows of numbers.
  await expect(page.locator('[data-test=explain-ladder]')).toContainText(/How sure it is/);
  await expect(page.locator('[data-test=explain-lens]')).toHaveCount(4);
  await expect(page.locator('canvas[data-strip]')).toHaveCount(0);
  // What the detectors do: the strongest ones for this letter, and all of them in their own panel.
  await expect(page.locator('[data-test=explain-units] .ucard').first()).toBeVisible({ timeout: 10_000 });
  await expect(page.locator('[data-test=units-team]')).toContainText(/make a real difference on their own/);
  await expect(page.locator('[data-test=unit-cards] .ucard').first()).toContainText(/Detector \d+/);
  await expect(page.locator('[data-test=units-note]')).toContainText(/letters of the learning data/);
  await expect(page.locator('[data-test=explain-units] [data-test=unit-off-now]').first()).toContainText(/Without it now/);
  await expect(page.locator('[data-test=unit-cards] [data-test=unit-off]').first()).toContainText(/Switched off at these moments/);
  // Expert: every number, and where it comes from; how fast each part changes.
  await page.locator('[data-test=expert]').check();
  await expect(page.locator('[data-test=overview] .flow-box')).toHaveCount(7);
  await expect(page.locator('[data-test=overview] .flow-box').first()).toContainText(/per step/);
  await page.locator('canvas[data-strip="dec0.cross:add"]').click({ position: { x: 7, y: 7 } });
  await expect(page.locator('[data-test=explain-detail]')).toContainText(/Where it comes from/);
  await page.locator('[data-test=explain-guesses] button').first().click();
  await expect(page.locator('[data-test=explain-detail]')).toContainText(/favourite-letter bonus/);
  await page.locator('[data-test=expert]').uncheck();

  // A learning example can be asked on the Ask tab.
  await page.click('[data-test=tab-data]');
  await page.locator('[data-test=data-table] td.learned:not(.name)').first().click();
  await page.click('[data-test=example-ask]');
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

test('the matrix tab shows the answer station by station and follows one number', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.addInitScript(() => localStorage.setItem('wombatlm.uiLang', 'en'));
  await page.goto('/');
  await page.click('[data-test=tab-matrix]');
  // The answer asked at start: reading first, all input rows come in from the letter table.
  await expect(page.locator('[data-test=matrix-caption]')).toContainText(/Reading the animal, station 1 of 5: letters/, { timeout: 15_000 });
  await expect(page.locator('.mx-stations button')).toHaveCount(5);
  await page.click('[data-test=matrix-next]');
  await expect(page.locator('[data-test=matrix-caption]')).toContainText(/\+ position/);
  // The first writing pass: seven stations, ending with the guess.
  await page.locator('.mx-passes [data-pass="1"]').click();
  await expect(page.locator('.mx-stations button')).toHaveCount(7);
  await page.locator('.mx-stations [data-station=guess]').click();
  await expect(page.locator('[data-test=matrix-caption]')).toContainText(/compared with every row of the letter table/);
  // Follow one number: click a square of the matrix.
  await expect(page.locator('[data-test=matrix-trail]')).toContainText(/click any square/);
  const box = (await page.locator('[data-test=matrix-canvas]').boundingBox())!;
  await page.mouse.click(box.x + box.width - 30, box.y + 34 + 10);
  await expect(page.locator('[data-test=matrix-trail] .mx-trail-step')).toHaveCount(6);
  await page.click('[data-test=matrix-mode-added]');
  await page.click('[data-test=matrix-play]');
  await expect(page.locator('[data-test=matrix-play]')).toHaveText(/Stop/);
  await page.click('[data-test=matrix-play]');
  expect(errors).toEqual([]);
});
