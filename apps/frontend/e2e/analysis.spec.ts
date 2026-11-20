import {expect, test, type Locator, type Page} from '@playwright/test';

import {
  assertNoPageScroll,
  mockApplication,
  prepareFourExamples,
  prepareTenFrames,
  startAnalysis,
} from './fixtures/analysis-responses';

async function focusByTab(page: Page, target: Locator, attempts = 80) {
  for (let index = 0; index < attempts; index += 1) {
    if (await target.evaluate((element) => element === document.activeElement).catch(() => false)) {
      return;
    }
    await page.keyboard.press('Tab');
  }
  throw new Error(`Keyboard focus did not reach ${await target.getAttribute('aria-label') ?? target}`);
}

test('login requires the complete terms on every new session before the workspace', async ({page}) => {
  const api = await mockApplication(page, {loginRequired: true});
  await page.goto('/');
  await expect(page).toHaveURL(/\/login$/);
  await page.getByLabel('Электронная почта').fill('doctor@example.test');
  await page.getByLabel('Пароль').fill('StrongPass123');
  await page.getByRole('button', {name: 'Войти'}).click();

  await expect(page.getByRole('heading', {name: 'Условия использования'})).toBeVisible();
  for (const heading of [
    '1. Назначение сервиса',
    '2. Целевая аудитория',
    '3. Ограничение ответственности',
    '4. Обработка данных',
    '5. Согласие пользователя',
  ]) {
    await expect(page.getByRole('heading', {name: heading})).toBeVisible();
  }
  await expect(page.getByText(/предварительная оценка, требующая интерпретации/)).toBeVisible();
  const accept = page.getByRole('button', {name: /Я прочитал\(а\)/});
  await focusByTab(page, accept);
  await page.keyboard.press('Enter');

  await expect(page.getByRole('main', {name: 'Рабочая область анализа'})).toBeVisible();
  expect(api.termsAccepts()).toBe(1);
});

test('one action loads all four examples and hides the offer', async ({page}) => {
  await mockApplication(page);
  await page.goto('/');
  const offer = page.getByRole('button', {name: 'Открыть примеры'});
  await offer.click();
  await expect(page.getByText('4 из 10')).toBeVisible();
  await expect(page.getByRole('button', {name: /Выбрать кадр/})).toHaveCount(4);
  await expect(offer).toHaveCount(0);
});

test('system information fixture matches the Flask response contract', async ({page}) => {
  await mockApplication(page);
  await page.goto('/');

  const payload = await page.evaluate(async () => {
    const response = await fetch('/api/v1/system/info');
    return response.json();
  });
  expect(payload).toEqual({
    application_version: '0.1.0',
    model_versions: ['v1', 'v2'],
    default_model_version: 'v2',
    schema_version: '0003',
    device: 'cpu',
  });
});

test('header exposes the application, three partner links, inline v2 model, and compact profile', async ({page}) => {
  await mockApplication(page);
  await page.goto('/');

  await expect(page.getByText('Spina Bifida', {exact: true})).toBeVisible();
  await expect(page.getByRole('link', {name: 'Yandex Cloud'})).toHaveAttribute('href', 'https://yandex.cloud/');
  await expect(page.getByRole('link', {name: 'Благотворительный фонд «Спина бифида»'})).toHaveAttribute('href', 'https://helpspinabifida.ru/');
  await expect(page.getByRole('link', {name: 'НМИЦ АГП им. В.И. Кулакова'})).toHaveAttribute('href', 'https://ncagp.ru/');
  await expect(page.getByRole('combobox', {name: 'Модель'})).toHaveValue('v2');
  expect(await page.locator('.model-control_inline').evaluate((element) => getComputedStyle(element).display)).toBe('flex');
  expect(await page.locator('.profile-menu__trigger').evaluate((element) => getComputedStyle(element).borderRadius)).toBe('50%');
});

test('rejected quality keeps the real assessment and derived image contract visible', async ({page}) => {
  const api = await mockApplication(page, {jobs: ['quality']});
  await prepareFourExamples(page);
  await startAnalysis(page);

  const payload = await page.evaluate(async () => {
    const response = await fetch('/api/v1/analyses/job-1');
    return response.json();
  });
  expect(payload.state).toBe('rejected_quality');
  expect(payload.result.frames[0]).toEqual(expect.objectContaining({
    analyzable: true,
    quality: {score: 0.18, accepted: false},
    plane: {type: 'sagittal', score: 0.91},
    roi: {
      box: [10, 20, 100, 140],
      display_box: [2, 8, 108, 152],
    },
    assessment: {category: 'no_expressed_signal', score: 0.14},
    warnings: ['quality_below_threshold'],
  }));
  expect(payload.result.frames[0].assets.map((asset: {kind: string}) => asset.kind)).toEqual([
    'original',
    'roi',
    'heatmap',
  ]);
  await expect(page.getByText('Неудовлетворительно', {exact: true})).toBeVisible();
  await expect(page.getByText('Не обнаружены', {exact: true})).toBeVisible();
  await expect(page.getByRole('progressbar', {name: 'Признаки патологии: 14%'})).toBeVisible();
  await expect(page.getByRole('img', {name: 'Оригинал'})).toBeVisible();
  await expect(page.getByRole('button', {name: 'ROI'})).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByRole('button', {name: 'Heatmap'})).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('.viewer-roi')).toBeVisible();
  await expect(page.locator('.viewer-heatmap')).toBeVisible();
  await page.getByRole('radio', {name: 'Требуется проверка'}).check();
  await page.getByLabel('Комментарий').fill('Проверить качество кадра');
  await page.getByRole('button', {name: 'Отправить'}).click();
  await expect(page.getByRole('status')).toHaveText('Обратная связь отправлена');
  expect(api.feedbackRequests).toEqual([
    {jobId: 'job-1', value: 'negative', comment: 'Проверить качество кадра'},
  ]);
});

test('ten files create independent mixed jobs and expose early terminal results', async ({page}) => {
  const api = await mockApplication(page, {
    jobs: [
      'negative',
      ['processing', 'negative'],
      'quality',
      'attention',
      'queued',
      'negative',
      'attention',
      'negative',
      'quality',
      'negative',
    ],
  });
  await prepareTenFrames(page);
  await startAnalysis(page);

  await expect(page.getByRole('progressbar', {name: 'Плоскость: 91%'})).toBeVisible();
  await expect(page.getByRole('button', {name: 'Выбрать кадр 2'})).toContainText(/Поиск области интереса|Анализ завершён/);
  await expect(page.getByRole('button', {name: 'Выбрать кадр 3'})).toContainText('Недостаточное качество');
  await expect(page.getByRole('button', {name: 'Выбрать кадр 5'})).toContainText('В очереди');
  await expect(page.getByText('Завершено 9 из 10')).toBeVisible();
  await expect(page.getByText('Обрабатывается: 0')).toBeVisible();
  await expect(page.getByText('В очереди: 1')).toBeVisible();
  await expect(page.getByRole('progressbar', {name: 'Прогресс пачки: 90%'})).toBeVisible();
  await page.getByRole('button', {name: 'Выбрать кадр 4'}).click();
  await expect(page.getByText('Обнаружены', {exact: true})).toBeVisible();
  await page.getByRole('button', {name: 'Выбрать кадр 3'}).click();
  await expect(page.getByText('Неудовлетворительно', {exact: true})).toBeVisible();

  expect(api.submissions).toHaveLength(1);
  expect(api.submissions[0].uploadedNames).toHaveLength(10);
  expect(api.submissions[0].modelVersion).toBe('v2');
  for (let index = 1; index <= 10; index += 1) {
    expect(api.jobReads(`job-${index}`)).toBeGreaterThan(0);
  }
});

test('changing v2 to v1 deletes the batch, preserves previews and resubmits', async ({page}) => {
  const api = await mockApplication(page, {defaultKind: 'negative'});
  await prepareFourExamples(page);
  const previews = await page.locator('.frame-card__select img').evaluateAll((images) =>
    images.map((image) => (image as HTMLImageElement).src),
  );
  await startAnalysis(page);
  await expect(page.getByRole('progressbar', {name: 'Плоскость: 91%'})).toBeVisible();

  await page.getByRole('combobox', {name: 'Модель'}).selectOption('v1');
  await expect.poll(() => api.deletedBatchIds).toContain('batch-1');
  await expect(page.getByText('4 из 10')).toBeVisible();
  expect(await page.locator('.frame-card__select img').evaluateAll((images) =>
    images.map((image) => (image as HTMLImageElement).src),
  )).toEqual(previews);

  await startAnalysis(page);
  await expect.poll(() => api.submissions.length).toBe(2);
  expect(api.submissions[1].modelVersion).toBe('v1');
  expect(api.submissions[1].uploadedNames).toEqual(api.submissions[0].uploadedNames);
});

test('one recoverable job retries without restarting completed siblings', async ({page}) => {
  const api = await mockApplication(page, {jobs: ['negative', 'recoverable', 'attention', 'negative']});
  await prepareFourExamples(page);
  await startAnalysis(page);
  await expect(page.getByRole('progressbar', {name: 'Плоскость: 91%'})).toBeVisible();
  const completedReads = api.jobReads('job-1');

  await page.getByRole('button', {name: 'Выбрать кадр 2'}).click();
  await expect(page.getByRole('button', {name: 'Повторить анализ'})).toBeVisible();
  await page.getByRole('button', {name: 'Повторить анализ'}).click();
  await expect(page.getByText('Не обнаружены', {exact: true})).toBeVisible();

  expect(api.retriedJobIds).toEqual(['job-2']);
  expect(api.submissions).toHaveLength(1);
  expect(api.jobReads('job-1')).toBe(completedReads);
});

test('feedback is persisted only for the selected frame', async ({page}) => {
  const api = await mockApplication(page, {jobs: ['negative', 'attention', 'negative', 'negative']});
  await prepareFourExamples(page);
  await startAnalysis(page);
  await page.getByRole('button', {name: 'Выбрать кадр 2'}).click();
  await expect(page.getByText('Обнаружены', {exact: true})).toBeVisible();
  await page.getByRole('radio', {name: 'Требуется проверка'}).check();
  await page.getByLabel('Комментарий').fill('Проверить выбранный кадр');
  await page.getByRole('button', {name: 'Отправить'}).click();
  await expect(page.getByRole('status')).toHaveText('Обратная связь отправлена');
  expect(api.feedbackRequests).toEqual([
    {jobId: 'job-2', value: 'negative', comment: 'Проверить выбранный кадр'},
  ]);

  await page.getByRole('button', {name: 'Выбрать кадр 1'}).click();
  await expect(page.getByText('Обратная связь отправлена')).toHaveCount(0);
  await page.getByRole('button', {name: 'Выбрать кадр 2'}).click();
  await expect(page.getByRole('status')).toHaveText('Обратная связь отправлена');
});

test('clear deletes the active batch and restores the example offer', async ({page}) => {
  const api = await mockApplication(page, {defaultKind: 'negative'});
  await prepareFourExamples(page);
  await startAnalysis(page);
  await expect(page.getByRole('progressbar', {name: 'Плоскость: 91%'})).toBeVisible();
  await page.getByRole('button', {name: 'Очистить'}).click();

  await expect(page.getByText('0 из 10')).toBeVisible();
  await expect(page.getByRole('button', {name: 'Открыть примеры'})).toBeVisible();
  expect(api.deletedBatchIds).toEqual(['batch-1']);
});

test('profile changes browser-local theme and persisted locale by keyboard', async ({page}) => {
  const api = await mockApplication(page, {initialLocale: 'en'});
  await page.goto('/');
  const profile = page.getByRole('button', {name: 'Profile'});
  await focusByTab(page, profile);
  await page.keyboard.press('Enter');
  const popup = page.getByTestId('profile-menu').locator('.profile-menu__popup');
  expect(await popup.evaluate((element) => getComputedStyle(element).backgroundColor)).not.toBe(
    'rgba(0, 0, 0, 0)',
  );
  const popupBounds = await popup.boundingBox();
  const offerBounds = await page.getByRole('button', {name: 'Open examples'}).boundingBox();
  if (popupBounds && offerBounds) {
    const overlapLeft = Math.max(popupBounds.x, offerBounds.x);
    const overlapRight = Math.min(
      popupBounds.x + popupBounds.width,
      offerBounds.x + offerBounds.width,
    );
    const overlapTop = Math.max(popupBounds.y, offerBounds.y);
    const overlapBottom = Math.min(
      popupBounds.y + popupBounds.height,
      offerBounds.y + offerBounds.height,
    );
    if (overlapLeft < overlapRight && overlapTop < overlapBottom) {
      expect(await page.evaluate(
        ({x, y}) => Boolean(document.elementFromPoint(x, y)?.closest('.profile-menu__popup')),
        {x: (overlapLeft + overlapRight) / 2, y: (overlapTop + overlapBottom) / 2},
      )).toBe(true);
    }
  }

  const theme = page.getByRole('combobox', {name: 'Theme'});
  await focusByTab(page, theme);
  await expect(theme).toBeFocused();
  await page.keyboard.press('l');
  await expect(theme).toHaveValue('light');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  expect(await page.evaluate(() => localStorage.getItem('theme'))).toBe('light');

  const locale = page.getByRole('combobox', {name: 'Language'});
  const stableLocaleControl = page.locator('.profile-menu__popup select').nth(1);
  await focusByTab(page, locale);
  await expect(locale).toBeFocused();
  await page.keyboard.press('o');
  await expect.poll(() => api.preferenceLocales).toEqual(['uz']);
  await expect(stableLocaleControl).toHaveValue('uz');
  expect(await page.evaluate(() => localStorage.getItem('locale'))).toBe('uz');
});

test('responsive workspace keeps ten cards and the profile popup inside the viewport', async ({page}) => {
  await mockApplication(page);
  await prepareTenFrames(page);
  expect(await page.evaluate(() => ({width: innerWidth, height: innerHeight}))).toEqual(
    expect.objectContaining({height: 900}),
  );
  expect(await assertNoPageScroll(page)).toBe(true);

  const rail = page.getByRole('complementary', {name: 'Кадры'});
  expect(await rail.evaluate((element) => element.scrollHeight > element.clientHeight)).toBe(true);
  await rail.evaluate((element) => { element.scrollTop = element.scrollHeight; });
  expect(await rail.evaluate((element) => element.scrollTop)).toBeGreaterThan(0);

  await page.getByRole('button', {name: 'Профиль'}).click();
  const popup = page.getByTestId('profile-menu').locator('.profile-menu__popup');
  await expect(popup).toBeVisible();
  expect(await popup.evaluate((element) => getComputedStyle(element).gridTemplateColumns.split(' ').length)).toBe(1);
  const bounds = await popup.boundingBox();
  if (!bounds) {
    throw new Error('Profile popup has no layout box');
  }
  const viewport = page.viewportSize();
  if (!viewport) {
    throw new Error('Responsive acceptance requires a fixed viewport');
  }
  expect(bounds.x).toBeGreaterThanOrEqual(0);
  expect(bounds.y).toBeGreaterThanOrEqual(0);
  expect(bounds.x + bounds.width).toBeLessThanOrEqual(viewport.width);
  expect(bounds.y + bounds.height).toBeLessThanOrEqual(viewport.height);
  const workspaceHeadingBounds = await page.getByRole('heading', {name: 'Анализ изображений'}).boundingBox();
  if (workspaceHeadingBounds) {
    const overlapLeft = Math.max(bounds.x, workspaceHeadingBounds.x);
    const overlapRight = Math.min(
      bounds.x + bounds.width,
      workspaceHeadingBounds.x + workspaceHeadingBounds.width,
    );
    const overlapTop = Math.max(bounds.y, workspaceHeadingBounds.y);
    const overlapBottom = Math.min(
      bounds.y + bounds.height,
      workspaceHeadingBounds.y + workspaceHeadingBounds.height,
    );
    if (overlapLeft < overlapRight && overlapTop < overlapBottom) {
      const popupOwnsOverlap = await page.evaluate(
        ({x, y}) => Boolean(document.elementFromPoint(x, y)?.closest('.profile-menu__popup')),
        {x: (overlapLeft + overlapRight) / 2, y: (overlapTop + overlapBottom) / 2},
      );
      expect(popupOwnsOverlap).toBe(true);
    }
  }
});

test('keyboard reaches model, frames, overlays, zoom, feedback and clear with semantic progress', async ({page}) => {
  await mockApplication(page, {jobs: ['negative', 'attention', 'negative', 'negative']});
  await prepareFourExamples(page);
  await startAnalysis(page);
  const planeProgress = page.getByRole('progressbar', {name: 'Плоскость: 91%'});
  await expect(planeProgress).toHaveAttribute('aria-valuemin', '0');
  await expect(planeProgress).toHaveAttribute('aria-valuemax', '100');
  await expect(planeProgress).toHaveAttribute('aria-valuenow', '91');

  await focusByTab(page, page.getByRole('combobox', {name: 'Модель'}));
  const secondFrame = page.getByRole('button', {name: 'Выбрать кадр 2'});
  await focusByTab(page, secondFrame);
  await page.keyboard.press('Enter');
  const roi = page.getByRole('button', {name: 'ROI'});
  await focusByTab(page, roi);
  await page.keyboard.press('Enter');
  await expect(roi).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('.viewer-roi')).toHaveCount(0);
  const heatmap = page.getByRole('button', {name: 'Heatmap'});
  await focusByTab(page, heatmap);
  await page.keyboard.press('Enter');
  await expect(heatmap).toHaveAttribute('aria-pressed', 'false');
  await expect(page.locator('.viewer-heatmap')).toHaveCount(0);
  const zoomIn = page.getByRole('button', {name: 'Увеличить масштаб'});
  await focusByTab(page, zoomIn);
  await page.keyboard.press('Enter');
  await expect(page.getByLabel('Масштаб 125%')).toBeVisible();
  const fit = page.getByRole('button', {name: 'По размеру'});
  await focusByTab(page, fit);
  await page.keyboard.press('Enter');
  await expect(page.getByLabel('Масштаб 100%')).toBeVisible();
  const positiveFeedback = page.getByRole('radio', {name: 'Результат полезен'});
  const feedback = page.getByRole('radio', {name: 'Требуется проверка'});
  await focusByTab(page, positiveFeedback);
  await page.keyboard.press('ArrowDown');
  await expect(feedback).toBeFocused();
  const submit = page.getByRole('button', {name: 'Отправить'});
  await focusByTab(page, submit);
  await page.keyboard.press('Enter');
  await expect(page.getByRole('status')).toHaveText('Обратная связь отправлена');
  const clear = page.getByRole('button', {name: 'Очистить'});
  await focusByTab(page, clear);
  await page.keyboard.press('Enter');
  await expect(page.getByRole('button', {name: 'Открыть примеры'})).toBeVisible();
});

test('keeps the preparation controls usable at 200 percent zoom', async ({page}) => {
  await mockApplication(page);
  const viewport = page.viewportSize();
  if (!viewport) {
    throw new Error('A fixed viewport is required for the zoom acceptance test');
  }
  await page.setViewportSize({
    width: Math.floor(viewport.width / 2),
    height: Math.floor(viewport.height / 2),
  });
  await page.goto('/');
  await expect(page.getByRole('button', {name: 'Открыть примеры'})).toBeVisible();
  await page.getByRole('button', {name: 'Открыть примеры'}).click();
  await expect(page.getByRole('button', {name: 'Запустить анализ'})).toBeVisible();
  expect(
    await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth),
  ).toBe(true);
});
