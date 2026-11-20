import {expect, test, type Page} from '@playwright/test';

import {
  assertNoPageScroll,
  mockApplication,
  prepareFourExamples,
  prepareTenFrames,
  startAnalysis,
  type JobKind,
} from './fixtures/analysis-responses';

type Theme = 'dark' | 'light';

async function useTheme(page: Page, theme: Theme) {
  await page.addInitScript((selectedTheme) => localStorage.setItem('theme', selectedTheme), theme);
}

async function snapshot(page: Page, name: string, viewer = false) {
  await page.evaluate(() => document.documentElement.setAttribute('data-visual-test', 'true'));
  await page.waitForFunction(() => {
    const theme = document.documentElement.dataset.theme;
    return theme === 'dark' || theme === 'light'
      ? document.body.classList.contains(`g-root_theme_${theme}`)
      : false;
  });
  await page.waitForFunction(() => {
    const header = document.querySelector<HTMLElement>('.app-header');
    const product = document.querySelector<HTMLElement>('.app-identity strong');
    const badge = Array.from(document.querySelectorAll<HTMLElement>('.app-header *')).find(
      (element) => element.textContent?.trim() === 'Только в исследовательских целях',
    );
    const profile = document.querySelector<HTMLElement>('.profile-menu__trigger');
    const model = document.querySelector<HTMLElement>('.model-control_inline');
    const partners = Array.from(document.querySelectorAll<HTMLAnchorElement>('.partner-logos a'));
    if (!header || !product || !badge || !profile || !model || partners.length !== 3) {
      return false;
    }
    const headerStyle = getComputedStyle(header);
    const productStyle = getComputedStyle(product);
    const badgeStyle = getComputedStyle(badge);
    const profileStyle = getComputedStyle(profile);
    return productStyle.color !== headerStyle.backgroundColor
      && badgeStyle.backgroundColor !== headerStyle.backgroundColor
      && profileStyle.borderRadius === '50%'
      && getComputedStyle(model).display === 'flex'
      && partners.every((partner) => partner.querySelector('img'));
  });
  await page.waitForFunction(() =>
    document.fonts.status === 'loaded' && Array.from(document.images).every((image) => image.complete),
  );
  expect(await assertNoPageScroll(page)).toBe(true);
  await expect(page).toHaveScreenshot(`${name}.png`, {
    animations: 'disabled',
    caret: 'hide',
    fullPage: false,
    maxDiffPixelRatio: 0.015,
  });
  if (viewer) {
    await expect(page.locator('.image-viewer')).toHaveScreenshot(`${name}-viewer.png`, {
      animations: 'disabled',
      caret: 'hide',
    });
  }
}

async function openResult(page: Page, theme: Theme, kind: JobKind) {
  await useTheme(page, theme);
  await mockApplication(page, {defaultKind: kind});
  await prepareFourExamples(page);
  await startAnalysis(page);
}

test('dark empty', async ({page}) => {
  await useTheme(page, 'dark');
  await mockApplication(page);
  await page.goto('/');
  await expect(page.getByText('Перетащите УЗИ-кадры сюда')).toBeVisible();
  await snapshot(page, 'dark-empty');
});

test('dark ten-frame', async ({page}) => {
  await useTheme(page, 'dark');
  await mockApplication(page);
  await prepareTenFrames(page);
  await snapshot(page, 'dark-ten-frame');
});

test('dark processing', async ({page}) => {
  await openResult(page, 'dark', 'processing');
  await expect(page.getByText('Поиск области интереса').first()).toBeVisible();
  await expect(page.getByText('Завершено 0 из 4')).toBeVisible();
  await expect(page.getByText('Обрабатывается: 4')).toBeVisible();
  await expect(page.getByText('В очереди: 0')).toBeVisible();
  await snapshot(page, 'dark-processing', true);
});

test('dark quality', async ({page}) => {
  await openResult(page, 'dark', 'quality');
  await expect(page.getByText('Неудовлетворительно', {exact: true})).toBeVisible();
  await expect(page.getByRole('progressbar', {name: 'Признаки патологии: 14%'})).toBeVisible();
  await snapshot(page, 'dark-quality', true);
});

test('dark negative', async ({page}) => {
  await openResult(page, 'dark', 'negative');
  await expect(page.getByText('Не обнаружены', {exact: true})).toBeVisible();
  await expect(page.locator('.viewer-roi')).toBeVisible();
  await expect(page.locator('.viewer-heatmap')).toBeVisible();
  await snapshot(page, 'dark-negative', true);
});

test('dark attention', async ({page}) => {
  await openResult(page, 'dark', 'attention');
  await expect(page.getByText('Обнаружены', {exact: true})).toBeVisible();
  await expect(page.locator('.viewer-roi')).toBeVisible();
  await expect(page.locator('.viewer-heatmap')).toBeVisible();
  await snapshot(page, 'dark-attention', true);
});

test('dark feedback', async ({page}) => {
  await openResult(page, 'dark', 'attention');
  await page.getByRole('radio', {name: 'Требуется проверка'}).check();
  await page.getByRole('button', {name: 'Отправить'}).click();
  await expect(page.getByRole('status')).toHaveText('Обратная связь отправлена');
  await snapshot(page, 'dark-feedback');
});

test('dark profile open', async ({page}) => {
  await useTheme(page, 'dark');
  await mockApplication(page);
  await page.goto('/');
  await page.getByRole('button', {name: 'Профиль'}).click();
  await expect(page.getByRole('dialog', {name: 'Профиль'})).toBeVisible();
  await snapshot(page, 'dark-profile-open');
});

test('light empty', async ({page}) => {
  await useTheme(page, 'light');
  await mockApplication(page);
  await page.goto('/');
  await expect(page.locator('html')).toHaveAttribute('data-theme', 'light');
  await snapshot(page, 'light-empty');
});

test('light result', async ({page}) => {
  await openResult(page, 'light', 'negative');
  await expect(page.getByText('Не обнаружены', {exact: true})).toBeVisible();
  await snapshot(page, 'light-result', true);
});

test('light profile open', async ({page}) => {
  await useTheme(page, 'light');
  await mockApplication(page);
  await page.goto('/');
  await page.getByRole('button', {name: 'Профиль'}).click();
  await expect(page.getByRole('dialog', {name: 'Профиль'})).toBeVisible();
  await snapshot(page, 'light-profile-open');
});
