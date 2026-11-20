import {render, screen} from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import {afterEach, describe, expect, it} from 'vitest';

import {AppProviders} from './providers';
import {useThemeSettings} from './theme';

function renderProviders() {
  function ThemeControl() {
    const {theme, setTheme} = useThemeSettings();
    return (
      <label>
        Тема
        <select aria-label="Тема" value={theme} onChange={(event) => setTheme(event.target.value as typeof theme)}>
          <option value="dark">Тёмная</option>
          <option value="light">Светлая</option>
        </select>
      </label>
    );
  }

  return render(
    <AppProviders>
      <ThemeControl />
    </AppProviders>,
  );
}

afterEach(() => {
  localStorage.clear();
  delete document.documentElement.dataset.theme;
});

describe('theme settings', () => {
  it('defaults to dark and synchronizes the document theme before rendering children', () => {
    localStorage.clear();

    renderProviders();

    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(document.body).toHaveClass('g-root_theme_dark');
  });

  it('restores a saved light choice and persists a later theme change', async () => {
    localStorage.setItem('theme', 'light');
    const user = userEvent.setup();

    renderProviders();

    expect(document.documentElement.dataset.theme).toBe('light');
    expect(document.body).toHaveClass('g-root_theme_light');
    await user.selectOptions(screen.getByLabelText('Тема'), 'dark');
    expect(document.documentElement.dataset.theme).toBe('dark');
    expect(localStorage.getItem('theme')).toBe('dark');
  });

  it('falls back to dark for an invalid persisted value', () => {
    localStorage.setItem('theme', 'contrast');

    renderProviders();

    expect(document.documentElement.dataset.theme).toBe('dark');
  });
});
