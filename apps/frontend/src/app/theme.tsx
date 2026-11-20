import {ThemeProvider} from '@gravity-ui/uikit';
import {createContext, useContext, useMemo, useState, type PropsWithChildren} from 'react';

export type ThemeName = 'dark' | 'light';

interface ThemeSettings {
  theme: ThemeName;
  setTheme: (theme: ThemeName) => void;
}

const ThemeSettingsContext = createContext<ThemeSettings | null>(null);

function initialTheme(): ThemeName {
  const theme = localStorage.getItem('theme') === 'light' ? 'light' : 'dark';
  document.documentElement.dataset.theme = theme;
  return theme;
}

export function ThemeSettingsProvider({children}: PropsWithChildren) {
  const [theme, updateTheme] = useState<ThemeName>(initialTheme);
  const settings = useMemo<ThemeSettings>(
    () => ({
      theme,
      setTheme(nextTheme) {
        localStorage.setItem('theme', nextTheme);
        document.documentElement.dataset.theme = nextTheme;
        updateTheme(nextTheme);
      },
    }),
    [theme],
  );

  return (
    <ThemeSettingsContext.Provider value={settings}>
      <ThemeProvider theme={theme}>{children}</ThemeProvider>
    </ThemeSettingsContext.Provider>
  );
}

export function useThemeSettings(): ThemeSettings {
  const settings = useContext(ThemeSettingsContext);
  if (!settings) {
    throw new Error('useThemeSettings must be used within ThemeSettingsProvider');
  }
  return settings;
}
