import Person from '@gravity-ui/icons/Person';
import {Button, Icon} from '@gravity-ui/uikit';
import {useQueryClient} from '@tanstack/react-query';
import {type ChangeEvent, useEffect, useMemo, useRef, useState} from 'react';
import {useTranslation} from 'react-i18next';
import {Link, useNavigate} from 'react-router-dom';

import {apiClient} from '../api/client';
import type {AuthResponse, Locale, UserDto} from '../api/contracts';
import {useThemeSettings} from '../app/theme';
import i18n from '../i18n';
import {currentUserQueryKey} from '../features/auth/useCurrentUser';

const LOCALES: Locale[] = ['ru', 'en', 'kk', 'uz'];

interface LocaleSelectorProps {
  currentLocale: Locale;
  onUpdated?: (user: UserDto) => void;
}

export function LocaleSelector({currentLocale, onUpdated}: LocaleSelectorProps) {
  const {t} = useTranslation();
  const [pending, setPending] = useState(false);

  async function changeLocale(event: ChangeEvent<HTMLSelectElement>) {
    const locale = event.target.value as Locale;
    setPending(true);
    try {
      const response = await apiClient.request<{user: UserDto}>('/api/v1/auth/preferences', {
        method: 'PATCH',
        json: {locale},
      });
      await i18n.changeLanguage(locale);
      localStorage.setItem('locale', locale);
      onUpdated?.(response.user);
    } catch {
      await i18n.changeLanguage(currentLocale);
    } finally {
      setPending(false);
    }
  }

  return (
    <label className="header-select">
      <span>{t('locale.label')}</span>
      <select
        aria-label={t('locale.label')}
        value={currentLocale}
        disabled={pending}
        onChange={changeLocale}
      >
        {LOCALES.map((locale) => (
          <option value={locale} key={locale}>
            {t(`locale.${locale}`)}
          </option>
        ))}
      </select>
    </label>
  );
}

export function ProfileMenu({user}: {user: UserDto}) {
  const {t} = useTranslation();
  const {theme, setTheme} = useThemeSettings();
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const initials = useMemo(() => {
    const localPart = user.email.split('@', 1)[0] ?? '';
    return (localPart.match(/[a-z0-9]/gi) ?? []).slice(0, 2).join('').toUpperCase();
  }, [user.email]);

  useEffect(() => {
    if (!open) {
      return;
    }
    function closeOnOutsidePointer(event: PointerEvent) {
      if (!rootRef.current?.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    function closeOnEscape(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        setOpen(false);
        triggerRef.current?.focus();
      }
    }
    document.addEventListener('pointerdown', closeOnOutsidePointer);
    document.addEventListener('keydown', closeOnEscape);
    return () => {
      document.removeEventListener('pointerdown', closeOnOutsidePointer);
      document.removeEventListener('keydown', closeOnEscape);
    };
  }, [open]);

  async function logout() {
    await apiClient.request<void>('/api/v1/auth/logout', {method: 'POST'});
    apiClient.clearSession();
    queryClient.clear();
    navigate('/login', {replace: true});
  }

  function updateCachedUser(updated: UserDto) {
    queryClient.setQueryData<AuthResponse>(currentUserQueryKey, (current) =>
      current ? {...current, user: updated} : undefined,
    );
  }

  return (
    <div className="profile-menu" data-testid="profile-menu" ref={rootRef}>
      <button
        type="button"
        ref={triggerRef}
        className="profile-menu__trigger"
        aria-label={t('auth.profile')}
        aria-haspopup="dialog"
        aria-expanded={open}
        title={user.email}
        onClick={() => setOpen((current) => !current)}
      >
        <span className="profile-avatar" aria-hidden="true">
          {initials || <Icon data={Person} size={18} />}
        </span>
      </button>
      {open ? (
        <div className="profile-menu__popup" role="dialog" aria-label={t('auth.profile')}>
          <span className="profile-menu__email" title={user.email}>{user.email}</span>
          <label className="header-select profile-menu__row">
            <span>{t('theme.label')}</span>
            <select
              aria-label={t('theme.label')}
              value={theme}
              onChange={(event) => setTheme(event.target.value as typeof theme)}
            >
              <option value="dark">{t('theme.dark')}</option>
              <option value="light">{t('theme.light')}</option>
            </select>
          </label>
          <div className="profile-menu__row">
            <LocaleSelector currentLocale={user.locale} onUpdated={updateCachedUser} />
          </div>
          <div className="profile-menu__row">
            <Link to="/change-password" onClick={() => setOpen(false)}>
              {t('account.change-password.title')}
            </Link>
          </div>
          {user.role === 'admin' ? <div className="profile-menu__row">
            <Link to="/admin" onClick={() => setOpen(false)}>{t('admin.panel')}</Link>
          </div> : null}
          <hr className="profile-menu__divider" />
          <div className="profile-menu__row profile-menu__logout">
            <Button view="flat-danger" onClick={logout}>{t('auth.logout')}</Button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
