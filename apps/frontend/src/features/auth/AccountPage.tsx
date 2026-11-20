import {Button, Card, TextInput} from '@gravity-ui/uikit';
import {useQueryClient} from '@tanstack/react-query';
import {type FormEvent, useEffect, useState} from 'react';
import {useTranslation} from 'react-i18next';
import {Link, useLocation, useNavigate} from 'react-router-dom';

import {ApiError, apiClient} from '../../api/client';
import type {Locale} from '../../api/contracts';

type Mode = 'register' | 'forgot-password' | 'verify-email' | 'reset-password' | 'change-password';

export function AccountPage({mode}: {mode: Mode}) {
  const {t, i18n} = useTranslation();
  const queryClient = useQueryClient();
  const location = useLocation();
  const navigate = useNavigate();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmation, setConfirmation] = useState('');
  const [currentPassword, setCurrentPassword] = useState('');
  const [locale, setLocale] = useState<Locale>(
    ['ru', 'en', 'kk', 'uz'].includes(i18n.language) ? i18n.language as Locale : 'ru',
  );
  const [token, setToken] = useState(() =>
    new URLSearchParams(location.hash.slice(1)).get('token') ?? '',
  );
  const [pending, setPending] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const hasPassword = ['register', 'reset-password', 'change-password'].includes(mode);
  const hasEmail = ['register', 'forgot-password'].includes(mode);
  const needsToken = ['verify-email', 'reset-password'].includes(mode);

  useEffect(() => {
    void apiClient.request('/api/v1/auth/me').catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!needsToken || !location.hash) {
      return;
    }
    // A new email can navigate to this same route without remounting the form.
    // Capture its token before removing the fragment through the router.
    setToken(new URLSearchParams(location.hash.slice(1)).get('token') ?? '');
    setError(null);
    setSuccess(false);
    navigate({pathname: location.pathname, search: location.search, hash: ''}, {replace: true});
  }, [needsToken, location.hash, location.pathname, location.search, navigate]);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError(null);
    if (hasPassword && password !== confirmation) {
      setError('password_mismatch');
      return;
    }
    setPending(true);
    try {
      await apiClient.request(`/api/v1/auth/${mode}`, {
        method: 'POST',
        json: {email, password, locale, token, current_password: currentPassword},
      });
      setPassword('');
      setConfirmation('');
      setCurrentPassword('');
      setSuccess(true);
      // Change/reset revoke every session; the login page refreshes auth state.
      if (mode === 'reset-password' || mode === 'change-password') {
        apiClient.clearSession();
        queryClient.clear();
      }
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.code : 'unknown');
    } finally {
      setPending(false);
    }
  }

  function passwordField(id: string, label: string, value: string, onUpdate: (v: string) => void,
                         current = false) {
    return <div className="form-field">
      <label htmlFor={id}>{label}</label>
      <TextInput id={id} type="password" autoComplete={current ? 'current-password' : 'new-password'}
        value={value} onUpdate={onUpdate} size="l"
        controlProps={{required: true, minLength: current ? undefined : 12, maxLength: 256}} />
    </div>;
  }

  return <main className="login-page">
    <Card className="login-card" type="container" view="raised" size="l">
      <div className="login-brand" aria-hidden="true">SB</div>
      <h1>{t(`account.${mode}.title`)}</h1>
      <p>{t(`account.${mode}.intro`)}</p>
      {success ? <p role="status">{t(`account.${mode}.success`)}</p> :
        <form className="login-form" onSubmit={(event) => void submit(event)}>
          {hasEmail ? <div className="form-field">
            <label htmlFor="account-email">{t('auth.email')}</label>
            <TextInput id="account-email" type="email" autoComplete="email" value={email}
              onUpdate={setEmail} size="l" controlProps={{required: true, maxLength: 320}} />
          </div> : null}
          {mode === 'change-password' ? passwordField('current-password',
            t('account.currentPassword'), currentPassword, setCurrentPassword, true) : null}
          {hasPassword ? <>
            {passwordField('account-password', t('account.newPassword'), password, setPassword)}
            {passwordField('confirm-password', t('account.confirmPassword'), confirmation, setConfirmation)}
            <p className="password-hint">{t('account.passwordHint')}</p>
          </> : null}
          {mode === 'register' ? <div className="form-field">
            <label htmlFor="register-locale">{t('locale.label')}</label>
            <select id="register-locale" value={locale}
              onChange={(event) => setLocale(event.target.value as Locale)}>
              {(['ru', 'en', 'kk', 'uz'] as const).map((item) =>
                <option value={item} key={item}>{t(`locale.${item}`)}</option>)}
            </select>
          </div> : null}
          {needsToken && !token ? <p role="alert">{t('errors.invalid_token')}</p> : null}
          {error ? <div className="form-error" role="alert">
            {t(`errors.${error}`, {defaultValue: t('errors.unknown')})}
          </div> : null}
          <Button type="submit" view="action" size="l" loading={pending}
            disabled={pending || (needsToken && !token)} width="max">
            {t(`account.${mode}.action`)}
          </Button>
        </form>}
      <nav className="account-links" aria-label={t('account.navigation')}>
        <Link to="/login">{t('account.backToLogin')}</Link>
      </nav>
    </Card>
  </main>;
}
