import {Button, Card, TextInput} from '@gravity-ui/uikit';
import {useMutation, useQueryClient} from '@tanstack/react-query';
import {type FormEvent, useEffect, useState} from 'react';
import {useTranslation} from 'react-i18next';
import {Link, useLocation, useNavigate} from 'react-router-dom';

import {ApiError, apiClient} from '../../api/client';
import type {AuthResponse} from '../../api/contracts';
import i18n from '../../i18n';
import {currentUserQueryKey} from './useCurrentUser';

export function LoginPage() {
  const {t} = useTranslation();
  const navigate = useNavigate();
  const location = useLocation();
  const queryClient = useQueryClient();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  useEffect(() => {
    void apiClient.request('/api/v1/auth/me').catch(() => undefined);
  }, []);

  const login = useMutation({
    mutationFn: () =>
      apiClient.request<AuthResponse>('/api/v1/auth/login', {
        method: 'POST',
        json: {email, password},
      }),
    onSuccess: async (response) => {
      queryClient.setQueryData(currentUserQueryKey, response);
      await i18n.changeLanguage(response.user.locale);
      const from = (location.state as {from?: string} | null)?.from;
      navigate(from && from !== '/login' ? from : '/', {replace: true});
    },
  });

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    login.mutate();
  }

  const errorCode = login.error instanceof ApiError ? login.error.code : 'unknown';

  return (
    <main className="login-page">
      <Card className="login-card" type="container" view="raised" size="l">
        <div className="login-brand" aria-hidden="true">SB</div>
        <h1>{t('auth.loginTitle')}</h1>
        <p>{t('app.service')}</p>
        <form className="login-form" onSubmit={submit}>
          <div className="form-field">
            <label htmlFor="login-email">{t('auth.email')}</label>
            <TextInput
              id="login-email"
              name="email"
              type="email"
              autoComplete="username"
              value={email}
              onUpdate={setEmail}
              size="l"
            />
          </div>
          <div className="form-field">
            <label htmlFor="login-password">{t('auth.password')}</label>
            <TextInput
              id="login-password"
              name="password"
              type="password"
              autoComplete="current-password"
              value={password}
              onUpdate={setPassword}
              size="l"
            />
          </div>
          {login.isError ? (
            <div className="form-error" role="alert">
              {t(`errors.${errorCode}`, {defaultValue: t('errors.unknown')})}
            </div>
          ) : null}
          <Button type="submit" view="action" size="l" loading={login.isPending} width="max">
            {login.isPending ? t('auth.submitting') : t('auth.submit')}
          </Button>
        </form>
        <nav className="account-links" aria-label={t('account.navigation')}>
          <Link to="/register">{t('account.register.title')}</Link>
          <Link to="/forgot-password">{t('account.forgot-password.title')}</Link>
        </nav>
      </Card>
    </main>
  );
}
