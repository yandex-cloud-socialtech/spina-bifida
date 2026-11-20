import {Button, TextInput} from '@gravity-ui/uikit';
import {useEffect, useState, type FormEvent} from 'react';
import {useTranslation} from 'react-i18next';

import type {CreateUserInput, Locale, UserRole} from '../../api/contracts';

export function UserForm({
  pending,
  resetToken,
  onSubmit,
}: {
  pending: boolean;
  resetToken: number;
  onSubmit: (input: CreateUserInput) => void;
}) {
  const {t} = useTranslation();
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [role, setRole] = useState<UserRole>('user');
  const [locale, setLocale] = useState<Locale>('ru');

  useEffect(() => {
    if (resetToken > 0) {
      setEmail('');
      setPassword('');
      setRole('user');
      setLocale('ru');
    }
  }, [resetToken]);

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onSubmit({email, password, role, locale});
  }

  return (
    <form className="admin-user-form" onSubmit={submit}>
      <h2>{t('admin.create')}</h2>
      <div className="admin-form-field">
        <label htmlFor="admin-user-email">{t('admin.userEmail')}</label>
        <TextInput
          id="admin-user-email"
          type="email"
          autoComplete="off"
          value={email}
          onUpdate={setEmail}
          controlProps={{required: true}}
        />
      </div>
      <div className="admin-form-field">
        <label htmlFor="admin-user-password">{t('admin.temporaryPassword')}</label>
        <TextInput
          id="admin-user-password"
          type="password"
          autoComplete="new-password"
          value={password}
          onUpdate={setPassword}
          controlProps={{required: true, minLength: 12}}
        />
      </div>
      <div className="admin-form-field">
        <label htmlFor="admin-user-role">{t('admin.role')}</label>
        <select
          id="admin-user-role"
          value={role}
          onChange={(event) => setRole(event.target.value as UserRole)}
        >
          <option value="user">{t('admin.roles.user')}</option>
          <option value="admin">{t('admin.roles.admin')}</option>
        </select>
      </div>
      <div className="admin-form-field">
        <label htmlFor="admin-user-locale">{t('admin.userLocale')}</label>
        <select
          id="admin-user-locale"
          value={locale}
          onChange={(event) => setLocale(event.target.value as Locale)}
        >
          {(['ru', 'en', 'kk', 'uz'] as const).map((value) => (
            <option key={value} value={value}>
              {t(`locale.${value}`)}
            </option>
          ))}
        </select>
      </div>
      <Button view="action" type="submit" loading={pending} disabled={pending}>
        {t('admin.create')}
      </Button>
    </form>
  );
}
