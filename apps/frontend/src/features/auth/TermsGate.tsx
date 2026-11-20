import {Button, Modal} from '@gravity-ui/uikit';
import {useMutation, useQueryClient} from '@tanstack/react-query';
import {type PropsWithChildren, useState} from 'react';
import {useTranslation} from 'react-i18next';

import {ApiError, apiClient} from '../../api/client';
import type {AuthResponse} from '../../api/contracts';
import {currentUserQueryKey} from './useCurrentUser';

const TERMS_SECTIONS = [
  {titleKey: 'terms.sections.purpose.title', paragraphKeys: ['terms.sections.purpose.body']},
  {titleKey: 'terms.sections.audience.title', paragraphKeys: ['terms.sections.audience.body']},
  {
    titleKey: 'terms.sections.liability.title',
    paragraphKeys: [
      'terms.sections.liability.efforts',
      'terms.sections.liability.results',
      'terms.sections.liability.consultation',
    ],
  },
  {
    titleKey: 'terms.sections.data.title',
    paragraphKeys: [
      'terms.sections.data.deidentify',
      'terms.sections.data.feedback',
      'terms.sections.data.logging',
      'terms.sections.data.disclosure',
    ],
  },
  {
    titleKey: 'terms.sections.consent.title',
    paragraphKeys: [
      'terms.sections.consent.intro',
      'terms.sections.consent.notice',
      'terms.sections.consent.specialist',
      'terms.sections.consent.decisions',
      'terms.sections.consent.data',
    ],
  },
] as const;

interface TermsAcceptanceResponse {
  terms_accepted: boolean;
  terms_version: string;
}

interface TermsGateProps extends PropsWithChildren {
  auth: AuthResponse;
}

export function TermsGate({auth, children}: TermsGateProps) {
  const {t} = useTranslation();
  const queryClient = useQueryClient();
  const [acceptedCsrfToken, setAcceptedCsrfToken] = useState<string | null>(null);
  const acceptedInCurrentSession = auth.terms_accepted || acceptedCsrfToken === auth.csrf_token;
  const acceptance = useMutation({
    mutationFn: () =>
      apiClient.request<TermsAcceptanceResponse>('/api/v1/auth/terms/accept', {method: 'POST'}),
    onSuccess: (response) => {
      setAcceptedCsrfToken(auth.csrf_token);
      queryClient.setQueryData<AuthResponse>(currentUserQueryKey, (current) =>
        current ? {...current, ...response} : current,
      );
    },
  });
  const errorCode = acceptance.error instanceof ApiError ? acceptance.error.code : 'unknown';

  if (acceptedInCurrentSession) {
    return <>{children}</>;
  }

  return (
    <Modal
      open
      disableEscapeKeyDown
      disableOutsideClick
      disableVisuallyHiddenDismiss
      contentClassName="terms-modal"
      contentOverflow="auto"
      aria-labelledby="terms-title"
    >
      <article className="terms-modal__content">
        <h2 id="terms-title">{t('terms.title')}</h2>
        <p>{t('terms.intro')}</p>
        {TERMS_SECTIONS.map(({titleKey, paragraphKeys}) => (
          <section key={titleKey}>
            <h3>{t(titleKey)}</h3>
            {paragraphKeys.map((key) => (
              <p key={key}>{t(key)}</p>
            ))}
          </section>
        ))}
        <p className="terms-modal__important">{t('terms.important')}</p>
        {acceptance.isError ? (
          <div className="form-error" role="alert">
            {t(`errors.${errorCode}`, {defaultValue: t('errors.unknown')})}
          </div>
        ) : null}
        <Button
          view="action"
          loading={acceptance.isPending}
          onClick={() => acceptance.mutate()}
          autoFocus
        >
          {t('terms.accept')}
        </Button>
      </article>
    </Modal>
  );
}
