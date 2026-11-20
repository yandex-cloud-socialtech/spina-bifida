import {useQuery} from '@tanstack/react-query';
import {useEffect} from 'react';

import {apiClient} from '../../api/client';
import type {AuthResponse} from '../../api/contracts';
import i18n from '../../i18n';

export const currentUserQueryKey = ['current-user'] as const;

export function useCurrentUser() {
  const query = useQuery({
    queryKey: currentUserQueryKey,
    queryFn: () => apiClient.request<AuthResponse>('/api/v1/auth/me'),
    staleTime: 30_000,
  });

  useEffect(() => {
    if (query.data?.user.locale && i18n.language !== query.data.user.locale) {
      void i18n.changeLanguage(query.data.user.locale);
    }
  }, [query.data?.user.locale]);

  return query;
}
