import {useQuery} from '@tanstack/react-query';

import {apiClient} from '../../api/client';
import type {SystemInfoDto} from '../../api/contracts';

export const systemInfoQueryKey = ['system-info'] as const;

export function useSystemInfo(enabled: boolean) {
  return useQuery({
    queryKey: systemInfoQueryKey,
    queryFn: () => apiClient.request<SystemInfoDto>('/api/v1/system/info'),
    enabled,
    staleTime: Number.POSITIVE_INFINITY,
  });
}
