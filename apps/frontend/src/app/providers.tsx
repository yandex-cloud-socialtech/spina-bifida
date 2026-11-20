import {QueryClient, QueryClientProvider} from '@tanstack/react-query';
import {useState, type PropsWithChildren} from 'react';

import {ThemeSettingsProvider} from './theme';

interface AppProvidersProps extends PropsWithChildren {
  queryClient?: QueryClient;
}

export function AppProviders({children, queryClient}: AppProvidersProps) {
  const [ownedQueryClient] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {retry: false},
        },
      }),
  );

  return (
    <ThemeSettingsProvider>
      <QueryClientProvider client={queryClient ?? ownedQueryClient}>{children}</QueryClientProvider>
    </ThemeSettingsProvider>
  );
}
