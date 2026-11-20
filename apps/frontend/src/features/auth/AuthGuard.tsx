import {Loader} from '@gravity-ui/uikit';
import {Navigate, Outlet, useLocation} from 'react-router-dom';
import {useTranslation} from 'react-i18next';

import {ApiError} from '../../api/client';
import {useCurrentUser} from './useCurrentUser';

export function AuthGuard() {
  const {t} = useTranslation();
  const location = useLocation();
  const currentUser = useCurrentUser();

  if (currentUser.isPending) {
    return (
      <div className="screen-status" role="status" aria-label={t('common.loading')}>
        <Loader size="l" />
      </div>
    );
  }
  if (currentUser.error instanceof ApiError && currentUser.error.status === 401) {
    return <Navigate to="/login" replace state={{from: location.pathname}} />;
  }
  if (currentUser.isError || !currentUser.data) {
    return <div className="screen-status screen-status_error">{t('errors.network_error')}</div>;
  }
  return <Outlet />;
}
