import {Loader} from '@gravity-ui/uikit';
import {useEffect, useState, useSyncExternalStore} from 'react';
import {useTranslation} from 'react-i18next';
import {Navigate, Outlet, Route, Routes, useOutletContext} from 'react-router-dom';

import {AppHeader} from '../components/AppHeader';
import {AnalysisWorkspace} from '../features/analysis/AnalysisWorkspace';
import {AnalysisController} from '../features/analysis/model';
import {useSystemInfo} from '../features/analysis/system-info';
import {UserManagementPage} from '../features/admin/UserManagementPage';
import {AccountPage} from '../features/auth/AccountPage';
import {AuthGuard} from '../features/auth/AuthGuard';
import {LoginPage} from '../features/auth/LoginPage';
import {TermsGate} from '../features/auth/TermsGate';
import {useCurrentUser} from '../features/auth/useCurrentUser';

function ApplicationLayout() {
  const currentUser = useCurrentUser();
  const systemInfo = useSystemInfo(Boolean(currentUser.data));
  const {t} = useTranslation();

  if (!currentUser.data) {
    return null;
  }
  if (systemInfo.isPending) {
    return (
      <div className="screen-status" role="status" aria-label={t('common.loading')}>
        <Loader size="l" />
      </div>
    );
  }
  if (systemInfo.isError || !systemInfo.data) {
    return <div className="screen-status screen-status_error">{t('errors.network_error')}</div>;
  }
  return <ConfiguredApplicationLayout defaultModelVersion={systemInfo.data.default_model_version} />;
}

function ConfiguredApplicationLayout({
  defaultModelVersion,
}: {
  defaultModelVersion: 'v1' | 'v2';
}) {
  const currentUser = useCurrentUser();
  const [controller] = useState(
    () => new AnalysisController({initialModelVersion: defaultModelVersion}),
  );
  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  );

  useEffect(() => {
    const lease = controller.attach();
    return () => controller.detach(lease);
  }, [controller]);

  if (!currentUser.data) {
    return null;
  }
  return (
    <TermsGate auth={currentUser.data}>
      <div className="app-frame">
        <AppHeader
          user={currentUser.data.user}
          modelVersion={snapshot.modelVersion}
          modelPending={
            snapshot.submitting ||
            snapshot.clearing ||
            snapshot.modelPending ||
            snapshot.mutationPending ||
            Boolean(snapshot.cleanupPendingBatchId)
          }
          onModelVersionChange={(version) => controller.changeModel(version)}
        />
        <Outlet context={controller} />
      </div>
    </TermsGate>
  );
}

function AnalysisRoute() {
  const controller = useOutletContext<AnalysisController>();
  return <AnalysisWorkspace controller={controller} />;
}

function AdminRoute() {
  const currentUser = useCurrentUser();
  if (currentUser.data?.user.role !== 'admin') {
    return <Navigate to="/" replace />;
  }
  return <UserManagementPage currentUser={currentUser.data.user} />;
}

export function AppRoutes() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      {(['register', 'forgot-password', 'verify-email', 'reset-password'] as const).map((mode) =>
        <Route key={mode} path={`/${mode}`} element={<AccountPage key={mode} mode={mode} />} />)}
      <Route element={<AuthGuard />}>
        <Route path="change-password" element={<AccountPage mode="change-password" />} />
        <Route element={<ApplicationLayout />}>
          <Route index element={<AnalysisRoute />} />
          <Route path="admin" element={<AdminRoute />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
