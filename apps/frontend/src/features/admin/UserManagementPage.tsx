import {Button, Modal} from '@gravity-ui/uikit';
import {useMutation, useQuery, useQueryClient} from '@tanstack/react-query';
import {useCallback, useRef, useState} from 'react';
import {useTranslation} from 'react-i18next';

import {ApiError, apiClient} from '../../api/client';
import type {
  CreateUserInput,
  UserDto,
  UserResponse,
  UsersResponse,
} from '../../api/contracts';
import {UserForm} from './UserForm';
import {UserTable} from './UserTable';

export const adminUsersQueryKey = ['admin-users'] as const;

type Confirmation = {user: UserDto} | null;

export function UserManagementPage({currentUser}: {currentUser: UserDto}) {
  const {t} = useTranslation();
  const queryClient = useQueryClient();
  const headingRef = useRef<HTMLHeadingElement>(null);
  const returnFocusRef = useRef<HTMLButtonElement | null>(null);
  const [confirmation, setConfirmation] = useState<Confirmation>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [formResetToken, setFormResetToken] = useState(0);
  const usersQuery = useQuery({
    queryKey: adminUsersQueryKey,
    queryFn: () => apiClient.request<UsersResponse>('/api/v1/admin/users'),
  });

  const mutation = useMutation({
    mutationFn: async (operation: () => Promise<UserResponse>) => operation(),
    onSuccess: async () => {
      await queryClient.invalidateQueries({queryKey: adminUsersQueryKey});
    },
  });

  const reportError = useCallback((error: unknown) => {
    setSuccess(null);
    setActionError(error instanceof ApiError ? error.code : 'unknown');
  }, []);

  async function createUser(input: CreateUserInput) {
    setActionError(null);
    try {
      await mutation.mutateAsync(() =>
        apiClient.request<UserResponse>('/api/v1/admin/users', {
          method: 'POST',
          json: input,
        }),
      );
      setFormResetToken((value) => value + 1);
      setSuccess(t('admin.success.created'));
    } catch (error) {
      reportError(error);
    }
  }

  async function toggleBlock(user: UserDto) {
    if (user.id === currentUser.id) {
      return;
    }
    setActionError(null);
    const blocked = user.status !== 'blocked';
    try {
      await mutation.mutateAsync(() =>
        apiClient.request<UserResponse>(`/api/v1/admin/users/${user.id}`, {
          method: 'PATCH',
          json: {blocked},
        }),
      );
      setSuccess(t(blocked ? 'admin.success.blocked' : 'admin.success.unblocked'));
    } catch (error) {
      reportError(error);
    }
  }

  async function moderate(user: UserDto, approved: boolean) {
    setActionError(null);
    try {
      await mutation.mutateAsync(() =>
        apiClient.request<UserResponse>(`/api/v1/admin/users/${user.id}/moderate`, {
          method: 'POST', json: {approved},
        }),
      );
      setSuccess(t(approved ? 'admin.success.approved' : 'admin.success.rejected'));
    } catch (error) {
      reportError(error);
    }
  }

  async function deleteUser(user: UserDto) {
    setActionError(null);
    try {
      await mutation.mutateAsync(() =>
        apiClient.request<UserResponse>(`/api/v1/admin/users/${user.id}`, {method: 'DELETE'}),
      );
      setSuccess(t('admin.success.deleted'));
      queueMicrotask(() => headingRef.current?.focus());
    } catch (error) {
      reportError(error);
    }
  }

  function requestDelete(user: UserDto, trigger: HTMLButtonElement) {
    returnFocusRef.current = trigger;
    setConfirmation({user});
  }

  function closeConfirmation(returnFocus = true) {
    setConfirmation(null);
    if (returnFocus) {
      queueMicrotask(() => returnFocusRef.current?.focus());
    }
  }

  async function confirmAction() {
    if (!confirmation) {
      return;
    }
    const pending = confirmation;
    closeConfirmation(false);
    await deleteUser(pending.user);
  }

  return (
    <main className="admin-page">
      <header className="admin-page__header">
        <div>
          <span className="eyebrow">{t('admin.eyebrow')}</span>
          <h1 ref={headingRef} tabIndex={-1}>{t('admin.title')}</h1>
          <p>{t('admin.intro')}</p>
        </div>
      </header>
      <UserForm
        pending={mutation.isPending}
        resetToken={formResetToken}
        onSubmit={(input) => void createUser(input)}
      />
      {actionError ? (
        <div className="admin-message admin-message_error" role="alert">
          {t(`errors.${actionError}`, {defaultValue: t('errors.unknown')})}
        </div>
      ) : null}
      {success ? (
        <div className="admin-message admin-message_success" role="status">
          {success}
        </div>
      ) : null}
      <section className="admin-users" aria-labelledby="admin-users-heading">
        <h2 id="admin-users-heading">{t('admin.listTitle')}</h2>
        {usersQuery.isPending ? <p>{t('admin.loading')}</p> : null}
        {usersQuery.isError ? (
          <div className="admin-message admin-message_error" role="alert">
            <p>{t('admin.listError')}</p>
            <Button onClick={() => void usersQuery.refetch()}>{t('common.retry')}</Button>
          </div>
        ) : null}
        {usersQuery.data ? (
          <UserTable
            users={usersQuery.data.users}
            currentUserId={currentUser.id}
            pendingUserId={mutation.isPending ? "*" : null}
            onToggleBlock={(user) => void toggleBlock(user)}
            onDelete={requestDelete}
            onModerate={(user, approved) => void moderate(user, approved)}
          />
        ) : null}
      </section>
      <Modal
        open={Boolean(confirmation)}
        aria-labelledby="admin-confirmation-title"
        onOpenChange={(open) => {
          if (!open) {
            closeConfirmation();
          }
        }}
      >
        {confirmation ? (
          <div className="admin-confirmation">
            <h2 id="admin-confirmation-title">
              {t('admin.confirmDeleteTitle')}
            </h2>
            <p>{t('admin.confirmDelete', {email: confirmation.user.email})}</p>
            <div className="admin-confirmation__actions">
              <Button onClick={() => closeConfirmation()}>{t('common.cancel')}</Button>
              <Button
                view="flat-danger"
                onClick={() => void confirmAction()}
              >
                {t('admin.delete')}
              </Button>
            </div>
          </div>
        ) : null}
      </Modal>
    </main>
  );
}
