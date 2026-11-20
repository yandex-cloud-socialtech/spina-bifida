import {Button, Label, Table, type TableColumnConfig} from '@gravity-ui/uikit';
import {useMemo, type MouseEvent} from 'react';
import {useTranslation} from 'react-i18next';

import type {UserDto} from '../../api/contracts';

type UserRow = UserDto & Record<string, unknown>;
type UserAction = (user: UserDto, trigger: HTMLButtonElement) => void;

export function UserTable({
  users,
  currentUserId,
  pendingUserId,
  onToggleBlock,
  onDelete,
  onModerate,
}: {
  users: UserDto[];
  currentUserId: string;
  pendingUserId: string | null;
  onToggleBlock: UserAction;
  onDelete: UserAction;
  onModerate: (user: UserDto, approved: boolean) => void;
}) {
  const {t} = useTranslation();
  const columns = useMemo<TableColumnConfig<UserRow>[]>(
    () => [
      {id: 'email', name: t('admin.email'), primary: true, template: (user) => user.email},
      {id: 'role', name: t('admin.role'), template: (user) => t(`admin.roles.${user.role}`)},
      {id: 'locale', name: t('admin.locale'), template: (user) => t(`locale.${user.locale}`)},
      {
        id: 'status',
        name: t('admin.status'),
        template: (user) => (
          <Label theme={user.status === 'active' ? 'success' : 'warning'}>
            {t(`admin.statuses.${user.status}`)}
          </Label>
        ),
      },
      {
        id: 'actions',
        name: t('admin.actions'),
        align: 'end',
        template: (user) => (
          <div className="admin-row-actions">
            {user.status === 'pending' ? <>
              <Button size="s" view="action" disabled={pendingUserId !== null}
                aria-label={t('admin.approveUser', {email: user.email})}
                onClick={() => onModerate(user, true)}>{t('admin.approve')}</Button>
              <Button size="s" view="flat-danger" disabled={pendingUserId !== null}
                aria-label={t('admin.rejectUser', {email: user.email})}
                onClick={() => onModerate(user, false)}>{t('admin.reject')}</Button>
            </> : null}
            {['active', 'blocked'].includes(user.status) ? <Button
              view="outlined"
              size="s"
              loading={(pendingUserId === "*" || pendingUserId === user.id)}
              disabled={pendingUserId !== null || user.id === currentUserId}
              title={user.id === currentUserId ? t('errors.self_block_forbidden') : undefined}
              aria-label={t(
                user.status === 'blocked' ? 'admin.unblockUser' : 'admin.blockUser',
                {email: user.email},
              )}
              onClick={(event: MouseEvent<HTMLButtonElement>) =>
                onToggleBlock(user, event.currentTarget)
              }
            >
              {t(user.status === 'blocked' ? 'admin.unblock' : 'admin.block')}
            </Button> : null}
            <Button
              view="flat-danger"
              size="s"
              disabled={(pendingUserId === "*" || pendingUserId === user.id) || user.status === 'deleted'}
              aria-label={t('admin.deleteUser', {email: user.email})}
              onClick={(event: MouseEvent<HTMLButtonElement>) =>
                onDelete(user, event.currentTarget)
              }
            >
              {t('admin.delete')}
            </Button>
          </div>
        ),
      },
    ],
    [currentUserId, onDelete, onModerate, onToggleBlock, pendingUserId, t],
  );

  return (
    <Table<UserRow>
      aria-label={t('admin.tableLabel')}
      data={users as UserRow[]}
      columns={columns}
      getRowDescriptor={(user) => ({id: user.id})}
      emptyMessage={t('admin.empty')}
      width="max"
      wordWrap
    />
  );
}
