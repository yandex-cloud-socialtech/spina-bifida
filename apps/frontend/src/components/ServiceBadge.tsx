import {Label} from '@gravity-ui/uikit';
import {useTranslation} from 'react-i18next';

export function ServiceBadge() {
  const {t} = useTranslation();
  return <Label theme="success">{t('serviceBadge.research')}</Label>;
}
