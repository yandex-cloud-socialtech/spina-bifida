import {useTranslation} from 'react-i18next';

import type {ModelVersion, UserDto} from '../api/contracts';
import kulakovLogo from '../assets/partners/kulakov-center.png';
import spinaBifidaFoundationLogo from '../assets/partners/spina-bifida-foundation.svg';
import yandexCloudLogo from '../assets/partners/yandex-cloud.svg';
import {ProfileMenu} from './ProfileMenu';
import {ServiceBadge} from './ServiceBadge';

export {LocaleSelector} from './ProfileMenu';

interface AppHeaderProps {
  user: UserDto;
  modelVersion: ModelVersion;
  modelPending: boolean;
  onModelVersionChange: (version: ModelVersion) => Promise<void>;
}

export function AppHeader({user, modelVersion, modelPending, onModelVersionChange}: AppHeaderProps) {
  const {t} = useTranslation();

  async function changeModel(version: ModelVersion) {
    await onModelVersionChange(version);
  }

  return (
    <header className="app-header">
      <div className="app-identity">
        <strong>{t('app.product')}</strong>
        <nav className="partner-logos" aria-label={t('partners.label')}>
          <a
            className="partner-logo partner-logo_yandex"
            href="https://yandex.cloud/"
            target="_blank"
            rel="noreferrer"
            aria-label={t('partners.yandexCloud')}
          >
            <img src={yandexCloudLogo} alt="" />
          </a>
          <a
            className="partner-logo partner-logo_foundation"
            href="https://helpspinabifida.ru/"
            target="_blank"
            rel="noreferrer"
            aria-label={t('partners.spinaBifidaFoundation')}
          >
            <img src={spinaBifidaFoundationLogo} alt="" />
          </a>
          <a
            className="partner-logo partner-logo_kulakov"
            href="https://ncagp.ru/"
            target="_blank"
            rel="noreferrer"
            aria-label={t('partners.kulakovCenter')}
          >
            <img src={kulakovLogo} alt="" />
          </a>
        </nav>
      </div>
      <ServiceBadge />
      <div className="header-actions">
        <label className="header-select model-control model-control_inline">
          <span>{t('model.label')}</span>
          <select
            aria-label={t('model.label')}
            value={modelVersion}
            disabled={modelPending}
            onChange={(event) => void changeModel(event.target.value as ModelVersion)}
          >
            <option value="v1">{t('model.v1')}</option>
            <option value="v2">{t('model.v2')}</option>
          </select>
        </label>
        <ProfileMenu user={user} />
      </div>
    </header>
  );
}
