import {Button} from '@gravity-ui/uikit';
import {useRef} from 'react';
import {useTranslation} from 'react-i18next';

import type {AnalysisController, AnalysisSnapshot} from '../model';

export function PreparationPanel({
  snapshot,
  controller,
}: {
  snapshot: AnalysisSnapshot;
  controller: AnalysisController;
}) {
  const {t} = useTranslation();
  const replacementRef = useRef<HTMLInputElement>(null);
  const selected = snapshot.frames.find((frame) => frame.id === snapshot.selectedFrameId);
  const isBusy =
    snapshot.submitting ||
    snapshot.clearing ||
    snapshot.modelPending ||
    snapshot.mutationPending ||
    Boolean(snapshot.cleanupPendingBatchId);
  const canStart =
    snapshot.frames.length > 0 &&
    !isBusy &&
    !snapshot.batchId &&
    !snapshot.cleanupPendingBatchId;
  const error = snapshot.error;

  return (
    <aside className="preparation-panel">
      <div>
        <span className="eyebrow">{t('analysis.preparation')}</span>
        <h1>{t('analysis.title')}</h1>
        <p>{t('analysis.intro')}</p>
      </div>
      <section className="preparation-section">
        <h2>{t('analysis.guidanceTitle')}</h2>
        <ul>
          <li>{t('analysis.guidanceSagittal')}</li>
          <li>{t('analysis.guidanceAxial')}</li>
          <li>{t('analysis.deidentifyWarning')}</li>
        </ul>
      </section>
      {error ? (
        <div className="analysis-error" role="alert">
          <strong>{t(`errors.${error.code}`, {defaultValue: t('errors.unknown')})}</strong>
          {error.fileName ? <span>{error.fileName}</span> : null}
          <p>
            {error.code === 'duplicate_image'
              ? t('analysis.duplicateRemediation')
              : t('analysis.errorRemediation')}
          </p>
        </div>
      ) : null}
      <div className="preparation-actions">
        {snapshot.cleanupPendingBatchId ? (
          <Button
            size="l"
            width="max"
            disabled={snapshot.mutationPending}
            loading={snapshot.mutationPending}
            onClick={() => void controller.retryCleanup()}
          >
            {t('analysis.retryCleanup')}
          </Button>
        ) : null}
        <Button
          view="action"
          size="l"
          width="max"
          disabled={!canStart}
          loading={snapshot.state === 'submitting'}
          onClick={() => void controller.start()}
        >
          {t('analysis.start')}
        </Button>
        {selected?.job?.state === 'failed_recoverable' ? (
          <Button
            size="l"
            width="max"
            disabled={isBusy || selected.retryPending}
            loading={selected.retryPending}
            onClick={() => void controller.retryFrame(selected.id)}
          >
            {t('analysis.retry')}
          </Button>
        ) : null}
        {selected ? (
          <Button view="flat" disabled={isBusy} onClick={() => replacementRef.current?.click()}>
            {t('analysis.replace')}
          </Button>
        ) : null}
        <input
          ref={replacementRef}
          className="visually-hidden"
          tabIndex={-1}
          aria-label={t('analysis.replace')}
          type="file"
          accept="image/jpeg,image/png,.jpg,.jpeg,.png"
          disabled={isBusy}
          onChange={(event) => {
            const file = event.target.files?.[0];
            if (file && selected) {
              void controller.replaceFrame(selected.id, file);
            }
            event.target.value = '';
          }}
        />
      </div>
    </aside>
  );
}
