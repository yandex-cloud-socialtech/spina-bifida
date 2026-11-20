import {Button} from '@gravity-ui/uikit';
import {useTranslation} from 'react-i18next';

import {
  FEEDBACK_ELIGIBLE_STATES,
  type AnalysisController,
  type AnalysisSnapshot,
} from '../model';
import {FeedbackPanel} from './FeedbackPanel';

export function AnalysisActions({
  snapshot,
  controller,
}: {
  snapshot: AnalysisSnapshot;
  controller: AnalysisController;
}) {
  const {t} = useTranslation();
  const selected = snapshot.frames.find((frame) => frame.id === snapshot.selectedFrameId);
  const isBusy = snapshot.mutationPending || Boolean(snapshot.cleanupPendingBatchId);

  return (
    <section className="result-actions">
      {selected?.job && FEEDBACK_ELIGIBLE_STATES.has(selected.job.state) ? (
        <FeedbackPanel
          key={`${selected.id}:${selected.jobId ?? ''}`}
          frame={selected}
          disabled={isBusy}
          controller={controller}
        />
      ) : null}
      {snapshot.error ? (
        <div className="analysis-error" role="alert">
          <strong>{t(`errors.${snapshot.error.code}`, {defaultValue: t('errors.unknown')})}</strong>
          <p>{t('analysis.errorRemediation')}</p>
        </div>
      ) : null}
      {snapshot.cleanupPendingBatchId ? (
        <Button
          disabled={snapshot.mutationPending}
          loading={snapshot.mutationPending}
          onClick={() => void controller.retryCleanup()}
        >
          {t('analysis.retryCleanup')}
        </Button>
      ) : null}
      <Button
        className="clear-action"
        view="outlined-danger"
        size="l"
        width="max"
        disabled={isBusy}
        loading={snapshot.clearing}
        onClick={() => void controller.clear()}
      >
        {t('analysis.clear')}
      </Button>
    </section>
  );
}
