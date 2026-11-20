import {useTranslation} from 'react-i18next';

import type {AnalysisController, AnalysisSnapshot} from '../model';
import {AnalysisActions} from './AnalysisActions';
import {ResultMetric} from './ResultMetric';

export function ResultSummary({
  snapshot,
  controller,
}: {
  snapshot: AnalysisSnapshot;
  controller: AnalysisController;
}) {
  const {t} = useTranslation();
  const selected = snapshot.frames.find((frame) => frame.id === snapshot.selectedFrameId);
  const result = selected?.job?.result;
  const frame = result?.frames.find(
    (candidate) => candidate.frame_id === selected?.backendFrameId,
  ) ?? result?.frames[0];
  const thresholds = result?.thresholds;

  return (
    <aside className="result-panel">
      <ResultMetric
        label={t('analysis.result.planeHeading')}
        notDetermined={t('analysis.result.notDetermined')}
        outcome={
          frame
            ? t(`medical.${frame.plane.type}`, {defaultValue: t('medical.unknown')})
            : t('analysis.result.notDetermined')
        }
        score={frame?.plane.score ?? null}
        direction="higher"
        threshold={thresholds?.plane_detection ?? null}
        warningMargin={thresholds?.warning_margin ?? 0}
      />
      <ResultMetric
        label={t('analysis.result.qualityHeading')}
        notDetermined={t('analysis.result.notDetermined')}
        outcome={
          frame
            ? t(
                frame.quality.accepted
                  ? 'analysis.result.qualitySatisfactory'
                  : 'analysis.result.qualityUnsatisfactory',
              )
            : t('analysis.result.notDetermined')
        }
        score={frame?.quality.score ?? null}
        direction="higher"
        threshold={thresholds?.quality ?? null}
        warningMargin={thresholds?.warning_margin ?? 0}
      />
      <ResultMetric
        label={t('analysis.result.assessmentHeading')}
        notDetermined={t('analysis.result.notDetermined')}
        outcome={
          frame?.assessment
            ? t(
                frame.assessment.category === 'attention_required'
                  ? 'analysis.result.pathologyDetected'
                  : 'analysis.result.pathologyNotDetected',
              )
            : t('analysis.result.notDetermined')
        }
        score={frame?.assessment?.score ?? null}
        direction="lower"
        threshold={thresholds?.assessment ?? null}
        warningMargin={thresholds?.warning_margin ?? 0}
      />
      <p className="result-disclaimer">{t('analysis.result.probabilityNote')}</p>
      <AnalysisActions snapshot={snapshot} controller={controller} />
    </aside>
  );
}
