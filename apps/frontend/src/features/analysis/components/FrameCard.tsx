import {useTranslation} from 'react-i18next';

import type {AnalysisController, DraftFrame} from '../model';

export function FrameCard({
  frame,
  index,
  selected,
  disabled,
  controller,
}: {
  frame: DraftFrame;
  index: number;
  selected: boolean;
  disabled: boolean;
  controller: AnalysisController;
}) {
  const {t} = useTranslation();
  return (
    <div className="frame-card">
      <button
        type="button"
        className="frame-card__select"
        aria-label={t('analysis.selectFrame', {count: index + 1})}
        aria-pressed={selected}
        onClick={() => controller.selectFrame(frame.id)}
      >
        <img src={frame.previewUrl} alt={frame.name} />
        <span>
          <b>{t('analysis.frame', {count: index + 1})}</b>
          <small>{frame.name}</small>
          {frame.job ? (
            <>
              <small>{t(`analysis.states.${frame.job.stage ?? frame.job.state}`)}</small>
              <small>{t('analysis.frameProgress', {count: frame.job.progress})}</small>
            </>
          ) : null}
        </span>
      </button>
      <button
        type="button"
        className="frame-card__remove"
        aria-label={t('analysis.removeFrame', {count: index + 1})}
        disabled={disabled}
        onClick={() => void controller.removeFrame(frame.id)}
      >
        ×
      </button>
    </div>
  );
}
