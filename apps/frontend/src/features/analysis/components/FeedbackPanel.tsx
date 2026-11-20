import {Button} from '@gravity-ui/uikit';
import {useState} from 'react';
import {useTranslation} from 'react-i18next';

import type {FeedbackValue} from '../../../api/contracts';
import type {AnalysisController, DraftFrame} from '../model';

export function FeedbackPanel({
  frame,
  disabled,
  controller,
}: {
  frame: DraftFrame;
  disabled: boolean;
  controller: AnalysisController;
}) {
  const {t} = useTranslation();
  const [value, setValue] = useState<FeedbackValue | null>(null);
  const [comment, setComment] = useState('');

  return (
    <div className="feedback-panel">
      <h3>{t('feedback.title')}</h3>
      {frame.feedback ? (
        <p role="status" className="feedback-sent">
          {t('feedback.sent')}
        </p>
      ) : (
        <>
          <fieldset>
            <legend className="visually-hidden">{t('feedback.title')}</legend>
            <label>
              <input
                type="radio"
                name="feedback"
                disabled={disabled}
                checked={value === 'positive'}
                onChange={() => setValue('positive')}
              />
              {t('feedback.positive')}
            </label>
            <label>
              <input
                type="radio"
                name="feedback"
                disabled={disabled}
                checked={value === 'negative'}
                onChange={() => setValue('negative')}
              />
              {t('feedback.negative')}
            </label>
          </fieldset>
          <label className="feedback-comment">
            <span>{t('feedback.comment')}</span>
            <textarea
              value={comment}
              disabled={disabled}
              onChange={(event) => setComment(event.target.value)}
            />
          </label>
          {frame.feedbackError ? (
            <p role="alert">{t(`errors.${frame.feedbackError}`, {defaultValue: t('errors.unknown')})}</p>
          ) : null}
          <Button
            view="action"
            disabled={!value || frame.feedbackPending || disabled}
            loading={frame.feedbackPending}
            onClick={() => {
              if (value) {
                void controller.submitFeedback(frame.id, value, comment);
              }
            }}
          >
            {t('feedback.submit')}
          </Button>
        </>
      )}
    </div>
  );
}
