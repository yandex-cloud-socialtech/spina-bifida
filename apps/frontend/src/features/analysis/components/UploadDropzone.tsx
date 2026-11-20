import {Button} from '@gravity-ui/uikit';
import {useRef, useState, type DragEvent} from 'react';
import {useTranslation} from 'react-i18next';

import type {AnalysisController} from '../model';

export function UploadDropzone({controller, compact = false, disabled = false, onOpenExamples}: {
  controller: AnalysisController;
  compact?: boolean;
  disabled?: boolean;
  onOpenExamples?: () => void;
}) {
  const {t} = useTranslation();
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);

  function drop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    if (disabled) {
      return;
    }
    void controller.addFiles(event.dataTransfer.files);
  }

  return (
    <div
      className={`upload-dropzone${dragging ? ' upload-dropzone_dragging' : ''}${compact ? ' upload-dropzone_compact' : ''}`}
      onDragEnter={(event) => {
        event.preventDefault();
        if (!disabled) {
          setDragging(true);
        }
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={() => setDragging(false)}
      onDrop={drop}
    >
      <span className="upload-mark" aria-hidden="true">＋</span>
      <strong>{t('analysis.dropTitle')}</strong>
      {!compact ? <p>{t('analysis.dropHint')}</p> : null}
      <div className="upload-choices">
        <Button
          view={compact ? 'outlined-action' : 'action'}
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          {t('analysis.upload')}
        </Button>
        {!compact && onOpenExamples ? (
          <>
            <span className="upload-choices__or">{t('common.or')}</span>
            <Button view="outlined" disabled={disabled} onClick={onOpenExamples}>
              {t('analysis.loadExamples')}
            </Button>
          </>
        ) : null}
      </div>
      <input
        ref={inputRef}
        className="visually-hidden"
        tabIndex={-1}
        aria-label={t('analysis.upload')}
        type="file"
        accept="image/jpeg,image/png,.jpg,.jpeg,.png"
        multiple
        disabled={disabled}
        onChange={(event) => {
          if (event.target.files) {
            void controller.addFiles(event.target.files);
          }
          event.target.value = '';
        }}
      />
    </div>
  );
}
