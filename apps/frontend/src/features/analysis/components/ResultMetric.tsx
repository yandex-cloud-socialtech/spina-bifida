export function ResultMetric({
  label,
  outcome,
  score,
  direction,
  threshold,
  warningMargin,
  notDetermined,
}: {
  label: string;
  outcome: string;
  score: number | null;
  direction: 'higher' | 'lower';
  threshold: number | null;
  warningMargin: number;
  notDetermined: string;
}) {
  const finiteScore = score !== null && Number.isFinite(score) ? score : null;
  const normalizedScore = finiteScore === null ? null : Math.min(1, Math.max(0, finiteScore));
  const percent = normalizedScore === null ? null : Math.round(normalizedScore * 100);
  const tone = normalizedScore === null
    ? null
    : metricTone(normalizedScore, threshold, warningMargin, direction);

  return (
    <section className="result-metric">
      <h2>{label}</h2>
      <strong>{score !== null && finiteScore === null ? notDetermined : outcome}</strong>
      {percent === null ? null : (
        <div
          role="progressbar"
          aria-label={`${label}: ${percent}%`}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percent}
          className="result-metric__bar"
        >
          <b>{percent}%</b>
          <span className="result-metric__scale" aria-hidden="true">
            <span
              className={`result-metric__fill result-metric__fill_${tone}`}
              style={{width: `${percent}%`}}
            />
          </span>
        </div>
      )}
    </section>
  );
}

function metricTone(
  score: number,
  threshold: number | null,
  warningMargin: number,
  direction: 'higher' | 'lower',
): 'success' | 'warning' | 'danger' {
  if (threshold === null || !Number.isFinite(threshold)) {
    return 'warning';
  }
  const normalizedThreshold = Math.min(1, Math.max(0, threshold));
  const normalizedMargin = Math.min(1, Math.max(0, warningMargin));
  const lowerBound = Math.max(0, normalizedThreshold - normalizedMargin);
  const upperBound = Math.min(1, normalizedThreshold + normalizedMargin);

  if (normalizedMargin > 0 && score >= lowerBound && score <= upperBound) {
    return 'warning';
  }
  if (direction === 'higher') {
    return score >= normalizedThreshold ? 'success' : 'danger';
  }
  return score < normalizedThreshold ? 'success' : 'danger';
}
