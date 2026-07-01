// src/components/ResultCard.jsx
import HighlightedText from './HighlightedText';

function CheckCircleIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="1.8" />
      <path d="M8 12.5L10.5 15L16 9" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function AlertTriangleIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 9v4M12 17h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M10.3 3.86L1.8 18a1.5 1.5 0 001.3 2.25h17.8a1.5 1.5 0 001.3-2.25L13.7 3.86a1.5 1.5 0 00-2.6 0z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

function ResultCard({ result }) {
  const isFake = result.label === 'FAKE';
  const pct = Math.round(result.confidence * 100);

  return (
    <div className={`result-card ${isFake ? 'result-fake' : 'result-real'}`}>
      <div className="result-header">
        <span className="result-icon">{isFake ? <AlertTriangleIcon /> : <CheckCircleIcon />}</span>
        <div>
          <h2 className="result-label">{isFake ? 'FAKE NEWS' : 'REAL NEWS'}</h2>
          <p className="result-subtitle">
            {isFake
              ? 'This article shows signs of misinformation.'
              : 'This article appears to be credible.'}
          </p>
        </div>
      </div>

      <div className="confidence-section">
        <span className="confidence-label">Confidence</span>
        <div className="confidence-bar-container">
          <div className="confidence-bar" style={{ width: `${pct}%` }} />
        </div>
        <span className="confidence-pct">{pct}%</span>
      </div>

      {result.article_text && (
        <div className="article-section">
          <h3>Analyzed Text</h3>
          <HighlightedText text={result.article_text} highlights={result.highlights} />
          {result.highlights?.length > 0 && (
            <p className="highlight-legend">
              <span className="highlighted-word">Highlighted words</span> flagged by AI.
            </p>
          )}
        </div>
      )}

      <p className="model-badge">Model: {result.model_used}</p>
    </div>
  );
}

export default ResultCard;