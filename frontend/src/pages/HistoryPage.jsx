// src/pages/HistoryPage.jsx
import { useState, useEffect } from 'react';
import { getHistory } from '../api/satyata';

function HistoryPage() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    getHistory()
      .then(setHistory)
      .catch((err) => setError(err.message || 'Could not load history. Is the backend running?'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="page-container">
        <h2 className="page-title">Verification History</h2>
        <p className="loading-text">Loading...</p>
      </div>
    );
  }

  return (
    <div className="page-container">
      <h2 className="page-title">Verification History</h2>

      {error && <p className="error-msg">{error}</p>}

      {!error && !history.length && (
        <div className="empty-state">No verifications yet. Go check some news!</div>
      )}

      {!error && history.length > 0 && (
        <div className="history-table-wrapper">
          <table className="history-table">
            <thead>
              <tr>
                <th>#</th>
                <th>Verdict</th>
                <th>Confidence</th>
                <th>Article Preview</th>
                <th>Date</th>
              </tr>
            </thead>
            <tbody>
              {history.map((item, i) => (
                <tr key={item.id}>
                  <td>{i + 1}</td>
                  <td>
                    <span className={`badge badge-${item.label.toLowerCase()}`}>
                      {item.label === 'FAKE' ? 'FAKE' : 'REAL'}
                    </span>
                  </td>
                  <td>{Math.round(item.confidence * 100)}%</td>
                  <td>{item.article_text.slice(0, 80)}...</td>
                  <td>{new Date(item.predicted_at).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default HistoryPage;