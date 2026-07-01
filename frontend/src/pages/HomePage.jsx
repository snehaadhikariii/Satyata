// src/pages/HomePage.jsx
import { useState } from 'react';
import heroImg from '../assets/hero.png';
import { analyzeArticle } from '../api/satyata';
import ResultCard from '../components/ResultCard';

function ClipboardIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <rect x="6" y="4" width="12" height="17" rx="2" stroke="currentColor" strokeWidth="1.6" />
      <path d="M9 4V3a1 1 0 011-1h4a1 1 0 011 1v1" stroke="currentColor" strokeWidth="1.6" />
    </svg>
  );
}

function LinkIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M9 12a3 3 0 003 3h1a3 3 0 000-6h-1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
      <path d="M15 12a3 3 0 00-3-3h-1a3 3 0 000 6h1" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M12 9v4M12 17h.01" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
      <path d="M10.3 3.86L1.8 18a1.5 1.5 0 001.3 2.25h17.8a1.5 1.5 0 001.3-2.25L13.7 3.86a1.5 1.5 0 00-2.6 0z" stroke="currentColor" strokeWidth="1.6" strokeLinejoin="round" />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M5 12h14M13 6l6 6-6 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function MountainPlaceholderIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
      <path d="M2 19L8 9L12 15L15 11L22 19H2Z" stroke="currentColor" strokeWidth="1.4" strokeLinejoin="round" />
      <circle cx="17" cy="6" r="2" stroke="currentColor" strokeWidth="1.4" />
    </svg>
  );
}

function HomePage() {
  const [type, setType] = useState('text');
  const [value, setValue] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    if (!value.trim()) {
      setError('Please enter text or a URL.');
      return;
    }
    setLoading(true);
    try {
      const data = await analyzeArticle(type === 'url' ? { url: value } : { text: value });
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong. Is the backend running?');
    } finally {
      setLoading(false);
    }
  };
  return (
    <div className="page-container">
      <div className="hero-illustration">
      <img src={heroImg} alt="Satyata Hero Illustration" className="hero-image" />
      </div>
      <div className="hero-text">
        <h1 className="hero-title">सत्यता — Satyata</h1>
        <p className="hero-subtitle">Bilingual Nepali-English Fake News Detection</p>
      </div>

      <div className="card input-card">
        <div className="input-toggle">
          <button
            type="button"
            className={type === 'text' ? 'toggle-btn active' : 'toggle-btn'}
            onClick={() => setType('text')}
          >
            <ClipboardIcon />
            Paste Text
          </button>
          <button
            type="button"
            className={type === 'url' ? 'toggle-btn active' : 'toggle-btn'}
            onClick={() => setType('url')}
          >
            <LinkIcon />
            Enter URL
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          {type === 'text' ? (
            <textarea
              className="input-area"
              rows={6}
              placeholder="Paste news link or text here... / यहाँ समाचारको लिंक वा पाठ टाँस्नुहोस्..."
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
          ) : (
            <input
              className="input-url"
              type="url"
              placeholder="https://ekantipur.com/news/..."
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
          )}

          {error && (
            <p className="error-msg">
              <AlertIcon />
              {error}
            </p>
          )}

          <button type="submit" className="submit-btn" disabled={loading}>
            {loading ? 'Analyzing...' : 'Verify Now'}
            {!loading && <ArrowIcon />}
          </button>
        </form>
      </div>

      {result && <ResultCard result={result} />}
    </div>
  );
}

export default HomePage;