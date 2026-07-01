// src/components/HighlightedText.jsx
function HighlightedText({ text, highlights }) {
  if (!highlights?.length) return <p className="article-text">{text}</p>;

  const suspicious = new Set(
    highlights.filter((h) => h.suspicious).map((h) => h.word.toLowerCase())
  );

  return (
    <p className="article-text">
      {text.split(' ').map((word, i) => {
        const clean = word.replace(/[^\w\s]/g, '').toLowerCase();
        return (
          <span
            key={i}
            className={suspicious.has(clean) ? 'highlighted-word' : ''}
            title={suspicious.has(clean) ? 'Suspicious word' : ''}
          >
            {word}{' '}
          </span>
        );
      })}
    </p>
  );
}

export default HighlightedText;