

import hashlib
import re
import requests
from bs4 import BeautifulSoup

FAKE_TRIGGER_WORDS = {
    'breaking', 'miracle', 'cure', 'shocking', 'secret', 'banned',
    'सनसनी', 'चमत्कार', 'खुलासा',
}


def scrape_url(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.content, 'html.parser')
    for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
        tag.decompose()
    art = (
        soup.find('article')
        or soup.find('div', class_=re.compile(r'article|content|story', re.I))
        or soup.find('main')
    )
    text = (art or soup).get_text(' ', strip=True)
    return re.sub(r'\s+', ' ', text).strip()[:3000]


def predict(text=None, url=None):
    article = scrape_url(url) if url else (text or '').strip()
    if len(article) < 20:
        raise ValueError('Text too short')

    words = article.split()
    lower_words = [w.lower() for w in words]
    trigger_hits = [w for w in lower_words if w.strip('.,!?').lower() in FAKE_TRIGGER_WORDS]

    digest = hashlib.sha256(article.encode('utf-8')).hexdigest()
    base_score = (int(digest[:8], 16) % 100) / 100

    if trigger_hits:
        score = max(base_score, 0.78)
        label = 'FAKE'
    else:
        score = max(base_score, 0.55) if base_score > 0.5 else (1 - base_score)
        label = 'FAKE' if base_score > 0.82 else 'REAL'

    highlights = [
        {'word': w, 'importance': 0.9, 'suspicious': True}
        for w in set(words) if w.strip('.,!?').lower() in FAKE_TRIGGER_WORDS
    ][:10]

    return {
        'label': label,
        'confidence': round(score, 4),
        'article_text': article,
        'highlights': highlights,
        'model_used': 'STUB (no model trained yet)',
    }