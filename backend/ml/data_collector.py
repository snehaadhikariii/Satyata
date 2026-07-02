"""
data_collector.py
Run: python -m ml.data_collector
Collects REAL news from Nepali sources + FAKE from fact-checkers.
Output: backend/ml/models/scraped_data.csv
"""

import requests, time, re, csv, os
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}
DELAY = 1.5          # seconds between requests — be polite
OUTPUT = os.path.join(os.path.dirname(__file__), 'models', 'scraped_data.csv')


def clean(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    return text.strip()[:2000]


def get_soup(url: str):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        return BeautifulSoup(r.content, 'html.parser')
    except Exception as e:
        print(f'  [SKIP] {url} — {e}')
        return None


def save_rows(rows: list, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df = pd.DataFrame(rows, columns=['text', 'title', 'url', 'source', 'label', 'language', 'collected_at'])
    df.drop_duplicates(subset='text').to_csv(path, index=False, encoding='utf-8-sig')
    print(f'\n✅  Saved {len(df)} rows → {path}')



def scrape_onlinekhabar(max_pages=5):
    """Scrapes Onlinekhabar news listing pages."""
    rows = []
    for page in range(1, max_pages + 1):
        url = f'https://www.onlinekhabar.com/news/page/{page}'
        soup = get_soup(url)
        if not soup:
            continue
        articles = soup.select('div.ok-single-post, article, .news-list li')
        for art in articles:
            a_tag = art.find('a', href=True)
            if not a_tag:
                continue
            title = clean(a_tag.get_text())
            link  = a_tag['href']
            # fetch article body
            art_soup = get_soup(link)
            if not art_soup:
                continue
            body_tag = (art_soup.find('div', class_=re.compile(r'content|article|post-body', re.I))
                        or art_soup.find('article'))
            body = clean(body_tag.get_text() if body_tag else title)
            full_text = f'{title}. {body}' if title not in body else body
            rows.append([full_text, title, link, 'onlinekhabar', 0, 'ne', datetime.now().isoformat()])
            time.sleep(DELAY)
        print(f'  Onlinekhabar page {page}: {len(rows)} total so far')
    return rows


def scrape_setopati(max_pages=5):
    rows = []
    for page in range(1, max_pages + 1):
        url = f'https://www.setopati.com/page/{page}'
        soup = get_soup(url)
        if not soup:
            continue
        for a_tag in soup.select('h2.entry-title a, h3.entry-title a, .post-title a'):
            title = clean(a_tag.get_text())
            link  = a_tag['href']
            art_soup = get_soup(link)
            if not art_soup:
                continue
            body_tag = art_soup.find('div', class_=re.compile(r'entry-content|post-content', re.I))
            body = clean(body_tag.get_text() if body_tag else title)
            full_text = f'{title}. {body}' if title not in body else body
            rows.append([full_text, title, link, 'setopati', 0, 'ne', datetime.now().isoformat()])
            time.sleep(DELAY)
        print(f'  Setopati page {page}: {len(rows)} total so far')
    return rows


def scrape_nepalitimes(max_pages=5):
    rows = []
    for page in range(1, max_pages + 1):
        url = f'https://nepalitimes.com/page/{page}'
        soup = get_soup(url)
        if not soup:
            continue
        for a_tag in soup.select('h2 a, h3 a, .post-title a'):
            title = clean(a_tag.get_text())
            link  = a_tag['href']
            art_soup = get_soup(link)
            if not art_soup:
                continue
            body_tag = art_soup.find('div', class_=re.compile(r'content|article-body', re.I))
            body = clean(body_tag.get_text() if body_tag else title)
            full_text = f'{title}. {body}' if title not in body else body
            rows.append([full_text, title, link, 'nepalitimes', 0, 'en', datetime.now().isoformat()])
            time.sleep(DELAY)
        print(f'  NepaliTimes page {page}: {len(rows)} total so far')
    return rows




def scrape_nepalcheck(max_pages=10):
    """Scrapes debunked claims from NepalCheck.org (label = 1)."""
    rows = []
    for page in range(1, max_pages + 1):
        url = f'https://nepalcheck.org/factcheck/page/{page}/'
        soup = get_soup(url)
        if not soup:
            break
        articles = soup.select('article, .post, h2.entry-title')
        if not articles:
            break
        for art in articles:
            a_tag = art.find('a', href=True) if art.name != 'a' else art
            if not a_tag:
                continue
            title = clean(a_tag.get_text())
            link  = a_tag['href'] if a_tag['href'].startswith('http') else 'https://nepalcheck.org' + a_tag['href']
            art_soup = get_soup(link)
            if not art_soup:
                continue
            claim_tag = (art_soup.find('blockquote')
                         or art_soup.find('div', class_=re.compile(r'claim|verdict|highlight', re.I))
                         or art_soup.find('div', class_=re.compile(r'entry-content|post-content', re.I)))
            claim_text = clean(claim_tag.get_text() if claim_tag else title)
            full_text = f'{title}. {claim_text}' if title not in claim_text else claim_text
            rows.append([full_text, title, link, 'nepalcheck', 1, 'ne/en', datetime.now().isoformat()])
            time.sleep(DELAY)
        print(f'  NepalCheck page {page}: {len(rows)} total so far')
    return rows


def scrape_nepalfactcheck(max_pages=10):
    """Scrapes debunked claims from NepalFactCheck.org (label = 1)."""
    rows = []
    for page in range(1, max_pages + 1):
        url = f'https://nepalfactcheck.org/page/{page}/'
        soup = get_soup(url)
        if not soup:
            break
        for a_tag in soup.select('h2.entry-title a, h3 a, .post-title a'):
            title = clean(a_tag.get_text())
            link  = a_tag['href']
            art_soup = get_soup(link)
            if not art_soup:
                continue
            body_tag = (art_soup.find('blockquote')
                        or art_soup.find('div', class_=re.compile(r'entry-content|post-content', re.I)))
            body = clean(body_tag.get_text() if body_tag else title)
            full_text = f'{title}. {body}' if title not in body else body
            rows.append([full_text, title, link, 'nepalfactcheck', 1, 'ne/en', datetime.now().isoformat()])
            time.sleep(DELAY)
        print(f'  NepalFactCheck page {page}: {len(rows)} total so far')
    return rows




def load_excel_sheet(path: str):
    """
    Load your manually collected Excel sheet.
    Expected columns: text, label  (and optionally: title, url, source, language)
    """
    df = pd.read_excel(path)
    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    rows = []
    for _, row in df.iterrows():
        text  = clean(str(row.get('text', '')))
        label = int(row.get('label', -1))
        title = str(row.get('title', ''))
        url   = str(row.get('url', ''))
        src   = str(row.get('source', 'manual'))
        lang  = str(row.get('language', 'ne/en'))
        if text and label in (0, 1):
            rows.append([text, title, url, src, label, lang, datetime.now().isoformat()])
    print(f'  Loaded {len(rows)} rows from Excel sheet')
    return rows



def main():
    all_rows = []

    
    excel_path = os.path.join(os.path.dirname(__file__), 'models', 'manual_data.xlsx')
    if os.path.exists(excel_path):
        print('\n[1] Loading your Excel sheet...')
        all_rows += load_excel_sheet(excel_path)
    else:
        print(f'\n[1] Excel not found at {excel_path} — skipping. Copy your file there.')

   
    print('\n[2] Scraping Onlinekhabar (REAL)...')
    all_rows += scrape_onlinekhabar(max_pages=3)

    print('\n[3] Scraping Setopati (REAL)...')
    all_rows += scrape_setopati(max_pages=3)

    print('\n[4] Scraping NepaliTimes (REAL)...')
    all_rows += scrape_nepalitimes(max_pages=3)

    
    print('\n[5] Scraping NepalCheck (FAKE)...')
    all_rows += scrape_nepalcheck(max_pages=8)

    print('\n[6] Scraping NepalFactCheck (FAKE)...')
    all_rows += scrape_nepalfactcheck(max_pages=8)

    save_rows(all_rows, OUTPUT)


    df = pd.read_csv(OUTPUT)
    print('\nClass balance:')
    print(df['label'].value_counts())
    print(f'\nTotal unique articles: {len(df)}')


if __name__ == '__main__':
    main()