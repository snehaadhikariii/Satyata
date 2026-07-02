"""
predict.py
Inference for Satyata's fake-news classifier. Loads the artifacts produced
by train.py (tfidf_vectorizer.pkl, lr_model.pkl / svm_model.pkl,
shap_background.pkl, metadata.json) ONCE per process and reuses them for
every request -- called from api/views.py's AnalyzeView.

If the model artifacts haven't been trained yet, falls back to a clearly
labeled heuristic so the API never hard-crashes.
"""
import json
import os
import pickle
import re

import numpy as np
import requests
from bs4 import BeautifulSoup

from ml import config, features

# --------------------------------------------------------------------------
# Lazy-loaded singletons -- loaded once, reused for every prediction.
# --------------------------------------------------------------------------
_tfidf_vectorizer = None
_model = None
_model_name = None
_metadata = None
_shap_explainer = None
_artifacts_missing = False


def _artifacts_available():
    required = [config.TFIDF_PATH, config.LR_MODEL_PATH, config.SVM_MODEL_PATH,
                config.METADATA_PATH]
    return all(os.path.exists(p) for p in required)


def _load_artifacts():
    global _tfidf_vectorizer, _model, _model_name, _metadata, _artifacts_missing

    if _model is not None or _artifacts_missing:
        return

    if not _artifacts_available():
        _artifacts_missing = True
        return

    with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
        _metadata = json.load(f)

    _tfidf_vectorizer = features.load_tfidf()

    _model_name = config.PRIMARY_MODEL
    model_path = config.SVM_MODEL_PATH if _model_name == "svm" else config.LR_MODEL_PATH
    with open(model_path, "rb") as f:
        _model = pickle.load(f)

    print(f"[predict] Loaded '{_model_name}' model "
          f"(trained {_metadata.get('trained_at')}, "
          f"{_metadata.get('total_feature_dim')} features)")


def _get_shap_explainer():
    """Builds (once) a shap.LinearExplainer from the averaged linear
    coefficients of the calibrated model + the saved background sample."""
    global _shap_explainer

    if _shap_explainer is not None:
        return _shap_explainer

    if not os.path.exists(config.BACKGROUND_PATH):
        return None

    try:
        import shap
    except ImportError:
        return None

    with open(config.BACKGROUND_PATH, "rb") as f:
        background = pickle.load(f)

    coefs, intercepts = [], []
    for calibrated_clf in _model.calibrated_classifiers_:
        est = calibrated_clf.estimator
        if not hasattr(est, "coef_"):
            return None  # base estimator isn't linear; skip explainability
        coefs.append(np.asarray(est.coef_).ravel())
        intercepts.append(np.ravel(est.intercept_)[0])

    mean_coef = np.mean(coefs, axis=0)
    mean_intercept = float(np.mean(intercepts))

    _shap_explainer = shap.LinearExplainer((mean_coef, mean_intercept), background)
    return _shap_explainer


# --------------------------------------------------------------------------
# Text acquisition
# --------------------------------------------------------------------------
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


# --------------------------------------------------------------------------
# Explainability
# --------------------------------------------------------------------------
def _explain(tfidf_vec, X):
    """Returns top contributing words toward the FAKE class using exact
    linear SHAP values over the TF-IDF portion of the feature vector.
    Embedding dimensions aren't per-word interpretable, so they're excluded
    from the highlight list (they still influence the prediction itself).

    `X` is the *already-assembled* full feature row (TF-IDF + embedding,
    same shape the model was trained/predicted on) -- reusing it here
    instead of reconstructing it avoids any risk of a feature-count
    mismatch with the model/background.
    """
    explainer = _get_shap_explainer()
    vocab = _tfidf_vectorizer.get_feature_names_out()
    tfidf_dense = tfidf_vec.toarray().ravel()
    present_idx = np.nonzero(tfidf_dense)[0]

    if explainer is None or len(present_idx) == 0:
        return []

    full_row = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    full_row = full_row.astype(np.float32).reshape(1, -1)

    shap_values = explainer.shap_values(full_row)[0]
    tfidf_shap = shap_values[:len(tfidf_dense)]

    contributions = [(vocab[i], float(tfidf_shap[i])) for i in present_idx]
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)

    highlights = []
    for word, value in contributions[:config.TOP_K_HIGHLIGHTS]:
        highlights.append({
            'word': word,
            'importance': round(abs(value), 4),
            'suspicious': value > 0,  # pushes prediction toward FAKE
        })
    return highlights


# --------------------------------------------------------------------------
# Fallback heuristic (used only if train.py hasn't been run yet)
# --------------------------------------------------------------------------
_FAKE_TRIGGER_WORDS = {
    'breaking', 'miracle', 'cure', 'shocking', 'secret', 'banned',
    'सनसनी', 'चमत्कार', 'खुलासा',
}


def _stub_predict(article):
    import hashlib
    words = article.split()
    lower_words = [w.strip('.,!?').lower() for w in words]
    trigger_hits = [w for w in lower_words if w in _FAKE_TRIGGER_WORDS]

    digest = hashlib.sha256(article.encode('utf-8')).hexdigest()
    base_score = (int(digest[:8], 16) % 100) / 100

    if trigger_hits:
        score, label = max(base_score, 0.78), 'FAKE'
    else:
        score = max(base_score, 0.55) if base_score > 0.5 else (1 - base_score)
        label = 'FAKE' if base_score > 0.82 else 'REAL'

    highlights = [
        {'word': w, 'importance': 0.9, 'suspicious': True}
        for w in set(words) if w.strip('.,!?').lower() in _FAKE_TRIGGER_WORDS
    ][:10]

    return {
        'label': label,
        'confidence': round(score, 4),
        'article_text': article,
        'highlights': highlights,
        'model_used': 'STUB (no model trained yet -- run `python -m ml.train`)',
    }


# --------------------------------------------------------------------------
# Public entrypoint (called by api/views.py)
# --------------------------------------------------------------------------
def predict(text=None, url=None):
    article = scrape_url(url) if url else (text or '').strip()
    if len(article) < 20:
        raise ValueError('Text too short')

    _load_artifacts()

    if _artifacts_missing:
        return _stub_predict(article)

    clean_text = features.clean_single_text(article)

    tfidf_vec = _tfidf_vectorizer.transform([clean_text])

    if _metadata.get("embedding_model"):
        embedding = features.TransformerEmbedder.embed(
            [clean_text], batch_size=1, show_progress=False
        )
        X = features.combine_features(tfidf_vec, embedding)
    else:
        X = tfidf_vec.astype(np.float32)

    proba = _model.predict_proba(X)[0]          # [P(REAL), P(FAKE)]
    pred_id = int(np.argmax(proba))
    label = config.ID2LABEL[pred_id]
    confidence = float(proba[pred_id])

    highlights = _explain(tfidf_vec, X)

    return {
        'label': label,
        'confidence': round(confidence, 4),
        'article_text': article,
        'highlights': highlights,
        'model_used': f'{_model_name.upper()} (trained {_metadata.get("trained_at")})',
    }