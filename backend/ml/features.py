"""
features.py
Feature extraction shared between train.py (fit + transform) and
predict.py (transform only). Keeping this logic in one place guarantees
that a live prediction is built from exactly the same feature pipeline
the models were trained on.

Feature layout for the hybrid matrix, in this fixed order:
    [ TF-IDF features (config.TFIDF_MAX_FEATURES cols) | XLM-R embedding (768 cols) ]
"""
import pickle

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from ml import config


# --------------------------------------------------------------------------
# Text preparation
# --------------------------------------------------------------------------
def build_text_column(df: pd.DataFrame) -> pd.Series:
    """Collapse (headline, body, full_text) columns into one clean text field."""
    def combine(row):
        if pd.notna(row.get("full_text")) and str(row["full_text"]).strip():
            return str(row["full_text"]).strip()
        parts = []
        if pd.notna(row.get("headline")) and str(row["headline"]).strip():
            parts.append(str(row["headline"]).strip())
        if pd.notna(row.get("body")) and str(row["body"]).strip():
            parts.append(str(row["body"]).strip())
        return " ".join(parts) if parts else ""

    return df.apply(combine, axis=1)


def clean_single_text(text: str) -> str:
    import re
    return re.sub(r"\s+", " ", (text or "")).strip()


# --------------------------------------------------------------------------
# TF-IDF
# --------------------------------------------------------------------------
def fit_tfidf(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)  # already L2-normalized per row
    return vectorizer, matrix


def load_tfidf():
    with open(config.TFIDF_PATH, "rb") as f:
        return pickle.load(f)


# --------------------------------------------------------------------------
# Transformer (XLM-RoBERTa) embeddings
# --------------------------------------------------------------------------
class TransformerEmbedder:
    """Loads xlm-roberta-base once per process and reuses it for every call.

    Loading a transformer is expensive (~1-2s + model download), so this is
    implemented as a lazy singleton rather than being reloaded per batch or
    per request.
    """
    _tokenizer = None
    _model = None
    _device = None

    @classmethod
    def _ensure_loaded(cls):
        if cls._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        cls._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[features] Loading {config.TRANSFORMER_MODEL_NAME} on {cls._device} ...")
        cls._tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
        cls._model = AutoModel.from_pretrained(
            config.TRANSFORMER_MODEL_NAME, low_cpu_mem_usage=True
        ).to(cls._device)
        cls._model.eval()

    @classmethod
    def embed(cls, texts, batch_size=None, show_progress=True):
        """Return an (n_texts, 768) float32 numpy array of mean-pooled,
        L2-normalized sentence embeddings."""
        import torch

        cls._ensure_loaded()
        batch_size = batch_size or config.BATCH_SIZE
        all_emb = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        iterator = range(0, len(texts), batch_size)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, total=total_batches, desc="Embedding")
            except ImportError:
                pass

        for i in iterator:
            batch = texts[i:i + batch_size]
            enc = cls._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.MAX_LENGTH,
                return_tensors="pt",
            ).to(cls._device)
            with torch.no_grad():
                out = cls._model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            # L2-normalize so embedding rows sit on the same scale as the
            # (already L2-normalized) TF-IDF rows -- keeps the downstream
            # linear classifiers well-conditioned.
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            all_emb.append(pooled.cpu().numpy())

        return np.vstack(all_emb).astype(np.float32)


def _mean_pool(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


# --------------------------------------------------------------------------
# Combining feature blocks
# --------------------------------------------------------------------------
def combine_features(tfidf_matrix, embedding_matrix):
    tfidf_matrix = tfidf_matrix.astype(np.float32)
    emb_sparse = csr_matrix(embedding_matrix)
    return hstack([tfidf_matrix, emb_sparse], format="csr")