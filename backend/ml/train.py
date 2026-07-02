"""
train.py
Complete training pipeline for Satyata's Nepali fake-news classifier.

Run from the `backend/` directory (mirrors ml/data_collector.py's convention):

    cd backend
    python -m ml.train                     # full pipeline: TF-IDF + XLM-R embeddings
    python -m ml.train --tfidf-only         # fast baseline, skips the transformer
    python -m ml.train --dataset some.csv   # train on a different CSV
    python -m ml.train --batch-size 32      # override embedding batch size
    python -m ml.train --test-size 0.15     # override the held-out test fraction

Input:
    backend/mlmodels/dataset.csv  (columns: category, label, headline, body, full_text)
    label must be one of "REAL" / "FAKE"

Output (written to backend/ml/models/):
    tfidf_vectorizer.pkl   fitted TfidfVectorizer
    lr_model.pkl           calibrated LogisticRegression  (has predict_proba)
    svm_model.pkl          calibrated LinearSVC           (has predict_proba)
    shap_background.pkl    small feature sample predict.py's SHAP explainer needs
    metadata.json          feature dims / label map / run info
    metrics.json           accuracy, precision, recall, f1, confusion matrix
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from ml import config, features


# --------------------------------------------------------------------------
def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)
    if "label" not in df.columns:
        raise ValueError("Dataset must have a 'label' column with REAL/FAKE values")

    df["text"] = features.build_text_column(df)
    before = len(df)
    df = df[df["text"].str.strip().ne("")].reset_index(drop=True)
    df = df[df["label"].isin(config.LABELS)].reset_index(drop=True)
    dropped = before - len(df)

    print(f"[1/6] Loaded {before} rows, dropped {dropped} empty/invalid -> {len(df)} remain")
    print(df["label"].value_counts().to_string())

    if df["label"].nunique() < 2:
        raise ValueError("Dataset needs both REAL and FAKE examples to train a classifier")

    return df


def evaluate(name, model, X_te, y_te):
    preds = model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    report = classification_report(
        y_te, preds, target_names=config.LABELS, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_te, preds).tolist()

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_te, preds, target_names=config.LABELS, zero_division=0))
    print("Confusion matrix (rows=actual, cols=predicted):")
    print(pd.DataFrame(cm, index=config.LABELS, columns=config.LABELS).to_string())

    return {
        "accuracy": acc,
        "precision_macro": precision_score(y_te, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(y_te, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(y_te, preds, average="macro", zero_division=0),
        "confusion_matrix": cm,
        "report": report,
    }


def save_shap_background(X_tr):
    """A small random sample of training features, used later by predict.py
    as the 'masker'/background distribution for shap.LinearExplainer."""
    rng = np.random.default_rng(config.RANDOM_STATE)
    bg_size = min(config.SHAP_BACKGROUND_SIZE, X_tr.shape[0])
    idx = rng.choice(X_tr.shape[0], size=bg_size, replace=False)
    background = X_tr[idx]
    with open(config.BACKGROUND_PATH, "wb") as f:
        pickle.dump(background, f)
    return bg_size


# --------------------------------------------------------------------------
def train(dataset_path=None, tfidf_only=False, batch_size=None, test_size=None,
          calibration_cv=None):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    dataset_path = dataset_path or config.DATASET_PATH
    test_size = test_size if test_size is not None else config.TEST_SIZE
    calibration_cv = calibration_cv or config.CALIBRATION_CV
    t0 = time.time()

    df = load_dataset(dataset_path)
    texts = df["text"].tolist()
    y = df["label"].map(config.LABEL2ID).to_numpy()

    # --- Features ----------------------------------------------------
    print("\n[2/6] Fitting TF-IDF vectorizer...")
    tfidf_vectorizer, tfidf_matrix = features.fit_tfidf(texts)
    with open(config.TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf_vectorizer, f)
    print(f" TF-IDF shape: {tfidf_matrix.shape}")

    if tfidf_only:
        print("\n[3/6] --tfidf-only set: skipping transformer embeddings")
        X = tfidf_matrix.astype(np.float32)
        embedding_dim = 0
    else:
        print(f"\n[3/6] Computing {config.TRANSFORMER_MODEL_NAME} embeddings "
              f"for {len(texts)} texts (this is the slow step)...")
        embedding_matrix = features.TransformerEmbedder.embed(texts, batch_size=batch_size)
        embedding_dim = embedding_matrix.shape[1]
        print(f" Embedding shape: {embedding_matrix.shape}")
        X = features.combine_features(tfidf_matrix, embedding_matrix)

    print(f"\n[4/6] Splitting off {test_size:.0%} stratified test set...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y,
    )
    print(f" Train: {X_tr.shape}   Test: {X_te.shape}")

    # --- Models --------------------------------------------------------
    print("\n[5/6] Training classifiers...")
    metrics = {}

    print(f" -> Logistic Regression (calibrated, cv={calibration_cv})")
    lr_base = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    lr = CalibratedClassifierCV(lr_base, method="sigmoid", cv=calibration_cv)
    lr.fit(X_tr, y_tr)
    metrics["logistic_regression"] = evaluate("Logistic Regression", lr, X_te, y_te)
    with open(config.LR_MODEL_PATH, "wb") as f:
        pickle.dump(lr, f)

    print(f"\n -> Linear SVM (calibrated, cv={calibration_cv})")
    svm_base = LinearSVC(C=1.0, max_iter=5000)
    svm = CalibratedClassifierCV(svm_base, method="sigmoid", cv=calibration_cv)
    svm.fit(X_tr, y_tr)
    metrics["svm"] = evaluate("Linear SVM", svm, X_te, y_te)
    with open(config.SVM_MODEL_PATH, "wb") as f:
        pickle.dump(svm, f)

    # --- Explainability background + metadata ------------------------
    print("\n[6/6] Saving SHAP background sample + metadata...")
    bg_size = save_shap_background(X_tr)

    metadata = {
        "tfidf_max_features": config.TFIDF_MAX_FEATURES,
        "tfidf_vocab_size": len(tfidf_vectorizer.vocabulary_),
        "embedding_model": None if tfidf_only else config.TRANSFORMER_MODEL_NAME,
        "embedding_dim": embedding_dim,
        "total_feature_dim": X.shape[1],
        "labels": config.LABELS,
        "label2id": config.LABEL2ID,
        "train_rows": int(X_tr.shape[0]),
        "test_rows": int(X_te.shape[0]),
        "shap_background_rows": bg_size,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_seconds": round(time.time() - t0, 1),
        "dataset_path": os.path.abspath(dataset_path),
    }
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    with open(config.METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. Artifacts saved to: {config.MODELS_DIR}")
    print(f" LR accuracy:  {metrics['logistic_regression']['accuracy']:.4f}")
    print(f" SVM accuracy: {metrics['svm']['accuracy']:.4f}")
    return metadata, metrics


# --------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train Satyata's fake-news classifier")
    p.add_argument("--dataset", default=None,
                    help="Path to training CSV (default: backend/mlmodels/dataset.csv)")
    p.add_argument("--tfidf-only", action="store_true",
                    help="Skip transformer embeddings for a fast TF-IDF-only baseline")
    p.add_argument("--batch-size", type=int, default=None,
                    help="Embedding batch size (default: env SATYATA_BATCH_SIZE or 16)")
    p.add_argument("--test-size", type=float, default=None,
                    help="Fraction of data held out for testing (default: 0.2)")
    p.add_argument("--calibration-cv", type=int, default=None,
                    help="CV folds for probability calibration (default: 3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset_path=args.dataset,
        tfidf_only=args.tfidf_only,
        batch_size=args.batch_size,
        test_size=args.test_size,
        calibration_cv=args.calibration_cv,
    )