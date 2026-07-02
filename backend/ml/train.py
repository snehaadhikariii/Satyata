import os
import pickle

import pandas as pd
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from transformers import AutoModel, AutoTokenizer
from scipy.sparse import hstack, csr_matrix

MODEL_NAME = "xlm-roberta-base"  
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MAX_LENGTH = 128
BATCH_SIZE = 2


def build_text_column(df: pd.DataFrame) -> pd.Series:
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


def get_embeddings(texts, tokenizer, model, device):
    all_emb = []
    model.eval()
    total = (len(texts) + BATCH_SIZE - 1) // BATCH_SIZE

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            out = model(**enc)
        all_emb.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        print(f" Batch {i // BATCH_SIZE + 1}/{total}")

    return np.vstack(all_emb)


DATASET_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "mlmodels", "dataset.csv")
)


def train():
    os.makedirs(MODELS_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    
    print("\n[1/5] Loading dataset...")
    local_dataset = os.path.join(MODELS_DIR, "dataset.csv")
    dataset_path = local_dataset if os.path.exists(local_dataset) else DATASET_PATH
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found. Checked: {local_dataset} and {DATASET_PATH}"
        )

    df = pd.read_csv(dataset_path)
    print(f" Loaded dataset from: {dataset_path}")
    print(f" Columns found: {df.columns.tolist()}")
    print(f" {len(df)} rows loaded")
    print(df["label"].value_counts())

    df["text"] = build_text_column(df)
    before = len(df)
    df = df[df["text"].str.strip().ne("")].reset_index(drop=True)
    print(f" Dropped {before - len(df)} empty rows => {len(df)} remain")

    texts = df["text"].tolist()
    labels = df["label"].tolist()

  
    print("\n[2/5] Computing TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    
    tfidf_feats = tfidf.fit_transform(texts)
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    print(f" TF-IDF shape: {tfidf_feats.shape}")

    
    print(f"\n[3/5] Loading {MODEL_NAME} with reduced CPU memory usage...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    rob = AutoModel.from_pretrained(MODEL_NAME, low_cpu_mem_usage=True).to(device)
    rob_feats = get_embeddings(texts, tok, rob, device)
    print(f" Embedding shape: {rob_feats.shape}")

   
    print("\n[4/5] Combining TF-IDF + RoBERTa features...")
    
    tfidf_feats = tfidf_feats.astype(np.float32)
    rob_sparse = csr_matrix(rob_feats.astype(np.float32))
    X = hstack([tfidf_feats, rob_sparse], format="csr")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )

    
    print("\n[5/5] Training classifiers...")
    svm = LinearSVC(C=1.0, max_iter=5000)
    svm.fit(X_tr, y_tr)
    svm_preds = svm.predict(X_te)
    print(f"SVM Accuracy: {accuracy_score(y_te, svm_preds):.4f}")
    print(
        classification_report(
            y_te, svm_preds, target_names=["REAL", "FAKE"]
        )
    )
    with open(os.path.join(MODELS_DIR, "svm_model.pkl"), "wb") as f:
        pickle.dump(svm, f)

    
    lr = LogisticRegression(max_iter=1000, C=1.0, solver="saga")
    lr.fit(X_tr, y_tr)
    lr_preds = lr.predict(X_te)
    print(f"LR Accuracy: {accuracy_score(y_te, lr_preds):.4f}")
    print(
        classification_report(
            y_te, lr_preds, target_names=["REAL", "FAKE"]
        )
    )
    with open(os.path.join(MODELS_DIR, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr, f)

    print("\nTraining complete! Model files saved to:", MODELS_DIR)


if __name__ == "__main__":
    train()