"""
config.py
Central configuration for the Satyata fake-news detection ML pipeline.

Both train.py (fit-time) and predict.py (inference-time) import this file
so that paths, feature sizes, and label mappings can never drift apart
between training and serving.
"""
import os

# --- Paths ---------------------------------------------------------------
ML_DIR       = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR   = os.path.join(ML_DIR, "models")
DATASET_PATH = os.path.abspath(os.path.join(ML_DIR, "..", "mlmodels", "dataset.csv"))

TFIDF_PATH      = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
LR_MODEL_PATH   = os.path.join(MODELS_DIR, "lr_model.pkl")
SVM_MODEL_PATH  = os.path.join(MODELS_DIR, "svm_model.pkl")
BACKGROUND_PATH = os.path.join(MODELS_DIR, "shap_background.pkl")
METADATA_PATH   = os.path.join(MODELS_DIR, "metadata.json")
METRICS_PATH    = os.path.join(MODELS_DIR, "metrics.json")

# --- Transformer embedding model -----------------------------------------
TRANSFORMER_MODEL_NAME = "xlm-roberta-base"   # multilingual, handles Nepali (Devanagari)
MAX_LENGTH = 128
BATCH_SIZE = int(os.environ.get("SATYATA_BATCH_SIZE", 16))

# --- TF-IDF ----------------------------------------------------------------
TFIDF_MAX_FEATURES = 5000
TFIDF_NGRAM_RANGE = (1, 2)

# --- Labels ------------------------------------------------------------
LABELS = ["REAL", "FAKE"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# --- Inference -----------------------------------------------------------
# Which trained model predict.py uses to answer requests: "svm" or "lr".
# SVM scored higher in your last run (97.51% vs 96.66%) so it's the default.
PRIMARY_MODEL = os.environ.get("SATYATA_PRIMARY_MODEL", "svm")

# --- Training ----------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
CALIBRATION_CV = 3          # folds used by CalibratedClassifierCV
SHAP_BACKGROUND_SIZE = 100  # rows kept for the SHAP masker at inference time
TOP_K_HIGHLIGHTS = 10