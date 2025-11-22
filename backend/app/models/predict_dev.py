# backend/app/models/predict_dev.py
"""
Development mode predictions and training helpers.

- train_from_storage() reads training_data.csv from storage and trains both:
    - TF-IDF + LogisticRegression (suitable for Excel-batch inputs)
    - BERT-embedding (sentence-transformer) + LogisticRegression (suitable for raw text inputs)
  Saves dev models to storage/model/dev_tfidf.pkl and dev_bert.pkl

- predict_on_excel(file_path, model='tfidf', use_dev=True) -> DataFrame with 3 columns:
    transaction, actual_cat, predicted_cat
"""

import os
from pathlib import Path
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from app.services.preprocessing import preprocess_series
from app.services.postprocess import format_dev_dataframe

BASE = Path(__file__).resolve().parents[1]
STORAGE = BASE / "storage"
MODEL_DIR = STORAGE / "model"
TRAIN_CSV = STORAGE / "training_data.csv"

DEV_TFIDF_PATH = MODEL_DIR / "dev_tfidf.pkl"
DEV_BERT_PATH = MODEL_DIR / "dev_bert.pkl"
DEV_BERT_ENCODER_PATH = MODEL_DIR / "dev_bert_encoder.pkl"  # saves label encoder if used

MODEL_DIR.mkdir(parents=True, exist_ok=True)


def train_from_storage(save_dev=True):
    """Train both TF-IDF+LogReg and BERT-embeddings+LogReg on training_data.csv"""
    if not TRAIN_CSV.exists():
        raise FileNotFoundError(f"Training data not found at {TRAIN_CSV}")

    df = pd.read_csv(TRAIN_CSV)
    # Expect columns: 'transaction' and 'actual_cat'
    if "transaction" not in df.columns or "actual_cat" not in df.columns:
        raise ValueError("training_data.csv must contain 'transaction' and 'actual_cat' columns")

    X = preprocess_series(df["transaction"])
    y = df["actual_cat"].astype(str).tolist()

    # Encode labels for BERT logistic regression
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # --- TF-IDF + Logistic Regression pipeline ---
    tfidf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1,2), max_features=10000)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))
    ])

    tfidf_pipeline.fit(X, y)

    # --- BERT Embeddings + Logistic Regression ---
    # use a light sentence transformer
    bert_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = bert_encoder.encode(X, show_progress_bar=True, convert_to_numpy=True)
    bert_clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    bert_clf.fit(embeddings, y_enc)

    # Save dev artifacts
    if save_dev:
        joblib.dump(tfidf_pipeline, DEV_TFIDF_PATH)
        joblib.dump(bert_clf, DEV_BERT_PATH)
        joblib.dump(le, DEV_BERT_ENCODER_PATH)
        # Save encoder (bert model) name is not required because we use SentenceTransformer when predicting

    # Optional: print classification report on a small holdout
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=42)
    # Evaluate tfidf
    preds_tfidf = tfidf_pipeline.predict(X_val)
    print("TFIDF classification report:")
    print(classification_report(y_val, preds_tfidf))
    # Evaluate bert
    emb_val = bert_encoder.encode(X_val, convert_to_numpy=True)
    preds_bert_enc = bert_clf.predict(emb_val)
    preds_bert = le.inverse_transform(preds_bert_enc)
    print("BERT-emb+LogReg classification report:")
    print(classification_report(y_val, preds_bert))

    return {"tfidf": DEV_TFIDF_PATH, "bert": DEV_BERT_PATH}


def _load_dev_models():
    if not DEV_TFIDF_PATH.exists() or not DEV_BERT_PATH.exists():
        raise FileNotFoundError("Dev model files not found; run train_from_storage()")
    tfidf = joblib.load(DEV_TFIDF_PATH)
    bert_clf = joblib.load(DEV_BERT_PATH)
    le = joblib.load(DEV_BERT_ENCODER_PATH)
    bert_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return tfidf, bert_encoder, bert_clf, le


def predict_on_excel(file_path: str):
    """
    file_path: path to Excel with columns 'transaction' and 'actual_cat'
    returns DataFrame with columns ['transaction','actual_cat','predicted_cat']
    (uses TF-IDF + LogReg)
    """
    df = pd.read_excel(file_path)
    if "transaction" not in df.columns or "actual_cat" not in df.columns:
        raise ValueError("Input Excel must have 'transaction' and 'actual_cat' columns for dev mode")

    texts = preprocess_series(df["transaction"])
    tfidf, _, _, _ = _load_dev_models()
    preds = tfidf.predict(texts)
    out_df = format_dev_dataframe(df["transaction"].astype(str).tolist(), df["actual_cat"].astype(str).tolist(), preds.tolist())
    return out_df


def predict_on_texts(texts):
    """
    texts: list[str]
    returns DataFrame with columns ['transaction','actual_cat','predicted_cat']
    Note: if actuals are not available, pass None or empty list; here during dev we expect actuals
    For dev testing raw texts, user must provide actual labels separately if desired.
    """
    tfidf, bert_encoder, bert_clf, le = _load_dev_models()
    cleaned = [t for t in texts]
    emb = bert_encoder.encode(cleaned, convert_to_numpy=True)
    preds_enc = bert_clf.predict(emb)
    preds = le.inverse_transform(preds_enc)
    # For dev, actuals are not available here; we return predicted only
    import pandas as pd
    df = pd.DataFrame({"transaction": cleaned, "predicted_cat": preds})
    # If you want actuals to be present during dev testing provide them externally
    return df
