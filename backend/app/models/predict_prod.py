# backend/app/models/predict_prod.py
"""
Production predictions:

- predict_on_excel(file_path) -> DataFrame with columns: transaction, predicted_cat (2 columns)
- predict_on_texts(list[str]) -> DataFrame with columns: transaction, predicted_cat (2 columns)

This code loads the final production models:
  storage/model/final_tfidf.pkl  and  storage/model/final_bert.pkl
"""

from pathlib import Path
import joblib
from app.services.preprocessing import preprocess_series
from app.services.postprocess import format_prod_dataframe
from sentence_transformers import SentenceTransformer
import os

BASE = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE / "storage" / "model"

FINAL_TFIDF = MODEL_DIR / "final_tfidf.pkl"
FINAL_BERT = MODEL_DIR / "final_bert.pkl"
FINAL_BERT_ENCODER = MODEL_DIR / "final_bert_encoder.pkl"  # label encoder if used

def _load_final_tfidf():
    if not FINAL_TFIDF.exists():
        raise FileNotFoundError("final_tfidf.pkl not found; deploy a trained prod model first")
    return joblib.load(FINAL_TFIDF)

def _load_final_bert():
    if not FINAL_BERT.exists():
        raise FileNotFoundError("final_bert.pkl not found; deploy a trained prod model first")
    clf = joblib.load(FINAL_BERT)
    # label encoder
    le = joblib.load(FINAL_BERT_ENCODER) if FINAL_BERT_ENCODER.exists() else None
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder, clf, le

def predict_on_excel(file_path: str):
    # Excel input -> use TF-IDF pipeline (fast, batch)
    import pandas as pd
    df = pd.read_excel(file_path)
    if "transaction" not in df.columns:
        raise ValueError("Input Excel must have a 'transaction' column")
    texts = preprocess_series(df["transaction"])
    tfidf = _load_final_tfidf()
    preds = tfidf.predict(texts)
    df_out = format_prod_dataframe(df["transaction"].astype(str).tolist(), preds.tolist())
    return df_out

def predict_on_texts(texts):
    # Raw texts -> use BERT embeddings + LogReg
    encoder, clf, le = _load_final_bert()
    cleaned = [t for t in texts]
    emb = encoder.encode(cleaned, convert_to_numpy=True)
    preds_enc = clf.predict(emb)
    if le is not None:
        preds = le.inverse_transform(preds_enc)
    else:
        preds = preds_enc
    df_out = format_prod_dataframe(cleaned, preds.tolist())
    return df_out
