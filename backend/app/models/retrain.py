# backend/app/models/retrain.py
"""
Retrain script used in development only.

Flow:
- Read storage/training_data.csv
- Optionally append error_buffer.json examples (if present)
- Train TF-IDF+LogReg and BERT-emb+LogReg (same as predict_dev.train_from_storage)
- Save outputs to dev_* files
- Optionally a helper to promote dev models to final production models
"""

import json
from pathlib import Path
import joblib
import pandas as pd
from .predict_dev import train_from_storage
from .predict_dev import DEV_TFIDF_PATH, DEV_BERT_PATH
from .predict_dev import DEV_BERT_ENCODER_PATH
from pathlib import Path
import shutil

BASE = Path(__file__).resolve().parents[1]
STORAGE = BASE / "storage"
MODEL_DIR = STORAGE / "model"
ERROR_BUFFER = STORAGE / "error_buffer.json"

FINAL_TFIDF = MODEL_DIR / "final_tfidf.pkl"
FINAL_BERT = MODEL_DIR / "final_bert.pkl"
FINAL_BERT_ENCODER = MODEL_DIR / "final_bert_encoder.pkl"

def append_error_buffer_to_training():
    """
    If error_buffer.json exists and contains items with 'transaction' and 'actual_cat',
    append them to training_data.csv so retraining uses them.
    """
    train_csv = STORAGE / "training_data.csv"
    if not train_csv.exists():
        return
    if not Path(ERROR_BUFFER).exists():
        return

    with open(ERROR_BUFFER, "r", encoding="utf-8") as f:
        buf = json.load(f)

    items = buf.get("wrong_examples", [])
    if not items:
        return

    df_append = pd.DataFrame(items)
    # Expecting df_append has columns 'transaction' and 'actual_cat'
    df_append = df_append.loc[:, ["transaction", "actual_cat"]]
    df_existing = pd.read_csv(train_csv)
    df_combined = pd.concat([df_existing, df_append], ignore_index=True)
    df_combined.to_csv(train_csv, index=False)

    # Optionally clear buffer
    with open(ERROR_BUFFER, "w", encoding="utf-8") as f:
        json.dump({"wrong_examples": []}, f)


def retrain_and_save_dev():
    """
    Combines append from error buffer and retrain using predict_dev.train_from_storage
    """
    append_error_buffer_to_training()
    train_from_storage(save_dev=True)
    return {"dev_tfidf": str(DEV_TFIDF_PATH), "dev_bert": str(DEV_BERT_PATH)}


def promote_dev_to_final():
    """
    Copy dev model files to final model paths (use when you are satisfied with dev performance).
    """
    if not DEV_TFIDF_PATH.exists() or not DEV_BERT_PATH.exists():
        raise FileNotFoundError("No dev models to promote. Train dev models first.")

    shutil.copy(DEV_TFIDF_PATH, FINAL_TFIDF)
    shutil.copy(DEV_BERT_PATH, FINAL_BERT)
    # encoder/label encoder (if present)
    if DEV_BERT_ENCODER_PATH.exists():
        shutil.copy(DEV_BERT_ENCODER_PATH, FINAL_BERT_ENCODER)
    return {"final_tfidf": str(FINAL_TFIDF), "final_bert": str(FINAL_BERT)}
