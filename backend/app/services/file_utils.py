# backend/app/services/file_utils.py
import os
from pathlib import Path
import pandas as pd
from datetime import datetime, timezone

BASE = Path(__file__).resolve().parents[1]  # backend/app
STATIC_DIR = BASE / "static"
UPLOAD_DIR = STATIC_DIR / "uploads"
RESULT_DIR = STATIC_DIR / "results"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file_obj, filename: str) -> str:
    """
    file_obj: a file-like object (from starlette UploadFile or requests)
    filename: desired filename (keep extension)
    returns saved absolute path
    """
    # make filename unique by appending timestamp
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_name = f"{stamp}_{filename}"
    out_path = UPLOAD_DIR / safe_name

    # file_obj could be a TemporaryUploadedFile or Byte stream
    with open(out_path, "wb") as f:
        f.write(file_obj.read())

    return str(out_path)


def save_results_dataframe(df: pd.DataFrame, base_filename: str = "predictions") -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    fname = f"{base_filename}_{stamp}.xlsx"
    out_path = RESULT_DIR / fname
    df.to_excel(out_path, index=False)
    return str(out_path)


def load_excel_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()
