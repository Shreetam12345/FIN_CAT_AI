# backend/app/models/preprocessing.py
import re
import string
from typing import List
from html import unescape

# basic text cleaner suitable for transaction strings
def clean_text(s: str) -> str:
    if s is None:
        return ""
    # unescape html entities
    s = unescape(str(s))
    # lowercase
    s = s.lower()
    # remove common noise tokens
    s = re.sub(r"http\S+|www\S+|@\S+", " ", s)
    # replace numbers with token if needed or keep them; here keep but normalize commas/currency symbols
    s = s.replace(",", "")
    s = re.sub(r"[\$\€\₹\£]", " ", s)
    # remove punctuation except hyphen and slash often present
    s = s.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s

def preprocess_series(series):
    """Apply clean_text across a pandas Series, return list[str]."""
    return series.fillna("").astype(str).apply(clean_text).tolist()
