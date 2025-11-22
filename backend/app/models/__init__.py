# backend/app/models/__init__.py
import os
from pathlib import Path

ENV = os.getenv("ENV", "dev")  # defaults to dev if not set

if ENV == "dev":
    # in dev mode import dev predict function
    from .predict_dev import predict_on_excel, predict_on_texts, train_from_storage
else:
    # in prod mode import prod predict function
    from .predict_prod import predict_on_excel, predict_on_texts

# Expose a generic name:
__all__ = ["predict_on_excel", "predict_on_texts"]
