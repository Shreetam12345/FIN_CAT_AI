# backend/train_dev_models.py
from app.models.predict_dev import train_from_storage

if __name__ == "__main__":
    train_from_storage(save_dev=True)
    print("Training finished.")
