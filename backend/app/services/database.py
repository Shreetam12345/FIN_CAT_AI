# backend/app/services/database.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime, timezone
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # backend/app
DB_DIR = os.path.join(BASE_DIR, "storage")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "app.db")

SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Upload(Base):
    __tablename__ = "uploads"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)   # local path or URL
    uploaded_at = Column(DateTime, default=datetime.now(timezone.utc))
    n_rows = Column(Integer, default=0)
    meta = Column(Text, nullable=True)  # optional JSON string for extra metadata

    results = relationship("Result", back_populates="upload")


class Result(Base):
    __tablename__ = "results"
    id = Column(Integer, primary_key=True, index=True)
    upload_id = Column(Integer, ForeignKey("uploads.id"), nullable=True)
    raw_text = Column(Text, nullable=False)
    predicted_category = Column(String, nullable=False)
    confidence = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))

    upload = relationship("Upload", back_populates="results")


def init_db():
    Base.metadata.create_all(bind=engine)


# convenience helper functions
def insert_upload(db_session, filename, filepath, n_rows=0, meta=None):
    u = Upload(filename=filename, filepath=filepath, n_rows=n_rows, meta=meta)
    db_session.add(u)
    db_session.commit()
    db_session.refresh(u)
    return u


def insert_results_bulk(db_session, upload_id, rows):
    """
    rows: iterable of dicts: {"raw_text":..., "predicted_category":..., "confidence":...}
    """
    objs = []
    for r in rows:
        objs.append(Result(upload_id=upload_id,
                           raw_text=r["raw_text"],
                           predicted_category=r["predicted_category"],
                           confidence=r.get("confidence", None)))
    db_session.bulk_save_objects(objs)
    db_session.commit()
    return len(objs)


def get_upload(db_session, upload_id):
    return db_session.query(Upload).filter(Upload.id == upload_id).first()
