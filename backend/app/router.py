from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import pandas as pd
import io
import os

from app.services import database as dbsvc
from app.services import file_utils
from app.models.gpt_classifier import gpt_predict_texts, gpt_predict_excel

router = APIRouter()


# -----------------------
# ENV endpoint
# -----------------------
@router.get("/api/env")
async def get_env():
    return {"env": os.getenv("ENV", "dev")}


# -----------------------
# Helper: safe GPT JSON response
# -----------------------
async def safe_gpt_texts(texts):
    try:
        results = await gpt_predict_texts(texts)
        # Ensure uniform structure: [{"transaction": "...", "category": "..."}]
        formatted = [{"transaction": t, "category": c} for t, c in zip(texts, results)]
        return formatted
    except Exception as e:
        return {"detail": str(e)}


async def safe_gpt_excel(df: pd.DataFrame):
    try:
        res_df = await gpt_predict_excel(df)
        return res_df
    except Exception as e:
        # If GPT fails, return a DataFrame with 'error' column
        df["category"] = f"Error: {e}"
        return df


# -----------------------
# GPT PROD endpoints
# -----------------------
@router.post("/api/predict_prod")
async def predict_prod_json(payload: dict):
    texts = payload.get("transactions", [])
    if not texts:
        raise HTTPException(status_code=400, detail="No transactions provided")

    results = await safe_gpt_texts(texts)

    # Save to DB only if results are valid
    if isinstance(results, list):
        db = dbsvc.SessionLocal()
        upload_row = dbsvc.insert_upload(
            db_session=db,
            filename="inline_json",
            filepath="inline",
            n_rows=len(texts)
        )
        dbsvc.insert_results_bulk(db, upload_row.id, results)
        db.close()

    return JSONResponse(content=results)


@router.post("/api/predict_prod_excel")
async def predict_prod_excel(file: UploadFile = File(...)):
    body = await file.read()
    saved_path = file_utils.save_uploaded_file(io.BytesIO(body), file.filename)

    df = pd.read_excel(io.BytesIO(body))
    res_df = await safe_gpt_excel(df)

    db = dbsvc.SessionLocal()
    upload_row = dbsvc.insert_upload(
        db_session=db,
        filename=file.filename,
        filepath=saved_path,
        n_rows=len(res_df)
    )
    results = res_df.to_dict(orient="records")
    dbsvc.insert_results_bulk(db, upload_row.id, results)
    db.close()

    output = io.BytesIO()
    res_df.to_excel(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=predictions.xlsx"}
    )


# -----------------------
# GPT DEV endpoints
# -----------------------
@router.post("/api/predict_dev")
async def predict_dev_json(payload: dict):
    texts = payload.get("transactions", [])
    if not texts:
        raise HTTPException(status_code=400, detail="No transactions provided")

    results = await safe_gpt_texts(texts)
    return JSONResponse(content=results)


@router.post("/api/predict_dev_excel")
async def predict_dev_excel(file: UploadFile = File(...)):
    body = await file.read()
    df = pd.read_excel(io.BytesIO(body))
    res_df = await safe_gpt_excel(df)

    output = io.BytesIO()
    res_df.to_excel(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=predictions.xlsx"}
    )
