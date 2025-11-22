from dotenv import load_dotenv
load_dotenv(dotenv_path=".env")


from fastapi import FastAPI
from contextlib import asynccontextmanager

from .router import router
from .services import database as dbsvc


@asynccontextmanager
async def lifespan(app: FastAPI):
    dbsvc.init_db()
    yield


app = FastAPI(title="FinTrans-AI Backend", lifespan=lifespan)

app.include_router(router)
