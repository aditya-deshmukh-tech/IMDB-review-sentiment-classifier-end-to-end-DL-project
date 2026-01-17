from fastapi import FastAPI
from app.api import health, IMDB_review_api, IMDB_review_ui
from app.core.logging_config import setup_logging
from app.services.model_store import load_DL_model, load_tokenizer

setup_logging()

app = FastAPI()

@app.on_event("startup")
def load_model():
    load_DL_model()
    load_tokenizer()

app.include_router(health.router, prefix="/api")
app.include_router(IMDB_review_api.router, prefix="/api/v1")
app.include_router(IMDB_review_ui.router, prefix="/ui")
        