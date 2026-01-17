from fastapi import APIRouter
import logging
from app.models.model_objs import Review
from app.services.IMDB_review_service import classify_IMDB_review_sentiment

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/review_sentiment")
async def predict(review: Review):
    return await classify_IMDB_review_sentiment(review)