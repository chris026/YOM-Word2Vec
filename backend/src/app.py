from __future__ import annotations

import os

from fastapi import FastAPI
from mangum import Mangum

from src.routes.health import router as health_router
from src.routes.recommendations import (
    router as recommendations_router,
    get_recommendation_service as recommendations_get_service,
)
from src.services.recommendations import RecommendationService

app = FastAPI(title="YOM Recommender Backend", version="0.4.0")

MODEL_ID = os.getenv("MODEL_ID", "christian_model_v1")

_service = RecommendationService(model_id=MODEL_ID)


def _get_recommendation_service() -> RecommendationService:
    return _service


app.dependency_overrides[recommendations_get_service] = _get_recommendation_service

app.include_router(health_router)
app.include_router(recommendations_router)

handler = Mangum(app)