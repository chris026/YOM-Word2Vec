from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Query

from src.models import (
    MultiRecommendationIn,
    MultiRecommendationOut,
    RecommendationOut,
)
from src.services.recommendations import RecommendationService

router = APIRouter(
    prefix="/recommendations",
    tags=["recommendations"],
)

DEFAULT_RECOMMENDATION_LIMIT = int(
    os.getenv("DEFAULT_RECOMMENDATION_LIMIT", "30")
)


def get_recommendation_service() -> RecommendationService:
    raise RuntimeError("RecommendationService dependency not configured")


@router.get("", response_model=list[RecommendationOut])
def get_recommendations(
    anchorId: str = Query(..., min_length=1),
    kioskId: str = Query(..., min_length=1),
    limit: int | None = Query(None, ge=1, le=100),
    service: RecommendationService = Depends(get_recommendation_service),
) -> list[RecommendationOut]:
    effective_limit = limit if limit is not None else DEFAULT_RECOMMENDATION_LIMIT

    return service.get_recommendations(
        anchor_id=anchorId,
        kiosk_id=kioskId,
        limit=effective_limit,
    )


@router.post("/multi", response_model=list[MultiRecommendationOut])
def get_multi_recommendations(
    items: list[MultiRecommendationIn],
    service: RecommendationService = Depends(get_recommendation_service),
) -> list[MultiRecommendationOut]:
    return service.get_multi_recommendations(items)