from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field


class RecommendationOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anchor_id: str
    kiosk_id: str
    product_id: str
    model_id: str
    recommendation_date: datetime


class MultiRecommendationIn(BaseModel):
    anchor_id: str = Field(min_length=1)
    kiosk_id: str = Field(min_length=1)


class MultiRecommendationOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    anchor_id: str
    kiosk_id: str
    recs: list[str]
    model_id: str
    recommendation_date: datetime


def now_utc() -> datetime:
    return datetime.now(timezone.utc)