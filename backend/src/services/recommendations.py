from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.models import (
    MultiRecommendationIn,
    MultiRecommendationOut,
    RecommendationOut,
    now_utc,
)

_GET_SINGLE_REC = None
_GET_MULTI_REC = None


def _get_single_rec_func():
    global _GET_SINGLE_REC
    if _GET_SINGLE_REC is None:
        from serve_bundle import getSingleRec
        _GET_SINGLE_REC = getSingleRec
    return _GET_SINGLE_REC


def _get_multi_rec_func():
    global _GET_MULTI_REC
    if _GET_MULTI_REC is None:
        from serve_bundle import getMultiRec
        _GET_MULTI_REC = getMultiRec
    return _GET_MULTI_REC


class RecommendationService:
    def __init__(self, model_id: str = "christian_model_v1") -> None:
        self._model_id = model_id

    def get_recommendations(
        self,
        anchor_id: str,
        kiosk_id: str,
        limit: int,
    ) -> list[RecommendationOut]:
        get_single_rec = _get_single_rec_func()

        df = get_single_rec(
            anchor_id=anchor_id,
            user_id=kiosk_id,
            topn=limit,
            addDebugInfo=False,
        )

        recommendation_date = now_utc()

        return [
            RecommendationOut(
                anchor_id=row["anchor_id"],
                kiosk_id=row["user_id"],
                product_id=row["product_id"],
                model_id=self._model_id,
                recommendation_date=recommendation_date,
            )
            for row in df.to_dicts()
        ]

    def get_multi_recommendations(
        self,
        items: list[MultiRecommendationIn],
    ) -> list[MultiRecommendationOut]:
        get_multi_rec = _get_multi_rec_func()

        if not items:
            return []

        input_df = pl.DataFrame(
            {
                "anchor_pid": [item.anchor_id for item in items],
                "userid": [item.kiosk_id for item in items],
            },
            schema={"anchor_pid": pl.Utf8, "userid": pl.Utf8},
        )

        df = get_multi_rec(input_df)
        recommendation_date = now_utc()

        return [
            MultiRecommendationOut(
                anchor_id=row["anchor_id"],
                kiosk_id=row["userid"],
                recs=row["recs"],
                model_id=self._model_id,
                recommendation_date=recommendation_date,
            )
            for row in df.to_dicts()
        ]