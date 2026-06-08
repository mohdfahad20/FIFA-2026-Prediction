from fastapi import APIRouter, Query
from ..services.prediction_service import get_match_prediction, get_score_prediction

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.get("")
def predict(
    home:    str  = Query(..., description="Home team name"),
    away:    str  = Query(..., description="Away team name"),
    neutral: bool = Query(True, description="Neutral venue"),
):
    return get_match_prediction(home, away, neutral)

@router.get("/score")
def score(
    home:    str  = Query(...),
    away:    str  = Query(...),
    neutral: bool = Query(True),
):
    return get_score_prediction(home, away, neutral)