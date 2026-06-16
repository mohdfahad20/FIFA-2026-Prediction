from fastapi import APIRouter
from ..services.predictions_service import get_daily_predictions

router = APIRouter(prefix="/predictions", tags=["predictions"])


@router.get("")
def predictions():
    """
    Returns tomorrow's match predictions and yesterday's
    prediction-vs-actual comparison.
    """
    return get_daily_predictions()