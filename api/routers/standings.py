from fastapi import APIRouter, Path
from ..services.standings_service import get_standings, get_group

router = APIRouter(prefix="/standings", tags=["standings"])

@router.get("")
def standings():
    return get_standings()

@router.get("/{group_letter}")
def group(group_letter: str = Path(..., description="Group letter A-L")):
    return get_group(group_letter)