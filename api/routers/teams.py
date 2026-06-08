from fastapi import APIRouter, Path, Query
from ..services.teams_service import get_all_teams, get_team_detail, get_h2h

router = APIRouter(prefix="/teams", tags=["teams"])

@router.get("")
def teams():
    return get_all_teams()

@router.get("/h2h")
def h2h(
    team1: str = Query(...),
    team2: str = Query(...),
):
    return get_h2h(team1, team2)

@router.get("/{team_name}")
def team(team_name: str = Path(...)):
    return get_team_detail(team_name)