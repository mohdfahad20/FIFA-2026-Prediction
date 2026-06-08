from fastapi import APIRouter
from ..services.simulation_service import get_latest_simulation, get_simulation_history

router = APIRouter(prefix="/simulation", tags=["simulation"])

@router.get("")
def simulation():
    return get_latest_simulation()

@router.get("/history")
def history():
    return get_simulation_history()