import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from model.predict       import predict_match, _load_model, _load_rank_cache
from score_model.predict_score import predict_score, _load_params

def get_match_prediction(home: str, away: str, neutral: bool = True) -> dict:
    ml = predict_match(home, away, is_neutral=neutral)
    ps = predict_score(home, away, is_neutral=neutral)
    return {
        "home_team":    home,
        "away_team":    away,
        "is_neutral":   neutral,
        "ml_model": {
            "win":  ml["win"],
            "draw": ml["draw"],
            "loss": ml["loss"],
        },
        "combined": {
            "win":  round((ml["win"]  + ps["win_prob"])  / 2, 4),
            "draw": round((ml["draw"] + ps["draw_prob"]) / 2, 4),
            "loss": round((ml["loss"] + ps["loss_prob"]) / 2, 4),
        },
        "home_rank": ml["home_rank"],
        "away_rank": ml["away_rank"],
    }

def get_score_prediction(home: str, away: str, neutral: bool = True) -> dict:
    return predict_score(home, away, is_neutral=neutral)