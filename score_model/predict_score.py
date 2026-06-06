"""
score_model/predict_score.py
============================
Dixon-Coles Poisson score predictor.

Loads poisson_params.pkl and for any matchup returns:
  - Expected goals for each team
  - Full scoreline probability matrix
  - Top N most likely scorelines
  - Win/draw/loss probabilities (derived from scoreline matrix)

Usage (CLI):
    python -m score_model.predict_score --home France --away Brazil
    python -m score_model.predict_score --home Spain --away Morocco --neutral false

Usage (import):
    from score_model.predict_score import predict_score
    result = predict_score("France", "Brazil")
"""

import pickle
import argparse
import numpy as np
from pathlib import Path
from scipy.stats import poisson

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PARAMS_PATH  = Path("score_model/poisson_params.pkl")
DEFAULT_ATT  = 1.0
DEFAULT_DEF  = 1.0

_params_cache = None


# ---------------------------------------------------------------------------
# Load params
# ---------------------------------------------------------------------------
def _load_params() -> dict:
    global _params_cache
    if _params_cache is None:
        if not PARAMS_PATH.exists():
            raise FileNotFoundError(
                f"Params not found: {PARAMS_PATH}. "
                "Run score_model/train_poisson.py first."
            )
        with open(PARAMS_PATH, "rb") as f:
            _params_cache = pickle.load(f)
    return _params_cache


# ---------------------------------------------------------------------------
# Core prediction
# ---------------------------------------------------------------------------
def predict_score(
    home_team: str,
    away_team: str,
    is_neutral: bool = True,
    top_n: int = 8,
) -> dict:
    """
    Predict scoreline probabilities using Dixon-Coles Poisson model.

    Args:
        home_team  : Name exactly as in DB (e.g. "France")
        away_team  : Name exactly as in DB
        is_neutral : True for all WC matches
        top_n      : Number of top scorelines to return

    Returns:
        {
            "home_team": "France",
            "away_team": "Brazil",
            "expected_goals": {"home": 1.72, "away": 1.31},
            "most_likely_score": "1-1",
            "top_scorelines": [
                {"score": "1-1", "probability": 0.118},
                ...
            ],
            "win_prob":  0.42,
            "draw_prob": 0.28,
            "loss_prob": 0.30,
        }
    """
    params     = _load_params()
    attack     = params["attack"]
    defense    = params["defense"]
    global_avg = params["global_avg"]
    home_adv   = 1.0 if is_neutral else params.get("home_advantage", 1.2)
    rho        = params.get("rho", -0.13)
    max_goals  = params.get("max_goals", 7)

    # Get team params with fallback
    att_h = attack.get(home_team, DEFAULT_ATT)
    def_h = defense.get(home_team, DEFAULT_DEF)
    att_a = attack.get(away_team, DEFAULT_ATT)
    def_a = defense.get(away_team, DEFAULT_DEF)

    if home_team not in attack:
        print(f"  [warn] No Poisson params for '{home_team}' — using defaults")
    if away_team not in attack:
        print(f"  [warn] No Poisson params for '{away_team}' — using defaults")

    # Expected goals
    lambda_h = att_h * def_a * global_avg * home_adv
    lambda_a = att_a * def_h * global_avg

    # Scoreline probability matrix (max_goals × max_goals)
    score_matrix = np.zeros((max_goals, max_goals))
    for i in range(max_goals):
        for j in range(max_goals):
            p = poisson.pmf(i, lambda_h) * poisson.pmf(j, lambda_a)
            # Dixon-Coles correction for low scores
            p *= _dc_correction(i, j, lambda_h, lambda_a, rho)
            score_matrix[i][j] = p

    # Normalise (correction factor slightly shifts total away from 1)
    score_matrix /= score_matrix.sum()

    # Win / draw / loss from matrix
    win_prob  = float(np.sum(np.tril(score_matrix, k=-1)))  # home scores more
    draw_prob = float(np.sum(np.diag(score_matrix)))         # equal scores
    loss_prob = float(np.sum(np.triu(score_matrix, k=1)))    # away scores more

    # Top scorelines
    flat = []
    for i in range(max_goals):
        for j in range(max_goals):
            flat.append((f"{i}-{j}", score_matrix[i][j]))
    flat.sort(key=lambda x: x[1], reverse=True)
    top_scorelines = [
        {"score": s, "probability": round(float(p), 4)}
        for s, p in flat[:top_n]
    ]
    most_likely = top_scorelines[0]["score"]

    return {
        "home_team":    home_team,
        "away_team":    away_team,
        "expected_goals": {
            "home": round(lambda_h, 3),
            "away": round(lambda_a, 3),
        },
        "most_likely_score": most_likely,
        "top_scorelines":    top_scorelines,
        "win_prob":  round(win_prob,  4),
        "draw_prob": round(draw_prob, 4),
        "loss_prob": round(loss_prob, 4),
    }


def _dc_correction(i: int, j: int,
                   lh: float, la: float, rho: float) -> float:
    """Dixon-Coles correction for low-scoring scorelines."""
    if   i == 0 and j == 0: return 1 - lh * la * rho
    elif i == 1 and j == 0: return 1 + la * rho
    elif i == 0 and j == 1: return 1 + lh * rho
    elif i == 1 and j == 1: return 1 - rho
    else:                   return 1.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--home",    required=True)
    parser.add_argument("--away",    required=True)
    parser.add_argument("--neutral", default="true")
    parser.add_argument("--top-n",   default=8, type=int)
    parser.add_argument("--params",  default="score_model/poisson_params.pkl")
    args = parser.parse_args()

    global PARAMS_PATH
    PARAMS_PATH = Path(args.params)

    is_neutral = args.neutral.lower() != "false"
    result     = predict_score(args.home, args.away,
                               is_neutral=is_neutral, top_n=args.top_n)

    print(f"\n{'='*50}")
    print(f"  {result['home_team']} vs {result['away_team']}")
    print(f"  Venue: {'Neutral' if is_neutral else 'Home advantage'}")
    print(f"{'='*50}")
    print(f"  Expected goals : {result['home_team']} {result['expected_goals']['home']:.2f}"
          f"  —  {result['away_team']} {result['expected_goals']['away']:.2f}")
    print(f"  Most likely    : {result['most_likely_score']}")
    print(f"\n  Top scorelines:")
    for s in result["top_scorelines"]:
        bar = "█" * int(s["probability"] * 100)
        print(f"    {s['score']:5s}  {s['probability']:.1%}  {bar}")
    print(f"\n  Win  ({result['home_team']:>10s}) : {result['win_prob']:.1%}")
    print(f"  Draw               : {result['draw_prob']:.1%}")
    print(f"  Loss ({result['away_team']:>10s}) : {result['loss_prob']:.1%}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()