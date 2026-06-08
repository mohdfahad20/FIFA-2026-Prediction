"""
model/predict.py
================
Thin wrapper around model.pkl.
Exposes predict_match() used by simulation, API, and score model.

Usage (CLI):
    python model/predict.py --home France --away Morocco
    python model/predict.py --home Brazil --away Argentina --neutral false

Usage (import):
    from model.predict import predict_match
    result = predict_match("France", "Morocco", is_neutral=True)
    # → {"win": 0.54, "draw": 0.26, "loss": 0.20}
"""

import pickle
import bisect
import sqlite3
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from model.train import SoftEnsemble, XGBWithWeight, LGBMWithWeight  # noqa: F401 — needed for pickle

# Add this class near the top of model/predict.py, after the imports:
class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "model.train"
        return super().find_class(module, name)

# ---------------------------------------------------------------------------
# Load model + rankings (cached at module level — loaded once)
# ---------------------------------------------------------------------------
_model_cache    = None
_rank_cache     = None   # {team: (sorted_dates, ranks)}
_conf_cache     = None   # {team: confederation}

MODEL_PATH = Path("model/model.pkl")
DB_PATH    = Path("fifa.db")


# Then change _load_model() to use it:
def _load_model():
    global _model_cache
    if _model_cache is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_PATH}. Run model/train.py first.")
        with open(MODEL_PATH, "rb") as f:
            _model_cache = _Unpickler(f).load()
    return _model_cache


def _load_rank_cache():
    global _rank_cache, _conf_cache
    if _rank_cache is not None:
        return _rank_cache, _conf_cache

    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}.")

    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("""
        SELECT team_name, rank_date, rank, confederation
        FROM rankings
        ORDER BY team_name, rank_date
    """).fetchall()
    conn.close()

    rank_cache = {}
    conf_cache = {}

    current_team = None
    dates, ranks, confs = [], [], []

    for team, rank_date, rank, confederation in rows:
        if team != current_team:
            if current_team:
                rank_cache[current_team] = (dates, ranks)
                conf_cache[current_team] = confs[-1] if confs else None
            current_team = team
            dates, ranks, confs = [], [], []
        dates.append(pd.Timestamp(rank_date))
        ranks.append(rank)
        confs.append(confederation)

    if current_team:
        rank_cache[current_team] = (dates, ranks)
        conf_cache[current_team] = confs[-1] if confs else None

    # ── Manual overrides for WC 2026 teams missing from rankings ──
    MANUAL_OVERRIDES = {
        "Bosnia-Herzegovina": (61,  "UEFA"),
        "Curaçao":            (86,  "CONCACAF"),
        "Iraq":               (63,  "AFC"),
    }
    ref_date = [pd.Timestamp("2026-04-01")]
    for team, (rank, conf) in MANUAL_OVERRIDES.items():
        if team not in rank_cache:
            rank_cache[team] = (ref_date, [rank])
            conf_cache[team] = conf

    _rank_cache = rank_cache
    _conf_cache = conf_cache
    return _rank_cache, _conf_cache


def _get_rank(team: str, match_date: pd.Timestamp, rank_cache: dict):
    if team not in rank_cache:
        return None
    dates, ranks = rank_cache[team]
    idx = bisect.bisect_right(dates, match_date) - 1
    return ranks[idx] if idx >= 0 else None


# ---------------------------------------------------------------------------
# Core feature builder for a single matchup
# ---------------------------------------------------------------------------
def _build_feature_row(
    home_team: str,
    away_team: str,
    is_neutral: bool,
    match_date: pd.Timestamp,
    home_form: float = 0.5,
    away_form: float = 0.5,
    home_goals_scored: float = 1.37,
    away_goals_scored: float = 1.37,
    home_goals_conceded: float = 1.37,
    away_goals_conceded: float = 1.37,
    h2h_winrate: float = 0.5,
    h2h_goal_diff: float = 0.0,
    tournament_weight: int = 3,
) -> tuple:
    """
    Builds a feature row dict ready for model prediction.
    Returns (feature_dict, home_rank, away_rank) — ranks returned
    for display purposes.
    """
    payload  = _load_model()
    rank_cache, conf_cache = _load_rank_cache()

    feature_cols  = payload["feature_cols"]   # numeric cols
    conf_cols     = payload["conf_cols"]       # confederation cols
    feature_names = payload["feature_names"]  # all cols after one-hot

    # Rankings
    home_rank = _get_rank(home_team, match_date, rank_cache)
    away_rank = _get_rank(away_team, match_date, rank_cache)

    # Fallback for unknown teams — use median rank (100)
    if home_rank is None:
        home_rank = 100
    if away_rank is None:
        away_rank = 100

    rank_diff = float(home_rank) - float(away_rank)

    # Confederation
    home_conf = conf_cache.get(home_team)
    away_conf = conf_cache.get(away_team)

    # Build numeric feature dict
    numeric = {
        "home_rank":               float(home_rank),
        "away_rank":               float(away_rank),
        "rank_diff":               rank_diff,
        "home_form":               home_form,
        "away_form":               away_form,
        "home_goals_scored_avg":   home_goals_scored,
        "away_goals_scored_avg":   away_goals_scored,
        "home_goals_conceded_avg": home_goals_conceded,
        "away_goals_conceded_avg": away_goals_conceded,
        "h2h_winrate_home":        h2h_winrate,
        "h2h_goal_diff":           h2h_goal_diff,
        "is_neutral":              int(is_neutral),
        "tournament_weight":       tournament_weight,
    }

    # One-hot confederation — match training schema exactly
    conf_dummies = {}
    for col in feature_names:
        if col.startswith("home_conf_") or col.startswith("away_conf_"):
            conf_dummies[col] = 0

    # Set the correct confederation flags
    if home_conf:
        key = f"home_conf_{home_conf}"
        if key in conf_dummies:
            conf_dummies[key] = 1
        else:
            key_nan = "home_conf_nan"
            if key_nan in conf_dummies:
                conf_dummies[key_nan] = 1

    if away_conf:
        key = f"away_conf_{away_conf}"
        if key in conf_dummies:
            conf_dummies[key] = 1
        else:
            key_nan = "away_conf_nan"
            if key_nan in conf_dummies:
                conf_dummies[key_nan] = 1

    # Assemble final row in exact feature order
    row = {}
    for col in feature_names:
        if col in numeric:
            row[col] = numeric[col]
        elif col in conf_dummies:
            row[col] = conf_dummies[col]
        else:
            row[col] = 0

    X = np.array([[row[col] for col in feature_names]])
    return X, home_rank, away_rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def predict_match(
    home_team: str,
    away_team: str,
    is_neutral: bool = True,
    match_date: pd.Timestamp = None,
    home_form: float = 0.5,
    away_form: float = 0.5,
    home_goals_scored: float = 1.37,
    away_goals_scored: float = 1.37,
    home_goals_conceded: float = 1.37,
    away_goals_conceded: float = 1.37,
    h2h_winrate: float = 0.5,
    h2h_goal_diff: float = 0.0,
    tournament_weight: int = 3,
) -> dict:
    """
    Predict win/draw/loss probabilities for a matchup.

    All WC 2026 matches are neutral=True.
    When called from simulation, pass pre-computed form/goals/h2h stats.
    When called standalone, uses neutral priors for unknown stats.

    Returns:
        {
            "home_team": "France",
            "away_team": "Morocco",
            "win":  0.54,   # home team wins
            "draw": 0.26,
            "loss": 0.20,   # home team loses
            "home_rank": 3,
            "away_rank": 12,
        }
    """
    if match_date is None:
        match_date = pd.Timestamp("2026-06-11")  # WC 2026 start date

    payload  = _load_model()
    ensemble = payload["ensemble"]

    X, home_rank, away_rank = _build_feature_row(
        home_team, away_team, is_neutral, match_date,
        home_form, away_form,
        home_goals_scored, away_goals_scored,
        home_goals_conceded, away_goals_conceded,
        h2h_winrate, h2h_goal_diff, tournament_weight,
    )

    proba = ensemble.predict_proba(X)[0]  # shape (3,) → [loss, draw, win]

    return {
        "home_team": home_team,
        "away_team": away_team,
        "win":       round(float(proba[2]), 4),
        "draw":      round(float(proba[1]), 4),
        "loss":      round(float(proba[0]), 4),
        "home_rank": int(home_rank),
        "away_rank": int(away_rank),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict a single match")
    parser.add_argument("--home",    required=True,  help="Home team name")
    parser.add_argument("--away",    required=True,  help="Away team name")
    parser.add_argument("--neutral", default="true", help="Neutral venue (true/false)")
    parser.add_argument("--model",   default="model/model.pkl")
    parser.add_argument("--db",      default="fifa.db")
    args = parser.parse_args()

    global MODEL_PATH, DB_PATH
    MODEL_PATH = Path(args.model)
    DB_PATH    = Path(args.db)

    is_neutral = args.neutral.lower() != "false"

    result = predict_match(args.home, args.away, is_neutral=is_neutral)

    print(f"\n{'='*45}")
    print(f"  {result['home_team']} vs {result['away_team']}")
    print(f"  Venue: {'Neutral' if is_neutral else 'Home advantage'}")
    print(f"  FIFA Rank: {result['home_team']} #{result['home_rank']}  "
          f"vs  {result['away_team']} #{result['away_rank']}")
    print(f"{'='*45}")
    print(f"  Win  ({result['home_team']:>12s}) : {result['win']:.1%}")
    print(f"  Draw                  : {result['draw']:.1%}")
    print(f"  Loss ({result['away_team']:>12s}) : {result['loss']:.1%}")
    print(f"{'='*45}\n")


if __name__ == "__main__":
    main()  