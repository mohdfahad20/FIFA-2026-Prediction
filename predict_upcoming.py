"""
predict_upcoming.py
===================
Predicts tomorrow's FIFA WC 2026 matches and stores them in the
predictions table. Also fills actual results for any completed
matches that have predictions but no actuals yet.

Usage:
    python predict_upcoming.py --db fifa.db
    python predict_upcoming.py --db fifa.db --date 2026-06-12  # override date

Run order in pipeline:
    1. scraper.py      (fills actual scores first)
    2. predict_upcoming.py  (fills actuals for completed, predicts tomorrow)
"""

import argparse
import json
import logging
import os
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.predict             import predict_match, _load_model, _load_rank_cache
from model.train               import SoftEnsemble, XGBWithWeight, LGBMWithWeight  # noqa
from score_model.predict_score import (
    predict_score,
    _load_params,
    most_likely_score_for_outcome,
    top_scorelines_for_outcome,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL    = "https://api.football-data.org/v4"
COMPETITION = "WC"

FD_TEAM_MAP: dict[str, str] = {
    "Korea Republic"         : "South Korea",
    "USA"                    : "United States",
    "United States"          : "United States",
    "Czechia"                : "Czech Republic",
    "Bosnia and Herzegovina" : "Bosnia-Herzegovina",
    "DR Congo"               : "DR Congo",
    "Cote d'Ivoire"          : "Ivory Coast",
    "Côte d'Ivoire"          : "Ivory Coast",
    "Cape Verde Islands"     : "Cape Verde",
}

STAGE_MAP: dict[str, str] = {
    "GROUP_STAGE"         : "Group Stage",
    "LAST_32"             : "Round of 32",
    "LAST_16"             : "Round of 16",
    "QUARTER_FINALS"      : "Quarter-finals",
    "SEMI_FINALS"         : "Semi-finals",
    "THIRD_PLACE_PLAYOFF" : "Third Place",
    "FINAL"               : "Final",
}

def _norm(name: str) -> str:
    return FD_TEAM_MAP.get(name.strip(), name.strip())

def make_match_id(date: str, home: str, away: str) -> str:
    return f"{date}_{home}_{away}".replace(" ", "_")

# ---------------------------------------------------------------------------
# DB setup — predictions table now includes pred_top_scorelines (JSON)
# ---------------------------------------------------------------------------
CREATE_PREDICTIONS_TABLE = """
CREATE TABLE IF NOT EXISTS predictions (
    match_id            TEXT PRIMARY KEY,
    date                TEXT,
    home_team           TEXT,
    away_team           TEXT,
    stage               TEXT,
    predicted_at        TEXT,
    pred_home_win       REAL,
    pred_draw           REAL,
    pred_away_win       REAL,
    pred_home_xg        REAL,
    pred_away_xg        REAL,
    pred_scoreline      TEXT,
    pred_top_scorelines TEXT,
    pred_winner         TEXT,
    actual_home         INTEGER,
    actual_away         INTEGER,
    actual_result       TEXT,
    was_correct         INTEGER
)
"""

def ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_PREDICTIONS_TABLE)
    # Migration: add column if upgrading from an older table version
    cols = [r[1] for r in conn.execute("PRAGMA table_info(predictions)").fetchall()]
    if "pred_top_scorelines" not in cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN pred_top_scorelines TEXT")
    conn.commit()

# ---------------------------------------------------------------------------
# football-data.org — fetch tomorrow's matches
# ---------------------------------------------------------------------------
def fetch_matches_for_date(target_date: str) -> list[dict]:
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        log.error("FOOTBALL_DATA_API_KEY not set.")
        sys.exit(1)

    headers = {"X-Auth-Token": api_key}
    url     = f"{BASE_URL}/competitions/{COMPETITION}/matches"
    params  = {"dateFrom": target_date, "dateTo": target_date}

    try:
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.error("API request failed: %s", exc)
        sys.exit(1)

    data    = resp.json()
    matches = data.get("matches", [])
    log.info("football-data.org: %d matches on %s", len(matches), target_date)
    return matches

# ---------------------------------------------------------------------------
# Generate prediction for one match
# ---------------------------------------------------------------------------
def predict_one(home_db: str, away_db: str) -> dict:
    """Run ML + Poisson and return combined prediction dict."""
    try:
        ml = predict_match(home_db, away_db, is_neutral=True)
    except Exception as exc:
        log.warning("ML predict failed for %s vs %s: %s", home_db, away_db, exc)
        ml = {"win": 0.333, "draw": 0.334, "loss": 0.333}

    try:
        ps = predict_score(home_db, away_db, is_neutral=True)
    except Exception as exc:
        log.warning("Poisson predict failed for %s vs %s: %s", home_db, away_db, exc)
        ps = {"win_prob": 0.333, "draw_prob": 0.334, "loss_prob": 0.333,
              "expected_goals": {"home": 1.3, "away": 1.1},
              "most_likely_score": "1-1",
              "top_scorelines": [{"score": "1-1", "probability": 1.0}]}

    home_win = round((ml["win"]  + ps["win_prob"])  / 2, 4)
    draw     = round((ml["draw"] + ps["draw_prob"]) / 2, 4)
    away_win = round((ml["loss"] + ps["loss_prob"]) / 2, 4)

    total    = home_win + draw + away_win
    home_win = round(home_win / total, 4)
    draw     = round(draw     / total, 4)
    away_win = round(away_win / total, 4)

    best = max([("home", home_win), ("draw", draw), ("away", away_win)],
               key=lambda x: x[1])

    top3 = top_scorelines_for_outcome(ps["top_scorelines"], best[0], n=3)

    return {
        "pred_home_win"       : home_win,
        "pred_draw"           : draw,
        "pred_away_win"       : away_win,
        "pred_home_xg"        : round(ps["expected_goals"]["home"], 2),
        "pred_away_xg"        : round(ps["expected_goals"]["away"], 2),
        "pred_scoreline"      : top3[0]["score"] if top3 else "1-1",
        "pred_top_scorelines" : json.dumps(top3),
        "pred_winner"         : best[0],
    }

# ---------------------------------------------------------------------------
# Upsert predictions for tomorrow's matches
# ---------------------------------------------------------------------------
def predict_tomorrow(conn: sqlite3.Connection, target_date: str) -> int:
    matches = fetch_matches_for_date(target_date)
    now     = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    count   = 0

    for m in matches:
        if m.get("status") not in ("TIMED", "SCHEDULED"):
            continue

        home_team = _norm(m.get("homeTeam", {}).get("name", "") or "")
        away_team = _norm(m.get("awayTeam", {}).get("name", "") or "")
        if not home_team or not away_team:
            continue

        raw_date  = m.get("utcDate", "")
        game_date = raw_date[:10] if raw_date else target_date
        stage     = STAGE_MAP.get(m.get("stage", "") or "", "")
        match_id  = make_match_id(game_date, home_team, away_team)

        log.info("Predicting: %s vs %s (%s)", home_team, away_team, game_date)
        pred = predict_one(home_team, away_team)

        conn.execute("""
            INSERT INTO predictions
                (match_id, date, home_team, away_team, stage, predicted_at,
                 pred_home_win, pred_draw, pred_away_win,
                 pred_home_xg, pred_away_xg, pred_scoreline,
                 pred_top_scorelines, pred_winner)
            VALUES
                (:match_id, :date, :home_team, :away_team, :stage, :predicted_at,
                 :pred_home_win, :pred_draw, :pred_away_win,
                 :pred_home_xg, :pred_away_xg, :pred_scoreline,
                 :pred_top_scorelines, :pred_winner)
            ON CONFLICT(match_id) DO UPDATE SET
                pred_home_win       = excluded.pred_home_win,
                pred_draw           = excluded.pred_draw,
                pred_away_win       = excluded.pred_away_win,
                pred_home_xg        = excluded.pred_home_xg,
                pred_away_xg        = excluded.pred_away_xg,
                pred_scoreline      = excluded.pred_scoreline,
                pred_top_scorelines = excluded.pred_top_scorelines,
                pred_winner         = excluded.pred_winner,
                predicted_at        = excluded.predicted_at
        """, {
            "match_id"     : match_id,
            "date"         : game_date,
            "home_team"    : home_team,
            "away_team"    : away_team,
            "stage"        : stage,
            "predicted_at" : now,
            **pred,
        })
        count += 1

    conn.commit()
    log.info("Upserted %d predictions for %s", count, target_date)
    return count

# ---------------------------------------------------------------------------
# Fill actuals
# ---------------------------------------------------------------------------
def fill_actuals(conn: sqlite3.Connection) -> int:
    rows = conn.execute("""
        SELECT p.match_id, p.pred_winner,
               m.home_score, m.away_score, m.result
        FROM predictions p
        JOIN matches m ON m.match_id = p.match_id
        WHERE p.actual_result IS NULL
          AND m.home_score IS NOT NULL
    """).fetchall()

    updated = 0
    for match_id, pred_winner, home_score, away_score, result in rows:
        actual_side = (
            "home" if result == "win"
            else "away" if result == "loss"
            else "draw"
        )
        was_correct = 1 if pred_winner == actual_side else 0

        conn.execute("""
            UPDATE predictions SET
                actual_home  = ?,
                actual_away  = ?,
                actual_result = ?,
                was_correct   = ?
            WHERE match_id = ?
        """, (home_score, away_score, actual_side, was_correct, match_id))
        updated += 1

        log.info("Filled actual: %s  pred=%s  actual=%s  correct=%s",
                 match_id, pred_winner, actual_side,
                 "✅" if was_correct else "❌")

    conn.commit()
    log.info("Filled actuals for %d matches", updated)
    return updated

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict tomorrow's WC matches.")
    parser.add_argument("--db",   default="fifa.db")
    parser.add_argument("--date", default=None,
                        help="Target date YYYY-MM-DD (default: tomorrow UTC)")
    args = parser.parse_args()

    if not Path(args.db).exists():
        log.error("DB not found: %s", args.db)
        sys.exit(1)

    if args.date:
        target_date = args.date
    else:
        tomorrow    = datetime.now(timezone.utc).date() + timedelta(days=1)
        target_date = str(tomorrow)

    log.info("Loading models...")
    _load_model()
    _load_rank_cache()
    _load_params()
    log.info("Models loaded.")

    conn = sqlite3.connect(args.db)
    try:
        ensure_table(conn)
        fill_actuals(conn)
        predict_tomorrow(conn, target_date)
    finally:
        conn.close()

    log.info("Done.")

if __name__ == "__main__":
    main()