"""
api/services/predictions_service.py
====================================
Reads the predictions table (populated by predict_upcoming.py) and
returns tomorrow's predictions + yesterday's prediction-vs-actual comparison.
"""

from datetime import datetime, timedelta, timezone

from ..core.database import get_conn


def _row_to_dict(row) -> dict:
    return {
        "home_team":      row["home_team"],
        "away_team":      row["away_team"],
        "stage":          row["stage"],
        "date":           row["date"],
        "pred_home_win":  row["pred_home_win"],
        "pred_draw":      row["pred_draw"],
        "pred_away_win":  row["pred_away_win"],
        "pred_home_xg":   row["pred_home_xg"],
        "pred_away_xg":   row["pred_away_xg"],
        "pred_scoreline": row["pred_scoreline"],
        "pred_winner":    row["pred_winner"],
        "actual_home":    row["actual_home"],
        "actual_away":    row["actual_away"],
        "actual_result":  row["actual_result"],
        "was_correct":    row["was_correct"],
    }


def get_daily_predictions() -> dict:
    """
    Returns:
        {
            "tomorrow":  [ ... predictions for tomorrow (UTC) ... ],
            "yesterday": [ ... predictions for yesterday with actuals filled ... ]
        }
    """
    today_utc     = datetime.now(timezone.utc).date()
    tomorrow_utc  = today_utc + timedelta(days=1)
    yesterday_utc = today_utc - timedelta(days=1)

    conn = get_conn()

    tomorrow_rows = conn.execute("""
        SELECT home_team, away_team, stage, date,
               pred_home_win, pred_draw, pred_away_win,
               pred_home_xg, pred_away_xg, pred_scoreline, pred_winner,
               actual_home, actual_away, actual_result, was_correct
        FROM predictions
        WHERE date = ?
        ORDER BY home_team
    """, (str(tomorrow_utc),)).fetchall()

    yesterday_rows = conn.execute("""
        SELECT home_team, away_team, stage, date,
               pred_home_win, pred_draw, pred_away_win,
               pred_home_xg, pred_away_xg, pred_scoreline, pred_winner,
               actual_home, actual_away, actual_result, was_correct
        FROM predictions
        WHERE date = ?
        ORDER BY home_team
    """, (str(yesterday_utc),)).fetchall()

    conn.close()

    return {
        "tomorrow":  [_row_to_dict(r) for r in tomorrow_rows],
        "yesterday": [_row_to_dict(r) for r in yesterday_rows],
    }