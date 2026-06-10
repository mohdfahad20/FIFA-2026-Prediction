"""
scraper/scraper.py
==================
Fetches FIFA World Cup 2026 match results from football-data.org and
upserts them into the local fifa.db SQLite database.

Usage:
    python -m scraper.scraper --db fifa.db          # yesterday + today (nightly default)
    python -m scraper.scraper --db fifa.db --full   # full tournament window

Environment (.env):
    FOOTBALL_DATA_API_KEY  — free tier key from football-data.org

API facts:
    Base URL : https://api.football-data.org/v4
    Endpoint : GET /competitions/WC/matches
    Auth     : X-Auth-Token header
    Rate     : 10 req/min on free tier
    WC code  : "WC"  (competition id 2000, season 2026)
    Statuses : FINISHED = completed match we want
               TIMED / SCHEDULED / IN_PLAY / PAUSED = skip
"""

import argparse
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone

import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_URL      = "https://api.football-data.org/v4"
COMPETITION   = "WC"
WC_START_DATE = "2026-06-11"
WC_END_DATE   = "2026-07-19"
TOURNAMENT    = "FIFA World Cup"

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


def _result(home: int, away: int) -> str:
    if home > away:  return "win"
    if home == away: return "draw"
    return "loss"


def _match_id(date: str, home: str, away: str) -> str:
    return f"{date}_{home}_{away}"


class FDClient:
    def __init__(self, api_key: str):
        self.session = requests.Session()
        self.session.headers.update({"X-Auth-Token": api_key})

    def get_wc_matches(self, date_from=None, date_to=None, status=None) -> list[dict]:
        params = {}
        if date_from: params["dateFrom"] = date_from
        if date_to:   params["dateTo"]   = date_to
        if status:    params["status"]   = status

        url = f"{BASE_URL}/competitions/{COMPETITION}/matches"

        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=30)
            except requests.RequestException as exc:
                log.error("Network error: %s", exc)
                raise

            if resp.status_code == 429:
                wait = int(resp.headers.get("X-RequestCounter-Reset", 60))
                log.warning("Rate-limited — waiting %ds", wait)
                time.sleep(wait)
                continue

            if resp.status_code == 403:
                log.error("403 Forbidden — check your FOOTBALL_DATA_API_KEY.")
                sys.exit(1)

            resp.raise_for_status()
            data = resp.json()
            matches = data.get("matches", [])
            log.info("football-data.org returned %d match records (played=%s)",
                     len(matches),
                     data.get("resultSet", {}).get("played", "?"))
            return matches

        log.error("Exhausted retries.")
        sys.exit(1)


def _parse_match(m: dict) -> dict | None:
    if m.get("status") != "FINISHED":
        return None

    home_team = _norm(m.get("homeTeam", {}).get("name", "") or "")
    away_team = _norm(m.get("awayTeam", {}).get("name", "") or "")
    if not home_team or not away_team:
        log.warning("Missing team name in match id=%s", m.get("id"))
        return None

    ft = m.get("score", {}).get("fullTime", {})
    try:
        home_score = int(ft.get("home") or 0)
        away_score = int(ft.get("away") or 0)
    except (TypeError, ValueError):
        log.warning("Unparseable score for %s vs %s", home_team, away_team)
        return None

    raw_date = m.get("utcDate", "")
    game_date = raw_date[:10] if raw_date else ""

    stage = STAGE_MAP.get(m.get("stage", "") or "", m.get("stage", "") or "")

    return {
        "match_id"   : _match_id(game_date, home_team, away_team),
        "date"       : game_date,
        "home_team"  : home_team,
        "away_team"  : away_team,
        "home_score" : home_score,
        "away_score" : away_score,
        "result"     : _result(home_score, away_score),
        "tournament" : TOURNAMENT,
        "city"       : "",
        "country"    : "Neutral",
        "is_neutral" : 1,
        "stage"      : stage,
    }


def upsert_matches(conn: sqlite3.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT OR REPLACE INTO matches
            (match_id, date, home_team, away_team,
             home_score, away_score, result,
             tournament, city, country, is_neutral, stage)
        VALUES
            (:match_id, :date, :home_team, :away_team,
             :home_score, :away_score, :result,
             :tournament, :city, :country, :is_neutral, :stage)
    """
    cur = conn.cursor()
    cur.executemany(sql, rows)
    conn.commit()
    return cur.rowcount


def run(db_path: str, date_from: str, date_to: str) -> int:
    api_key = os.environ.get("FOOTBALL_DATA_API_KEY", "").strip()
    if not api_key:
        log.error("FOOTBALL_DATA_API_KEY env var not set.")
        sys.exit(1)

    log.info("Fetching FINISHED WC matches  %s -> %s", date_from, date_to)
    client = FDClient(api_key)
    raw = client.get_wc_matches(date_from=date_from, date_to=date_to, status="FINISHED")

    rows = [r for m in raw if (r := _parse_match(m)) is not None]
    log.info("Parsed %d completed WC matches", len(rows))

    if not rows:
        log.info("Nothing to upsert — no completed matches in window.")
        return 0

    for r in rows:
        log.info("  %s  %s %d-%d %s",
                 r["date"], r["home_team"],
                 r["home_score"], r["away_score"],
                 r["away_team"])

    conn = sqlite3.connect(db_path)
    try:
        n = upsert_matches(conn, rows)
    finally:
        conn.close()

    log.info("Upserted %d rows -> %s", n, db_path)
    return n


def _default_window() -> tuple[str, str]:
    today     = datetime.now(timezone.utc).date()
    yesterday = today - timedelta(days=1)
    return str(yesterday), str(today)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape FIFA WC 2026 results.")
    parser.add_argument("--db",    default="fifa.db")
    parser.add_argument("--start", default=None, help="YYYY-MM-DD (default: yesterday)")
    parser.add_argument("--end",   default=None, help="YYYY-MM-DD (default: today)")
    parser.add_argument("--full",  action="store_true",
                        help="Fetch full tournament window (June 11 - July 19)")
    args = parser.parse_args()

    if args.full:
        start, end = WC_START_DATE, WC_END_DATE
    else:
        d_start, d_end = _default_window()
        start = args.start or d_start
        end   = args.end   or d_end

    try:
        run(args.db, start, end)
    except Exception as exc:
        log.exception("Fatal error: %s", exc)
        sys.exit(1)
