"""
load_historical_data.py
=======================
Loads Kaggle CSVs into fifa.db:
  - results.csv     → matches table
  - fifa_rankings.csv → rankings table

Usage:
    python data/load_historical_data.py \
        --results  data/results.csv \
        --rankings data/fifa_rankings.csv \
        --db       fifa.db

After this script:
  - matches table  : ~49k rows (all history)
  - rankings table : ~70k rows (one per team per ranking date)
  - features table : empty shell (populated by features/features.py)
"""

import sqlite3
import argparse
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Team name normalisation
# Rankings CSV uses slightly different names from results CSV in some cases.
# Add to this dict whenever features.py reports a missing ranking.
# ---------------------------------------------------------------------------
NAME_MAP = {
    # Asia
    "Korea Republic":              "South Korea",
    "Korea DPR":                   "North Korea",
    "IR Iran":                     "Iran",
    "China":                       "China PR",
    "Chinese Taipei":              "Taiwan",
    "Kyrgyz Republic":             "Kyrgyzstan",
    "Brunei Darussalam":           "Brunei",

    # Europe
    "Türkiye":                     "Turkey",
    "Czechia":                     "Czech Republic",
    "Bosnia-Herzegovina":          "Bosnia and Herzegovina",

    # Africa
    "Côte d'Ivoire":               "Ivory Coast",
    "Congo DR":                    "DR Congo",
    "Cabo Verde":                  "Cape Verde",

    # Americas
    "USA":                         "United States",

    # Caribbean
    "St. Kitts and Nevis":         "Saint Kitts and Nevis",
    "St. Lucia":                   "Saint Lucia",
    "St. Vincent / Grenadines":    "Saint Vincent and the Grenadines",

    "The Gambia":                   "Gambia",
    "St Lucia":                     "Saint Lucia",
    "St Vincent and the Grenadines":"Saint Vincent and the Grenadines",
    "Hong Kong, China":             "Hong Kong",
    "St Kitts and Nevis":           "Saint Kitts and Nevis",
    "US Virgin Islands":            "United States Virgin Islands",
}

# ---------------------------------------------------------------------------
# Tournament weight — used in features.py, stored here as reference
# ---------------------------------------------------------------------------
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":                       3,
    "UEFA Euro":                            3,
    "Copa América":                         3,
    "AFC Asian Cup":                        2,
    "Africa Cup of Nations":                2,
    "CONCACAF Gold Cup":                    2,
    "OFC Nations Cup":                      2,
    "FIFA World Cup qualification":         2,
    "UEFA Euro qualification":              2,
    "Copa América qualification":           2,
    "AFC Asian Cup qualification":          2,
    "Africa Cup of Nations qualification":  2,
    "CONCACAF Gold Cup qualification":      2,
    "Friendly":                             1,
}


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id    TEXT PRIMARY KEY,
        date        TEXT NOT NULL,
        home_team   TEXT NOT NULL,
        away_team   TEXT NOT NULL,
        home_score  INTEGER,
        away_score  INTEGER,
        result      TEXT,           -- 'win'|'draw'|'loss' from home team POV
        tournament  TEXT,
        city        TEXT,
        country     TEXT,
        is_neutral  INTEGER,        -- 0 or 1
        stage       TEXT            -- 'group'|'R32'|'R16'|'QF'|'SF'|'Final'|NULL
    );

    CREATE TABLE IF NOT EXISTS rankings (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        rank_date       TEXT NOT NULL,
        team_name       TEXT NOT NULL,
        team_code       TEXT,
        rank            INTEGER,
        total_points    REAL,
        confederation   TEXT
    );

    CREATE TABLE IF NOT EXISTS features (
        match_id                TEXT PRIMARY KEY,
        date                    TEXT,
        home_team               TEXT,
        away_team               TEXT,
        home_rank               INTEGER,
        away_rank               INTEGER,
        rank_diff               REAL,
        home_form               REAL,
        away_form               REAL,
        home_goals_scored_avg   REAL,
        away_goals_scored_avg   REAL,
        home_goals_conceded_avg REAL,
        away_goals_conceded_avg REAL,
        h2h_winrate_home        REAL,
        h2h_goal_diff           REAL,
        is_neutral              INTEGER,
        tournament_weight       INTEGER,
        home_confederation      TEXT,
        away_confederation      TEXT,
        target                  INTEGER     -- 0=loss, 1=draw, 2=win
    );

    CREATE TABLE IF NOT EXISTS simulation_results (
        run_id              TEXT PRIMARY KEY,
        run_at              TEXT,
        n_simulations       INTEGER,
        matches_played      INTEGER,
        matches_remaining   INTEGER,
        results_json        TEXT
    );

    CREATE TABLE IF NOT EXISTS poisson_params (
        team                TEXT PRIMARY KEY,
        attack_strength     REAL,
        defense_weakness    REAL,
        updated_at          TEXT
    );
    """)
    conn.commit()


def load_results(path: str, conn: sqlite3.Connection) -> int:
    print(f"[matches] Reading {path} ...")
    df = pd.read_csv(path)

    # Normalise column types
    df["date"]       = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce").astype("Int64")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce").astype("Int64")
    df["neutral"]    = df["neutral"].astype(str).str.upper().isin(["TRUE", "1", "YES"])

    # Derive result from home team perspective
    def get_result(row):
        if pd.isna(row["home_score"]) or pd.isna(row["away_score"]):
            return None
        if row["home_score"] > row["away_score"]:
            return "win"
        elif row["home_score"] == row["away_score"]:
            return "draw"
        else:
            return "loss"

    df["result"]     = df.apply(get_result, axis=1)
    df["is_neutral"] = df["neutral"].astype(int)
    df["match_id"]   = (
        df["date"] + "_" +
        df["home_team"].str.replace(" ", "_") + "_" +
        df["away_team"].str.replace(" ", "_")
    )
    df["stage"] = None  # populated later for WC 2026 live matches

    rows = df[[
        "match_id", "date", "home_team", "away_team",
        "home_score", "away_score", "result",
        "tournament", "city", "country", "is_neutral", "stage"
    ]].to_dict(orient="records")

    inserted = 0
    skipped  = 0
    for row in rows:
        try:
            conn.execute("""
                INSERT OR IGNORE INTO matches
                (match_id, date, home_team, away_team, home_score, away_score,
                 result, tournament, city, country, is_neutral, stage)
                VALUES
                (:match_id, :date, :home_team, :away_team, :home_score, :away_score,
                 :result, :tournament, :city, :country, :is_neutral, :stage)
            """, row)
            inserted += conn.execute(
                "SELECT changes()"
            ).fetchone()[0]
        except Exception as e:
            skipped += 1
            if skipped <= 5:
                print(f"  [warn] Skipped row: {e} | {row['match_id']}")

    conn.commit()
    print(f"[matches] Inserted {inserted} rows, skipped {skipped} duplicates.")
    return inserted


def load_rankings(path: str, conn: sqlite3.Connection) -> int:
    print(f"[rankings] Reading {path} ...")
    df = pd.read_csv(path)

    # Detect column names (Kaggle dataset has slight variations)
    # Expected: rank_date or date, team_name or country_full, rank or rank_position
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ("rank_date", "date"):
            col_map["rank_date"] = col
        elif cl in ("country_full", "team_name", "name"):
            col_map["team_name"] = col
        elif cl in ("country_abrv", "team_code", "code", "country_abrv"):
            col_map["team_code"] = col
        elif cl in ("rank", "rank_position", "current_rank"):
            col_map["rank"] = col
        elif cl in ("total_points", "points", "total_points"):
            col_map["total_points"] = col
        elif cl in ("confederation",):
            col_map["confederation"] = col

    print(f"[rankings] Detected columns: {col_map}")
    print(f"[rankings] All columns in file: {list(df.columns)}")

    # Rename to standard names
    df = df.rename(columns={v: k for k, v in col_map.items()})

    # Ensure required columns exist
    for req in ("rank_date", "team_name", "rank"):
        if req not in df.columns:
            raise ValueError(
                f"Column '{req}' not found. Available: {list(df.columns)}\n"
                f"Update col_map in load_rankings() to match your CSV headers."
            )

    df["rank_date"] = pd.to_datetime(df["rank_date"], dayfirst=False, errors="coerce")
    df = df.dropna(subset=["rank_date"])
    df["rank_date"] = df["rank_date"].dt.strftime("%Y-%m-%d")

    # Apply name normalisation
    df["team_name"] = df["team_name"].map(
        lambda n: NAME_MAP.get(n, n)
    )

    df["rank"]         = pd.to_numeric(df.get("rank", pd.Series()), errors="coerce").astype("Int64")
    df["total_points"] = pd.to_numeric(df.get("total_points", pd.Series()), errors="coerce")
    df["team_code"]    = df.get("team_code", pd.Series(dtype=str))
    df["confederation"]= df.get("confederation", pd.Series(dtype=str))

    rows = df[[
        "rank_date", "team_name", "team_code", "rank", "total_points", "confederation"
    ]].to_dict(orient="records")

    conn.execute("DELETE FROM rankings")   # full reload is fine, ~70k rows is fast
    conn.executemany("""
        INSERT INTO rankings (rank_date, team_name, team_code, rank, total_points, confederation)
        VALUES (:rank_date, :team_name, :team_code, :rank, :total_points, :confederation)
    """, rows)
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
    print(f"[rankings] Loaded {count} rows.")
    return count


def print_summary(conn: sqlite3.Connection) -> None:
    print("\n" + "="*50)
    print("DATABASE SUMMARY")
    print("="*50)

    m = conn.execute("SELECT COUNT(*) FROM matches").fetchone()[0]
    r = conn.execute("SELECT COUNT(*) FROM rankings").fetchone()[0]
    print(f"  matches  : {m:,}")
    print(f"  rankings : {r:,}")

    # Date range of matches
    lo, hi = conn.execute(
        "SELECT MIN(date), MAX(date) FROM matches"
    ).fetchone()
    print(f"  match dates : {lo}  →  {hi}")

    # Result distribution
    print("\n  Result distribution (home team POV):")
    for result, cnt in conn.execute(
        "SELECT result, COUNT(*) FROM matches GROUP BY result ORDER BY COUNT(*) DESC"
    ).fetchall():
        print(f"    {result or 'NULL':8s} : {cnt:,}")

    # Top tournaments
    print("\n  Top 10 tournaments by match count:")
    for t, cnt in conn.execute(
        "SELECT tournament, COUNT(*) FROM matches GROUP BY tournament ORDER BY COUNT(*) DESC LIMIT 10"
    ).fetchall():
        print(f"    {cnt:5d}  {t}")

    # Ranking date range
    lo_r, hi_r = conn.execute(
        "SELECT MIN(rank_date), MAX(rank_date) FROM rankings"
    ).fetchone()
    print(f"\n  ranking dates : {lo_r}  →  {hi_r}")

    # Teams in rankings but not in matches (top 10)
    unmatched = conn.execute("""
        SELECT DISTINCT r.team_name
        FROM rankings r
        WHERE NOT EXISTS (
            SELECT 1 FROM matches m
            WHERE m.home_team = r.team_name OR m.away_team = r.team_name
        )
        LIMIT 10
    """).fetchall()
    if unmatched:
        print(f"\n  [warn] Teams in rankings not found in matches (top 10):")
        for (t,) in unmatched:
            print(f"    {t}")
        print("  → Add these to NAME_MAP in load_historical_data.py if needed.")

    print("="*50 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Load Kaggle CSVs into fifa.db")
    parser.add_argument("--results",   default="data/results.csv",      help="Path to results.csv")
    parser.add_argument("--rankings",  default="data/fifa_ranking-2026-04-01.csv", help="Path to FIFA rankings CSV")
    parser.add_argument("--db",        default="fifa.db",                help="SQLite DB path")
    args = parser.parse_args()

    db_path = Path(args.db)
    print(f"[init] Opening database: {db_path.resolve()}")
    conn = sqlite3.connect(db_path)

    create_schema(conn)
    load_results(args.results, conn)
    load_rankings(args.rankings, conn)
    print_summary(conn)

    conn.close()
    print("[done] fifa.db is ready. Next: python features/features.py --db fifa.db")


if __name__ == "__main__":
    main()