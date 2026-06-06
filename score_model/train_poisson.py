"""
score_model/train_poisson.py
============================
Fits Dixon-Coles Poisson model for score prediction.

Fixes applied:
  1. FIFA-ranked teams only — filters out non-FIFA micro-nations
  2. Time decay — recent matches weighted more (half-life = 3 years)
  3. min-matches = 20 after FIFA filter

Usage:
    python -m score_model.train_poisson --db fifa.db
    python -m score_model.train_poisson --db fifa.db --min-year 2010 --min-matches 20
"""

import sqlite3
import pickle
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

MAX_GOALS   = 7
HOME_ADV    = 1.2
DEFAULT_ATT = 1.0
DEFAULT_DEF = 1.0
DECAY_YEARS = 3      # half-life for time decay


# ---------------------------------------------------------------------------
# Load + filter matches
# ---------------------------------------------------------------------------
def load_matches(db_path: str, min_year: int = 2010) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)

    df = pd.read_sql(f"""
        SELECT date, home_team, away_team,
               home_score, away_score, is_neutral
        FROM matches
        WHERE home_score IS NOT NULL
          AND away_score IS NOT NULL
          AND strftime('%Y', date) >= '{min_year}'
        ORDER BY date ASC
    """, conn, parse_dates=["date"])

    # Load FIFA-ranked teams
    ranked_teams = set(pd.read_sql(
        "SELECT DISTINCT team_name FROM rankings", conn
    )["team_name"].tolist())

    conn.close()

    print(f"[data] Loaded {len(df):,} matches from {min_year} onwards.")

    # Filter to FIFA-ranked teams only
    before = len(df)
    df = df[
        df["home_team"].isin(ranked_teams) &
        df["away_team"].isin(ranked_teams)
    ].copy()
    print(f"[filter] FIFA-ranked teams only: {before:,} → {len(df):,} matches "
          f"(removed {before - len(df):,} non-FIFA)")

    # Time decay — exponential, half-life = DECAY_YEARS
    reference_date  = df["date"].max()
    df["days_ago"]  = (reference_date - df["date"]).dt.days
    df["weight"]    = np.exp(-df["days_ago"] / (365 * DECAY_YEARS))

    # Weighted global averages
    total_weight          = df["weight"].sum()
    global_avg_home       = (df["home_score"] * df["weight"]).sum() / total_weight
    global_avg_away       = (df["away_score"] * df["weight"]).sum() / total_weight
    global_avg            = (global_avg_home + global_avg_away) / 2

    print(f"[data] Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"[data] Weighted avg: home={global_avg_home:.3f}  "
          f"away={global_avg_away:.3f}  combined={global_avg:.3f}")

    # Store globals on df for access downstream
    df.attrs["global_avg"]      = global_avg
    df.attrs["global_avg_home"] = global_avg_home
    df.attrs["global_avg_away"] = global_avg_away

    return df


# ---------------------------------------------------------------------------
# Compute attack / defense params
# ---------------------------------------------------------------------------
def compute_team_params(df: pd.DataFrame, min_matches: int = 20) -> dict:
    global_avg      = df.attrs["global_avg"]
    global_avg_home = df.attrs["global_avg_home"]
    global_avg_away = df.attrs["global_avg_away"]

    # Per-team weighted stats
    teams = set(df["home_team"]) | set(df["away_team"])
    stats = {t: {"scored": 0.0, "conceded": 0.0,
                 "weight": 0.0, "matches": 0} for t in teams}

    for _, row in df.iterrows():
        w   = row["weight"]
        ht  = row["home_team"]
        at  = row["away_team"]
        hs  = row["home_score"]
        as_ = row["away_score"]

        stats[ht]["scored"]   += hs  * w
        stats[ht]["conceded"] += as_ * w
        stats[ht]["weight"]   += w
        stats[ht]["matches"]  += 1

        stats[at]["scored"]   += as_ * w
        stats[at]["conceded"] += hs  * w
        stats[at]["weight"]   += w
        stats[at]["matches"]  += 1

    # Filter by min matches
    qualified = {t: s for t, s in stats.items()
                 if s["matches"] >= min_matches}
    skipped   = len(stats) - len(qualified)
    print(f"\n[params] Teams qualified (>={min_matches} matches): "
          f"{len(qualified)}  (skipped {skipped})")

    # Direct MLE — weighted avg goals / global avg
    attack  = {}
    defense = {}
    for t, s in qualified.items():
        avg_scored   = s["scored"]   / s["weight"]
        avg_conceded = s["conceded"] / s["weight"]
        attack[t]    = avg_scored   / global_avg
        defense[t]   = avg_conceded / global_avg

    return {
        "attack":          attack,
        "defense":         defense,
        "global_avg":      global_avg,
        "global_avg_home": global_avg_home,
        "global_avg_away": global_avg_away,
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_params(team_params: dict, out_path: str, db_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # Write to DB
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM poisson_params")
    rows = [
        (team,
         team_params["attack"].get(team, DEFAULT_ATT),
         team_params["defense"].get(team, DEFAULT_DEF),
         now)
        for team in set(team_params["attack"]) | set(team_params["defense"])
    ]
    conn.executemany("""
        INSERT INTO poisson_params (team, attack_strength, defense_weakness, updated_at)
        VALUES (?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

    payload = {
        "attack":          team_params["attack"],
        "defense":         team_params["defense"],
        "global_avg":      team_params["global_avg"],
        "global_avg_home": team_params["global_avg_home"],
        "global_avg_away": team_params["global_avg_away"],
        "rho":             -0.13,
        "home_advantage":  HOME_ADV,
        "max_goals":       MAX_GOALS,
        "decay_years":     DECAY_YEARS,
        "updated_at":      now,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\n[save] Params → {path.resolve()}")
    print(f"       DB updated ({len(rows)} teams)")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def print_summary(team_params: dict) -> None:
    attack  = team_params["attack"]
    defense = team_params["defense"]

    print(f"\n{'='*55}")
    print("POISSON PARAMS SUMMARY")
    print(f"{'='*55}")
    print(f"  Teams fitted: {len(attack)}")

    top_att = sorted(attack.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\n  Top 10 attack strength:")
    for team, val in top_att:
        print(f"    {team:30s}: {val:.3f}")

    top_def = sorted(defense.items(), key=lambda x: x[1])[:10]
    print(f"\n  Top 10 defense (lowest = hardest to score against):")
    for team, val in top_def:
        print(f"    {team:30s}: {val:.3f}")

    wc_teams = ["France", "Brazil", "Argentina", "England",
                "Spain", "Germany", "Morocco", "United States",
                "Portugal", "Netherlands", "Japan", "Senegal"]
    print(f"\n  WC 2026 key teams:")
    print(f"  {'Team':25s} {'Attack':>8s} {'Defense':>9s}  {'xG scored':>10s}")
    print(f"  {'-'*57}")
    g = team_params["global_avg"]
    for t in wc_teams:
        att  = attack.get(t, DEFAULT_ATT)
        def_ = defense.get(t, DEFAULT_DEF)
        xg   = att * g   # expected goals vs average opponent
        marker = " ← missing" if t not in attack else ""
        print(f"  {t:25s} {att:8.3f} {def_:9.3f}  {xg:10.3f}{marker}")

    print(f"{'='*55}")
    print("\n[done] Next: python -m score_model.predict_score --home France --away Brazil")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",          default="fifa.db")
    parser.add_argument("--out",         default="score_model/poisson_params.pkl")
    parser.add_argument("--min-year",    default=2010, type=int)
    parser.add_argument("--min-matches", default=20,   type=int)
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise FileNotFoundError(f"DB not found: {args.db}")

    print(f"[init] DB: {Path(args.db).resolve()}\n")

    df          = load_matches(args.db, min_year=args.min_year)
    team_params = compute_team_params(df, min_matches=args.min_matches)
    print_summary(team_params)
    save_params(team_params, args.out, args.db)


if __name__ == "__main__":
    main()