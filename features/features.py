"""
features/features.py
====================
Vectorised feature builder — runs in seconds not minutes.

Strategy:
  - Form / goals scored / goals conceded → pandas groupby + rolling (vectorised)
  - H2H → precomputed lookup dict per team pair (one pass)
  - Rankings → bisect lookup (same as before)
  - Everything joined back to matches at the end

Usage:
    python features/features.py --db fifa.db
"""

import sqlite3
import argparse
import bisect
import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Tournament weight
# ---------------------------------------------------------------------------
TOURNAMENT_WEIGHTS = {
    "FIFA World Cup":                       3,
    "UEFA Euro":                            3,
    "Copa América":                         3,
    "AFC Asian Cup":                        3,
    "Africa Cup of Nations":                3,
    "Gold Cup":                             3,
    "OFC Nations Cup":                      3,
    "FIFA World Cup qualification":         2,
    "UEFA Euro qualification":              2,
    "Copa América qualification":           2,
    "AFC Asian Cup qualification":          2,
    "Africa Cup of Nations qualification":  2,
    "African Cup of Nations qualification": 2,
    "CONCACAF Nations League":              2,
    "UEFA Nations League":                  2,
}

def get_weight(tournament: str) -> int:
    t = tournament.lower()
    # Check qualifiers BEFORE their parent tournament
    if "qualif" in t:
        return 2
    for key, w in TOURNAMENT_WEIGHTS.items():
        if key.lower() in t:
            return w
    return 1


# ---------------------------------------------------------------------------
# Step 1 — load raw data
# ---------------------------------------------------------------------------
def load_data(conn: sqlite3.Connection, start_year: int):
    print("[1/6] Loading matches and rankings from DB...")

    all_matches = pd.read_sql("""
        SELECT match_id, date, home_team, away_team,
               home_score, away_score, result,
               tournament, is_neutral
        FROM matches
        WHERE home_score IS NOT NULL
          AND away_score IS NOT NULL
        ORDER BY date ASC
    """, conn, parse_dates=["date"])

    rankings = pd.read_sql("""
        SELECT team_name, rank_date, rank, confederation
        FROM rankings
        ORDER BY team_name, rank_date
    """, conn, parse_dates=["rank_date"])

    modern = all_matches[all_matches["date"].dt.year >= start_year].copy()
    print(f"   All matches  : {len(all_matches):,}")
    print(f"   Modern (>={start_year}): {len(modern):,}")
    print(f"   Rankings rows: {len(rankings):,}")
    return all_matches, modern, rankings


# ---------------------------------------------------------------------------
# Step 2 — vectorised rolling stats per team
# ---------------------------------------------------------------------------
def build_rolling_stats(all_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (date, team) with columns:
        form, goals_scored_avg, goals_conceded_avg
    Uses shift(1) so stats are computed from BEFORE the match (no leakage).
    """
    print("[2/6] Computing rolling stats (vectorised)...")

    # Expand each match into two rows — one per team
    home = all_matches[["date", "home_team", "home_score", "away_score", "result"]].copy()
    home.columns = ["date", "team", "scored", "conceded", "result"]
    home["win"] = (home["result"] == "win").astype(float)

    away = all_matches[["date", "away_team", "away_score", "home_score", "result"]].copy()
    away.columns = ["date", "team", "scored", "conceded", "result"]
    away["win"] = (away["result"] == "loss").astype(float)  # away team won

    long = pd.concat([home, away], ignore_index=True)
    long = long.sort_values(["team", "date"]).reset_index(drop=True)

    # Rolling per team — shift(1) ensures we only use PAST matches
    grp = long.groupby("team")
    long["form"]               = grp["win"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    long["goals_scored_avg"]   = grp["scored"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    long["goals_conceded_avg"] = grp["conceded"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )

    # Fill NaN (first match ever for a team) with neutral priors
    long["form"]               = long["form"].fillna(0.5)
    long["goals_scored_avg"]   = long["goals_scored_avg"].fillna(1.0)
    long["goals_conceded_avg"] = long["goals_conceded_avg"].fillna(1.0)

    stats = long[["date", "team", "form", "goals_scored_avg", "goals_conceded_avg"]]
    print(f"   Rolling stats computed for {long['team'].nunique()} teams.")
    return stats


# ---------------------------------------------------------------------------
# Step 3 — H2H lookup (precomputed per team pair)
# ---------------------------------------------------------------------------
def build_h2h_lookup(all_matches: pd.DataFrame) -> dict:
    """
    Precomputes H2H history for every team pair into a dict:
        h2h[(teamA, teamB)] = list of (date, goal_diff_from_A_pov, a_won)
    sorted by date ascending so we can bisect by match_date.
    """
    print("[3/6] Building H2H lookup...")

    h2h = {}

    for _, row in all_matches.iterrows():
        ht, at = row["home_team"], row["away_team"]
        hs, as_ = row["home_score"], row["away_score"]
        date = row["date"]

        if pd.isna(hs) or pd.isna(as_):
            continue

        gd = hs - as_
        home_won = 1 if row["result"] == "win" else 0

        # Store from home team's POV under key (home, away)
        key = (ht, at)
        if key not in h2h:
            h2h[key] = []
        h2h[key].append((date, gd, home_won))

        # Store from away team's POV under key (away, home)
        key2 = (at, ht)
        if key2 not in h2h:
            h2h[key2] = []
        h2h[key2].append((date, -gd, 1 - home_won))

    # Sort each list by date
    for key in h2h:
        h2h[key].sort(key=lambda x: x[0])

    print(f"   H2H pairs computed: {len(h2h):,}")
    return h2h


def get_h2h_stats(home_team: str, away_team: str,
                  match_date: pd.Timestamp, h2h: dict):
    """Return (winrate, avg_goal_diff) for home_team vs away_team before match_date."""
    key = (home_team, away_team)
    if key not in h2h:
        return 0.5, 0.0

    records = [(d, gd, w) for d, gd, w in h2h[key] if d < match_date]
    if not records:
        return 0.5, 0.0

    winrate   = float(np.mean([w for _, _, w in records]))
    goal_diff = float(np.mean([gd for _, gd, _ in records]))
    return winrate, goal_diff


# ---------------------------------------------------------------------------
# Step 4 — rankings bisect lookup
# ---------------------------------------------------------------------------
def build_rank_lookup(rankings: pd.DataFrame):
    print("[4/6] Building rank lookup...")

    rank_lookup = {}
    conf_lookup = {}

    for team, grp in rankings.groupby("team_name"):
        grp_s = grp.sort_values("rank_date")
        rank_lookup[team] = (
            list(grp_s["rank_date"]),
            list(grp_s["rank"])
        )
        conf_lookup[team] = grp_s["confederation"].iloc[-1]

    print(f"   Teams in rank lookup: {len(rank_lookup):,}")
    return rank_lookup, conf_lookup


def get_rank(team: str, match_date: pd.Timestamp, rank_lookup: dict):
    if team not in rank_lookup:
        return None
    dates_list, ranks_list = rank_lookup[team]
    idx = bisect.bisect_right(dates_list, match_date) - 1
    return ranks_list[idx] if idx >= 0 else None


# ---------------------------------------------------------------------------
# Step 5 — assemble features table
# ---------------------------------------------------------------------------
def assemble_features(modern: pd.DataFrame, all_matches: pd.DataFrame,
                      rolling_stats: pd.DataFrame, h2h: dict,
                      rank_lookup: dict, conf_lookup: dict) -> pd.DataFrame:
    print("[5/6] Assembling features table...")

    # Join rolling stats for home team
    home_stats = rolling_stats.rename(columns={
        "team":               "home_team",
        "form":               "home_form",
        "goals_scored_avg":   "home_goals_scored_avg",
        "goals_conceded_avg": "home_goals_conceded_avg",
    })
    away_stats = rolling_stats.rename(columns={
        "team":               "away_team",
        "form":               "away_form",
        "goals_scored_avg":   "away_goals_scored_avg",
        "goals_conceded_avg": "away_goals_conceded_avg",
    })

    df = modern.copy()
    df = df.merge(
        home_stats[["date", "home_team", "home_form",
                    "home_goals_scored_avg", "home_goals_conceded_avg"]],
        on=["date", "home_team"], how="left"
    )
    df = df.merge(
        away_stats[["date", "away_team", "away_form",
                    "away_goals_scored_avg", "away_goals_conceded_avg"]],
        on=["date", "away_team"], how="left"
    )

    # Rankings + confederation (needs bisect — vectorised via apply on small col)
    df["home_rank"] = df.apply(
        lambda r: get_rank(r["home_team"], r["date"], rank_lookup), axis=1
    )
    df["away_rank"] = df.apply(
        lambda r: get_rank(r["away_team"], r["date"], rank_lookup), axis=1
    )
    df["home_confederation"] = df["home_team"].map(conf_lookup)
    df["away_confederation"] = df["away_team"].map(conf_lookup)

    # Drop rows where either team has no ranking
    before = len(df)
    df = df.dropna(subset=["home_rank", "away_rank"])
    after  = len(df)
    print(f"   Dropped {before - after:,} rows with missing rankings.")

    # Rank diff
    df["rank_diff"] = df["home_rank"] - df["away_rank"]

    # H2H (still needs per-row lookup — but h2h dict is precomputed so it's fast)
    print("   Computing H2H stats...")
    h2h_results = df.apply(
        lambda r: get_h2h_stats(r["home_team"], r["away_team"], r["date"], h2h),
        axis=1
    )
    df["h2h_winrate_home"] = h2h_results.apply(lambda x: x[0])
    df["h2h_goal_diff"]    = h2h_results.apply(lambda x: x[1])

    # Tournament weight
    df["tournament_weight"] = df["tournament"].apply(get_weight)

    # Target
    df["target"] = df["result"].map({"win": 2, "draw": 1, "loss": 0})

    # Fill any remaining NaN stats with priors
    df["home_form"]               = df["home_form"].fillna(0.5)
    df["away_form"]               = df["away_form"].fillna(0.5)
    df["home_goals_scored_avg"]   = df["home_goals_scored_avg"].fillna(1.0)
    df["away_goals_scored_avg"]   = df["away_goals_scored_avg"].fillna(1.0)
    df["home_goals_conceded_avg"] = df["home_goals_conceded_avg"].fillna(1.0)
    df["away_goals_conceded_avg"] = df["away_goals_conceded_avg"].fillna(1.0)

    print(f"   Final feature rows: {len(df):,}")
    return df


# ---------------------------------------------------------------------------
# Step 6 — write to DB + summary
# ---------------------------------------------------------------------------
def write_to_db(df: pd.DataFrame, conn: sqlite3.Connection) -> None:
    print("[6/6] Writing to features table...")

    conn.execute("DELETE FROM features")
    conn.commit()

    cols = [
        "match_id", "date", "home_team", "away_team",
        "home_rank", "away_rank", "rank_diff",
        "home_form", "away_form",
        "home_goals_scored_avg", "away_goals_scored_avg",
        "home_goals_conceded_avg", "away_goals_conceded_avg",
        "h2h_winrate_home", "h2h_goal_diff",
        "is_neutral", "tournament_weight",
        "home_confederation", "away_confederation",
        "target"
    ]

    out = df[cols].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")

    out.drop_duplicates(subset=["match_id"]).to_sql(
    "features", conn, if_exists="append", index=False, chunksize=500
    )
    conn.commit()

    total = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    print(f"\n{'='*55}")
    print("FEATURES TABLE SUMMARY")
    print(f"{'='*55}")
    print(f"  Total rows : {total:,}")

    print("\n  Target distribution:")
    for target, label in [(2, "win"), (1, "draw"), (0, "loss")]:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM features WHERE target=?", (target,)
        ).fetchone()[0]
        print(f"    {label:5s} (target={target}) : {cnt:,}  ({cnt/total:.1%})")

    lo, hi = conn.execute(
        "SELECT MIN(date), MAX(date) FROM features"
    ).fetchone()
    print(f"\n  Date range : {lo}  →  {hi}")

    print("\n  Tournament weight distribution:")
    for w, label in [(3, "Major"), (2, "Qualifier"), (1, "Friendly")]:
        cnt = conn.execute(
            "SELECT COUNT(*) FROM features WHERE tournament_weight=?", (w,)
        ).fetchone()[0]
        print(f"    weight={w} ({label:10s}) : {cnt:,}  ({cnt/total:.1%})")

    print("\n  Feature ranges (sanity check):")
    for col in ["rank_diff", "home_form", "away_form",
                "home_goals_scored_avg", "h2h_winrate_home"]:
        mn, mx, avg = conn.execute(
            f"SELECT MIN({col}), MAX({col}), AVG({col}) FROM features"
        ).fetchone()
        print(f"    {col:28s}: min={mn:6.2f}  max={mx:6.2f}  avg={avg:6.2f}")

    print(f"{'='*55}")
    print("\n[done] Next: python model/train.py --db fifa.db")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",         default="fifa.db")
    parser.add_argument("--start-year", default=2000, type=int)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}. Run load_historical_data.py first.")

    print(f"[init] Opening: {db_path.resolve()}\n")
    conn = sqlite3.connect(db_path)

    all_matches, modern, rankings = load_data(conn, args.start_year)
    rolling_stats                 = build_rolling_stats(all_matches)
    h2h                           = build_h2h_lookup(all_matches)
    rank_lookup, conf_lookup      = build_rank_lookup(rankings)
    df                            = assemble_features(
                                        modern, all_matches, rolling_stats,
                                        h2h, rank_lookup, conf_lookup
                                    )
    write_to_db(df, conn)
    conn.close()


if __name__ == "__main__":
    main()