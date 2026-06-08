import bisect
import pandas as pd
from ..core.database  import get_conn
from ..core.model_loader import get_poisson

GROUPS = {
    "A": ["Mexico","South Africa","South Korea","Czechia"],
    "B": ["Canada","Switzerland","Qatar","Bosnia and Herzegovina"],
    "C": ["Brazil","Morocco","Haiti","Scotland"],
    "D": ["United States","Paraguay","Australia","Turkey"],
    "E": ["Germany","Curaçao","Ivory Coast","Ecuador"],
    "F": ["Netherlands","Japan","Tunisia","Sweden"],
    "G": ["Belgium","Egypt","Iran","New Zealand"],
    "H": ["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    "I": ["France","Senegal","Norway","Iraq"],
    "J": ["Argentina","Algeria","Austria","Jordan"],
    "K": ["Portugal","Uzbekistan","Colombia","DR Congo"],
    "L": ["England","Croatia","Ghana","Panama"],
}
DB_NAMES = {
    "Czechia":"Czech Republic","Bosnia and Herzegovina":"Bosnia-Herzegovina",
    "Turkey":"Turkey","Iran":"Iran","Ivory Coast":"Ivory Coast",
    "DR Congo":"DR Congo","Cape Verde":"Cape Verde",
    "United States":"United States","South Korea":"South Korea",
}
def db_name(t): return DB_NAMES.get(t, t)

TEAM_TO_GROUP = {t: g for g, ts in GROUPS.items() for t in ts}

def get_all_teams() -> list:
    conn    = get_conn()
    poisson = get_poisson()
    ref     = pd.Timestamp("2026-04-01")

    # Build rank lookup
    rows = conn.execute("""
        SELECT team_name, rank_date, rank, confederation
        FROM rankings ORDER BY team_name, rank_date
    """).fetchall()
    rank_lookup = {}
    for row in rows:
        t = row["team_name"]
        if t not in rank_lookup:
            rank_lookup[t] = ([], [])
        rank_lookup[t][0].append(pd.Timestamp(row["rank_date"]))
        rank_lookup[t][1].append(row["rank"])
    conn.close()

    def get_rank(team):
        dbt = db_name(team)
        if dbt not in rank_lookup: return None
        dates, ranks = rank_lookup[dbt]
        idx = bisect.bisect_right(dates, ref) - 1
        return int(ranks[idx]) if idx >= 0 else None

    teams_out = []
    for grp, teams in GROUPS.items():
        for team in teams:
            dbt  = db_name(team)
            rank = get_rank(team)
            att  = poisson["attack"].get(dbt,  1.0)
            defn = poisson["defense"].get(dbt, 1.0)
            teams_out.append({
                "team":             team,
                "group":            grp,
                "fifa_rank":        rank,
                "attack_strength":  round(att,  3),
                "defense_weakness": round(defn, 3),
            })

    teams_out.sort(key=lambda x: (x["fifa_rank"] or 999))
    return teams_out

def get_team_detail(team_name: str) -> dict:
    conn    = get_conn()
    poisson = get_poisson()
    dbt     = db_name(team_name)

    # Recent matches
    recent = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score,
               tournament, result
        FROM matches
        WHERE (home_team=? OR away_team=?)
          AND home_score IS NOT NULL
        ORDER BY date DESC LIMIT 10
    """, (dbt, dbt)).fetchall()

    # WC 2026 fixtures
    wc_fixtures = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score, result
        FROM matches
        WHERE date >= '2026-06-11'
          AND tournament = 'FIFA World Cup'
          AND (home_team=? OR away_team=?)
        ORDER BY date ASC
    """, (dbt, dbt)).fetchall()

    conn.close()

    return {
        "team":             team_name,
        "group":            TEAM_TO_GROUP.get(team_name),
        "attack_strength":  round(poisson["attack"].get(dbt,  1.0), 3),
        "defense_weakness": round(poisson["defense"].get(dbt, 1.0), 3),
        "recent_matches":   [dict(r) for r in recent],
        "wc_fixtures":      [dict(r) for r in wc_fixtures],
    }

def get_h2h(team1: str, team2: str) -> dict:
    conn = get_conn()
    db1, db2 = db_name(team1), db_name(team2)

    rows = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score,
               result, tournament
        FROM matches
        WHERE ((home_team=? AND away_team=?) OR (home_team=? AND away_team=?))
          AND home_score IS NOT NULL
        ORDER BY date DESC
    """, (db1, db2, db2, db1)).fetchall()
    conn.close()

    t1_wins = t2_wins = draws = 0
    for r in rows:
        if r["home_team"] == db1:
            if r["result"] == "win":   t1_wins += 1
            elif r["result"] == "draw": draws   += 1
            else:                       t2_wins += 1
        else:
            if r["result"] == "loss":  t1_wins += 1
            elif r["result"] == "draw": draws   += 1
            else:                       t2_wins += 1

    total = len(rows)
    return {
        "team1": team1, "team2": team2,
        "total_matches": total,
        "team1_wins":    t1_wins,
        "team2_wins":    t2_wins,
        "draws":         draws,
        "team1_winrate": round(t1_wins / total, 3) if total else 0,
        "matches":       [dict(r) for r in rows[:10]],
    }