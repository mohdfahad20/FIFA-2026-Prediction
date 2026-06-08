from ..core.database import get_conn

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
    "Czechia":                "Czech Republic",
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Turkey":                 "Turkey",
    "Iran":                   "Iran",
    "Ivory Coast":            "Ivory Coast",
    "DR Congo":               "DR Congo",
    "Cape Verde":             "Cape Verde",
    "United States":          "United States",
    "South Korea":            "South Korea",
}

def db_name(team): return DB_NAMES.get(team, team)

def get_standings() -> dict:
    conn = get_conn()
    groups_out = {}

    for grp, teams in GROUPS.items():
        standings = []
        for team in teams:
            dbt = db_name(team)
            rows = conn.execute("""
                SELECT home_score, away_score, result
                FROM matches
                WHERE date >= '2026-06-11'
                  AND tournament = 'FIFA World Cup'
                  AND (home_team = ? OR away_team = ?)
                  AND home_score IS NOT NULL
            """, (dbt, dbt)).fetchall()

            pts, gd, gf, ga, played, w, d, l = 0, 0, 0, 0, 0, 0, 0, 0
            for row in rows:
                played += 1
                is_home = conn.execute(
                    "SELECT home_team FROM matches WHERE home_team=? AND date>='2026-06-11'",
                    (dbt,)
                ).fetchone() is not None

                hs, as_ = row["home_score"], row["away_score"]
                if row["result"] == "win":
                    pts += 3; w += 1
                    scored, conceded = (hs, as_) if is_home else (as_, hs)
                elif row["result"] == "draw":
                    pts += 1; d += 1
                    scored, conceded = hs, as_
                else:
                    l += 1
                    scored, conceded = (hs, as_) if is_home else (as_, hs)
                gf += scored; ga += conceded; gd += scored - conceded

            standings.append({
                "team":   team,
                "played": played,
                "w": w, "d": d, "l": l,
                "pts": pts, "gf": gf, "ga": ga, "gd": gd,
            })

        standings.sort(key=lambda x: (x["pts"], x["gd"], x["gf"]), reverse=True)
        groups_out[grp] = standings

    # Recent WC results
    recent = conn.execute("""
        SELECT date, home_team, away_team, home_score, away_score
        FROM matches
        WHERE date >= '2026-06-11'
          AND tournament = 'FIFA World Cup'
          AND home_score IS NOT NULL
        ORDER BY date DESC LIMIT 20
    """).fetchall()

    conn.close()
    return {
        "groups":         groups_out,
        "recent_results": [dict(r) for r in recent],
    }

def get_group(group_letter: str) -> dict:
    all_standings = get_standings()
    grp = group_letter.upper()
    if grp not in all_standings["groups"]:
        return {"error": f"Group {grp} not found."}
    return {
        "group":    grp,
        "teams":    GROUPS.get(grp, []),
        "standings": all_standings["groups"][grp],
    }