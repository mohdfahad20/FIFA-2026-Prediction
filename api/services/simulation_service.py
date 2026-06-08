import json
from ..core.database import get_conn

def get_latest_simulation() -> dict:
    conn = get_conn()
    row  = conn.execute("""
        SELECT run_id, run_at, n_simulations,
               matches_played, matches_remaining, results_json
        FROM simulation_results
        ORDER BY run_at DESC LIMIT 1
    """).fetchone()
    conn.close()

    if not row:
        return {"error": "No simulation results yet."}

    results = json.loads(row["results_json"])
    return {
        "run_id":             row["run_id"],
        "run_at":             row["run_at"],
        "n_simulations":      row["n_simulations"],
        "matches_played":     row["matches_played"],
        "matches_remaining":  row["matches_remaining"],
        "results":            results,
    }

def get_simulation_history() -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT run_id, run_at, n_simulations, matches_played, results_json
        FROM simulation_results
        ORDER BY run_at ASC
    """).fetchall()
    conn.close()

    history = []
    for row in rows:
        results = json.loads(row["results_json"])
        # Return top 10 teams per run for trend chart
        top10 = dict(list(results.items())[:10])
        history.append({
            "run_id":         row["run_id"],
            "run_at":         row["run_at"],
            "matches_played": row["matches_played"],
            "top10":          top10,
        })
    return history