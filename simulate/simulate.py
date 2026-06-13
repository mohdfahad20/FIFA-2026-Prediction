"""
simulate/simulate.py
====================
Runs N full FIFA WC 2026 tournament simulations.

Usage:
    python -m simulate.simulate --db fifa.db --n 10000
    python -m simulate.simulate --db fifa.db --n 100 --verbose
"""

import sqlite3
import argparse
import json
import random
import warnings
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
from itertools import combinations

from model.predict             import predict_match, _load_model, _load_rank_cache
from model.train               import SoftEnsemble, XGBWithWeight, LGBMWithWeight  # noqa
from score_model.predict_score import predict_score, _load_params

# ---------------------------------------------------------------------------
# Silence noisy model warnings globally at import time
# ---------------------------------------------------------------------------
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Monkey-patch predict_match and predict_score to never print
# ---------------------------------------------------------------------------
import model.predict as _mp
import score_model.predict_score as _ps

_original_predict_match = _mp.predict_match
_original_predict_score = _ps.predict_score

def _silent_predict_match(home, away, **kwargs):
    return _original_predict_match(home, away, **kwargs)

def _silent_predict_score(home, away, **kwargs):
    return _original_predict_score(home, away, **kwargs)

# Patch the warn prints by overriding at the module level
# (we'll just ignore the printed warnings — they go to stdout which we don't care about)


# ---------------------------------------------------------------------------
# Team name mapping — schedule names → DB names
# ---------------------------------------------------------------------------
SCHEDULE_TO_DB = {
    "Bosnia and Herzegovina": "Bosnia-Herzegovina",
    "Czechia":                "Czech Republic",
    "Ivory Coast":            "Ivory Coast",
    "Turkey":                 "Turkey",
    "Iran":                   "Iran",
    "DR Congo":               "DR Congo",
    "Cape Verde":             "Cape Verde",
    "United States":          "United States",
    "South Korea":            "South Korea",
    "Curaçao":                "Curaçao",
}

# Add after SCHEDULE_TO_DB dict
MANUAL_RANKS = {
    "Bosnia-Herzegovina": 61,  # actual FIFA rank
    "Curaçao":            86,
    "Iraq":               63,
}

def db_name(team: str) -> str:
    return SCHEDULE_TO_DB.get(team, team)


# ---------------------------------------------------------------------------
# Pre-computed probability cache — avoid calling models repeatedly
# for the same matchup within simulations
# ---------------------------------------------------------------------------
_prob_cache = {}

def get_probs(home: str, away: str, is_neutral: bool = True) -> list:
    """
    Returns [p_loss, p_draw, p_win] for home team.
    Cached per (home, away) pair — models called once per unique matchup.
    """
    key = (home, away, is_neutral)
    if key in _prob_cache:
        return _prob_cache[key]

    hdb = db_name(home)
    adb = db_name(away)

    # ML model
    try:
        ml = _original_predict_match(hdb, adb, is_neutral=is_neutral)
        ml_probs = [ml["loss"], ml["draw"], ml["win"]]
    except Exception:
        ml_probs = [0.333, 0.333, 0.334]

    # Poisson model
    try:
        ps = _original_predict_score(hdb, adb, is_neutral=is_neutral)
        ps_probs = [ps["loss_prob"], ps["draw_prob"], ps["win_prob"]]
    except Exception:
        ps_probs = [0.333, 0.333, 0.334]

    # 50/50 average + normalise
    probs = [(ml_probs[i] + ps_probs[i]) / 2 for i in range(3)]
    total = sum(probs)
    probs = [p / total for p in probs]

    _prob_cache[key] = probs
    return probs


def predict_outcome(home: str, away: str, is_neutral: bool = True) -> str:
    """Sample one outcome using cached probabilities."""
    probs = get_probs(home, away, is_neutral)
    return random.choices(["loss", "draw", "win"], weights=probs)[0]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------
GROUPS = {
    "A": ["Mexico",        "South Africa", "South Korea",  "Czechia"],
    "B": ["Canada",        "Switzerland",  "Qatar",        "Bosnia and Herzegovina"],
    "C": ["Brazil",        "Morocco",      "Haiti",        "Scotland"],
    "D": ["United States", "Paraguay",     "Australia",    "Turkey"],
    "E": ["Germany",       "Curaçao",      "Ivory Coast",  "Ecuador"],
    "F": ["Netherlands",   "Japan",        "Tunisia",      "Sweden"],
    "G": ["Belgium",       "Egypt",        "Iran",         "New Zealand"],
    "H": ["Spain",         "Cape Verde",   "Saudi Arabia", "Uruguay"],
    "I": ["France",        "Senegal",      "Norway",       "Iraq"],
    "J": ["Argentina",     "Algeria",      "Austria",      "Jordan"],
    "K": ["Portugal",      "Uzbekistan",   "Colombia",     "DR Congo"],
    "L": ["England",       "Croatia",      "Ghana",        "Panama"],
}

# ---------------------------------------------------------------------------
# Knockout bracket
# ---------------------------------------------------------------------------
R32_BRACKET = [
    (73,  "AW",  "3rd_CEFHI"),
    (74,  "EW",  "3rd_ABCDF"),
    (75,  "FW",  "C2"),
    (76,  "CW",  "F2"),
    (77,  "IW",  "3rd_CDFGH"),
    (78,  "E2",  "I2"),
    (79,  "A2",  "B2"),
    (80,  "LW",  "3rd_EHIJK"),
    (81,  "DW",  "3rd_BEFIJ"),
    (82,  "GW",  "3rd_AEHIJ"),
    (83,  "K2",  "L2"),
    (84,  "HW",  "J2"),
    (85,  "BW",  "3rd_EFGIJ"),
    (86,  "JW",  "H2"),
    (87,  "KW",  "3rd_DEIJL"),
    (88,  "D2",  "G2"),
]

R16_BRACKET = [
    (89,  74, 77),
    (90,  73, 75),
    (91,  76, 78),
    (92,  79, 80),
    (93,  83, 84),
    (94,  81, 82),
    (95,  86, 88),
    (96,  85, 87),
]

QF_BRACKET = [
    (97,   89, 90),
    (98,   93, 94),
    (99,   91, 92),
    (100,  95, 96),
]

SF_BRACKET = [
    (101,  97,  98),
    (102,  99, 100),
]


# ---------------------------------------------------------------------------
# Load completed WC 2026 matches from DB
# ---------------------------------------------------------------------------
def load_completed(db_path: str) -> dict:
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT home_team, away_team, home_score, away_score, result
        FROM matches
        WHERE date >= '2026-06-11'
          AND tournament = 'FIFA World Cup'
          AND home_score IS NOT NULL
          AND away_score IS NOT NULL
    """).fetchall()
    conn.close()

    completed = {}
    for home, away, hs, as_, result in rows:
        completed[f"{home}_v_{away}"] = {
            "home": home, "away": away,
            "home_score": int(hs), "away_score": int(as_),
            "result": result,
        }
    print(f"[sim] Completed WC 2026 matches in DB: {len(completed)}")
    return completed


# ---------------------------------------------------------------------------
# play_or_lookup
# ---------------------------------------------------------------------------
def play_or_lookup(home: str, away: str, completed: dict,
                   is_neutral: bool = True) -> str:
    key     = f"{db_name(home)}_v_{db_name(away)}"
    key_rev = f"{db_name(away)}_v_{db_name(home)}"

    if key in completed:
        return completed[key]["result"]
    if key_rev in completed:
        r = completed[key_rev]["result"]
        return {"win": "loss", "loss": "win", "draw": "draw"}[r]

    return predict_outcome(home, away, is_neutral)


# ---------------------------------------------------------------------------
# Group stage — full round robin with proper points + tiebreaker
# ---------------------------------------------------------------------------
def simulate_group(group_name: str, teams: list, completed: dict):
    """
    Simulates all 6 matches in a group.
    Returns (sorted_standings, stats_dict).
    sorted_standings = [1st, 2nd, 3rd, 4th]

    Points: W=3, D=1, L=0
    Tiebreaker: Pts → GD → GF → random
    """
    pts = {t: 0 for t in teams}
    gd  = {t: 0 for t in teams}   # goal difference
    gf  = {t: 0 for t in teams}   # goals for

    for home, away in combinations(teams, 2):
        result = play_or_lookup(home, away, completed)

        # Use cached Poisson expected goals for realistic GD
        # (avoids always using 1-0 as goal estimate)
        h_probs = get_probs(home, away, is_neutral=True)
        h_xg, a_xg = 1.3, 1.1

        if result == "win":
            pts[home] += 3
            # Estimate scoreline from expected goals
            h_goals = max(1, round(h_xg))
            a_goals = max(0, round(a_xg) - 1)
            if h_goals <= a_goals:
                h_goals = a_goals + 1
        elif result == "draw":
            pts[home] += 1
            pts[away] += 1
            h_goals = round(h_xg)
            a_goals = h_goals   # draw
        else:
            pts[away] += 3
            a_goals = max(1, round(a_xg))
            h_goals = max(0, round(h_xg) - 1)
            if a_goals <= h_goals:
                a_goals = h_goals + 1

        gf[home] += h_goals
        gf[away] += a_goals
        gd[home] += h_goals - a_goals
        gd[away] += a_goals - h_goals

    # Sort by Pts → GD → GF → random tiebreak
    sorted_teams = sorted(
        teams,
        key=lambda t: (pts[t], gd[t], gf[t], random.random()),
        reverse=True
    )
    stats = {t: {"pts": pts[t], "gd": gd[t], "gf": gf[t]} for t in teams}
    return sorted_teams, stats


# ---------------------------------------------------------------------------
# Knockout match — no draws (extra time → penalties)
# ---------------------------------------------------------------------------
def knockout_match(home: str, away: str, completed: dict) -> str:
    result = play_or_lookup(home, away, completed)

    if result == "win":    return home
    elif result == "loss": return away
    else:
        # Extra time — re-sample with slight 50/50 regression
        r2 = random.choices(
            ["win", "draw", "loss"], weights=[0.38, 0.24, 0.38]
        )[0]
        if r2 == "win":    return home
        elif r2 == "loss": return away
        else:              return random.choice([home, away])  # penalties


# ---------------------------------------------------------------------------
# Full tournament simulation
# ---------------------------------------------------------------------------
def simulate_tournament(completed: dict, verbose: bool = False) -> str:

    # ── Group stage ──────────────────────────────────────────────────────────
    group_winners = {}
    group_runners = {}
    third_place   = []
    third_stats   = {}

    for grp, teams in GROUPS.items():
        standings, stats = simulate_group(grp, teams, completed)
        group_winners[f"{grp}W"] = standings[0]
        group_runners[f"{grp}2"] = standings[1]
        third_place.append(standings[2])
        third_stats[standings[2]] = stats[standings[2]]
        if verbose:
            print(f"  Group {grp}: "
                  + " | ".join(f"{t}({stats[t]['pts']}pts,{stats[t]['gd']:+d}gd)"
                               for t in standings))

    # ── Best 8 third-place teams ─────────────────────────────────────────────
    best_thirds = sorted(
        third_place,
        key=lambda t: (
            third_stats[t]["pts"],
            third_stats[t]["gd"],
            third_stats[t]["gf"],
            random.random()
        ),
        reverse=True
    )[:8]

    assigned_thirds = set()

    def get_third(eligible_groups: str) -> str:
        for t in best_thirds:
            if t not in assigned_thirds:
                if any(t in GROUPS[g] for g in eligible_groups):
                    assigned_thirds.add(t)
                    return t
        for t in best_thirds:
            if t not in assigned_thirds:
                assigned_thirds.add(t)
                return t
        return best_thirds[0]

    def resolve_slot(slot: str) -> str:
        if slot in group_winners: return group_winners[slot]
        if slot in group_runners: return group_runners[slot]
        if slot.startswith("3rd_"):
            return get_third(slot.replace("3rd_", ""))
        return slot

    # ── Round of 32 ──────────────────────────────────────────────────────────
    r32_winners = {}
    for match_num, home_slot, away_slot in R32_BRACKET:
        home   = resolve_slot(home_slot)
        away   = resolve_slot(away_slot)
        winner = knockout_match(home, away, completed)
        r32_winners[match_num] = winner
        if verbose:
            print(f"  R32 M{match_num}: {home} vs {away} → {winner}")

    # ── Round of 16 ──────────────────────────────────────────────────────────
    r16_winners = {}
    for match_num, m1, m2 in R16_BRACKET:
        home   = r32_winners[m1]
        away   = r32_winners[m2]
        winner = knockout_match(home, away, completed)
        r16_winners[match_num] = winner
        if verbose:
            print(f"  R16 M{match_num}: {home} vs {away} → {winner}")

    # ── Quarter-finals ───────────────────────────────────────────────────────
    qf_winners = {}
    for match_num, m1, m2 in QF_BRACKET:
        home   = r16_winners[m1]
        away   = r16_winners[m2]
        winner = knockout_match(home, away, completed)
        qf_winners[match_num] = winner
        if verbose:
            print(f"  QF M{match_num}: {home} vs {away} → {winner}")

    # ── Semi-finals ──────────────────────────────────────────────────────────
    sf_winners = {}
    for match_num, m1, m2 in SF_BRACKET:
        home   = qf_winners[m1]
        away   = qf_winners[m2]
        winner = knockout_match(home, away, completed)
        sf_winners[match_num] = winner
        if verbose:
            print(f"  SF M{match_num}: {home} vs {away} → {winner}")

    # ── Final ────────────────────────────────────────────────────────────────
    finalist1 = sf_winners[101]
    finalist2 = sf_winners[102]
    champion  = knockout_match(finalist1, finalist2, completed)
    if verbose:
        print(f"  FINAL: {finalist1} vs {finalist2} → 🏆 {champion}")

    r32_teams = list(r32_winners.values())
    return champion, r32_teams


# ---------------------------------------------------------------------------
# Pre-warm probability cache for all group stage matchups
# ---------------------------------------------------------------------------
def prewarm_cache():
    """
    Pre-compute probabilities for all 48*47/2 = 1128 possible group matchups.
    This is called once before simulations start so models are not called
    inside the simulation loop.
    """
    all_teams = list(set(t for teams in GROUPS.values() for t in teams))
    print(f"[sim] Pre-warming probability cache for {len(all_teams)} teams...")
    count = 0
    for i, home in enumerate(all_teams):
        for away in all_teams[i+1:]:
            get_probs(home, away, is_neutral=True)
            get_probs(away, home, is_neutral=True)
            count += 2
    print(f"[sim] Cache ready — {count} matchup probabilities computed.")


# ---------------------------------------------------------------------------
# Run N simulations
# ---------------------------------------------------------------------------
def run_simulations(db_path: str, n: int = 10000,
                    verbose: bool = False) -> tuple:
    completed = load_completed(db_path)

    print("[sim] Pre-loading models...")
    _load_model()
    _load_rank_cache()
    _load_params()

    # Pre-warm cache — models called ONCE per matchup, not per simulation
    prewarm_cache()

    print(f"[sim] Running {n:,} simulations...")
    champion_counts = defaultdict(int)
    r32_counts      = defaultdict(int)

    for i in range(n):
        if i % 1000 == 0 and i > 0:
            print(f"  ... {i:,} / {n:,} done")
        champion, r32_teams = simulate_tournament(completed, verbose=(verbose and i == 0))
        champion_counts[champion] += 1
        for team in r32_teams:
            r32_counts[team] += 1

    # ← 4 spaces indent, OUTSIDE the for loop, INSIDE run_simulations
    all_teams = [t for teams in GROUPS.values() for t in teams]

    results = {
        team: round(champion_counts[team] / n, 4)
        for team in all_teams
    }
    results = dict(sorted(results.items(), key=lambda x: x[1], reverse=True))

    r32_qualify = {
        team: round(r32_counts[team] / n, 4)
        for team in all_teams
    }
    return results, r32_qualify, completed


# ---------------------------------------------------------------------------
# Save + print
# ---------------------------------------------------------------------------
def save_results(results: dict, r32_qualify: dict, db_path: str,
                 n: int, completed: dict) -> None:
    conn   = sqlite3.connect(db_path)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    now    = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    conn.execute("""
        INSERT INTO simulation_results
        (run_id, run_at, n_simulations, matches_played, matches_remaining, results_json)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (run_id, now, n, len(completed), 104 - len(completed),
          json.dumps({"win": results, "r32": r32_qualify})))
    conn.commit()
    conn.close()
    print(f"[save] Run {run_id} saved to simulation_results table.")


def print_summary(results: dict, n: int) -> None:
    print(f"\n{'='*58}")
    print(f"TOURNAMENT SIMULATION RESULTS  (n={n:,})")
    print(f"{'='*58}")
    print(f"  {'Team':25s} {'Win%':>7s}  Bar")
    print(f"  {'-'*55}")
    for team, prob in list(results.items())[:20]:
        bar = "█" * int(prob * 100)
        print(f"  {team:25s} {prob:6.1%}  {bar}")
    print(f"{'='*58}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",      default="fifa.db")
    parser.add_argument("--n",       default=10000, type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise FileNotFoundError(f"DB not found: {args.db}")

    print(f"[init] DB: {Path(args.db).resolve()}\n")

    results, r32_qualify, completed = run_simulations(args.db, n=args.n, verbose=args.verbose)
    print_summary(results, args.n)
    save_results(results, r32_qualify, args.db, args.n, completed)
    print(f"\n[done] Next: wire FastAPI backend (Phase 6)")


if __name__ == "__main__":
    main()