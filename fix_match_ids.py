"""
fix_match_ids.py
================
One-time script to fix malformed match_ids in both the matches and
predictions tables caused by inconsistent space→underscore normalization
and team name mapping differences.

Run ONCE locally, then re-upload to GitHub Release.

Issues fixed:
1. match_ids with spaces instead of underscores (e.g. 'Saudi Arabia' → 'Saudi_Arabia')
2. 'Congo DR' → 'DR_Congo' normalization
3. Removes duplicate matches caused by date-shifting (keeps the scored one)
"""

import sqlite3

conn = sqlite3.connect("fifa.db")
conn.row_factory = sqlite3.Row

def make_match_id(date, home, away):
    return f"{date}_{home}_{away}".replace(" ", "_")

print("=" * 60)
print("Step 1: Fix spaces in match_ids in matches table")
print("=" * 60)

rows = conn.execute("""
    SELECT match_id, date, home_team, away_team, home_score, away_score,
           result, tournament, city, country, is_neutral, stage
    FROM matches
    WHERE match_id LIKE '%2026-06%'
""").fetchall()

fixed = 0
for r in rows:
    correct_id = make_match_id(r["date"], r["home_team"], r["away_team"])
    if r["match_id"] != correct_id:
        print(f"  FIX: '{r['match_id']}' → '{correct_id}'")
        # Check if correct_id already exists
        existing = conn.execute(
            "SELECT match_id, home_score FROM matches WHERE match_id=?", (correct_id,)
        ).fetchone()
        if existing:
            # If existing has a score and current doesn't, delete current (bad one)
            if existing["home_score"] is not None and r["home_score"] is None:
                conn.execute("DELETE FROM matches WHERE match_id=?", (r["match_id"],))
                print(f"    → Deleted duplicate (existing has score)")
            elif existing["home_score"] is None and r["home_score"] is not None:
                # Current has score, existing doesn't — update existing, delete current
                conn.execute("UPDATE matches SET home_score=?, away_score=?, result=? WHERE match_id=?",
                             (r["home_score"], r["away_score"], r["result"], correct_id))
                conn.execute("DELETE FROM matches WHERE match_id=?", (r["match_id"],))
                print(f"    → Merged score into existing, deleted bad row")
            else:
                # Both have same score or neither — just delete the bad one
                conn.execute("DELETE FROM matches WHERE match_id=?", (r["match_id"],))
                print(f"    → Deleted duplicate")
        else:
            conn.execute("UPDATE matches SET match_id=? WHERE match_id=?",
                         (correct_id, r["match_id"]))
            print(f"    → Renamed")
        fixed += 1

conn.commit()
print(f"Fixed {fixed} match_ids in matches table.\n")

print("=" * 60)
print("Step 2: Fix 'Congo DR' → 'DR Congo' in matches table")
print("=" * 60)

# Fix team names that got stored wrong
wrong_name_fixes = [
    ("Congo DR", "DR Congo"),
    ("Congo_DR", "DR Congo"),  # in match_id
]
for wrong, correct in wrong_name_fixes:
    n = conn.execute("SELECT COUNT(*) FROM matches WHERE home_team=?", (wrong,)).fetchone()[0]
    if n > 0:
        conn.execute("UPDATE matches SET home_team=? WHERE home_team=?", (correct, wrong))
        print(f"  Fixed {n} home_team '{wrong}' → '{correct}'")
    n = conn.execute("SELECT COUNT(*) FROM matches WHERE away_team=?", (wrong,)).fetchone()[0]
    if n > 0:
        conn.execute("UPDATE matches SET away_team=? WHERE away_team=?", (correct, wrong))
        print(f"  Fixed {n} away_team '{wrong}' → '{correct}'")

# Now rebuild match_ids for DR Congo rows
rows = conn.execute("""
    SELECT match_id, date, home_team, away_team FROM matches
    WHERE (home_team='DR Congo' OR away_team='DR Congo')
    AND date >= '2026-06-11'
""").fetchall()
for r in rows:
    correct_id = make_match_id(r["date"], r["home_team"], r["away_team"])
    if r["match_id"] != correct_id:
        existing = conn.execute("SELECT match_id FROM matches WHERE match_id=?", (correct_id,)).fetchone()
        if not existing:
            conn.execute("UPDATE matches SET match_id=? WHERE match_id=?",
                         (correct_id, r["match_id"]))
            print(f"  Renamed: {r['match_id']} → {correct_id}")
        else:
            conn.execute("DELETE FROM matches WHERE match_id=?", (r["match_id"],))
            print(f"  Deleted duplicate: {r['match_id']}")

conn.commit()
print("Done.\n")

print("=" * 60)
print("Step 3: Fix predictions match_ids to align with matches table")
print("=" * 60)

pred_rows = conn.execute("""
    SELECT match_id, date, home_team, away_team
    FROM predictions WHERE date >= '2026-06-11'
""").fetchall()

for r in pred_rows:
    correct_id = make_match_id(r["date"], r["home_team"], r["away_team"])
    if r["match_id"] != correct_id:
        print(f"  FIX predictions: '{r['match_id']}' → '{correct_id}'")
        existing = conn.execute("SELECT match_id FROM predictions WHERE match_id=?", (correct_id,)).fetchone()
        if not existing:
            conn.execute("UPDATE predictions SET match_id=? WHERE match_id=?",
                         (correct_id, r["match_id"]))
        else:
            conn.execute("DELETE FROM predictions WHERE match_id=?", (r["match_id"],))
        print(f"    → Fixed")

conn.commit()
print("Done.\n")

print("=" * 60)
print("Step 4: Run fill_actuals manually to catch up")
print("=" * 60)

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
            actual_home=?, actual_away=?, actual_result=?, was_correct=?
        WHERE match_id=?
    """, (home_score, away_score, actual_side, was_correct, match_id))
    updated += 1
    print(f"  Filled: {match_id}  pred={pred_winner}  actual={actual_side}  {'✅' if was_correct else '❌'}")

conn.commit()
print(f"\nFilled actuals for {updated} matches.")

print("\n" + "=" * 60)
print("FINAL STATE — predictions with actuals:")
print("=" * 60)
final = conn.execute("""
    SELECT match_id, pred_winner, actual_result, was_correct
    FROM predictions
    WHERE actual_result IS NOT NULL
    ORDER BY date
""").fetchall()
for r in final:
    icon = "✅" if r["was_correct"] else "❌"
    print(f"  {icon} {r['match_id']}  pred={r['pred_winner']}  actual={r['actual_result']}")

conn.close()
print("\nDone. Run: python run_pipeline.py --skip-data --skip-ml to refresh artifacts.")