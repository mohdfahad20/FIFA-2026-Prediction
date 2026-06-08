"""
Builds wc2026_schedule.csv directly from hardcoded data
fetched from roadtrips.com — complete, confirmed schedule.
"""
import pandas as pd

# Complete group assignments (confirmed after March 31 qualifying)
GROUPS = {
    "A": ["Mexico", "South Africa", "South Korea", "Czechia"],
    "B": ["Canada", "Switzerland", "Qatar", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["United States", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Tunisia", "Sweden"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Norway", "Iraq"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

# All 72 group stage + knockout matches
# Format: (match_num, date, home, away, group, stage)
MATCHES = [
    # ── GROUP STAGE ──────────────────────────────────────────
    (1,  "2026-06-11", "Mexico",                "South Africa",          "A", "group"),
    (2,  "2026-06-11", "South Korea",           "Czechia",               "A", "group"),
    (3,  "2026-06-12", "Canada",                "Bosnia and Herzegovina","B", "group"),
    (4,  "2026-06-12", "United States",         "Paraguay",              "D", "group"),
    (5,  "2026-06-13", "Haiti",                 "Scotland",              "C", "group"),
    (6,  "2026-06-13", "Australia",             "Turkey",                "D", "group"),
    (7,  "2026-06-13", "Brazil",                "Morocco",               "C", "group"),
    (8,  "2026-06-13", "Qatar",                 "Switzerland",           "B", "group"),
    (9,  "2026-06-14", "Ivory Coast",           "Ecuador",               "E", "group"),
    (10, "2026-06-14", "Germany",               "Curaçao",               "E", "group"),
    (11, "2026-06-14", "Netherlands",           "Japan",                 "F", "group"),
    (12, "2026-06-14", "Sweden",                "Tunisia",               "F", "group"),
    (13, "2026-06-15", "Saudi Arabia",          "Uruguay",               "H", "group"),
    (14, "2026-06-15", "Spain",                 "Cape Verde",            "H", "group"),
    (15, "2026-06-15", "Iran",                  "New Zealand",           "G", "group"),
    (16, "2026-06-15", "Belgium",               "Egypt",                 "G", "group"),
    (17, "2026-06-16", "France",                "Senegal",               "I", "group"),
    (18, "2026-06-16", "Iraq",                  "Norway",                "I", "group"),
    (19, "2026-06-16", "Argentina",             "Algeria",               "J", "group"),
    (20, "2026-06-16", "Austria",               "Jordan",                "J", "group"),
    (21, "2026-06-17", "Ghana",                 "Panama",                "L", "group"),
    (22, "2026-06-17", "England",               "Croatia",               "L", "group"),
    (23, "2026-06-17", "Portugal",              "DR Congo",              "K", "group"),
    (24, "2026-06-17", "Uzbekistan",            "Colombia",              "K", "group"),
    (25, "2026-06-18", "Czechia",               "South Africa",          "A", "group"),
    (26, "2026-06-18", "Switzerland",           "Bosnia and Herzegovina","B", "group"),
    (27, "2026-06-18", "Canada",                "Qatar",                 "B", "group"),
    (28, "2026-06-18", "Mexico",                "South Korea",           "A", "group"),
    (29, "2026-06-19", "Brazil",                "Haiti",                 "C", "group"),
    (30, "2026-06-19", "Scotland",              "Morocco",               "C", "group"),
    (31, "2026-06-19", "Turkey",                "Paraguay",              "D", "group"),
    (32, "2026-06-19", "United States",         "Australia",             "D", "group"),
    (33, "2026-06-20", "Germany",               "Ivory Coast",           "E", "group"),
    (34, "2026-06-20", "Ecuador",               "Curaçao",               "E", "group"),
    (35, "2026-06-20", "Netherlands",           "Sweden",                "F", "group"),
    (36, "2026-06-20", "Tunisia",               "Japan",                 "F", "group"),
    (37, "2026-06-21", "Uruguay",               "Cape Verde",            "H", "group"),
    (38, "2026-06-21", "Spain",                 "Saudi Arabia",          "H", "group"),
    (39, "2026-06-21", "Belgium",               "Iran",                  "G", "group"),
    (40, "2026-06-21", "New Zealand",           "Egypt",                 "G", "group"),
    (41, "2026-06-22", "Norway",                "Senegal",               "I", "group"),
    (42, "2026-06-22", "France",                "Iraq",                  "I", "group"),
    (43, "2026-06-22", "Argentina",             "Austria",               "J", "group"),
    (44, "2026-06-22", "Jordan",                "Algeria",               "J", "group"),
    (45, "2026-06-23", "England",               "Ghana",                 "L", "group"),
    (46, "2026-06-23", "Panama",                "Croatia",               "L", "group"),
    (47, "2026-06-23", "Portugal",              "Uzbekistan",            "K", "group"),
    (48, "2026-06-23", "Colombia",              "DR Congo",              "K", "group"),
    (49, "2026-06-24", "Scotland",              "Brazil",                "C", "group"),
    (50, "2026-06-24", "Morocco",               "Haiti",                 "C", "group"),
    (51, "2026-06-24", "Switzerland",           "Canada",                "B", "group"),
    (52, "2026-06-24", "Bosnia and Herzegovina","Qatar",                 "B", "group"),
    (53, "2026-06-24", "Czechia",               "Mexico",                "A", "group"),
    (54, "2026-06-24", "South Africa",          "South Korea",           "A", "group"),
    (55, "2026-06-25", "Curaçao",               "Ivory Coast",           "E", "group"),
    (56, "2026-06-25", "Ecuador",               "Germany",               "E", "group"),
    (57, "2026-06-25", "Japan",                 "Sweden",                "F", "group"),
    (58, "2026-06-25", "Tunisia",               "Netherlands",           "F", "group"),
    (59, "2026-06-25", "Turkey",                "United States",         "D", "group"),
    (60, "2026-06-25", "Paraguay",              "Australia",             "D", "group"),
    (61, "2026-06-26", "Norway",                "France",                "I", "group"),
    (62, "2026-06-26", "Senegal",               "Iraq",                  "I", "group"),
    (63, "2026-06-26", "Egypt",                 "Iran",                  "G", "group"),
    (64, "2026-06-26", "New Zealand",           "Belgium",               "G", "group"),
    (65, "2026-06-26", "Cape Verde",            "Saudi Arabia",          "H", "group"),
    (66, "2026-06-26", "Uruguay",               "Spain",                 "H", "group"),
    (67, "2026-06-27", "Panama",                "England",               "L", "group"),
    (68, "2026-06-27", "Croatia",               "Ghana",                 "L", "group"),
    (69, "2026-06-27", "Algeria",               "Austria",               "J", "group"),
    (70, "2026-06-27", "Jordan",                "Argentina",             "J", "group"),
    (71, "2026-06-27", "Colombia",              "Portugal",              "K", "group"),
    (72, "2026-06-27", "DR Congo",              "Uzbekistan",            "K", "group"),
    # ── ROUND OF 32 ──────────────────────────────────────────
    (73, "2026-06-28", "Group A Runner-Up",     "Group B Runner-Up",     None, "R32"),
    (74, "2026-06-29", "Group E Winner",        "Best 3rd A/B/C/D/F",    None, "R32"),
    (75, "2026-06-29", "Group F Winner",        "Group C Runner-Up",     None, "R32"),
    (76, "2026-06-29", "Group C Winner",        "Group F Runner-Up",     None, "R32"),
    (77, "2026-06-30", "Group I Winner",        "Best 3rd C/D/F/G/H",    None, "R32"),
    (78, "2026-06-30", "Group E Runner-Up",     "Group I Runner-Up",     None, "R32"),
    (79, "2026-06-30", "Group A Winner",        "Best 3rd C/E/F/H/I",    None, "R32"),
    (80, "2026-07-01", "Group L Winner",        "Best 3rd E/H/I/J/K",    None, "R32"),
    (81, "2026-07-01", "Group D Winner",        "Best 3rd B/E/F/I/J",    None, "R32"),
    (82, "2026-07-01", "Group G Winner",        "Best 3rd A/E/H/I/J",    None, "R32"),
    (83, "2026-07-02", "Group K Runner-Up",     "Group L Runner-Up",     None, "R32"),
    (84, "2026-07-02", "Group H Winner",        "Group J Runner-Up",     None, "R32"),
    (85, "2026-07-02", "Group B Winner",        "Best 3rd E/F/G/I/J",    None, "R32"),
    (86, "2026-07-03", "Group J Winner",        "Group H Runner-Up",     None, "R32"),
    (87, "2026-07-03", "Group K Winner",        "Best 3rd D/E/I/J/L",    None, "R32"),
    (88, "2026-07-03", "Group D Runner-Up",     "Group G Runner-Up",     None, "R32"),
    # ── ROUND OF 16 ──────────────────────────────────────────
    (89, "2026-07-04", "Winner M74",            "Winner M77",            None, "R16"),
    (90, "2026-07-04", "Winner M73",            "Winner M75",            None, "R16"),
    (91, "2026-07-05", "Winner M76",            "Winner M78",            None, "R16"),
    (92, "2026-07-05", "Winner M79",            "Winner M80",            None, "R16"),
    (93, "2026-07-06", "Winner M83",            "Winner M84",            None, "R16"),
    (94, "2026-07-06", "Winner M81",            "Winner M82",            None, "R16"),
    (95, "2026-07-07", "Winner M86",            "Winner M88",            None, "R16"),
    (96, "2026-07-07", "Winner M85",            "Winner M87",            None, "R16"),
    # ── QUARTER-FINALS ───────────────────────────────────────
    (97,  "2026-07-09", "Winner M89",           "Winner M90",            None, "QF"),
    (98,  "2026-07-10", "Winner M93",           "Winner M94",            None, "QF"),
    (99,  "2026-07-11", "Winner M91",           "Winner M92",            None, "QF"),
    (100, "2026-07-11", "Winner M95",           "Winner M96",            None, "QF"),
    # ── SEMI-FINALS ──────────────────────────────────────────
    (101, "2026-07-14", "Winner M97",           "Winner M98",            None, "SF"),
    (102, "2026-07-15", "Winner M99",           "Winner M100",           None, "SF"),
    # ── THIRD PLACE ──────────────────────────────────────────
    (103, "2026-07-18", "Loser M101",           "Loser M102",            None, "3rd"),
    # ── FINAL ────────────────────────────────────────────────
    (104, "2026-07-19", "Winner M101",          "Winner M102",           None, "Final"),
]

df = pd.DataFrame(MATCHES, columns=[
    "match_num", "date", "home", "away", "group", "stage"
])
df["date"] = pd.to_datetime(df["date"])

print("=" * 55)
print("WC 2026 SCHEDULE VERIFICATION")
print("=" * 55)
print(f"Total matches     : {len(df)}  (expected 104)")
print(f"Group stage       : {(df['stage']=='group').sum()}  (expected 72)")
print(f"R32               : {(df['stage']=='R32').sum()}  (expected 16)")
print(f"R16               : {(df['stage']=='R16').sum()}  (expected 8)")
print(f"QF                : {(df['stage']=='QF').sum()}  (expected 4)")
print(f"SF                : {(df['stage']=='SF').sum()}  (expected 2)")
print(f"Final/3rd         : {df['stage'].isin(['Final','3rd']).sum()}  (expected 2)")
print(f"Date range        : {df['date'].min().date()} → {df['date'].max().date()}")

# Verify each team plays exactly 3 group stage matches
group_df = df[df["stage"] == "group"]
all_teams = []
for _, row in group_df.iterrows():
    all_teams.extend([row["home"], row["away"]])
team_counts = pd.Series(all_teams).value_counts()
teams_3 = (team_counts == 3).sum()
print(f"\nTeams with exactly 3 group matches: {teams_3}  (expected 48)")
if teams_3 != 48:
    print("  Teams NOT at 3 matches:")
    print(team_counts[team_counts != 3].to_string())

# Verify groups have 4 teams each
print(f"\nGroup sizes:")
for grp, teams in GROUPS.items():
    print(f"  Group {grp}: {teams}  ({len(teams)} teams)")

# Check teams in schedule match GROUPS dict
sched_teams = set(team_counts.index)
dict_teams  = set(t for teams in GROUPS.values() for t in teams)
missing = dict_teams - sched_teams
extra   = sched_teams - dict_teams
if missing: print(f"\n⚠️  In GROUPS but not in schedule: {missing}")
if extra:   print(f"⚠️  In schedule but not in GROUPS: {extra}")

print(f"\n{'='*55}")
checks = [
    ("104 total matches",          len(df) == 104),
    ("72 group matches",           (df['stage']=='group').sum() == 72),
    ("48 teams play 3 matches",    teams_3 == 48),
    ("12 groups of 4",             all(len(t)==4 for t in GROUPS.values())),
    ("Starts June 11",             df['date'].min().date().isoformat() == "2026-06-11"),
    ("Final July 19",              df['date'].max().date().isoformat() == "2026-07-19"),
]
all_pass = True
for label, result in checks:
    status = "✅ PASS" if result else "❌ FAIL"
    if not result: all_pass = False
    print(f"  {status}  {label}")

print(f"\n{'✅ ALL GOOD' if all_pass else '❌ FIX NEEDED'}")

# Save
df.to_csv("data/wc2026_schedule.csv", index=False)
print(f"\n[saved] data/wc2026_schedule.csv")