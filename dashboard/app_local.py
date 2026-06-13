"""
dashboard/app_local.py
======================
FIFA WC 2026 Prediction Dashboard — Modern Minimal Theme
Reads DB + models directly (local dev mode).
"""

import sys
import json
import sqlite3
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model.predict             import predict_match, _load_model, _load_rank_cache
from model.train               import SoftEnsemble, XGBWithWeight, LGBMWithWeight  # noqa
from score_model.predict_score import predict_score, _load_params

DB_PATH = Path(__file__).resolve().parent.parent / "fifa.db"

st.set_page_config(
    page_title="FIFA WC 2026 · Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; background-color: #f8fafc !important; color: #0f172a !important; }
.nav-bar { display:flex;align-items:center;justify-content:space-between;padding:1rem 0 1.5rem;border-bottom:2px solid #16a34a;margin-bottom:2rem; }
.nav-title { font-size:1.5rem;font-weight:700;color:#0f172a;letter-spacing:-0.5px; }
.nav-title span { color:#16a34a; }
.nav-badge { background:#dcfce7;color:#15803d;font-size:0.75rem;font-weight:600;padding:0.3rem 0.8rem;border-radius:999px;letter-spacing:0.5px; }
.card { background:#ffffff;border:1px solid #e2e8f0;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;box-shadow:0 1px 3px rgba(0,0,0,0.06); }
.card-label { font-size:0.72rem;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:#64748b;margin-bottom:0.4rem; }
.card-value { font-size:2rem;font-weight:700;color:#0f172a;line-height:1.1; }
.card-value.green { color:#16a34a; }
.card-sub { font-size:0.8rem;color:#94a3b8;margin-top:0.2rem; }
.section-title { font-size:1.1rem;font-weight:700;color:#0f172a;margin-bottom:0.25rem; }
.section-sub { font-size:0.8rem;color:#64748b;margin-bottom:1.25rem; }
.prob-wrap { margin-bottom:0.75rem; }
.prob-header { display:flex;justify-content:space-between;margin-bottom:0.3rem; }
.prob-team { font-size:0.85rem;font-weight:500;color:#0f172a; }
.prob-pct { font-size:0.85rem;font-weight:700; }
.prob-track { background:#f1f5f9;border-radius:6px;height:8px;overflow:hidden; }
.prob-fill { height:8px;border-radius:6px; }
.stTabs [data-baseweb="tab-list"] { background:#f1f5f9 !important;border-radius:10px !important;padding:4px !important;gap:2px !important;border-bottom:none !important; }
.stTabs [data-baseweb="tab"] { background:transparent !important;color:#64748b !important;border-radius:8px !important;font-size:0.82rem !important;font-weight:600 !important;padding:0.5rem 1.1rem !important;border:none !important; }
.stTabs [aria-selected="true"] { background:#ffffff !important;color:#16a34a !important;box-shadow:0 1px 3px rgba(0,0,0,0.1) !important; }
.stButton > button { background:#16a34a !important;color:#ffffff !important;border:none !important;font-weight:600 !important;font-size:0.85rem !important;border-radius:8px !important;padding:0.55rem 1.5rem !important; }
.stButton > button:hover { background:#15803d !important; }
.stSelectbox > div > div { background:#ffffff !important;border:1px solid #e2e8f0 !important;border-radius:8px !important;color:#0f172a !important;font-size:0.9rem !important; }
hr { border:none !important;border-top:1px solid #e2e8f0 !important;margin:1.5rem 0 !important; }
.rank-badge { display:inline-block;background:#f0fdf4;color:#16a34a;font-size:0.75rem;font-weight:700;padding:0.15rem 0.5rem;border-radius:4px;border:1px solid #bbf7d0; }
.vs-text { text-align:center;font-size:0.85rem;font-weight:700;color:#94a3b8;letter-spacing:2px;padding-top:2rem; }
</style>
""", unsafe_allow_html=True)

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@st.cache_resource
def load_models():
    _load_model(); _load_rank_cache(); _load_params()

@st.cache_data(ttl=60)
def get_simulation():
    conn = get_conn()
    row  = conn.execute("""
        SELECT run_id, run_at, n_simulations, matches_played,
               matches_remaining, results_json
        FROM simulation_results ORDER BY run_at DESC LIMIT 1
    """).fetchone()
    conn.close()
    if not row: return {}
    d = dict(row)
    raw = json.loads(d["results_json"])
    # Handle both new format {"win":..., "r32":...} and legacy flat dict
    if isinstance(raw, dict) and "win" in raw:
        d["results"]     = raw["win"]
        d["r32_qualify"] = raw.get("r32", {})
    else:
        d["results"]     = raw
        d["r32_qualify"] = {}
    return d

@st.cache_data(ttl=60)
def get_matches_played():
    conn = get_conn()
    n = conn.execute("""
        SELECT COUNT(*) FROM matches
        WHERE date>='2026-06-11' AND tournament='FIFA World Cup'
        AND home_score IS NOT NULL
    """).fetchone()[0]
    conn.close(); return n

def prob_bar(label, prob, color, pct_color):
    st.markdown(f"""
    <div class="prob-wrap">
        <div class="prob-header">
            <span class="prob-team">{label}</span>
            <span class="prob-pct" style="color:{pct_color}">{prob*100:.1f}%</span>
        </div>
        <div class="prob-track">
            <div class="prob-fill" style="width:{prob*100:.1f}%;background:{color}"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

load_models()

played = get_matches_played()
sim    = get_simulation()

st.markdown(f"""
<div class="nav-bar">
    <div class="nav-title">⚽ FIFA WC <span>2026</span> Predictor</div>
    <div class="nav-badge">🟢 {played} / 104 matches played</div>
</div>
""", unsafe_allow_html=True)

if sim.get("results"):
    top_team = list(sim["results"].keys())[0]
    top_prob = list(sim["results"].values())[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="card"><div class="card-label">🏆 Tournament Favourite</div>
            <div class="card-value">{top_team}</div>
            <div class="card-sub">Based on {sim.get("n_simulations",0):,} simulations</div></div>""",
            unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="card"><div class="card-label">📈 Win Probability</div>
            <div class="card-value green">{top_prob:.1%}</div>
            <div class="card-sub">Monte Carlo estimate</div></div>""",
            unsafe_allow_html=True)
    with c3:
        remaining = sim.get("matches_remaining", 104)
        st.markdown(f"""<div class="card"><div class="card-label">⏳ Matches Remaining</div>
            <div class="card-value">{remaining}</div>
            <div class="card-sub">of 104 total</div></div>""",
            unsafe_allow_html=True)
    with c4:
        run_at = sim.get("run_at","")[:16].replace("T"," ")
        st.markdown(f"""<div class="card"><div class="card-label">🔄 Last Updated</div>
            <div class="card-value" style="font-size:1.2rem">{run_at}</div>
            <div class="card-sub">UTC</div></div>""",
            unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

ALL_TEAMS = sorted([
    "Argentina","Algeria","Austria","Australia","Belgium","Bosnia and Herzegovina",
    "Brazil","Canada","Cape Verde","Colombia","Croatia","Curaçao","Czechia",
    "DR Congo","Ecuador","Egypt","England","France","Germany","Ghana",
    "Haiti","Iran","Iraq","Ivory Coast","Japan","Jordan","Mexico",
    "Morocco","Netherlands","New Zealand","Norway","Panama","Paraguay",
    "Portugal","Qatar","Saudi Arabia","Scotland","Senegal","South Africa",
    "South Korea","Spain","Sweden","Switzerland","Tunisia","Turkey",
    "United States","Uruguay","Uzbekistan",
])

GROUPS = {
    "A":["Mexico","South Africa","South Korea","Czechia"],
    "B":["Canada","Switzerland","Qatar","Bosnia and Herzegovina"],
    "C":["Brazil","Morocco","Haiti","Scotland"],
    "D":["United States","Paraguay","Australia","Turkey"],
    "E":["Germany","Curaçao","Ivory Coast","Ecuador"],
    "F":["Netherlands","Japan","Tunisia","Sweden"],
    "G":["Belgium","Egypt","Iran","New Zealand"],
    "H":["Spain","Cape Verde","Saudi Arabia","Uruguay"],
    "I":["France","Senegal","Norway","Iraq"],
    "J":["Argentina","Algeria","Austria","Jordan"],
    "K":["Portugal","Uzbekistan","Colombia","DR Congo"],
    "L":["England","Croatia","Ghana","Panama"],
}
DB_NAMES = {
    "Czechia":"Czech Republic","Bosnia and Herzegovina":"Bosnia-Herzegovina",
    "Turkey":"Turkey","Iran":"Iran","Ivory Coast":"Ivory Coast",
    "DR Congo":"DR Congo","Cape Verde":"Cape Verde",
    "United States":"United States","South Korea":"South Korea",
}
def dbn(t): return DB_NAMES.get(t, t)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "  🏆  Standings  ",
    "  🔑  Qualification  ",
    "  🎯  Match Predictor  ",
    "  ⚽  Score Predictor  ",
    "  📊  Tournament Odds  ",
    "  🤖  Model Info  ",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — GROUP STANDINGS
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">Group Stage Standings</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Top 2 per group qualify · Best 8 third-place teams also advance</div>', unsafe_allow_html=True)
    conn = get_conn()
    cols = st.columns(3)
    for i, (grp, teams) in enumerate(sorted(GROUPS.items())):
        rows = []
        for team in teams:
            dbt = dbn(team)
            ms  = conn.execute("""
                SELECT home_team,away_team,home_score,away_score,result
                FROM matches WHERE date>='2026-06-11'
                AND tournament='FIFA World Cup'
                AND (home_team=? OR away_team=?) AND home_score IS NOT NULL
            """, (dbt, dbt)).fetchall()
            p=w=d=l=gf=ga=pts=0
            for m in ms:
                p+=1
                if m["home_team"]==dbt:
                    gf+=m["home_score"]; ga+=m["away_score"]
                    if m["result"]=="win": pts+=3;w+=1
                    elif m["result"]=="draw": pts+=1;d+=1
                    else: l+=1
                else:
                    gf+=m["away_score"]; ga+=m["home_score"]
                    if m["result"]=="loss": pts+=3;w+=1
                    elif m["result"]=="draw": pts+=1;d+=1
                    else: l+=1
            rows.append({"Team":team,"P":p,"W":w,"D":d,"L":l,"GF":gf,"GA":ga,"GD":gf-ga,"Pts":pts})
        rows.sort(key=lambda x:(x["Pts"],x["GD"],x["GF"]),reverse=True)
        with cols[i%3]:
            st.markdown(f"**Group {grp}**")
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True, height=185)
    conn.close()

    conn2 = get_conn()
    recent = conn2.execute("""
        SELECT date, home_team, away_team, home_score, away_score
        FROM matches WHERE date>='2026-06-11'
        AND tournament='FIFA World Cup' AND home_score IS NOT NULL
        ORDER BY date DESC LIMIT 10
    """).fetchall()
    conn2.close()
    if recent:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="section-title">Recent Results</div>', unsafe_allow_html=True)
        for r in recent:
            col_l, col_s, col_r = st.columns([3,1,3])
            with col_l:
                st.markdown(f"<div style='text-align:right;font-weight:600'>{r['home_team']}</div>", unsafe_allow_html=True)
            with col_s:
                st.markdown(f"<div style='text-align:center;font-weight:700;color:#16a34a'>{r['home_score']} – {r['away_score']}</div>", unsafe_allow_html=True)
            with col_r:
                st.markdown(f"<div style='font-weight:600'>{r['away_team']}</div>", unsafe_allow_html=True)
    else:
        st.info("No live match data yet — tournament starts June 11, 2026.")

# ═══════════════════════════════════════════════════════════════
# TAB 2 — QUALIFICATION ODDS  (NEW)
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">Round of 32 Qualification Probabilities</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Based on {sim.get("n_simulations",0):,} simulations · Chance each team advances from the group stage</div>',
                unsafe_allow_html=True)

    r32 = sim.get("r32_qualify", {})

    if not r32:
        st.warning("Qualification data not available yet — re-run the simulation pipeline to generate it.")
        st.code("python run_pipeline.py --skip-data --skip-ml --n 1000")
    else:
        cols = st.columns(3)
        for i, (grp, teams) in enumerate(sorted(GROUPS.items())):
            rows = []
            for team in teams:
                dbt  = dbn(team)
                prob = r32.get(dbt, r32.get(team, 0.0))
                rows.append({"team": team, "prob": prob})
            rows.sort(key=lambda x: x["prob"], reverse=True)

            with cols[i % 3]:
                st.markdown(f"**Group {grp}**")
                COLORS = [
                    ("#16a34a", "#f0fdf4", "#bbf7d0"),   # 1st — dark green
                    ("#4ade80", "#f0fdf4", "#dcfce7"),   # 2nd — light green
                    ("#f59e0b", "#fffbeb", "#fde68a"),   # 3rd — amber
                    ("#ef4444", "#fff1f2", "#fecdd3"),   # 4th — red
                ]
                LABELS = ["1st", "2nd", "3rd", "4th"]
                for j, row in enumerate(rows):
                    bar_color, bg, border = COLORS[j]
                    pct = row["prob"] * 100
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {border};border-radius:8px;
                                padding:0.6rem 0.75rem;margin-bottom:0.4rem">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:0.3rem">
                            <span style="font-size:0.85rem;font-weight:600;color:#0f172a">{row['team']}</span>
                            <span style="font-size:0.8rem;font-weight:700;color:{bar_color}">{pct:.1f}%</span>
                        </div>
                        <div style="background:#e2e8f0;border-radius:4px;height:6px">
                            <div style="width:{pct:.1f}%;background:{bar_color};height:6px;border-radius:4px"></div>
                        </div>
                        <div style="font-size:0.7rem;color:#94a3b8;margin-top:0.2rem">
                            Projected {LABELS[j]}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">Match Outcome Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">ML ensemble + Dixon-Coles Poisson · 50/50 weighted average</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([5,1,5])
    with c1: home = st.selectbox("Home Team", ALL_TEAMS, index=ALL_TEAMS.index("France"), key="ph")
    with c2: st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with c3: away = st.selectbox("Away Team", ALL_TEAMS, index=ALL_TEAMS.index("Brazil"), key="pa")
    col_tog, col_btn = st.columns([3,1])
    with col_tog: neutral = st.toggle("Neutral venue", value=True)
    with col_btn: predict_clicked = st.button("Predict Match", use_container_width=True)
    if predict_clicked:
        with st.spinner(""):
            ml = predict_match(home, away, is_neutral=neutral)
            ps = predict_score(home, away, is_neutral=neutral)
        combined = {
            "win":  round((ml["win"]  + ps["win_prob"])  / 2, 4),
            "draw": round((ml["draw"] + ps["draw_prob"]) / 2, 4),
            "loss": round((ml["loss"] + ps["loss_prob"]) / 2, 4),
        }
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="display:flex;gap:1rem;margin-bottom:1.25rem;align-items:center">
            <span style="font-weight:600">{home}</span>
            <span class="rank-badge">#{ml['home_rank']}</span>
            <span style="color:#94a3b8;font-size:0.8rem">vs</span>
            <span style="font-weight:600">{away}</span>
            <span class="rank-badge">#{ml['away_rank']}</span>
        </div>
        """, unsafe_allow_html=True)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="card"><div class="card-label">Combined (ML + Poisson)</div></div>', unsafe_allow_html=True)
            prob_bar(f"{home} win", combined["win"],  "#16a34a", "#16a34a")
            prob_bar("Draw",         combined["draw"], "#f59e0b", "#b45309")
            prob_bar(f"{away} win", combined["loss"], "#ef4444", "#b91c1c")
        with col_b:
            st.markdown('<div class="card"><div class="card-label">ML Model only</div></div>', unsafe_allow_html=True)
            prob_bar(f"{home} win", ml["win"],  "#16a34a", "#16a34a")
            prob_bar("Draw",         ml["draw"], "#f59e0b", "#b45309")
            prob_bar(f"{away} win", ml["loss"], "#ef4444", "#b91c1c")
        best = max([("win",combined["win"]),("draw",combined["draw"]),("loss",combined["loss"])], key=lambda x:x[1])
        verdict = (f"{home} are favoured to win" if best[0]=="win"
                   else f"{away} are favoured to win" if best[0]=="loss"
                   else "A draw is the most likely outcome")
        st.markdown(f"""
        <div class="card" style="border-left:4px solid #16a34a;margin-top:0.5rem">
            <div class="card-label">Verdict</div>
            <div style="font-size:1rem;font-weight:600;color:#0f172a;margin-top:0.2rem">
                {verdict} <span style="color:#16a34a">({best[1]:.1%})</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — SCORE PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">Score Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Dixon-Coles Poisson model with 3-year time-decayed attack/defense parameters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([5,1,5])
    with c1: sh = st.selectbox("Home Team", ALL_TEAMS, index=ALL_TEAMS.index("Spain"), key="sh")
    with c2: st.markdown('<div class="vs-text">VS</div>', unsafe_allow_html=True)
    with c3: sa = st.selectbox("Away Team", ALL_TEAMS, index=ALL_TEAMS.index("Morocco"), key="sa")
    _, col_btn2 = st.columns([3,1])
    with col_btn2: score_clicked = st.button("Predict Score", use_container_width=True)
    if score_clicked:
        with st.spinner(""):
            result = predict_score(sh, sa, is_neutral=True)
        xg = result["expected_goals"]
        st.markdown("<hr>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""<div class="card" style="text-align:center">
                <div class="card-label">{sh} Expected Goals</div>
                <div class="card-value green">{xg['home']:.2f}</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="card" style="text-align:center;border:2px solid #16a34a">
                <div class="card-label">Most Likely Score</div>
                <div class="card-value">{result['most_likely_score']}</div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="card" style="text-align:center">
                <div class="card-label">{sa} Expected Goals</div>
                <div class="card-value green">{xg['away']:.2f}</div></div>""", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        df_s = pd.DataFrame(result["top_scorelines"])
        df_s["pct"] = (df_s["probability"] * 100).round(1)
        fig = px.bar(df_s, x="score", y="pct", text="pct", color="pct",
                     color_continuous_scale=[[0,"#dcfce7"],[0.5,"#4ade80"],[1,"#16a34a"]],
                     labels={"pct":"Probability (%)","score":"Scoreline"})
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          textfont=dict(size=12,color="#0f172a"))
        fig.update_layout(plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",
                          font=dict(family="Inter",color="#0f172a"),
                          showlegend=False,coloraxis_showscale=False,
                          margin=dict(t=30,b=10,l=10,r=10),height=320,
                          xaxis=dict(showgrid=False,title=""),
                          yaxis=dict(showgrid=False,showticklabels=False,title=""))
        fig.update_xaxes(tickfont=dict(size=13,color="#0f172a"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="section-title" style="font-size:0.9rem">Outcome Probabilities (Poisson)</div>', unsafe_allow_html=True)
        prob_bar(f"{sh} win",  result["win_prob"],  "#16a34a", "#16a34a")
        prob_bar("Draw",        result["draw_prob"], "#f59e0b", "#b45309")
        prob_bar(f"{sa} win",  result["loss_prob"], "#ef4444", "#b91c1c")

# ═══════════════════════════════════════════════════════════════
# TAB 5 — TOURNAMENT ODDS
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-title">Tournament Win Probabilities</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="section-sub">Based on {sim.get("n_simulations",0):,} Monte Carlo simulations of the full bracket</div>', unsafe_allow_html=True)
    if sim.get("results"):
        results = sim["results"]
        df = pd.DataFrame([
            {"Rank":i+1,"Team":t,"Win %":round(p*100,1)}
            for i,(t,p) in enumerate(results.items()) if p>0
        ])
        col_chart, col_table = st.columns([3,2])
        with col_chart:
            top20 = df.head(20)
            colors = ["#16a34a" if i<3 else "#4ade80" if i<8 else "#bbf7d0" for i in range(len(top20))]
            fig = go.Figure(go.Bar(
                x=top20["Win %"], y=top20["Team"], orientation="h",
                marker_color=colors,
                text=[f"{v}%" for v in top20["Win %"]],
                textposition="outside",
                textfont=dict(size=11,color="#0f172a",family="Inter"),
            ))
            fig.update_layout(plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",
                              font=dict(family="Inter",color="#0f172a"),
                              yaxis=dict(autorange="reversed",showgrid=False,tickfont=dict(size=12)),
                              xaxis=dict(showgrid=True,gridcolor="#f1f5f9",showticklabels=False,title=""),
                              margin=dict(l=10,r=70,t=10,b=10),height=520)
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(df[["Rank","Team","Win %"]], hide_index=True, use_container_width=True, height=520,
                column_config={"Win %":st.column_config.ProgressColumn("Win %",min_value=0,max_value=df["Win %"].max(),format="%.1f%%")})
        conn3 = get_conn()
        runs = conn3.execute("SELECT run_at, matches_played, results_json FROM simulation_results ORDER BY run_at ASC").fetchall()
        conn3.close()
        if len(runs) > 1:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown('<div class="section-title">Probability Trend</div>', unsafe_allow_html=True)
            st.markdown('<div class="section-sub">How win probabilities shift as real results come in</div>', unsafe_allow_html=True)
            top5 = list(results.keys())[:5]
            trend = []
            for run in runs:
                raw = json.loads(run["results_json"])
                r_json = raw["win"] if isinstance(raw, dict) and "win" in raw else raw
                for team in top5:
                    trend.append({"Date":run["run_at"][:10],"Team":team,"Win %":round(r_json.get(team,0)*100,1)})
            trend_df = pd.DataFrame(trend)
            fig2 = px.line(trend_df, x="Date", y="Win %", color="Team",
                           color_discrete_sequence=["#16a34a","#0ea5e9","#f59e0b","#8b5cf6","#ef4444"],
                           markers=True)
            fig2.update_layout(plot_bgcolor="#ffffff",paper_bgcolor="#ffffff",
                               font=dict(family="Inter",color="#0f172a"),
                               xaxis=dict(showgrid=False),yaxis=dict(showgrid=True,gridcolor="#f1f5f9"),
                               legend=dict(orientation="h",yanchor="bottom",y=1.02),
                               margin=dict(t=40,b=10),height=300)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run the simulation first: `python -m simulate.simulate --db fifa.db --n 10000`")

# ═══════════════════════════════════════════════════════════════
# TAB 6 — MODEL INFO
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown('<div class="section-title">How the Models Work</div>', unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div class="card">
<div class="card-label">ML Ensemble Model</div>
<div style="margin-top:0.75rem;font-size:0.9rem;line-height:1.8;color:#334155">

**Architecture** — Soft-voting ensemble of MLP, XGBoost, and LightGBM

**Training data** — 22,000+ international matches, 2000–2022

**Test set** — 2023–2026 held out (time-based split)

**Performance** — AUC 0.74 · Log-loss 0.88

**Draw handling** — Class weight 2.0 upsamples the draw class

**Top features:** FIFA rank difference (28%), H2H goal difference (9%), neutral venue flag (5%)

</div>
</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class="card">
<div class="card-label">Dixon-Coles Poisson Score Model</div>
<div style="margin-top:0.75rem;font-size:0.9rem;line-height:1.8;color:#334155">

**Method** — MLE of per-team attack/defense strength

**Training data** — 14,724 FIFA-ranked matches (2010–2026)

**Time decay** — Exponential, 3-year half-life

**DC correction** — ρ = −0.13 for low-scoring scorelines

**Notable params:** Morocco defense: 0.303 · Spain attack: 1.849 · Argentina defense: 0.374

</div>
</div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
<div class="card" style="border-left:4px solid #16a34a">
<div class="card-label">Tournament Simulation</div>
<div style="margin-top:0.75rem;font-size:0.9rem;line-height:1.8;color:#334155">

10,000 Monte Carlo simulations · ML + Poisson averaged 50/50 · Probability cache prewarmed

Group stage: round-robin · Pts → GD → GF tiebreaker · Best 8 third-place teams advance

Knockout: extra time → penalties (50/50) · Real results locked in nightly

</div>
</div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">All 48 Teams — Poisson Parameters</div>', unsafe_allow_html=True)
    poisson = _load_params()
    attack  = poisson["attack"]
    defense = poisson["defense"]
    team_rows = []
    for grp, teams in sorted(GROUPS.items()):
        for team in teams:
            dbt = dbn(team)
            team_rows.append({"Group":grp,"Team":team,
                               "Attack":round(attack.get(dbt,1.0),3),
                               "Defense":round(defense.get(dbt,1.0),3)})
    teams_df = pd.DataFrame(team_rows).sort_values("Attack",ascending=False)
    st.dataframe(teams_df, hide_index=True, use_container_width=True,
        column_config={
            "Attack":  st.column_config.ProgressColumn("Attack",  min_value=0, max_value=2.5, format="%.3f"),
            "Defense": st.column_config.ProgressColumn("Defense", min_value=0, max_value=2.0, format="%.3f"),
        })