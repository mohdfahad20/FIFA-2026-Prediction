"""
dashboard/app_local.py
======================
Same as app.py but reads DB directly (no API needed).
Use this for local development.

Run:
    streamlit run dashboard/app_local.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import warnings
warnings.filterwarnings("ignore")

import json
import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from model.predict             import predict_match, _load_model, _load_rank_cache
from model.train               import SoftEnsemble, XGBWithWeight, LGBMWithWeight  # noqa
from score_model.predict_score import predict_score, _load_params

DB_PATH = Path(__file__).resolve().parent.parent / "fifa.db"

# ── Same CSS as app.py ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');
:root{--green:#00d26a;--dark:#0a0a0f;--card:#13131a;--border:#1e1e2e;--muted:#6b7280;--text:#f0f0f0;}
html,body,[class*="css"]{background-color:var(--dark)!important;color:var(--text)!important;font-family:'Inter',sans-serif!important;}
.wc-header{text-align:center;padding:2.5rem 0 1.5rem;border-bottom:1px solid var(--border);margin-bottom:2rem;}
.wc-header h1{font-family:'Bebas Neue',sans-serif;font-size:3.5rem;letter-spacing:4px;color:var(--green);margin:0;line-height:1;}
.wc-header p{color:var(--muted);font-size:0.85rem;letter-spacing:2px;text-transform:uppercase;margin-top:0.4rem;}
.stat-card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:0.8rem;}
.stat-card .label{font-size:0.72rem;text-transform:uppercase;letter-spacing:1.5px;color:var(--muted);margin-bottom:0.3rem;}
.stat-card .value{font-size:1.8rem;font-weight:600;color:var(--green);}
.prob-row{display:flex;align-items:center;margin-bottom:0.9rem;gap:0.8rem;}
.prob-label{width:80px;font-size:0.85rem;color:var(--muted);}
.prob-bar-wrap{flex:1;background:var(--border);border-radius:4px;height:10px;}
.prob-bar{height:10px;border-radius:4px;}
.prob-val{width:45px;text-align:right;font-size:0.85rem;font-weight:600;}
.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border-radius:8px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--muted)!important;border-radius:6px!important;font-size:0.85rem!important;font-weight:500!important;padding:0.5rem 1.2rem!important;}
.stTabs [aria-selected="true"]{background:var(--green)!important;color:var(--dark)!important;}
button[kind="primary"],.stButton>button{background:var(--green)!important;color:var(--dark)!important;border:none!important;font-weight:600!important;border-radius:6px!important;}
.stSelectbox>div>div{background:var(--card)!important;border:1px solid var(--border)!important;color:var(--text)!important;}
hr{border-color:var(--border)!important;}
</style>
""", unsafe_allow_html=True)

# ── DB helpers ────────────────────────────────────────────────────────────────
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

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
    d["results"] = json.loads(d["results_json"])
    return d

@st.cache_data(ttl=60)
def get_matches_played():
    conn = get_conn()
    n = conn.execute("""
        SELECT COUNT(*) FROM matches
        WHERE date >= '2026-06-11' AND tournament='FIFA World Cup'
        AND home_score IS NOT NULL
    """).fetchone()[0]
    conn.close()
    return n

def prob_bar(label, prob, color):
    st.markdown(f"""
    <div class="prob-row">
        <div class="prob-label">{label}</div>
        <div class="prob-bar-wrap"><div class="prob-bar" style="width:{prob*100:.1f}%;background:{color}"></div></div>
        <div class="prob-val">{prob*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

# ── Pre-load models ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    _load_model()
    _load_rank_cache()
    _load_params()

load_models()

# ── Header ────────────────────────────────────────────────────────────────────
played = get_matches_played()
st.markdown(f"""
<div class="wc-header">
    <h1>⚽ FIFA WC 2026</h1>
    <p>AI-Powered Tournament Predictor · Matches played: {played} / 104</p>
</div>
""", unsafe_allow_html=True)

# ── Top stats ─────────────────────────────────────────────────────────────────
sim = get_simulation()
if sim.get("results"):
    top_team = list(sim["results"].keys())[0]
    top_prob = list(sim["results"].values())[0]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="label">Tournament favourite</div><div class="value">{top_team}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="label">Win probability</div><div class="value">{top_prob:.1%}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="label">Simulations run</div><div class="value">{sim.get("n_simulations",0):,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="label">Matches remaining</div><div class="value">{sim.get("matches_remaining",104)}</div></div>', unsafe_allow_html=True)

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

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Group Standings","🎯 Match Predictor",
    "⚽ Score Predictor","📊 Tournament Odds","🤖 Model Info",
])

# TAB 1 — STANDINGS
with tab1:
    st.markdown("### Group Stage Standings")
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
    DB_NAMES = {"Czechia":"Czech Republic","Bosnia and Herzegovina":"Bosnia-Herzegovina",
                "Turkey":"Turkey","Iran":"Iran","Ivory Coast":"Ivory Coast",
                "DR Congo":"DR Congo","Cape Verde":"Cape Verde",
                "United States":"United States","South Korea":"South Korea"}
    def dbn(t): return DB_NAMES.get(t, t)

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
            """, (dbt,dbt)).fetchall()
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
            st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True,height=185)
    conn.close()

# TAB 2 — MATCH PREDICTOR
with tab2:
    st.markdown("### Match Outcome Predictor")
    c1,c2,c3 = st.columns([2,1,2])
    with c1: home = st.selectbox("Home",ALL_TEAMS,index=ALL_TEAMS.index("France"),key="lhome")
    with c2: st.markdown("<br><div style='text-align:center;font-size:1.5rem;color:#6b7280'>VS</div>",unsafe_allow_html=True)
    with c3: away = st.selectbox("Away",ALL_TEAMS,index=ALL_TEAMS.index("Brazil"),key="laway")
    neutral = st.toggle("Neutral venue",value=True,key="lneutral")

    if st.button("Predict",key="lpredict"):
        with st.spinner("Predicting..."):
            ml = predict_match(home, away, is_neutral=neutral)
            ps = predict_score(home, away, is_neutral=neutral)
        combined = {
            "win":  round((ml["win"]  + ps["win_prob"])  / 2, 4),
            "draw": round((ml["draw"] + ps["draw_prob"]) / 2, 4),
            "loss": round((ml["loss"] + ps["loss_prob"]) / 2, 4),
        }
        ca, cb = st.columns(2)
        with ca:
            st.markdown("**Combined**")
            prob_bar(f"{home} Win", combined["win"],  "#00d26a")
            prob_bar("Draw",         combined["draw"], "#f59e0b")
            prob_bar(f"{away} Win", combined["loss"], "#ef4444")
        with cb:
            st.markdown("**ML Model**")
            prob_bar(f"{home} Win", ml["win"],  "#00d26a")
            prob_bar("Draw",         ml["draw"], "#f59e0b")
            prob_bar(f"{away} Win", ml["loss"], "#ef4444")

# TAB 3 — SCORE PREDICTOR
with tab3:
    st.markdown("### Score Predictor")
    c1,c2,c3 = st.columns([2,1,2])
    with c1: sh = st.selectbox("Home",ALL_TEAMS,index=ALL_TEAMS.index("Spain"),key="lsh")
    with c2: st.markdown("<br><div style='text-align:center;font-size:1.5rem;color:#6b7280'>VS</div>",unsafe_allow_html=True)
    with c3: sa = st.selectbox("Away",ALL_TEAMS,index=ALL_TEAMS.index("Morocco"),key="lsa")

    if st.button("Predict Score",key="lscore"):
        with st.spinner("Calculating..."):
            result = predict_score(sh, sa, is_neutral=True)
        xg = result["expected_goals"]
        ca,cb,cc = st.columns(3)
        with ca: st.markdown(f'<div class="stat-card"><div class="label">{sh} xG</div><div class="value">{xg["home"]}</div></div>',unsafe_allow_html=True)
        with cb: st.markdown(f'<div class="stat-card"><div class="label">Most Likely</div><div class="value">{result["most_likely_score"]}</div></div>',unsafe_allow_html=True)
        with cc: st.markdown(f'<div class="stat-card"><div class="label">{sa} xG</div><div class="value">{xg["away"]}</div></div>',unsafe_allow_html=True)

        df_s = pd.DataFrame(result["top_scorelines"])
        df_s["pct"] = (df_s["probability"]*100).round(1)
        fig = px.bar(df_s,x="score",y="pct",text="pct",
                     color="pct",color_continuous_scale=[[0,"#1e1e2e"],[1,"#00d26a"]],
                     labels={"pct":"Probability (%)","score":"Scoreline"})
        fig.update_traces(texttemplate="%{text}%",textposition="outside")
        fig.update_layout(plot_bgcolor="#13131a",paper_bgcolor="#13131a",font_color="#f0f0f0",
                          showlegend=False,coloraxis_showscale=False,margin=dict(t=20,b=20),height=350)
        fig.update_xaxes(showgrid=False); fig.update_yaxes(showgrid=False,showticklabels=False)
        st.plotly_chart(fig,use_container_width=True)
        prob_bar(f"{sh} Win",result["win_prob"],"#00d26a")
        prob_bar("Draw",result["draw_prob"],"#f59e0b")
        prob_bar(f"{sa} Win",result["loss_prob"],"#ef4444")

# TAB 4 — TOURNAMENT ODDS
with tab4:
    st.markdown("### Tournament Win Probabilities")
    if sim.get("results"):
        df = pd.DataFrame([
            {"Team":t,"Win %":round(p*100,1),"Probability":p}
            for t,p in sim["results"].items() if p>0
        ]).sort_values("Win %",ascending=False).reset_index(drop=True)
        df.index += 1
        cc,ct = st.columns([3,2])
        with cc:
            top20 = df.head(20)
            fig = go.Figure(go.Bar(
                x=top20["Win %"],y=top20["Team"],orientation="h",
                marker=dict(color=top20["Win %"],colorscale=[[0,"#1e1e2e"],[0.5,"#059669"],[1,"#00d26a"]]),
                text=[f"{v}%" for v in top20["Win %"]],textposition="outside",
                textfont=dict(color="#f0f0f0",size=11),
            ))
            fig.update_layout(plot_bgcolor="#13131a",paper_bgcolor="#13131a",
                              font_color="#f0f0f0",
                              yaxis=dict(autorange="reversed",showgrid=False),
                              xaxis=dict(showgrid=False,showticklabels=False),
                              margin=dict(l=10,r=60,t=10,b=10),height=500)
            st.plotly_chart(fig,use_container_width=True)
        with ct:
            st.dataframe(df[["Team","Win %"]].head(20),hide_index=False,
                         use_container_width=True,height=500)

# TAB 5 — MODEL INFO
with tab5:
    st.markdown("### How the Models Work")
    c1,c2 = st.columns(2)
    with c1:
        st.markdown("""
**ML Ensemble Model**
- Soft-voting: MLP + XGBoost + LightGBM
- 22,000+ matches (2000–2022 train, 2023+ test)
- AUC: 0.74 · Log-loss: 0.88
- Draw class weight: 2.0
- Top feature: FIFA rank difference (28%)
        """)
    with c2:
        st.markdown("""
**Dixon-Coles Poisson Model**
- 14,724 FIFA-ranked matches (2010–2026)
- 3-year exponential time decay
- ρ = −0.13 low-score correction
- 7×7 scoreline probability matrix
- Attack/defense params per team
        """)
    st.markdown("---")
    st.markdown("**Simulation:** 10,000 Monte Carlo runs · ML + Poisson 50/50 · Extra time + penalties on KO draws")