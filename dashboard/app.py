"""
dashboard/app.py
================
Streamlit dashboard for FIFA WC 2026 Prediction System.
Calls FastAPI backend (for Streamlit Cloud deployment).
"""

import os
import time
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, timezone

API_BASE = os.getenv("API_BASE_URL", "https://fifa-predictor-api-xjnm.onrender.com")

st.set_page_config(
    page_title="FIFA WC 2026 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;500;600&display=swap');
:root { --green:#00d26a; --dark:#0a0a0f; --card:#13131a; --border:#1e1e2e; --muted:#6b7280; --text:#f0f0f0; }
html, body, [class*="css"] { background-color:var(--dark) !important; color:var(--text) !important; font-family:'Inter',sans-serif !important; }
.wc-header { text-align:center; padding:2.5rem 0 1.5rem; border-bottom:1px solid var(--border); margin-bottom:2rem; }
.wc-header h1 { font-family:'Bebas Neue',sans-serif; font-size:3.5rem; letter-spacing:4px; color:var(--green); margin:0; line-height:1; }
.wc-header p { color:var(--muted); font-size:0.85rem; letter-spacing:2px; text-transform:uppercase; margin-top:0.4rem; }
.stat-card { background:var(--card); border:1px solid var(--border); border-radius:10px; padding:1.2rem 1.4rem; margin-bottom:0.8rem; }
.stat-card .label { font-size:0.72rem; text-transform:uppercase; letter-spacing:1.5px; color:var(--muted); margin-bottom:0.3rem; }
.stat-card .value { font-size:1.8rem; font-weight:600; color:var(--green); }
.stTabs [data-baseweb="tab-list"] { background:var(--card) !important; border-radius:8px; padding:4px; gap:4px; }
.stTabs [data-baseweb="tab"] { background:transparent !important; color:var(--muted) !important; border-radius:6px !important; font-size:0.85rem !important; font-weight:500 !important; padding:0.5rem 1.2rem !important; }
.stTabs [aria-selected="true"] { background:var(--green) !important; color:var(--dark) !important; }
.prob-row { display:flex; align-items:center; margin-bottom:0.9rem; gap:0.8rem; }
.prob-label { width:80px; font-size:0.85rem; color:var(--muted); }
.prob-bar-wrap { flex:1; background:var(--border); border-radius:4px; height:10px; }
.prob-bar { height:10px; border-radius:4px; }
.prob-val { width:45px; text-align:right; font-size:0.85rem; font-weight:600; }
button[kind="primary"], .stButton > button { background:var(--green) !important; color:var(--dark) !important; border:none !important; font-weight:600 !important; border-radius:6px !important; }
.stSelectbox > div > div { background:var(--card) !important; border:1px solid var(--border) !important; color:var(--text) !important; }
hr { border-color:var(--border) !important; }
</style>
""", unsafe_allow_html=True)

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

@st.cache_data(ttl=300)
def api_get(path, params=None):
    for attempt in range(3):
        try:
            r = requests.get(f"{API_BASE}{path}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except Exception:
            if attempt < 2:
                time.sleep(3)
    return None

def prob_bar(label, prob, color):
    st.markdown(f"""
    <div class="prob-row">
        <div class="prob-label">{label}</div>
        <div class="prob-bar-wrap"><div class="prob-bar" style="width:{prob*100:.1f}%;background:{color}"></div></div>
        <div class="prob-val">{prob*100:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

# ── Backend wake-up handler ───────────────────────────────────────────────────
health = api_get("/health")
if not health:
    st.warning("🚀 Backend is waking up on Render... please wait 30 seconds")
    time.sleep(5)
    st.rerun()

st.markdown(f"""
<div class="wc-header">
    <h1>⚽ FIFA WC 2026</h1>
    <p>AI-Powered Tournament Predictor · Matches played: {health.get('matches_played', 0)} / 104</p>
</div>
""", unsafe_allow_html=True)

sim = api_get("/simulation") or {}
_raw = sim.get("results", {})
if isinstance(_raw, dict) and "win" in _raw:
    sim_results = _raw["win"]
    sim_r32     = _raw.get("r32", {})
else:
    sim_results = _raw
    sim_r32     = {}

if sim_results:
    top_team  = list(sim_results.keys())[0]
    top_prob  = list(sim_results.values())[0]
    n_sims    = sim.get("n_simulations", 0)
    remaining = sim.get("matches_remaining", 104)
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="stat-card"><div class="label">Tournament favourite</div><div class="value">{top_team}</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-card"><div class="label">Win probability</div><div class="value">{top_prob:.1%}</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="stat-card"><div class="label">Simulations run</div><div class="value">{n_sims:,}</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="stat-card"><div class="label">Matches remaining</div><div class="value">{remaining}</div></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "🏆 Group Standings",
    "🔑 Qualification",
    "📅 Predictions",
    "🎯 Match Predictor",
    "⚽ Score Predictor",
    "📊 Tournament Odds",
    "🤖 Model Info",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — GROUP STANDINGS
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Group Stage Standings")
    st.caption("Top 2 per group qualify automatically · Best 8 third-place teams also advance")
    data   = api_get("/standings") or {}
    groups = data.get("groups", {})
    if not groups:
        st.info("No live match data yet — tournament starts June 11, 2026.")
    else:
        cols = st.columns(3)
        for i, (grp, rows) in enumerate(sorted(groups.items())):
            with cols[i % 3]:
                st.markdown(f"**Group {grp}**")
                df = pd.DataFrame(rows)
                if not df.empty:
                    df = df[["team","played","w","d","l","gf","ga","gd","pts"]]
                    df.columns = ["Team","P","W","D","L","GF","GA","GD","Pts"]
                    st.dataframe(df, hide_index=True, use_container_width=True, height=180)
    recent = data.get("recent_results", [])
    if recent:
        st.markdown("---")
        st.markdown("### Recent Results")
        for r in recent[:8]:
            st.markdown(
                f"**{r['home_team']}** {r.get('home_score','')} – {r.get('away_score','')} **{r['away_team']}** "
                f"<span style='color:#6b7280;font-size:0.8rem'>{r['date']}</span>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 2 — QUALIFICATION ODDS
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Round of 32 Qualification Probabilities")
    st.caption(f"Based on {sim.get('n_simulations', 0):,} simulations · Chance each team advances from group stage")
    if not sim_r32:
        st.warning("Qualification data not available — re-run the pipeline to generate it.")
    else:
        cols = st.columns(3)
        COLORS = [
            ("#00d26a", "#0d1f16", "#0a3320"),
            ("#4ade80", "#0d1f16", "#0a2e1a"),
            ("#f59e0b", "#1f1800", "#2e2200"),
            ("#ef4444", "#1f0a0a", "#2e0f0f"),
        ]
        LABELS = ["1st", "2nd", "3rd", "4th"]
        for i, (grp, teams) in enumerate(sorted(GROUPS.items())):
            rows = []
            for team in teams:
                prob = sim_r32.get(dbn(team), sim_r32.get(team, 0.0))
                rows.append({"team": team, "prob": prob})
            rows.sort(key=lambda x: x["prob"], reverse=True)
            with cols[i % 3]:
                st.markdown(f"**Group {grp}**")
                for j, row in enumerate(rows):
                    bar_color, bg, border = COLORS[j]
                    pct = row["prob"] * 100
                    st.markdown(f"""
                    <div style="background:{bg};border:1px solid {border};border-radius:8px;
                                padding:0.6rem 0.75rem;margin-bottom:0.4rem">
                        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.3rem">
                            <span style="font-size:0.85rem;font-weight:600;color:#f0f0f0">{row['team']}</span>
                            <span style="font-size:0.8rem;font-weight:700;color:{bar_color}">{pct:.1f}%</span>
                        </div>
                        <div style="background:#1e1e2e;border-radius:4px;height:6px">
                            <div style="width:{pct:.1f}%;background:{bar_color};height:6px;border-radius:4px"></div>
                        </div>
                        <div style="font-size:0.7rem;color:#6b7280;margin-top:0.2rem">Projected {LABELS[j]}</div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIONS  (NEW)
# ═══════════════════════════════════════════════════════════════
with tab3:
    today_utc     = datetime.now(timezone.utc).date()
    tomorrow_utc  = today_utc + timedelta(days=1)
    yesterday_utc = today_utc - timedelta(days=1)

    st.markdown("### 📅 Match Predictions")
    st.caption("Tomorrow's predictions · Yesterday's results vs actual")

    preds_data = api_get("/predictions") or {}
    tomorrow_rows  = preds_data.get("tomorrow", [])
    yesterday_rows = preds_data.get("yesterday", [])

    # ── Tomorrow ─────────────────────────────────────────────────────────────
    st.markdown(f"### 🔮 Tomorrow — {tomorrow_utc.strftime('%B %d, %Y')}")
    if not tomorrow_rows:
        st.info("No predictions yet for tomorrow. They will appear after tonight's pipeline run (02:00 UTC).")
    else:
        cols = st.columns(2)
        for i, r in enumerate(tomorrow_rows):
            hw = r["pred_home_win"]
            dr = r["pred_draw"]
            aw = r["pred_away_win"]
            fav_prob = max(hw, dr, aw)
            fav = (r["home_team"] if hw == fav_prob
                   else r["away_team"] if aw == fav_prob
                   else "Draw")
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);border-top:3px solid #00d26a;
                            border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.75rem;">
                        <span style="font-size:0.7rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">{r['stage']}</span>
                        <span style="font-size:0.7rem;color:var(--muted);">{r['date']}</span>
                    </div>
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem;">
                        <div style="text-align:center;flex:1;">
                            <div style="font-size:0.95rem;font-weight:700;color:var(--text);">{r['home_team']}</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#00d26a;">{hw:.0%}</div>
                            <div style="font-size:0.7rem;color:var(--muted);">xG {r['pred_home_xg']}</div>
                        </div>
                        <div style="text-align:center;padding:0 0.5rem;">
                            <div style="font-size:0.8rem;font-weight:700;color:var(--muted);">VS</div>
                            <div style="font-size:0.75rem;font-weight:600;color:#f59e0b;margin-top:0.2rem;">Draw {dr:.0%}</div>
                        </div>
                        <div style="text-align:center;flex:1;">
                            <div style="font-size:0.95rem;font-weight:700;color:var(--text);">{r['away_team']}</div>
                            <div style="font-size:1.4rem;font-weight:700;color:#ef4444;">{aw:.0%}</div>
                            <div style="font-size:0.7rem;color:var(--muted);">xG {r['pred_away_xg']}</div>
                        </div>
                    </div>
                    <div style="background:#1e1e2e;border-radius:6px;padding:0.5rem 0.75rem;
                                display:flex;justify-content:space-between;align-items:center;">
                        <span style="font-size:0.8rem;color:var(--muted);">
                            🎯 Likely score: <strong style="color:var(--text);">{r['pred_scoreline']}</strong>
                        </span>
                        <span style="font-size:0.75rem;font-weight:700;color:#00d26a;">
                            Tip: {fav} ({fav_prob:.0%})
                        </span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── Yesterday ─────────────────────────────────────────────────────────────
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(f"### 📊 Yesterday — {yesterday_utc.strftime('%B %d, %Y')} · Prediction vs Actual")
    if not yesterday_rows:
        st.info("No completed predictions for yesterday yet.")
    else:
        cols2 = st.columns(2)
        for i, r in enumerate(yesterday_rows):
            correct    = r.get("was_correct")
            has_actual = r.get("actual_home") is not None

            if has_actual:
                border_color = "#00d26a" if correct else "#ef4444"
                badge        = "✅ CORRECT" if correct else "❌ WRONG"
                badge_color  = "#00d26a"  if correct else "#ef4444"
                badge_bg     = "#0d1f16"  if correct else "#1f0a0a"
            else:
                border_color = "#1e1e2e"
                badge        = "⏳ PENDING"
                badge_color  = "#f59e0b"
                badge_bg     = "#1f1800"

            pred_label = (r["home_team"]  if r["pred_winner"] == "home"
                          else r["away_team"] if r["pred_winner"] == "away"
                          else "Draw")

            actual_score_html = (
                f'<div style="font-size:0.9rem;font-weight:700;color:var(--text);">'
                f'{r["actual_home"]} – {r["actual_away"]}</div>'
                f'<div style="font-size:0.75rem;color:var(--muted);">'
                f'{(r.get("actual_result") or "").capitalize()}</div>'
                if has_actual else
                '<div style="font-size:0.85rem;color:var(--muted);">Pending</div>'
            )
            actual_bg = (
                "#0d1f16" if (correct and has_actual)
                else "#1f0a0a" if (has_actual and not correct)
                else "#1e1e2e"
            )

            with cols2[i % 2]:
                st.markdown(f"""
                <div style="background:var(--card);border:1px solid var(--border);border-top:3px solid {border_color};
                            border-radius:10px;padding:1.2rem;margin-bottom:1rem;">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
                        <span style="font-size:0.7rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:1px;">{r['stage']}</span>
                        <span style="background:{badge_bg};color:{badge_color};font-size:0.7rem;
                                     font-weight:700;padding:0.2rem 0.6rem;border-radius:999px;">{badge}</span>
                    </div>
                    <div style="font-size:1rem;font-weight:700;color:var(--text);text-align:center;margin-bottom:0.75rem;">
                        {r['home_team']} vs {r['away_team']}
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:0.5rem;">
                        <div style="background:#1e1e2e;border-radius:8px;padding:0.6rem;text-align:center;">
                            <div style="font-size:0.65rem;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:0.3rem;">🔮 Predicted</div>
                            <div style="font-size:0.9rem;font-weight:700;color:var(--text);">{pred_label}</div>
                            <div style="font-size:0.75rem;color:var(--muted);">Score: {r['pred_scoreline']}</div>
                            <div style="font-size:0.7rem;color:var(--muted);">xG {r['pred_home_xg']} – {r['pred_away_xg']}</div>
                        </div>
                        <div style="background:{actual_bg};border-radius:8px;padding:0.6rem;text-align:center;">
                            <div style="font-size:0.65rem;font-weight:600;color:var(--muted);text-transform:uppercase;margin-bottom:0.3rem;">⚽ Actual</div>
                            {actual_score_html}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — MATCH PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### Match Outcome Predictor")
    st.caption("Uses ML ensemble + Dixon-Coles Poisson model (50/50 weighted)")
    col1,col2,col3 = st.columns([2,1,2])
    with col1: home = st.selectbox("Home Team", ALL_TEAMS, index=ALL_TEAMS.index("France"), key="pred_home")
    with col2: st.markdown("<br><div style='text-align:center;font-size:1.5rem;color:#6b7280'>VS</div>", unsafe_allow_html=True)
    with col3: away = st.selectbox("Away Team", ALL_TEAMS, index=ALL_TEAMS.index("Brazil"), key="pred_away")
    neutral = st.toggle("Neutral venue", value=True)
    if st.button("Predict", key="btn_predict"):
        with st.spinner("Predicting..."):
            result = api_get("/predict", {"home": home, "away": away, "neutral": str(neutral).lower()})
        if result:
            c  = result["combined"]
            ml = result["ml_model"]
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Combined (ML + Poisson)**")
                prob_bar(f"{home} Win",  c["win"],  "#00d26a")
                prob_bar("Draw",          c["draw"], "#f59e0b")
                prob_bar(f"{away} Win",  c["loss"], "#ef4444")
            with col_b:
                st.markdown("**ML Model only**")
                prob_bar(f"{home} Win",  ml["win"],  "#00d26a")
                prob_bar("Draw",          ml["draw"], "#f59e0b")
                prob_bar(f"{away} Win",  ml["loss"], "#ef4444")
            st.markdown(f"""
            <div class="stat-card" style="margin-top:1rem">
                <div class="label">FIFA Rankings</div>
                <div style="font-size:1rem;margin-top:0.3rem">
                    {home} <strong>#{result['home_rank']}</strong>
                    &nbsp;·&nbsp;
                    {away} <strong>#{result['away_rank']}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 5 — SCORE PREDICTOR
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### Score Predictor")
    st.caption("Dixon-Coles Poisson model with time-decayed attack/defense parameters")
    col1,col2,col3 = st.columns([2,1,2])
    with col1: s_home = st.selectbox("Home Team", ALL_TEAMS, index=ALL_TEAMS.index("Spain"), key="score_home")
    with col2: st.markdown("<br><div style='text-align:center;font-size:1.5rem;color:#6b7280'>VS</div>", unsafe_allow_html=True)
    with col3: s_away = st.selectbox("Away Team", ALL_TEAMS, index=ALL_TEAMS.index("Morocco"), key="score_away")
    if st.button("Predict Score", key="btn_score"):
        with st.spinner("Calculating..."):
            result = api_get("/predict/score", {"home": s_home, "away": s_away})
        if result:
            xg = result["expected_goals"]
            col_a,col_b,col_c = st.columns(3)
            with col_a:
                st.markdown(f'<div class="stat-card"><div class="label">{s_home} xG</div><div class="value">{xg["home"]}</div></div>', unsafe_allow_html=True)
            with col_b:
                st.markdown(f'<div class="stat-card"><div class="label">Most Likely Score</div><div class="value">{result["most_likely_score"]}</div></div>', unsafe_allow_html=True)
            with col_c:
                st.markdown(f'<div class="stat-card"><div class="label">{s_away} xG</div><div class="value">{xg["away"]}</div></div>', unsafe_allow_html=True)
            st.markdown("---")
            scores_df = pd.DataFrame(result["top_scorelines"])
            scores_df["probability_pct"] = (scores_df["probability"] * 100).round(1)
            fig = px.bar(scores_df, x="score", y="probability_pct", text="probability_pct",
                         color="probability_pct",
                         color_continuous_scale=[[0,"#1e1e2e"],[1,"#00d26a"]],
                         labels={"probability_pct":"Probability (%)","score":"Scoreline"})
            fig.update_traces(texttemplate="%{text}%", textposition="outside")
            fig.update_layout(plot_bgcolor="#13131a",paper_bgcolor="#13131a",
                              font_color="#f0f0f0",showlegend=False,coloraxis_showscale=False,
                              margin=dict(t=20,b=20),height=350)
            fig.update_xaxes(showgrid=False)
            fig.update_yaxes(showgrid=False,showticklabels=False)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("**Outcome probabilities (Poisson)**")
            prob_bar(f"{s_home} Win", result["win_prob"],  "#00d26a")
            prob_bar("Draw",           result["draw_prob"], "#f59e0b")
            prob_bar(f"{s_away} Win", result["loss_prob"], "#ef4444")

# ═══════════════════════════════════════════════════════════════
# TAB 6 — TOURNAMENT ODDS
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.markdown("### Tournament Win Probabilities")
    st.caption(f"Based on {sim.get('n_simulations',0):,} Monte Carlo simulations")
    if sim_results:
        df = pd.DataFrame([
            {"Team":t,"Win %":round(p*100,1),"Probability":p}
            for t,p in sim_results.items() if p>0
        ]).sort_values("Win %",ascending=False).reset_index(drop=True)
        df.index += 1
        col_chart,col_table = st.columns([3,2])
        with col_chart:
            top20 = df.head(20)
            fig = go.Figure(go.Bar(
                x=top20["Win %"], y=top20["Team"], orientation="h",
                marker=dict(color=top20["Win %"],colorscale=[[0,"#1e1e2e"],[0.5,"#059669"],[1,"#00d26a"]]),
                text=[f"{v}%" for v in top20["Win %"]],
                textposition="outside",textfont=dict(color="#f0f0f0",size=11),
            ))
            fig.update_layout(plot_bgcolor="#13131a",paper_bgcolor="#13131a",font_color="#f0f0f0",
                              yaxis=dict(autorange="reversed",showgrid=False),
                              xaxis=dict(showgrid=False,showticklabels=False),
                              margin=dict(l=10,r=60,t=10,b=10),height=500)
            st.plotly_chart(fig, use_container_width=True)
        with col_table:
            st.dataframe(df[["Team","Win %"]].head(20), hide_index=False,
                         use_container_width=True, height=500)
        history = api_get("/simulation/history") or []
        if len(history) > 1:
            st.markdown("---")
            st.markdown("### Probability Trend")
            top5 = list(sim_results.keys())[:5]
            trend_rows = []
            for run in history:
                for team in top5:
                    prob = run["top10"].get(team, 0)
                    trend_rows.append({"Run":run["run_at"][:10],"Team":team,"Win %":round(prob*100,1)})
            trend_df = pd.DataFrame(trend_rows)
            fig2 = px.line(trend_df, x="Run", y="Win %", color="Team",
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig2.update_layout(plot_bgcolor="#13131a",paper_bgcolor="#13131a",
                               font_color="#f0f0f0",margin=dict(t=10,b=10),height=300)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Run the simulation first.")

# ═══════════════════════════════════════════════════════════════
# TAB 7 — MODEL INFO
# ═══════════════════════════════════════════════════════════════
with tab7:
    st.markdown("### How the Models Work")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("""
**ML Ensemble Model**

- **Architecture:** Soft-voting ensemble of MLP, XGBoost, LightGBM
- **Training data:** 22,000+ international matches (2000–2022)
- **Test set:** 2023–2026 (time-based split)
- **AUC:** 0.74 · **Log-loss:** 0.88
- **Draw handling:** Class weight 2.0
- **Top features:** FIFA rank diff, H2H goal diff, neutral venue, confederation
        """)
    with col2:
        st.markdown("""
**Dixon-Coles Poisson Score Model**

- **Method:** MLE of attack/defense parameters per team
- **Training data:** 14,724 FIFA-ranked matches (2010–2026)
- **Time decay:** Exponential, 3-year half-life
- **DC correction:** ρ=−0.13 for low-scoring scorelines

| Team | Attack | Defense |
|------|--------|---------|
| Spain | 1.85 | 0.55 |
| Argentina | 1.58 | 0.37 |
| Morocco | 1.53 | 0.30 |
| France | 1.60 | 0.66 |
        """)
    st.markdown("---")
    st.markdown("""
**Tournament Simulation**

10,000 Monte Carlo simulations · ML + Poisson averaged 50/50

Group stage: round-robin · Pts → GD → GF tiebreaker · Best 8 third-place teams advance

Knockout: extra time → penalties (50/50) · Real completed matches locked in nightly
    """)
    st.markdown("---")
    st.markdown("### All 48 Teams — FIFA Rankings & Poisson Parameters")
    teams_data = api_get("/teams") or []
    if teams_data:
        teams_df = pd.DataFrame(teams_data)
        teams_df = teams_df.rename(columns={"team":"Team","group":"Group",
                                             "fifa_rank":"FIFA Rank",
                                             "attack_strength":"Attack",
                                             "defense_weakness":"Defense"})
        st.dataframe(teams_df, hide_index=True, use_container_width=True)