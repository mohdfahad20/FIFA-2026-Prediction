# ⚽ FIFA World Cup 2026 — ML Prediction System

![Python](https://img.shields.io/badge/Python-3.13-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-latest-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An end-to-end machine learning system for predicting FIFA World Cup 2026 match outcomes, simulating the full tournament, and serving predictions via a REST API and interactive dashboard.

---

## 🏆 Live Demo

| Service | URL |
|---------|-----|
| Dashboard | [Streamlit Cloud](#) |
| API | [Render](#) |
| API Docs | [Render/docs](#) |

---

## 📸 Features

- **Match Outcome Prediction** — Win/Draw/Loss probabilities using an MLP + XGBoost + LightGBM soft-voting ensemble (AUC 0.74)
- **Score Prediction** — Expected goals and most likely scorelines using a Dixon-Coles Poisson model with time decay
- **Tournament Simulation** — 10,000-run Monte Carlo simulation of all 104 matches, updated nightly
- **Live Standings** — Group tables computed from real match results as the tournament progresses
- **Nightly Pipeline** — GitHub Actions scrapes results every night, retrains Poisson model, re-simulates, and redeploys the API automatically

---

## 🧠 Model Architecture

```
Historical Data (49k matches, 70k rankings rows)
        ↓
Feature Engineering (23,545 rows, 14 features)
        ↓
┌─────────────────────────────────────┐
│         Soft Voting Ensemble        │
│  MLP (1.020) + XGB (0.992)          │
│           + LGBM (0.988)            │
└─────────────────────────────────────┘
        ↓                    ↓
  3-Class Output       Dixon-Coles
  Win / Draw / Loss    Poisson Model
        ↓                    ↓
         50/50 Combined Prediction
                  ↓
        Monte Carlo Simulation
           (10,000 runs)
```

**Key metrics:**
- Macro AUC: **0.7407** (target >0.68)
- Log-loss: **0.8817** (random baseline: 1.0986)
- Draw prediction: 790 predicted vs 776 actual ✅

---

## 🗂️ Project Structure

```
FIFA-2026-Prediction/
│
├── data/
│   ├── results.csv                    ← Kaggle: 49k international matches
│   ├── fifa_ranking-2026-04-01.csv    ← Kaggle: FIFA rankings to Apr 2026
│   ├── load_historical_data.py        ← Loads CSVs into SQLite
│   └── build_schedule.py             ← WC 2026 schedule (hardcoded)
│
├── features/
│   └── features.py                   ← Vectorised feature builder
│
├── model/
│   ├── train.py                      ← 3-phase training (baseline→tune→ensemble)
│   ├── predict.py                    ← predict_match() wrapper
│   └── model.pkl                     ← generated (not committed)
│
├── score_model/
│   ├── train_poisson.py              ← Dixon-Coles with time decay
│   ├── predict_score.py              ← scoreline + xG predictions
│   └── poisson_params.pkl            ← generated (not committed)
│
├── simulate/
│   └── simulate.py                   ← Full 104-match Monte Carlo simulator
│
├── api/
│   ├── main.py                       ← FastAPI app (9 endpoints)
│   ├── startup.py                    ← Downloads artifacts on Render boot
│   ├── core/                         ← DB, model loader, config
│   ├── routers/                      ← predict, simulation, standings, teams
│   └── services/                     ← Business logic per router
│
├── dashboard/
│   ├── app.py                        ← Streamlit (API mode — Streamlit Cloud)
│   └── app_local.py                  ← Streamlit (local DB mode)
│
├── scraper/
│   └── scraper.py                    ← Fetches WC results from football-data.org
│
├── run_pipeline.py                   ← Local pipeline runner (all 5 steps)
├── render.yaml                       ← Render deployment config
├── requirements.txt
└── .github/workflows/
    └── update_fifa_2026.yml          ← Nightly CI/CD pipeline
```

---

## ⚙️ Local Setup

### Prerequisites
- Python 3.13+
- Kaggle account (for data download)
- football-data.org API key (free)

### Installation

```bash
git clone https://github.com/mohdfahad20/FIFA-2026-Prediction.git
cd FIFA-2026-Prediction

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:
```
FOOTBALL_DATA_API_KEY=your_key_here
```

### Run the full pipeline

```bash
# Full pipeline from scratch (~8 minutes, ML training dominates)
python run_pipeline.py

# Skip data loading if fifa.db already exists
python run_pipeline.py --skip-data

# Quick test (1000 sims, skip ML retraining)
python run_pipeline.py --skip-data --skip-ml --n 1000
```

### Run locally

```bash
# Terminal 1 — API
uvicorn api.main:app --reload --port 8000

# Terminal 2 — Dashboard
streamlit run dashboard/app_local.py
```

Open `http://localhost:8501` for the dashboard, `http://localhost:8000/docs` for the API.

---

## 🔌 API Endpoints

Base URL: `https://your-render-url.onrender.com`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Status, matches played, last simulation |
| GET | `/predict?home=France&away=Brazil` | Win/Draw/Loss probabilities |
| GET | `/predict/score?home=France&away=Brazil` | xG, scorelines, outcome probs |
| GET | `/simulation` | Latest tournament simulation (all 48 teams) |
| GET | `/simulation/history` | Simulation trend over time |
| GET | `/standings` | All 12 group tables |
| GET | `/standings/{A-L}` | Single group table |
| GET | `/teams` | All 48 teams with rank + Poisson params |
| GET | `/teams/{name}` | Team detail, fixtures, form |
| GET | `/teams/h2h?team1=France&team2=Brazil` | Head-to-head history |

---

## 🔄 Nightly Pipeline (GitHub Actions)

Runs at **02:00 UTC daily** (07:30 IST) throughout the tournament:

```
Restore DB from GitHub Release
        ↓
Scrape yesterday's results (football-data.org)
        ↓
Rebuild features table
        ↓
Retrain Poisson model
        ↓
Run 10,000 Monte Carlo simulations
        ↓
Upload artifacts.zip to GitHub Release
        ↓
Trigger Render redeploy
```

### Required GitHub Secrets

| Secret | Description |
|--------|-------------|
| `FOOTBALL_DATA_API_KEY` | Free key from football-data.org |
| `RENDER_DEPLOY_HOOK_FIFA` | Render service deploy hook URL |
| `GITHUB_TOKEN` | Auto-provided by Actions |

---

## 📊 WC 2026 Groups

| Group | Teams |
|-------|-------|
| A | Mexico, South Africa, South Korea, Czechia |
| B | Canada, Switzerland, Qatar, Bosnia and Herzegovina |
| C | Brazil, Morocco, Haiti, Scotland |
| D | United States, Paraguay, Australia, Turkey |
| E | Germany, Curaçao, Ivory Coast, Ecuador |
| F | Netherlands, Japan, Tunisia, Sweden |
| G | Belgium, Egypt, Iran, New Zealand |
| H | Spain, Cape Verde, Saudi Arabia, Uruguay |
| I | France, Senegal, Norway, Iraq |
| J | Argentina, Algeria, Austria, Jordan |
| K | Portugal, Uzbekistan, Colombia, DR Congo |
| L | England, Croatia, Ghana, Panama |

---

## 🏅 Pre-Tournament Simulation Results

Top 10 teams by championship probability (10,000 simulations, pre-tournament):

| Team | Win % |
|------|-------|
| Argentina | 8.9% |
| Spain | 7.5% |
| England | 7.2% |
| Morocco | 5.8% |
| Portugal | 5.5% |
| Japan | 5.1% |
| France | 4.1% |
| Brazil | 4.0% |
| Belgium | 3.7% |
| Norway | 3.6% |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.13 |
| Database | SQLite |
| ML Models | scikit-learn, XGBoost, LightGBM |
| Score Model | SciPy (Poisson / Dixon-Coles) |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Plotly |
| Live Data | football-data.org API |
| CI/CD | GitHub Actions |
| API Hosting | Render |
| Dashboard Hosting | Streamlit Cloud |

---

## 📁 Data Sources

- [Kaggle — International Football Results 1872-2026](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
- [Kaggle — FIFA World Rankings](https://www.kaggle.com/datasets/cashncarry/fifaworldranking)
- [football-data.org](https://www.football-data.org/) — Live WC 2026 results

---

## 🙏 Acknowledgements

Modelled after a prior IPL 2026 Live Prediction System. Reuses the same FastAPI + Streamlit + GitHub Actions + Render infrastructure pattern.

---

*FIFA WC 2026 · June 11 – July 19, 2026 · USA, Canada, Mexico*