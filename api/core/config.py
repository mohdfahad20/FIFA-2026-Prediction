from pathlib import Path

BASE_DIR     = Path(__file__).resolve().parent.parent.parent
DB_PATH      = BASE_DIR / "fifa.db"
MODEL_PATH   = BASE_DIR / "model" / "model.pkl"
POISSON_PATH = BASE_DIR / "score_model" / "poisson_params.pkl"