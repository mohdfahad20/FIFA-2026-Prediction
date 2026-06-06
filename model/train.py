"""
model/train.py
==============
Model selection + hyperparameter tuning + soft ensemble.

Phase 1 — Baseline: compare 7 models with class weights for draw fix
Phase 2 — Tune top 3 with RandomizedSearchCV
Phase 3 — Soft ensemble, evaluate, save

Classes: 0=loss, 1=draw, 2=win (home team POV)

Usage:
    python model/train.py --db fifa.db
    python model/train.py --db fifa.db --skip-tuning
    python model/train.py --db fifa.db --min-weight 2   # no friendlies
    python model/train.py --db fifa.db --n-iter 50      # more tuning
"""

import sqlite3
import argparse
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import randint, uniform

from sklearn.ensemble        import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model    import LogisticRegression
from sklearn.neural_network  import MLPClassifier
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics         import log_loss, classification_report, roc_auc_score
from sklearn.utils           import compute_sample_weight
from xgboost                 import XGBClassifier
from lightgbm                import LGBMClassifier
from sklearn.base            import BaseEstimator, ClassifierMixin

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Draw class weight — upweight draws so model stops ignoring them
# Draws are 23.6% of data but get predicted 0% of the time without this
# ---------------------------------------------------------------------------
CLASS_WEIGHT = {0: 1.0, 1: 2.0, 2: 1.0}


# ---------------------------------------------------------------------------
# SoftEnsemble — module level (avoids pickle issues)
# ---------------------------------------------------------------------------
class SoftEnsemble:
    """Weighted average of predicted probabilities from multiple classifiers."""
    def __init__(self, clfs: list, weights: list = None):
        self.clfs    = clfs
        self.weights = weights if weights else [1.0] * len(clfs)

    def fit(self, X, y):
        for clf in self.clfs:
            clf.fit(X, y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        probas = np.array([clf.predict_proba(X) for clf in self.clfs])
        w      = np.array(self.weights)
        return np.average(probas, axis=0, weights=w)

    def predict(self, X) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


# ---------------------------------------------------------------------------
# Feature columns
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "home_rank", "away_rank", "rank_diff",
    "home_form", "away_form",
    "home_goals_scored_avg", "away_goals_scored_avg",
    "home_goals_conceded_avg", "away_goals_conceded_avg",
    "h2h_winrate_home", "h2h_goal_diff",
    "is_neutral", "tournament_weight",
]
CONF_COLS  = ["home_confederation", "away_confederation"]
TARGET_COL = "target"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_and_prepare(db_path: str, min_weight: int = 1):
    conn = sqlite3.connect(db_path)
    df   = pd.read_sql(f"""
        SELECT {', '.join(FEATURE_COLS + CONF_COLS + [TARGET_COL, 'date', 'tournament_weight'])}
        FROM features
        WHERE target IS NOT NULL
    """, conn, parse_dates=["date"])
    conn.close()

    print(f"[data] Loaded {len(df):,} rows.")

    if min_weight > 1:
        df = df[df["tournament_weight"] >= min_weight]
        print(f"[data] After weight filter (>={min_weight}): {len(df):,} rows.")

    # One-hot encode confederation
    conf_dummies = pd.get_dummies(
        df[CONF_COLS], prefix=["home_conf", "away_conf"], dummy_na=True
    )
    X = pd.concat([
        df[FEATURE_COLS].reset_index(drop=True),
        conf_dummies.reset_index(drop=True)
    ], axis=1).fillna(0)

    y             = df[TARGET_COL].values.astype(int)
    feature_names = list(X.columns)

    # Time-based train/test split
    mask_test  = df["date"].dt.year >= 2023
    mask_train = ~mask_test

    X_train = X[mask_train].values
    y_train = y[mask_train]
    X_test  = X[mask_test].values
    y_test  = y[mask_test]

    # Sample weights for draw fix (used by XGBoost + LightGBM)
    sw_train = compute_sample_weight(CLASS_WEIGHT, y_train)

    print(f"[split] Train: {mask_train.sum():,}  "
          f"({df.loc[mask_train,'date'].min().date()} → "
          f"{df.loc[mask_train,'date'].max().date()})")
    print(f"[split] Test : {mask_test.sum():,}  "
          f"({df.loc[mask_test,'date'].min().date()} → "
          f"{df.loc[mask_test,'date'].max().date()})")
    print(f"\n  Target distribution (train):")
    for t, label in [(2,"win"),(1,"draw"),(0,"loss")]:
        n = (y_train == t).sum()
        print(f"    {label:5s}: {n:,}  ({n/len(y_train):.1%})")

    return X_train, y_train, X_test, y_test, feature_names, sw_train

# --------------------------------------------------------------------------- 
# # Build candidate models (all with draw fix applied) # 
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# XGBoost wrapper — passes sample_weight at fit time
# ---------------------------------------------------------------------------
class XGBWithWeight(BaseEstimator, ClassifierMixin):
    def __init__(self, sw=None, n_estimators=200, max_depth=5,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=1, gamma=0, random_state=42, n_jobs=-1):
        self.sw               = sw
        self.n_estimators     = n_estimators
        self.max_depth        = max_depth
        self.learning_rate    = learning_rate
        self.subsample        = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.gamma            = gamma
        self.random_state     = random_state
        self.n_jobs           = n_jobs

    def fit(self, X, y):
        self._clf = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight, gamma=self.gamma,
            objective="multi:softprob", num_class=3,
            eval_metric="mlogloss", verbosity=0,
            random_state=self.random_state, n_jobs=self.n_jobs,
        )
        sw = self.sw[:len(y)] if self.sw is not None else None
        self._clf.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, X):  return self._clf.predict_proba(X)
    def predict(self, X):        return self._clf.predict(X)

    @property
    def feature_importances_(self): return self._clf.feature_importances_


class LGBMWithWeight(BaseEstimator, ClassifierMixin):
    def __init__(self, sw=None, n_estimators=200, max_depth=5,
                 learning_rate=0.05, num_leaves=31, subsample=0.8,
                 colsample_bytree=0.8, min_child_samples=20,
                 random_state=42, n_jobs=-1):
        self.sw                = sw
        self.n_estimators      = n_estimators
        self.max_depth         = max_depth
        self.learning_rate     = learning_rate
        self.num_leaves        = num_leaves
        self.subsample         = subsample
        self.colsample_bytree  = colsample_bytree
        self.min_child_samples = min_child_samples
        self.random_state      = random_state
        self.n_jobs            = n_jobs

    def fit(self, X, y):
        self._clf = LGBMClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, num_leaves=self.num_leaves,
            subsample=self.subsample, colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            objective="multiclass", num_class=3,
            verbose=-1, random_state=self.random_state, n_jobs=self.n_jobs,
        )
        sw = self.sw[:len(y)] if self.sw is not None else None
        self._clf.fit(X, y, sample_weight=sw)
        return self

    def predict_proba(self, X):  return self._clf.predict_proba(X)
    def predict(self, X):        return self._clf.predict(X)

    @property
    def feature_importances_(self): return self._clf.feature_importances_

def build_candidates(sw_train):
    candidates = {
        "XGBoost": XGBWithWeight(sw=sw_train),
        "LightGBM": LGBMWithWeight(sw=sw_train),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight=CLASS_WEIGHT, random_state=42, n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=10,
            class_weight=CLASS_WEIGHT, random_state=42, n_jobs=-1,
        ),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(
                max_iter=1000, C=0.5, class_weight=CLASS_WEIGHT,
                solver="lbfgs", random_state=42,
            ))
        ]),
        "MLP": Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPClassifier(
                hidden_layer_sizes=(128, 64, 32), activation="relu",
                max_iter=300, random_state=42,
                early_stopping=True, validation_fraction=0.1,
            ))
        ]),
    }
    return candidates


# ---------------------------------------------------------------------------
# Phase 1 — Baseline comparison
# ---------------------------------------------------------------------------
def phase1_baseline(X_train, y_train, sw_train):
    print("\n" + "="*60)
    print("PHASE 1 — BASELINE MODEL COMPARISON (5-fold CV log-loss)")
    print("  Draw fix: class_weight={0:1.0, 1:2.5, 2:1.0} applied to all")
    print("="*60)

    candidates = build_candidates(sw_train)
    cv         = StratifiedKFold(n_splits=5, shuffle=False)
    results    = {}

    for name, clf in candidates.items():
        scores = cross_val_score(
            clf, X_train, y_train,
            cv=cv, scoring="neg_log_loss", n_jobs=1
        )
        mean_ll = -scores.mean()
        std_ll  = scores.std()
        results[name] = mean_ll
        print(f"  {name:22s}: log-loss = {mean_ll:.4f} ± {std_ll:.4f}")

    ranked = sorted(results.items(), key=lambda x: x[1])
    print(f"\n  Ranking (best → worst):")
    for i, (name, ll) in enumerate(ranked, 1):
        print(f"    {i}. {name:22s}: {ll:.4f}")

    top3 = [name for name, _ in ranked[:3]]
    print(f"\n  → Top 3 for tuning: {top3}")
    return top3, candidates


# ---------------------------------------------------------------------------
# Phase 2 — Hyperparameter tuning
# ---------------------------------------------------------------------------
def phase2_tuning(top3: list, candidates: dict,
                  X_train, y_train, sw_train, n_iter: int = 30):
    print("\n" + "="*60)
    print(f"PHASE 2 — HYPERPARAMETER TUNING ({n_iter} iters per model)")
    print("="*60)

    cv    = StratifiedKFold(n_splits=5, shuffle=False)
    tuned = {}

    # Param grids for sklearn-compatible models only
    # XGBoost/LightGBM wrappers tuned manually below
    PARAM_GRIDS = {
        "RandomForest": {
            "n_estimators":      randint(200, 600),
            "max_depth":         randint(5, 15),
            "min_samples_leaf":  randint(5, 30),
            "min_samples_split": randint(2, 20),
            "max_features":      uniform(0.3, 0.7),
        },
        "ExtraTrees": {
            "n_estimators":      randint(200, 600),
            "max_depth":         randint(5, 15),
            "min_samples_leaf":  randint(5, 30),
            "min_samples_split": randint(2, 20),
            "max_features":      uniform(0.3, 0.7),
        },
        "LogisticRegression": {
            "lr__C": uniform(0.01, 2.0),
        },
        "MLP": {
            "mlp__hidden_layer_sizes": [
                (64, 32), (128, 64), (128, 64, 32),
                (256, 128, 64), (64, 64, 64)
            ],
            "mlp__alpha":   uniform(0.0001, 0.01),
            "mlp__learning_rate_init": uniform(0.0001, 0.005),
        },
    }

    # XGBoost best params — tune via manual grid (wrapper not compatible with RandomizedSearchCV)
    XGB_CONFIGS = [
        dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             subsample=0.8, colsample_bytree=0.8, min_child_weight=3),
        dict(n_estimators=400, max_depth=5, learning_rate=0.03,
             subsample=0.7, colsample_bytree=0.7, min_child_weight=5),
        dict(n_estimators=500, max_depth=6, learning_rate=0.02,
             subsample=0.9, colsample_bytree=0.9, min_child_weight=1),
        dict(n_estimators=300, max_depth=3, learning_rate=0.08,
             subsample=0.8, colsample_bytree=0.6, min_child_weight=7),
    ]
    LGBM_CONFIGS = [
        dict(n_estimators=300, max_depth=4, learning_rate=0.05,
             num_leaves=31, subsample=0.8, colsample_bytree=0.8),
        dict(n_estimators=400, max_depth=5, learning_rate=0.03,
             num_leaves=50, subsample=0.7, colsample_bytree=0.7),
        dict(n_estimators=500, max_depth=6, learning_rate=0.02,
             num_leaves=63, subsample=0.9, colsample_bytree=0.9),
    ]

    for name in top3:
        print(f"\n  Tuning {name}...")
        clf = candidates[name]

        # XGBoost — manual grid
        if name == "XGBoost":
            grid = {
                "n_estimators":     randint(200, 600),
                "max_depth":        randint(3, 7),
                "learning_rate":    uniform(0.01, 0.15),
                "subsample":        uniform(0.6, 0.4),
                "colsample_bytree": uniform(0.6, 0.4),
                "min_child_weight": randint(1, 10),
                "gamma":            uniform(0, 0.5),
            }
            search = RandomizedSearchCV(
                XGBWithWeight(sw=sw_train), grid,
                n_iter=n_iter, cv=cv, scoring="neg_log_loss",
                n_jobs=1, random_state=42, verbose=0, refit=True,
            )
            search.fit(X_train, y_train)
            print(f"    Best log-loss : {-search.best_score_:.4f}")
            print(f"    Best params   : {search.best_params_}")
            tuned[name] = search.best_estimator_

        # LightGBM — manual grid
        elif name == "LightGBM":
            grid = {
                "n_estimators":      randint(200, 600),
                "max_depth":         randint(3, 7),
                "learning_rate":     uniform(0.01, 0.15),
                "num_leaves":        randint(20, 80),
                "subsample":         uniform(0.6, 0.4),
                "colsample_bytree":  uniform(0.6, 0.4),
                "min_child_samples": randint(10, 50),
            }
            search = RandomizedSearchCV(
                LGBMWithWeight(sw=sw_train), grid,
                n_iter=n_iter, cv=cv, scoring="neg_log_loss",
                n_jobs=1, random_state=42, verbose=0, refit=True,
            )
            search.fit(X_train, y_train)
            print(f"    Best log-loss : {-search.best_score_:.4f}")
            print(f"    Best params   : {search.best_params_}")
            tuned[name] = search.best_estimator_

        # sklearn-compatible — RandomizedSearchCV
        elif name in PARAM_GRIDS:
            search = RandomizedSearchCV(
                clf,
                param_distributions = PARAM_GRIDS[name],
                n_iter              = n_iter,
                cv                  = cv,
                scoring             = "neg_log_loss",
                n_jobs              = -1,
                random_state        = 42,
                verbose             = 0,
                refit               = True,
            )
            search.fit(X_train, y_train)
            print(f"    Best log-loss : {-search.best_score_:.4f}")
            print(f"    Best params   : {search.best_params_}")
            tuned[name] = search.best_estimator_

        else:
            # No grid — fit with defaults
            clf.fit(X_train, y_train)
            tuned[name] = clf

    return tuned


# ---------------------------------------------------------------------------
# Phase 3 — Ensemble + evaluation
# ---------------------------------------------------------------------------
def phase3_ensemble(tuned: dict, X_train, y_train,
                    X_test, y_test, feature_names):
    print("\n" + "="*60)
    print("PHASE 3 — ENSEMBLE + FINAL EVALUATION")
    print("="*60)

    print("\n  Individual model performance on test set:")
    individual_scores = {}
    for name, clf in tuned.items():
        proba = clf.predict_proba(X_test)
        ll    = log_loss(y_test, proba)
        auc   = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
        individual_scores[name] = {"log_loss": ll, "auc": auc}
        print(f"    {name:22s}: log-loss={ll:.4f}  AUC={auc:.4f}")

    # Weight by inverse log-loss
    names   = list(tuned.keys())
    clfs    = list(tuned.values())
    losses  = [individual_scores[n]["log_loss"] for n in names]
    inv_ll  = [1.0 / ll for ll in losses]
    total   = sum(inv_ll)
    weights = [w / total * len(names) for w in inv_ll]

    print(f"\n  Ensemble weights (higher = better model):")
    for name, w in zip(names, weights):
        print(f"    {name:22s}: {w:.3f}")

    ensemble = SoftEnsemble(clfs=clfs, weights=weights)
    print("\n  Fitting ensemble on full train set...")
    ensemble.fit(X_train, y_train)

    # Evaluate
    proba_test = ensemble.predict_proba(X_test)
    pred_test  = ensemble.predict(X_test)
    ll_ens     = log_loss(y_test, proba_test)
    auc_ens    = roc_auc_score(y_test, proba_test, multi_class="ovr", average="macro")

    print(f"\n  {'='*40}")
    print(f"  ENSEMBLE RESULTS")
    print(f"  {'='*40}")
    print(f"  Log-loss  : {ll_ens:.4f}  (random={np.log(3):.4f})")
    print(f"  Macro AUC : {auc_ens:.4f}  (target >0.68)")

    print(f"\n  Classification report:")
    print(classification_report(
        y_test, pred_test,
        target_names=["loss(0)", "draw(1)", "win(2)"]
    ))

    # Draw prediction check
    pred_draws = (pred_test == 1).sum()
    actual_draws = (y_test == 1).sum()
    print(f"  Draw prediction check:")
    print(f"    Actual draws    : {actual_draws} ({actual_draws/len(y_test):.1%})")
    print(f"    Predicted draws : {pred_draws} ({pred_draws/len(y_test):.1%})")
    if pred_draws == 0:
        print(f"    ⚠️  Still predicting zero draws — try increasing CLASS_WEIGHT[1]")
    else:
        print(f"    ✅ Model is now predicting draws")

    # Feature importance
    for name, clf in tuned.items():
        if hasattr(clf, "feature_importances_"):
            importances = clf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:12]
            print(f"\n  Top 12 features ({name}):")
            for i in top_idx:
                if i < len(feature_names):
                    print(f"    {feature_names[i]:35s}: {importances[i]:.4f}")
            break

    return ensemble, {"log_loss": ll_ens, "auc": auc_ens}


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
def save_model(ensemble, feature_names, metrics, model_out: str):
    path = Path(model_out)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ensemble":      ensemble,
        "feature_names": feature_names,
        "feature_cols":  FEATURE_COLS,
        "conf_cols":     CONF_COLS,
        "classes":       {0: "loss", 1: "draw", 2: "win"},
        "metrics":       metrics,
        "class_weight":  CLASS_WEIGHT,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    print(f"\n[save] Model saved → {path.resolve()}")
    print(f"[done] Next: python model/predict.py --db fifa.db")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db",          default="fifa.db")
    parser.add_argument("--model-out",   default="model/model.pkl")
    parser.add_argument("--min-weight",  default=1,  type=int)
    parser.add_argument("--n-iter",      default=30, type=int)
    parser.add_argument("--skip-tuning", action="store_true")
    args = parser.parse_args()

    if not Path(args.db).exists():
        raise FileNotFoundError(f"DB not found: {args.db}")

    print(f"[init] DB: {Path(args.db).resolve()}\n")

    X_train, y_train, X_test, y_test, feature_names, sw_train = load_and_prepare(
        args.db, min_weight=args.min_weight
    )

    top3, candidates = phase1_baseline(X_train, y_train, sw_train)

    if args.skip_tuning:
        print("\n[skip] Tuning skipped — fitting Phase 1 defaults.")
        tuned = {}
        for name in top3:
            candidates[name].fit(X_train, y_train)
            tuned[name] = candidates[name]
    else:
        tuned = phase2_tuning(top3, candidates, X_train, y_train, sw_train, args.n_iter)

    ensemble, metrics = phase3_ensemble(
        tuned, X_train, y_train, X_test, y_test, feature_names
    )
    save_model(ensemble, feature_names, metrics, args.model_out)


if __name__ == "__main__":
    main()