"""LightGBM rating predictor trained on (persona x item) features.

Used as a fast deterministic baseline alongside the LLM reasoner's prediction.
The final API blends both.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import MODELS_DIR, PROCESSED_DIR
from app.persona.features import extract

MODEL_PATH = MODELS_DIR / "rating_lgbm.pkl"
FEATURE_ORDER = [
    "user_avg_rating",
    "user_std_rating",
    "user_n_reviews",
    "user_harshness",
    "user_optimism",
    "user_consistency",
    "user_price_focus",
    "user_delivery_sensitivity",
    "user_packaging_sensitivity",
    "user_service_sensitivity",
    "user_quality_sensitivity",
    "user_emotional_intensity",
    "user_night_share",
    "user_weekend_positivity",
    "item_avg_rating",
    "item_n_reviews",
    "rating_delta",  # user_avg - item_avg, predictive of bias
]


def _row_features(persona: dict, item_row: dict) -> dict:
    comm = persona.get("communication_style", {})
    behav = persona.get("behavioral_profile", {})
    econ = persona.get("economic_profile", {})
    temp = persona.get("temporal_profile", {})
    stats = persona.get("stats", {})
    user_avg = stats.get("avg_rating", 3.0)
    item_avg = item_row.get("avg_rating", 3.0)
    hour_dist = temp.get("hour_distribution", {}) or {}
    return {
        "user_avg_rating": user_avg,
        "user_std_rating": stats.get("std_rating", 1.0),
        "user_n_reviews": stats.get("n_reviews", 0),
        "user_harshness": behav.get("harshness", 0.5),
        "user_optimism": behav.get("optimism", 0.5),
        "user_consistency": behav.get("consistency", 0.5),
        "user_price_focus": econ.get("price_focus", 0.0),
        "user_delivery_sensitivity": behav.get("delivery_sensitivity", 0.0),
        "user_packaging_sensitivity": behav.get("packaging_sensitivity", 0.0),
        "user_service_sensitivity": behav.get("service_sensitivity", 0.0),
        "user_quality_sensitivity": behav.get("quality_sensitivity", 0.0),
        "user_emotional_intensity": comm.get("emotional_intensity", 0.3),
        "user_night_share": hour_dist.get("late_night", 0.0),
        "user_weekend_positivity": temp.get("weekend_positivity", 0.0),
        "item_avg_rating": item_avg,
        "item_n_reviews": item_row.get("n_reviews", 0),
        "rating_delta": user_avg - item_avg,
    }


def train_and_save() -> dict:
    """Train on the processed reviews. Returns metrics dict."""
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split

    reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet").set_index("item_id")

    # Build per-user persona once
    print("[train] extracting personas...")
    personas: dict[str, dict] = {}
    for uid, grp in reviews.groupby("user_id"):
        personas[uid] = extract(uid, grp).to_dict()

    rows = []
    targets = []
    print("[train] building feature matrix...")
    for _, r in reviews.iterrows():
        p = personas[r["user_id"]]
        item_row = items.loc[r["item_id"]].to_dict() if r["item_id"] in items.index else {}
        feats = _row_features(p, item_row)
        rows.append([feats[k] for k in FEATURE_ORDER])
        targets.append(r["rating"])
    X = pd.DataFrame(rows, columns=FEATURE_ORDER).astype(np.float32)
    y = np.array(targets, dtype=np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True)
    model = lgb.LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_data_in_leaf=20,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_te, y_te)], callbacks=[lgb.early_stopping(20)])
    pred = model.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae = float(np.mean(np.abs(pred - y_te)))

    joblib.dump({"model": model, "feature_order": FEATURE_ORDER}, MODEL_PATH)
    print(f"[train] RMSE={rmse:.3f}  MAE={mae:.3f}  -> saved {MODEL_PATH}")
    return {"rmse": rmse, "mae": mae, "n_train": len(X_tr), "n_test": len(X_te)}


def _load():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


def predict(persona: dict, item_row: dict) -> float | None:
    bundle = _load()
    if bundle is None:
        return None
    model = bundle["model"]
    order = bundle["feature_order"]
    feats = _row_features(persona, item_row)
    x = pd.DataFrame([{k: feats[k] for k in order}])
    raw = float(model.predict(x)[0])
    return float(np.clip(raw, 1.0, 5.0))


if __name__ == "__main__":
    train_and_save()
