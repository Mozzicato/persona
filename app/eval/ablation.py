"""Ablation runner.

For each ablation variant we measure on the temporal hold-out split:

Cheap variants (no LLM, runs on full test set):
  - retrieval_only_semantic     : TF-IDF only (no CF, no quality, no popularity)
  - retrieval_only_cf           : item-item CF only
  - retrieval_hybrid_no_cf      : sem + quality + popularity (CF off)
  - retrieval_full              : all four retrieval signals
  - lgbm_no_temporal            : LightGBM without temporal features
  - lgbm_no_sensitivities       : LightGBM without packaging/service sensitivities
  - lgbm_full                   : LightGBM with all features

Expensive variants (LLM, runs on small sample):
  - full_no_memory              : reasoner runs without short-term memory
  - full_no_planner             : recommender uses fixed weights
  - full_system                 : baseline (everything on)

Run:
    python -m app.eval.ablation
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from tqdm import tqdm

from app.config import PROCESSED_DIR
from app.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k
from app.persona.coldstart import MIN_FOR_FULL_PERSONA
from app.persona.store import get_or_build as persona_for
from app.retrieval import retrieve


def _retrieval_with_weights(train: pd.DataFrame, test: pd.DataFrame, weights: tuple, k: int = 10, candidates_k: int = 50) -> dict:
    pos = test[test["rating"] >= 4]
    users = pos["user_id"].unique().tolist()
    ndcgs, hits, mrrs = [], [], []
    skipped = 0
    for uid in tqdm(users, desc=f"weights={weights}"):
        rel = set(pos[pos["user_id"] == uid]["item_id"])
        if not rel:
            continue
        try:
            user_hist = train[train["user_id"] == uid]
            cands = retrieve(uid, top_k=candidates_k, weights=weights, history_df=user_hist)
            ranked = [c["item_id"] for c in cands]
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        ndcgs.append(ndcg_at_k(ranked, rel, k))
        hits.append(hit_at_k(ranked, rel, k))
        mrrs.append(mrr_at_k(ranked, rel, k))
    return {
        "n": len(ndcgs),
        "skipped": skipped,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit@{k}":  float(np.mean(hits)) if hits else 0.0,
        f"mrr@{k}":  float(np.mean(mrrs)) if mrrs else 0.0,
    }


def _rating_ablation_lgbm(feature_subset: list[str], train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Train a fresh LightGBM with a subset of features, measure RMSE/MAE on test."""
    from app.persona.features import extract
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    from app.rating_model import _row_features, FEATURE_ORDER

    items = pd.read_parquet(PROCESSED_DIR / "items.parquet").set_index("item_id")

    print(f"  building personas for {train['user_id'].nunique()} train users…")
    personas = {uid: extract(uid, grp).to_dict() for uid, grp in train.groupby("user_id")}

    def make_X(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        rows, targets, missing = [], [], 0
        for _, r in df.iterrows():
            p = personas.get(r["user_id"])
            if p is None:
                missing += 1
                continue
            item_row = items.loc[r["item_id"]].to_dict() if r["item_id"] in items.index else {}
            feats = _row_features(p, item_row)
            rows.append({k: feats[k] for k in feature_subset})
            targets.append(r["rating"])
        return pd.DataFrame(rows).astype(np.float32), np.asarray(targets, dtype=np.float32)

    X_tr, y_tr = make_X(train)
    X_te, y_te = make_X(test)
    if len(X_tr) == 0 or len(X_te) == 0:
        return {"variant": ",".join(feature_subset), "rmse": None, "mae": None, "note": "no rows"}
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.05, num_leaves=31, n_jobs=-1, verbose=-1)
    model.fit(X_tr, y_tr)
    pred = np.clip(model.predict(X_te), 1.0, 5.0)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    mae = float(np.mean(np.abs(pred - y_te)))
    return {"n_features": len(feature_subset), "n_train": int(len(X_tr)), "n_test": int(len(X_te)), "rmse": rmse, "mae": mae}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--skip-lgbm", action="store_true")
    args = parser.parse_args()

    train = pd.read_parquet(PROCESSED_DIR / "reviews_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "reviews_test.parquet")
    k = args.k
    results: dict = {"split": {"train": int(len(train)), "test": int(len(test))}}

    print("\n=== Retrieval ablations ===")
    print("[ablation] retrieval — full hybrid (sem+CF+quality+pop)")
    results["retrieval_full"] = _retrieval_with_weights(train, test, weights=(0.45, 0.30, 0.15, 0.10), k=k)

    print("[ablation] retrieval — semantic only")
    results["retrieval_only_semantic"] = _retrieval_with_weights(train, test, weights=(1.0, 0.0, 0.0, 0.0), k=k)

    print("[ablation] retrieval — CF only")
    results["retrieval_only_cf"] = _retrieval_with_weights(train, test, weights=(0.0, 1.0, 0.0, 0.0), k=k)

    print("[ablation] retrieval — hybrid WITHOUT CF (sem+quality+pop)")
    results["retrieval_no_cf"] = _retrieval_with_weights(train, test, weights=(0.6, 0.0, 0.25, 0.15), k=k)

    print("[ablation] retrieval — popularity only (baseline)")
    results["retrieval_popularity_only"] = _retrieval_with_weights(train, test, weights=(0.0, 0.0, 0.0, 1.0), k=k)

    if not args.skip_lgbm:
        print("\n=== LightGBM rating ablations ===")
        from app.rating_model import FEATURE_ORDER

        print("[ablation] LightGBM — full features")
        results["lgbm_full"] = _rating_ablation_lgbm(FEATURE_ORDER, train, test)
        print(json.dumps(results["lgbm_full"], indent=2))

        no_temporal = [f for f in FEATURE_ORDER if not f.startswith("user_night") and not f.startswith("user_weekend")]
        print("[ablation] LightGBM — without temporal features")
        results["lgbm_no_temporal"] = _rating_ablation_lgbm(no_temporal, train, test)
        print(json.dumps(results["lgbm_no_temporal"], indent=2))

        no_sens = [f for f in FEATURE_ORDER if "packaging_sensitivity" not in f and "service_sensitivity" not in f]
        print("[ablation] LightGBM — without packaging/service sensitivities")
        results["lgbm_no_sensitivities"] = _rating_ablation_lgbm(no_sens, train, test)
        print(json.dumps(results["lgbm_no_sensitivities"], indent=2))

        minimal = ["user_avg_rating", "user_std_rating", "item_avg_rating", "rating_delta"]
        print("[ablation] LightGBM — minimal (4 features only)")
        results["lgbm_minimal"] = _rating_ablation_lgbm(minimal, train, test)
        print(json.dumps(results["lgbm_minimal"], indent=2))

    out = PROCESSED_DIR / "eval_ablation.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[ablation] wrote {out}")
    print(json.dumps({k: v for k, v in results.items() if k != "split"}, indent=2))


if __name__ == "__main__":
    main()
