"""Rating prediction evaluation.

Compares three predictors on the temporal hold-out set:

  baseline_user_mean : predict the user's historical average rating
  lgbm               : LightGBM regressor (app/rating_model.py)
  llm_reasoner       : the LLM reasoner's predicted_rating
                       (runs on a SMALL sample because each prediction is one
                        Groq call)

Metrics: RMSE, MAE.
"""
from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from app import context as ctx_mod
from app import memory as memory_mod
from app import rating_model
from app.config import PROCESSED_DIR
from app.persona.coldstart import MIN_FOR_FULL_PERSONA
from app.persona.store import get_or_build as persona_for
from app.reasoner import reason


def _rmse(y_true: list[float], y_pred: list[float]) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true: list[float], y_pred: list[float]) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def evaluate_baseline(test: pd.DataFrame, train: pd.DataFrame) -> dict:
    user_means = train.groupby("user_id")["rating"].mean()
    preds, truths = [], []
    for _, row in test.iterrows():
        u = row["user_id"]
        preds.append(float(user_means.get(u, 3.5)))
        truths.append(float(row["rating"]))
    return {"variant": "baseline_user_mean", "n": len(test), "rmse": _rmse(truths, preds), "mae": _mae(truths, preds)}


def evaluate_lgbm(test: pd.DataFrame) -> dict:
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet").set_index("item_id")
    preds, truths = [], []
    skipped = 0
    for _, row in tqdm(test.iterrows(), total=len(test), desc="lgbm"):
        u, i = row["user_id"], row["item_id"]
        try:
            p = persona_for(u, refine=False)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        item_row = items.loc[i].to_dict() if i in items.index else {}
        pred = rating_model.predict(p, item_row)
        if pred is None:
            skipped += 1
            continue
        preds.append(pred)
        truths.append(float(row["rating"]))
    return {"variant": "lgbm", "n": len(preds), "skipped": skipped, "rmse": _rmse(truths, preds), "mae": _mae(truths, preds)}


def evaluate_llm(test: pd.DataFrame, sample_size: int = 30) -> dict:
    """LLM reasoner predictions on a small random sample."""
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet").set_index("item_id")
    rng = np.random.default_rng(42)
    sample = test.sample(n=min(sample_size, len(test)), random_state=42)
    preds, truths, latencies = [], [], []
    for _, row in tqdm(sample.iterrows(), total=len(sample), desc="llm_reasoner"):
        u, i = row["user_id"], row["item_id"]
        try:
            persona = persona_for(u, refine=True)
            memory = memory_mod.get_or_build(u)
            context = ctx_mod.normalize({})
            item_payload: dict[str, Any] = {
                "name": items.loc[i].get("sample_summary", i) if i in items.index else i,
                "item_id": i,
                "domain": items.loc[i].get("domain", "food") if i in items.index else "food",
                "avg_rating": items.loc[i].get("avg_rating", None) if i in items.index else None,
            }
            t0 = time.time()
            out = reason(persona, memory, context, item_payload)
            latencies.append(time.time() - t0)
            preds.append(float(out["predicted_rating"]))
            truths.append(float(row["rating"]))
        except Exception as e:  # noqa: BLE001
            print(f"  ! skipped {u}/{i}: {e}")
            continue
    return {
        "variant": "llm_reasoner",
        "n": len(preds),
        "rmse": _rmse(truths, preds) if preds else None,
        "mae": _mae(truths, preds) if preds else None,
        "avg_latency_sec": float(np.mean(latencies)) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm-sample", type=int, default=30)
    parser.add_argument("--skip-llm", action="store_true")
    args = parser.parse_args()

    print("[eval] loading splits…")
    train = pd.read_parquet(PROCESSED_DIR / "reviews_train.parquet")
    test = pd.read_parquet(PROCESSED_DIR / "reviews_test.parquet")
    print(f"[eval] train={len(train):,} test={len(test):,} (users={test['user_id'].nunique():,})")

    results: dict = {}

    print("[eval] baseline user-mean…")
    results["baseline_user_mean"] = evaluate_baseline(test, train)
    print(json.dumps(results["baseline_user_mean"], indent=2))

    print("[eval] LightGBM…")
    results["lgbm"] = evaluate_lgbm(test)
    print(json.dumps(results["lgbm"], indent=2))

    if not args.skip_llm:
        print(f"[eval] LLM reasoner (sample={args.llm_sample})…")
        results["llm_reasoner"] = evaluate_llm(test, sample_size=args.llm_sample)
        print(json.dumps(results["llm_reasoner"], indent=2))

    out_path = PROCESSED_DIR / "eval_rating.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
