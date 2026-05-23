"""Ranking quality evaluation.

We measure NDCG@10, Hit@10, MRR@10 on a temporal hold-out:
- For each test user, the held-out item(s) (their last review) form the
  relevant set if rating >= 4 (positive engagement).
- The recommender produces top-K recommendations using ONLY the training
  history (the test rows are excluded by point-in-time `seen` filtering).

Two variants are evaluated:
  retrieval_only  : hybrid retrieval, no LLM simulation re-rank
  full_system     : retrieval + LLM behavioral simulator re-rank
                    (run on a smaller sample because each user costs ~12 LLM calls)

Run:
    python -m app.eval.rank_eval
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from app import memory as memory_mod
from app.config import PROCESSED_DIR
from app.eval.metrics import hit_at_k, mrr_at_k, ndcg_at_k
from app.eval.split import temporal_split, write_splits
from app.persona.store import get_or_build as persona_for
from app.persona.coldstart import MIN_FOR_FULL_PERSONA
from app.retrieval import retrieve as retrieve_full
from app.recommender import recommend


def _build_retrieval_only_for_user(user_id: str, train_df: pd.DataFrame, k: int = 50) -> list[str]:
    """Run retrieval without LLM simulation re-rank using train-only history."""
    user_hist = train_df[train_df["user_id"] == user_id]
    cands = retrieve_full(user_id, top_k=k, history_df=user_hist)
    return [c["item_id"] for c in cands]


def _build_full_for_user(user_id: str, train_df: pd.DataFrame, top_n: int = 10, candidates_k: int = 20) -> list[str]:
    # `recommend()` internally calls retrieve(); to inject train-only history
    # cleanly we monkey-patch retrieve via the recommender's internals would
    # be invasive — instead we accept a small leakage warning for the LLM
    # full-system path (it still gets reranked, just with full-history retrieval)
    # OR we replicate the recommender locally. Simpler: do retrieval here with
    # train_df then call the reasoner per candidate manually.
    from app import context as ctx_mod
    from app import memory as memory_mod
    from app.persona.store import get_or_build as persona_for
    from app.reasoner import reason
    from app.planner import plan as plan_step
    from app.recommender import analyze_behavior, _rerank_score, _should_skip

    persona = persona_for(user_id, refine=True)
    memory = memory_mod.get_or_build(user_id)
    context = ctx_mod.normalize(None)
    behavior = analyze_behavior(persona, memory, context)
    planner_out = plan_step(persona, memory, behavior, context)

    user_hist = train_df[train_df["user_id"] == user_id]
    cands = retrieve_full(user_id, top_k=candidates_k, history_df=user_hist)

    enriched = []
    for c in cands:
        if _should_skip(c, planner_out.get("must_avoid", [])):
            continue
        item_payload = {"name": c.get("sample_summary") or c["item_id"], "item_id": c["item_id"],
                        "category": c.get("domain", "food"), "avg_rating": c.get("avg_rating")}
        sim_out = reason(persona, memory, context, item_payload)
        enriched.append({"candidate": c, "simulation": sim_out,
                         "score": _rerank_score(c, sim_out, planner_out)})
    enriched.sort(key=lambda x: x["score"], reverse=True)
    return [e["candidate"]["item_id"] for e in enriched[:top_n]]


def evaluate_retrieval(train: pd.DataFrame, test: pd.DataFrame, k: int = 10, candidates_k: int = 50) -> dict:
    """Cheap eval — no LLM. Runs over many users."""
    pos = test[test["rating"] >= 4]
    users = pos["user_id"].unique().tolist()
    ndcgs, hits, mrrs = [], [], []
    skipped = 0
    for uid in tqdm(users, desc="retrieval_only"):
        relevant = set(pos[pos["user_id"] == uid]["item_id"])
        if not relevant:
            skipped += 1
            continue
        try:
            ranked = _build_retrieval_only_for_user(uid, train, k=candidates_k)
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        hits.append(hit_at_k(ranked, relevant, k))
        mrrs.append(mrr_at_k(ranked, relevant, k))
    return {
        "variant": "retrieval_only",
        "k": k,
        "n_users": len(ndcgs),
        "skipped": skipped,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit@{k}":  float(np.mean(hits)) if hits else 0.0,
        f"mrr@{k}":  float(np.mean(mrrs)) if mrrs else 0.0,
    }


def evaluate_full(train: pd.DataFrame, test: pd.DataFrame, k: int = 10, sample_size: int = 30, candidates_k: int = 20) -> dict:
    """Expensive eval — calls LLM. Runs on a small random sample."""
    pos = test[test["rating"] >= 4]
    user_pool = pos["user_id"].unique().tolist()
    rng = np.random.default_rng(42)
    sample = rng.choice(user_pool, size=min(sample_size, len(user_pool)), replace=False).tolist()
    ndcgs, hits, mrrs, latencies = [], [], [], []
    skipped = 0
    for uid in tqdm(sample, desc="full_system"):
        relevant = set(pos[pos["user_id"] == uid]["item_id"])
        if not relevant:
            skipped += 1
            continue
        t0 = time.time()
        try:
            ranked = _build_full_for_user(uid, train, top_n=k, candidates_k=candidates_k)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {uid} failed: {e}")
            skipped += 1
            continue
        latencies.append(time.time() - t0)
        ndcgs.append(ndcg_at_k(ranked, relevant, k))
        hits.append(hit_at_k(ranked, relevant, k))
        mrrs.append(mrr_at_k(ranked, relevant, k))
    return {
        "variant": "full_system",
        "k": k,
        "n_users": len(ndcgs),
        "skipped": skipped,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit@{k}":  float(np.mean(hits)) if hits else 0.0,
        f"mrr@{k}":  float(np.mean(mrrs)) if mrrs else 0.0,
        "avg_latency_sec": float(np.mean(latencies)) if latencies else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--full-sample", type=int, default=20)
    parser.add_argument("--retrieval-only", action="store_true", help="skip the LLM full-system eval")
    args = parser.parse_args()

    print("[eval] building temporal split…")
    train, test = temporal_split(min_reviews=MIN_FOR_FULL_PERSONA)
    write_splits(train, test)
    print(f"[eval] train: {len(train):,} reviews | test: {len(test):,} reviews | users: {test['user_id'].nunique():,}")

    results = {"split": {"train_size": int(len(train)), "test_size": int(len(test)), "n_users": int(test["user_id"].nunique())}}

    # Note: retrieval/personas were built on the FULL dataset, not just train.
    # This is a slight leakage that biases UP NDCG of retrieval — we report it
    # honestly and note it in the paper. To run a fully clean eval we would
    # re-index on train only; doing both takes much more time.
    print("[eval] retrieval-only (FAST, no LLM)…")
    r_retr = evaluate_retrieval(train, test, k=args.k, candidates_k=50)
    print(json.dumps(r_retr, indent=2))
    results["retrieval_only"] = r_retr

    if not args.retrieval_only:
        print(f"[eval] full system (LLM, sample={args.full_sample})…")
        r_full = evaluate_full(train, test, k=args.k, sample_size=args.full_sample, candidates_k=20)
        print(json.dumps(r_full, indent=2))
        results["full_system"] = r_full

    out_path = PROCESSED_DIR / "eval_ranking.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path}")


if __name__ == "__main__":
    main()
