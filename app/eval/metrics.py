"""Ranking metrics: NDCG@k, Hit@k, MRR@k.

Each function takes:
- recommended_ids: list of item_ids in ranked order produced by the system
- relevant_ids: set of item_ids that are actually relevant (e.g. held-out
  positives the user rated >= 4)
"""
from __future__ import annotations

import math


def hit_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int = 10) -> int:
    return 1 if any(r in relevant_ids for r in recommended_ids[:k]) else 0


def mrr_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    for i, r in enumerate(recommended_ids[:k], start=1):
        if r in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(recommended_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    """Binary-relevance NDCG@k. rel_i = 1 if recommended_ids[i] in relevant_ids."""
    dcg = 0.0
    for i, r in enumerate(recommended_ids[:k], start=1):
        if r in relevant_ids:
            dcg += 1.0 / math.log2(i + 1)
    n_rel = min(len(relevant_ids), k)
    if n_rel == 0:
        return 0.0
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
    return dcg / idcg if idcg > 0 else 0.0
