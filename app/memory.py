"""Behavioral memory: long-term + short-term + contextual state per user.

Long-term: stable preferences derived from full history.
Short-term: mood + explicit recent_experiences tags from the LAST K reviews.
Contextual: situational flags supplied per-request (rainy, salary_week, etc.).

The PRD asks for a memory graph; we model it as a layered dict with explicit
tagged experiences ([late_delivery, bad_packaging, ...]) so downstream agents
can reason over recent friction without re-reading full review text.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from app.config import MEMORY_DIR
from app.persona.features import (
    DELIVERY_TERMS,
    NEG_TERMS,
    PACKAGING_TERMS,
    POS_TERMS,
    PRICE_TERMS,
    QUALITY_TERMS,
    SERVICE_TERMS,
)
from app.persona.store import user_reviews


# Tag rules: (tag name, regex/keyword set, rating-condition predicate)
_TAG_RULES: list[tuple[str, set[str], int]] = [
    ("late_delivery", {"late", "slow", "delay", "took forever", "took too long"}, 3),
    ("bad_packaging", {"damaged", "leaked", "crushed", "broken", "open", "torn", "smashed"}, 3),
    ("bad_quality", {"stale", "rotten", "moldy", "spoiled", "rancid", "off"}, 3),
    ("bad_service", {"rude", "unhelpful", "no response", "refused refund", "no refund"}, 3),
    ("overpriced", {"overpriced", "not worth", "too expensive", "rip off"}, 3),
    ("great_value", {"great deal", "worth it", "bargain", "great price"}, 4),
    ("loved_quality", {"delicious", "perfect", "amazing", "fantastic", "love this"}, 4),
]


def _path(user_id: str) -> Path:
    return MEMORY_DIR / f"{user_id.replace('/', '_')}.json"


def _recent_signal(reviews: pd.DataFrame, tokens: set[str]) -> float:
    if reviews.empty:
        return 0.0
    return float(reviews["text"].str.lower().apply(lambda t: any(tok in t for tok in tokens)).mean())


def _tag_review(text: str, rating: int) -> list[str]:
    """Return tags that fire on a single review."""
    t = text.lower()
    tags = []
    for tag, tokens, threshold in _TAG_RULES:
        hit = any(tok in t for tok in tokens)
        if not hit:
            continue
        if threshold <= 3 and rating <= threshold:
            tags.append(tag)
        elif threshold >= 4 and rating >= threshold:
            tags.append(tag)
    return tags


def build(user_id: str, last_k: int = 5) -> dict:
    from app.persona.coldstart import neutral_memory
    from app.persona.store import _reviews

    df = _reviews()
    sub = df[df["user_id"] == user_id]
    if sub.empty:
        out = neutral_memory(user_id)
        _path(user_id).write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out
    reviews = sub.sort_values("timestamp")
    recent = reviews.tail(last_k)
    recent_avg = float(recent["rating"].mean())
    long_avg = float(reviews["rating"].mean())

    drift = recent_avg - long_avg  # negative = trending grumpier
    recent_frustration = _recent_signal(recent, NEG_TERMS)
    recent_joy = _recent_signal(recent, POS_TERMS)
    recent_delivery_complaints = _recent_signal(recent[recent["rating"] <= 2], DELIVERY_TERMS)
    recent_price_complaints = _recent_signal(recent[recent["rating"] <= 2], PRICE_TERMS)
    recent_packaging_complaints = _recent_signal(recent[recent["rating"] <= 2], PACKAGING_TERMS)
    recent_service_complaints = _recent_signal(recent[recent["rating"] <= 2], SERVICE_TERMS)

    # Explicit per-review experience tags from the recent window
    experiences: list[dict] = []
    tag_counts: dict[str, int] = {}
    for _, row in recent.iterrows():
        tags = _tag_review(str(row["text"]), int(row["rating"]))
        if not tags:
            continue
        experiences.append(
            {
                "timestamp": str(row["timestamp"]),
                "rating": int(row["rating"]),
                "tags": tags,
                "summary": str(row.get("summary") or "")[:80],
            }
        )
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Open friction = any negative tag fired in recent window
    negative_tags = {"late_delivery", "bad_packaging", "bad_quality", "bad_service", "overpriced"}
    open_friction = sorted({t for t in tag_counts if t in negative_tags})

    mood = "neutral"
    if drift <= -0.7 or recent_frustration > 0.4 or open_friction:
        mood = "frustrated"
    elif drift >= 0.7 or recent_joy > 0.5:
        mood = "upbeat"

    memory = {
        "user_id": user_id,
        "long_term": {
            "avg_rating": round(long_avg, 3),
            "n_reviews_total": int(len(reviews)),
        },
        "short_term": {
            "recent_avg_rating": round(recent_avg, 3),
            "rating_drift": round(drift, 3),
            "mood": mood,
            "recent_frustration": round(recent_frustration, 3),
            "recent_joy": round(recent_joy, 3),
            "recent_delivery_complaints": round(recent_delivery_complaints, 3),
            "recent_price_complaints": round(recent_price_complaints, 3),
            "recent_packaging_complaints": round(recent_packaging_complaints, 3),
            "recent_service_complaints": round(recent_service_complaints, 3),
            "open_friction": open_friction,
            "tag_counts": tag_counts,
            "recent_experiences": experiences,
            "last_k_ratings": recent["rating"].tolist(),
            "last_k_summaries": recent["summary"].fillna("").tolist(),
        },
    }
    _path(user_id).write_text(json.dumps(memory, indent=2), encoding="utf-8")
    return memory


def get_or_build(user_id: str) -> dict:
    p = _path(user_id)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return build(user_id)
