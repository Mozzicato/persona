"""Persona persistence + lazy lookup. Personas are cached as JSON per user."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd

from app.config import PERSONA_DIR, PROCESSED_DIR
from app.persona.features import Persona, extract, rating_distribution
from app.persona.refine import refine_persona

REVIEWS_PARQUET = PROCESSED_DIR / "reviews.parquet"


@lru_cache(maxsize=1)
def _reviews() -> pd.DataFrame:
    if not REVIEWS_PARQUET.exists():
        raise FileNotFoundError(
            f"{REVIEWS_PARQUET} missing. Run `python -m app.data.ingest` first."
        )
    return pd.read_parquet(REVIEWS_PARQUET)


def user_reviews(user_id: str) -> pd.DataFrame:
    df = _reviews()
    sub = df[df["user_id"] == user_id]
    if sub.empty:
        raise KeyError(f"user_id {user_id!r} not found in dataset")
    return sub


def _path(user_id: str) -> Path:
    safe = user_id.replace("/", "_")
    return PERSONA_DIR / f"{safe}.json"


def get_or_build(
    user_id: str,
    *,
    refine: bool = True,
    force: bool = False,
    cold_start_hints: dict | None = None,
) -> dict:
    """Build (or load cached) persona for `user_id`.

    Cold-start contract:
    - 0 reviews                          → neutral baseline persona
    - 1-4 reviews                        → partial persona (stats only)
    - >= 5 (`MIN_FOR_FULL_PERSONA`)      → full persona with LLM refinement
    """
    from app.persona.coldstart import (
        MIN_FOR_FULL_PERSONA,
        neutral_persona,
        partial_persona,
    )

    p = _path(user_id)
    if p.exists() and not force:
        return json.loads(p.read_text(encoding="utf-8"))

    df = _reviews()
    sub = df[df["user_id"] == user_id]

    if sub.empty:
        out = neutral_persona(user_id, cold_start_hints)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    if len(sub) < MIN_FOR_FULL_PERSONA:
        out = partial_persona(user_id, sub)
        p.write_text(json.dumps(out, indent=2), encoding="utf-8")
        return out

    persona: Persona = extract(user_id, sub)
    out = persona.to_dict()
    out["cold_start"] = False
    out["rating_distribution"] = rating_distribution(sub)

    if refine:
        out["llm_traits"] = refine_persona(sub)
        out["food_preferences"]["likes_keywords"] = out["llm_traits"].get("likes_keywords", [])
        out["food_preferences"]["dislikes_keywords"] = out["llm_traits"].get("dislikes_keywords", [])

    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def list_user_ids(limit: int | None = None) -> list[str]:
    ids = _reviews()["user_id"].unique().tolist()
    return ids[:limit] if limit else ids
