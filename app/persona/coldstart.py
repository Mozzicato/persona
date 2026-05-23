"""Cold-start persona path.

When a user has zero or very few reviews in our catalogue, we cannot extract
a behavioral persona from history. The cold-start handler:

1. Accepts optional caller-supplied hints (preferences, demographics).
2. Builds a neutral baseline persona with mid-range traits.
3. If 1 <= n_reviews < MIN_FOR_FULL_PERSONA, builds a *partial* persona
   (deterministic stats only, no LLM refinement) and flags it as such.
4. Lets the downstream LLM reasoner work off a defensible default rather than
   crash.

This is the "Cold-Start" criterion in Task B (25 pts in the brief).
"""
from __future__ import annotations

import pandas as pd

from app.persona.features import Persona, extract, rating_distribution

MIN_FOR_FULL_PERSONA = 5  # Below this, we don't trust LLM refinement


def neutral_persona(user_id: str, hints: dict | None = None) -> dict:
    """Build a baseline persona for a completely unknown user.

    `hints` may include any of: likes (list[str]), dislikes (list[str]),
    budget_sensitive (bool), verbosity (low|medium|high), archetype (str),
    preferred_domain (str).
    """
    hints = hints or {}
    likes = list(hints.get("likes", []))[:6]
    dislikes = list(hints.get("dislikes", []))[:6]
    return {
        "user_id": user_id,
        "cold_start": True,
        "communication_style": {
            "verbosity": hints.get("verbosity", "medium"),
            "avg_review_length": 200.0,
            "exclamation_rate": 0.5,
            "allcaps_rate": 0.0,
            "emoji_rate": 0.0,
            "emotional_intensity": 0.25,
        },
        "economic_profile": {
            "price_focus": 0.2,
            "budget_sensitive": bool(hints.get("budget_sensitive", True)),
        },
        "behavioral_profile": {
            "harshness": 0.4,
            "optimism": 0.55,
            "consistency": 0.6,
            "delivery_sensitivity": 0.3,
            "packaging_sensitivity": 0.2,
            "service_sensitivity": 0.25,
            "quality_sensitivity": 0.4,
        },
        "temporal_profile": {
            "peak_time_bucket": "evening",
            "hour_distribution": {"morning": 0.2, "afternoon": 0.3, "evening": 0.35, "late_night": 0.15},
            "night_reviewer": False,
            "weekend_positivity": 0.0,
            "festive_generosity": 0.0,
        },
        "food_preferences": {
            "likes_keywords": likes,
            "dislikes_keywords": dislikes,
        },
        "stats": {
            "n_reviews": 0,
            "avg_rating": 3.5,
            "std_rating": 1.0,
            "min_rating": 1,
            "max_rating": 5,
        },
        "rating_distribution": {1: 0.1, 2: 0.1, 3: 0.2, 4: 0.3, 5: 0.3},
        "llm_traits": {
            "likes_keywords": likes,
            "dislikes_keywords": dislikes,
            "sarcasm": 0.1,
            "sensitivities": ["taste", "freshness"] if not hints.get("preferred_domain") else [],
            "reviewer_archetype": hints.get("archetype", "pragmatic"),
            "summary": "Cold-start user — baseline persona, no review history yet.",
        },
    }


def neutral_memory(user_id: str) -> dict:
    """Behavioral memory baseline for a user with no recent activity."""
    return {
        "user_id": user_id,
        "cold_start": True,
        "long_term": {"avg_rating": 3.5, "n_reviews_total": 0},
        "short_term": {
            "recent_avg_rating": 3.5,
            "rating_drift": 0.0,
            "mood": "neutral",
            "recent_frustration": 0.0,
            "recent_joy": 0.0,
            "recent_delivery_complaints": 0.0,
            "recent_price_complaints": 0.0,
            "recent_packaging_complaints": 0.0,
            "recent_service_complaints": 0.0,
            "open_friction": [],
            "tag_counts": {},
            "recent_experiences": [],
            "last_k_ratings": [],
            "last_k_summaries": [],
        },
    }


def partial_persona(user_id: str, reviews: pd.DataFrame) -> dict:
    """For users with 1 <= n < MIN_FOR_FULL_PERSONA reviews — stats only."""
    p = extract(user_id, reviews).to_dict()
    p["cold_start"] = True
    p["rating_distribution"] = rating_distribution(reviews)
    p["llm_traits"] = {
        "likes_keywords": [],
        "dislikes_keywords": [],
        "sarcasm": 0.0,
        "sensitivities": [],
        "reviewer_archetype": "pragmatic",
        "summary": f"Partial persona — only {len(reviews)} review(s) available.",
    }
    return p
