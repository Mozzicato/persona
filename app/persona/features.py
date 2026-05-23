"""Deterministic behavioral feature extraction from a user's review history.

This runs without an LLM call so it's fast over thousands of users. The output
is later optionally refined by an LLM (see `persona.refine`).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

EXCLAIM_RE = re.compile(r"!")
ALLCAPS_RE = re.compile(r"\b[A-Z]{3,}\b")
EMOJI_RE = re.compile(
    "["  # rough emoji block
    "\U0001F300-\U0001FAFF"
    "\U00002700-\U000027BF"
    "]"
)
PRICE_TERMS = {"price", "expensive", "cheap", "value", "cost", "worth", "money", "overpriced", "bargain", "deal"}
DELIVERY_TERMS = {"delivery", "shipping", "arrived", "shipment", "late", "delay", "rider", "courier"}
PACKAGING_TERMS = {"package", "packaging", "packed", "box", "wrapped", "wrapper", "container", "seal", "leaked", "leaking", "crushed", "damaged"}
SERVICE_TERMS = {"customer service", "support", "rude", "polite", "responsive", "helpful", "refund", "replacement", "complaint", "agent"}
QUALITY_TERMS = {"quality", "fresh", "taste", "flavor", "stale", "delicious", "awful", "perfect", "bland"}
NEG_TERMS = {"bad", "terrible", "awful", "horrible", "worst", "disappointed", "ruined", "stale", "rotten", "waste"}
POS_TERMS = {"great", "excellent", "love", "perfect", "amazing", "best", "delicious", "wonderful", "fantastic"}

# Festive / seasonal months in Nigerian context: December (Christmas/holiday),
# April (Easter often), late Ramadan/Eid varies but we treat Dec as primary festive.
FESTIVE_MONTHS = {12}


def _ratio(series: pd.Series, tokens: set[str]) -> float:
    if series.empty:
        return 0.0
    hits = series.str.lower().apply(lambda t: any(tok in t for tok in tokens)).mean()
    return float(hits)


def _clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def _temporal_traits(reviews: pd.DataFrame) -> dict:
    """Detect timing-based reviewer behaviors.

    - night_reviewer: posts most reviews 22:00-05:00
    - weekend_positivity: avg rating on Sat/Sun vs weekday delta
    - festive_generosity: avg rating in December vs rest-of-year delta
    """
    ts = pd.to_datetime(reviews["timestamp"])
    hours = ts.dt.hour
    weekdays = ts.dt.dayofweek  # 0=Mon, 6=Sun
    months = ts.dt.month

    night_share = float(((hours >= 22) | (hours < 5)).mean())
    morning_share = float(((hours >= 5) & (hours < 12)).mean())
    afternoon_share = float(((hours >= 12) & (hours < 17)).mean())
    evening_share = float(((hours >= 17) & (hours < 22)).mean())

    is_weekend = weekdays >= 5
    weekend_avg = float(reviews.loc[is_weekend, "rating"].mean()) if is_weekend.any() else float("nan")
    weekday_avg = float(reviews.loc[~is_weekend, "rating"].mean()) if (~is_weekend).any() else float("nan")
    weekend_delta = 0.0
    if not np.isnan(weekend_avg) and not np.isnan(weekday_avg):
        weekend_delta = weekend_avg - weekday_avg

    is_festive = months.isin(FESTIVE_MONTHS)
    festive_avg = float(reviews.loc[is_festive, "rating"].mean()) if is_festive.any() else float("nan")
    nonfestive_avg = float(reviews.loc[~is_festive, "rating"].mean()) if (~is_festive).any() else float("nan")
    festive_delta = 0.0
    if not np.isnan(festive_avg) and not np.isnan(nonfestive_avg):
        festive_delta = festive_avg - nonfestive_avg

    # Categorical archetype derived from highest hour-bucket share
    bucket_shares = {
        "morning": morning_share,
        "afternoon": afternoon_share,
        "evening": evening_share,
        "late_night": night_share,
    }
    peak_time = max(bucket_shares, key=bucket_shares.get)

    return {
        "peak_time_bucket": peak_time,
        "hour_distribution": {k: round(v, 3) for k, v in bucket_shares.items()},
        "night_reviewer": night_share >= 0.3,
        "weekend_positivity": round(weekend_delta, 3),
        "festive_generosity": round(festive_delta, 3),
    }


@dataclass
class Persona:
    user_id: str
    communication_style: dict
    economic_profile: dict
    behavioral_profile: dict
    temporal_profile: dict
    food_preferences: dict
    stats: dict

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "communication_style": self.communication_style,
            "economic_profile": self.economic_profile,
            "behavioral_profile": self.behavioral_profile,
            "temporal_profile": self.temporal_profile,
            "food_preferences": self.food_preferences,
            "stats": self.stats,
        }


def extract(user_id: str, reviews: pd.DataFrame) -> Persona:
    """Build a persona from a user's review subset (rows for that user)."""
    n = len(reviews)
    avg_rating = float(reviews["rating"].mean())
    std_rating = float(reviews["rating"].std() or 0.0)
    avg_len = float(reviews["text"].str.len().mean())
    exclaim_rate = float(reviews["text"].apply(lambda t: len(EXCLAIM_RE.findall(t))).mean())
    caps_rate = float(reviews["text"].apply(lambda t: len(ALLCAPS_RE.findall(t))).mean())
    emoji_rate = float(reviews["text"].apply(lambda t: len(EMOJI_RE.findall(t))).mean())

    # Harshness: lower avg rating + presence of negative terms
    harshness = _clip01((5.0 - avg_rating) / 4.0 * 0.7 + _ratio(reviews["text"], NEG_TERMS) * 0.3)
    optimism = _clip01((avg_rating - 1.0) / 4.0 * 0.6 + _ratio(reviews["text"], POS_TERMS) * 0.4)

    price_focus = _ratio(reviews["text"], PRICE_TERMS)
    delivery_focus = _ratio(reviews["text"], DELIVERY_TERMS)
    packaging_focus = _ratio(reviews["text"], PACKAGING_TERMS)
    service_focus = _ratio(reviews["text"], SERVICE_TERMS)
    quality_focus = _ratio(reviews["text"], QUALITY_TERMS)

    verbosity = "high" if avg_len > 400 else ("medium" if avg_len > 150 else "low")
    emotional_intensity = _clip01(exclaim_rate / 5.0 + caps_rate / 3.0)

    # consistency: low std means consistent reviewer
    consistency = _clip01(1.0 - (std_rating / 2.0))

    temporal = _temporal_traits(reviews)

    persona = Persona(
        user_id=user_id,
        communication_style={
            "verbosity": verbosity,
            "avg_review_length": round(avg_len, 1),
            "exclamation_rate": round(exclaim_rate, 3),
            "allcaps_rate": round(caps_rate, 3),
            "emoji_rate": round(emoji_rate, 3),
            "emotional_intensity": round(emotional_intensity, 3),
        },
        economic_profile={
            "price_focus": round(price_focus, 3),
            "budget_sensitive": price_focus > 0.15,
        },
        behavioral_profile={
            "harshness": round(harshness, 3),
            "optimism": round(optimism, 3),
            "consistency": round(consistency, 3),
            "delivery_sensitivity": round(delivery_focus, 3),
            "packaging_sensitivity": round(packaging_focus, 3),
            "service_sensitivity": round(service_focus, 3),
            "quality_sensitivity": round(quality_focus, 3),
        },
        temporal_profile=temporal,
        food_preferences={
            # Filled by LLM refinement later; leave empty stubs here
            "likes_keywords": [],
            "dislikes_keywords": [],
        },
        stats={
            "n_reviews": int(n),
            "avg_rating": round(avg_rating, 3),
            "std_rating": round(std_rating, 3),
            "min_rating": int(reviews["rating"].min()),
            "max_rating": int(reviews["rating"].max()),
        },
    )
    return persona


def rating_distribution(reviews: pd.DataFrame) -> dict[int, float]:
    counts = reviews["rating"].value_counts(normalize=True).to_dict()
    return {int(k): float(v) for k, v in counts.items()}
