"""Component tests that don't hit the LLM.

These exercise the deterministic parts (features, memory tags, context
auto-detection, retrieval scoring shape) so a regression is caught fast.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import pytest

from app import context as ctx_mod
from app.persona import features as feat_mod


@pytest.fixture
def fake_reviews() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["u"] * 6,
            "item_id": [f"i{i}" for i in range(6)],
            "rating": [1, 5, 2, 4, 1, 5],
            "text": [
                "This was terrible and stale, the delivery was late too.",
                "Great taste! Loved it!",
                "Overpriced, not worth it. Packaging was damaged.",
                "Pretty decent quality overall.",
                "Awful service, refused refund.",
                "Delicious and arrived on time.",
            ],
            "summary": ["bad", "good", "meh", "ok", "rude", "yum"],
            "timestamp": pd.to_datetime(
                [
                    "2024-01-01 23:30",  # late night
                    "2024-01-06 14:00",  # weekend afternoon
                    "2024-12-15 21:00",  # festive evening
                    "2024-03-10 09:00",  # weekday morning
                    "2024-12-25 19:00",  # festive evening
                    "2024-07-20 22:00",  # weekend night
                ]
            ),
        }
    )


def test_persona_extracts_packaging_and_service_sensitivities(fake_reviews):
    p = feat_mod.extract("u", fake_reviews).to_dict()
    behav = p["behavioral_profile"]
    assert behav["packaging_sensitivity"] > 0  # "Packaging was damaged"
    assert behav["service_sensitivity"] > 0  # "refused refund" + "service"
    assert behav["delivery_sensitivity"] > 0  # "delivery was late"


def test_persona_extracts_temporal_traits(fake_reviews):
    p = feat_mod.extract("u", fake_reviews).to_dict()
    temp = p["temporal_profile"]
    assert "peak_time_bucket" in temp
    assert set(temp["hour_distribution"].keys()) == {"morning", "afternoon", "evening", "late_night"}
    # 2 of 6 reviews in December -> festive_generosity is computed
    assert "festive_generosity" in temp


def test_context_auto_detects_salary_week():
    # day 27 in any month -> salary_week + end_of_month
    fake_now = datetime(2026, 3, 27, 12, 0)
    ctx = ctx_mod.normalize({}, now=fake_now)
    assert "salary_week" in ctx["nigerian_flags"]
    assert "end_of_month" in ctx["nigerian_flags"]
    assert "festive" not in ctx["nigerian_flags"]


def test_context_auto_detects_festive_december():
    fake_now = datetime(2026, 12, 5, 20, 0)
    ctx = ctx_mod.normalize({}, now=fake_now)
    assert "festive" in ctx["nigerian_flags"]
    assert ctx["time_bucket"] == "evening"


def test_user_can_override_auto_flag():
    fake_now = datetime(2026, 3, 27, 12, 0)
    ctx = ctx_mod.normalize({"salary_week": False}, now=fake_now)
    assert "salary_week" not in ctx["nigerian_flags"]


def test_memory_tags_late_delivery_in_recent_reviews(monkeypatch, fake_reviews, tmp_path):
    """Memory tagger should fire `late_delivery` on the negative delivery review."""
    from app import memory as memory_mod

    # memory.build now reads from app.persona.store._reviews internally
    monkeypatch.setattr("app.persona.store._reviews", lambda: fake_reviews)
    monkeypatch.setattr(memory_mod, "MEMORY_DIR", tmp_path)
    monkeypatch.setattr(memory_mod, "_path", lambda uid: tmp_path / f"{uid}.json")

    mem = memory_mod.build("u", last_k=6)
    tags = mem["short_term"]["tag_counts"]
    # "delivery was late too" with rating 1 -> late_delivery
    assert tags.get("late_delivery", 0) >= 1
    # "Packaging was damaged" with rating 2 -> bad_packaging
    assert tags.get("bad_packaging", 0) >= 1
    # "Awful service, refused refund" with rating 1 -> bad_service
    assert tags.get("bad_service", 0) >= 1
    # "Overpriced, not worth it" with rating 2 -> overpriced
    assert tags.get("overpriced", 0) >= 1
    assert mem["short_term"]["mood"] == "frustrated"
