"""Final reproducibility smoke test.

Exercises every endpoint behavior at least once: persona load (hot user),
cold-start, Task A, Task B same-domain, Task B cross-domain.

Run after the data + index + model artifacts are built:
    python -m app.data.ingest
    python -m app.data.playstore
    python -m app.data.unify
    python -m app.data.nigerian_voice
    python -m app.retrieval
    python -m app.rating_model
    python scripts/final_smoke.py
"""
from __future__ import annotations

import json
import sys

from app import context as ctx_mod
from app import memory as memory_mod
from app.generator import generate as gen_review
from app.persona.store import get_or_build as persona_for
from app.persona.store import list_user_ids
from app.reasoner import reason
from app.recommender import recommend


def pretty(label: str, payload) -> None:
    print(f"\n===== {label} =====")
    print(json.dumps(payload, indent=2, default=str)[:3000])


def section(title: str) -> None:
    print(f"\n{'='*70}\n  {title}\n{'='*70}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    ids = list_user_ids(limit=5)
    if not ids:
        sys.exit("No users — run the data pipeline first.")
    uid = ids[0]

    section("1. PERSONA (hot user from Amazon Fine Food)")
    persona = persona_for(uid, refine=True)
    pretty("PERSONA slice", {k: persona[k] for k in ("communication_style","behavioral_profile","economic_profile","temporal_profile","llm_traits","stats") if k in persona})

    section("2. MEMORY (with tagged experiences)")
    pretty("MEMORY short_term", memory_mod.get_or_build(uid).get("short_term"))

    section("3. CONTEXT (auto-detected Nigerian flags from today's date)")
    ctx_today = ctx_mod.normalize({"weather": "rainy"})
    pretty("CONTEXT", ctx_today)

    section("4. TASK A — Generate review for Chowdeck on a rainy night")
    item = {"name": "Chowdeck", "category": "food_delivery", "item_id": "com.chowdeck.app"}
    r = reason(persona, memory_mod.get_or_build(uid), ctx_today, item)
    review = gen_review(persona, item, r, ctx_today)
    pretty("REASONER", r)
    print(f"\n--- GENERATED REVIEW (rating={r['predicted_rating']}) ---\n{review}\n")

    section("5. TASK B — Same-domain recommendations (food)")
    rec_same = recommend(uid, {"time": "evening"}, top_n=3, candidates_k=10)
    pretty("plan", rec_same.get("plan"))
    for i, rec in enumerate(rec_same["recommendations"], 1):
        print(f"  [{i}] {rec['item'][:80]}  rating={rec['predicted_rating']}  score={rec['score']}")
        print(f"       reason: {rec['reason'][:200]}...")

    section("6. TASK B — CROSS-DOMAIN recommendations (food user -> apps)")
    rec_cross = recommend(uid, {"time": "evening", "mood": "tired"}, top_n=3, candidates_k=10, cross_domain=True)
    print(f"target home domains excluded; recommending across: ", end="")
    print({r["item_id"] for r in rec_cross["recommendations"]} or "(none)")
    for i, rec in enumerate(rec_cross["recommendations"], 1):
        print(f"  [{i}] {rec['item'][:80]}  rating={rec['predicted_rating']}  score={rec['score']}")
        print(f"       reason: {rec['reason'][:200]}...")

    section("7. COLD START — brand new user, no history")
    cs_persona = persona_for("brand_new_xyz", refine=False, cold_start_hints={"budget_sensitive": True, "likes": ["spicy", "fast delivery"]})
    pretty("cold-start persona", {k: cs_persona[k] for k in ("cold_start","communication_style","behavioral_profile","food_preferences","stats")})
    cs_rec = recommend("brand_new_xyz", {"time": "night"}, top_n=3, candidates_k=8)
    print("Cold-start recommendations:")
    for i, rec in enumerate(cs_rec["recommendations"], 1):
        print(f"  [{i}] {rec['item'][:80]}  rating={rec['predicted_rating']}")

    print("\n✓ Smoke test complete — all paths exercised.")


if __name__ == "__main__":
    main()
