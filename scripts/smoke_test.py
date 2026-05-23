"""End-to-end smoke test against a real user.

Run after `python -m app.data.ingest && python -m app.retrieval` and (optionally)
`python -m app.rating_model`.
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
    print(json.dumps(payload, indent=2, default=str))


def main() -> None:
    ids = list_user_ids(limit=5)
    if not ids:
        sys.exit("No users in dataset — run `python -m app.data.ingest` first.")
    uid = ids[0]
    print(f"[smoke] using user_id={uid}")

    persona = persona_for(uid, refine=True)
    pretty("PERSONA", {k: persona[k] for k in ("communication_style", "behavioral_profile", "economic_profile", "temporal_profile", "stats", "llm_traits") if k in persona})

    memory = memory_mod.get_or_build(uid)
    pretty("MEMORY", memory.get("short_term"))

    context = ctx_mod.normalize({"time": "night", "weather": "rainy", "traffic_heavy": True})
    pretty("CONTEXT", context)

    # Task A
    item = {"name": "Mega Chicken Wings", "category": "restaurant", "price_range": "medium"}
    r = reason(persona, memory, context, item)
    pretty("REASONER", r)
    review = gen_review(persona, item, r, context)
    pretty("REVIEW (Task A)", {"rating": r["predicted_rating"], "review": review})

    # Task B
    recs = recommend(uid, {"time": "night", "mood": "tired"}, top_n=3)
    pretty("RECOMMENDATIONS (Task B)", recs)


if __name__ == "__main__":
    main()
