"""Full end-to-end LLM smoke test: /simulate-review + /recommend + cold-start.

Hits Groq. Takes ~30-60 s total.
"""
from __future__ import annotations

import json
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def step(name: str, method: str, path: str, body: dict | None = None):
    print(f"\n[smoke-llm] {name}\n  {method} {path}")
    t0 = time.time()
    if method == "GET":
        r = client.get(path)
    else:
        r = client.post(path, json=body or {})
    dt = (time.time() - t0) * 1000
    if r.status_code != 200:
        print(f"  FAIL {r.status_code}: {r.text[:200]}")
        sys.exit(1)
    print(f"  OK ({dt:.0f} ms, {len(r.text)} bytes)")
    return r.json()


def main() -> None:
    h = step("health", "GET", "/health")
    assert h.get("status") == "ok"

    users = step("list users", "GET", "/users?limit=5")
    uid = users["user_ids"][0]
    print(f"  using uid={uid}")

    # Task A: simulate review for a Nigerian app context
    a = step(
        "Task A: simulate-review (rainy + heavy traffic)",
        "POST",
        "/simulate-review",
        {
            "user_id": uid,
            "item": {"name": "Chowdeck", "category": "food_delivery", "item_id": "com.chowdeck.app"},
            "context": {"time": "night", "weather": "rainy", "traffic_heavy": True},
        },
    )
    print(f"  rating={a['rating']} | emotion={a['emotional_state']}")
    print(f"  review: {a['review'][:200]}")
    assert isinstance(a["rating"], int) and 1 <= a["rating"] <= 5
    assert len(a["review"]) > 20

    # Task B: same-domain recommendation
    b = step(
        "Task B: recommend (same domain)",
        "POST",
        "/recommend",
        {"user_id": uid, "context": {"time": "night", "mood": "tired"}, "top_n": 3},
    )
    print(f"  plan: {b['plan']['plan_summary']}")
    print(f"  top-3:")
    for r in b["recommendations"]:
        print(f"    - [{r['retrieval_source']}] {str(r['item'])[:60]} | rating={r['predicted_rating']} | {r.get('emotional_state')}")
    assert len(b["recommendations"]) >= 1

    # Task B: CROSS-DOMAIN — food user gets Nigerian apps
    c = step(
        "Task B: cross-domain (food user -> Nigerian apps)",
        "POST",
        "/recommend",
        {"user_id": uid, "context": {"time": "evening"}, "top_n": 3, "cross_domain": True},
    )
    print(f"  domains served: {[r.get('item') for r in c['recommendations']]}")
    assert c.get("cross_domain") is True

    # Cold-start: brand new user
    d = step(
        "Cold-start: simulate-review for unknown user",
        "POST",
        "/simulate-review",
        {
            "user_id": "totally_new_user_xyz123",
            "item": {"name": "Bolt Food", "category": "food_delivery"},
            "context": {"time": "afternoon", "salary_week": True},
            "cold_start_hints": {"budget_sensitive": True, "likes": ["affordable", "fast"]},
        },
    )
    print(f"  rating={d['rating']} | emotion={d['emotional_state']}")
    print(f"  review: {d['review'][:200]}")
    assert isinstance(d["rating"], int)

    print("\n[smoke-llm] ALL CHECKS PASSED — full pipeline working.")


if __name__ == "__main__":
    main()
