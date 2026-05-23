"""Non-LLM smoke test: imports + retrieval + cold-start + LightGBM.

Burns zero Groq tokens. Confirms the deterministic pipeline still works
after the recent file edits.
"""
from __future__ import annotations

import sys
import time
import traceback

sys.stdout.reconfigure(encoding="utf-8")


def step(name: str, fn):
    print(f"\n[smoke] {name} ... ", end="", flush=True)
    t0 = time.time()
    try:
        result = fn()
        dt = (time.time() - t0) * 1000
        print(f"OK ({dt:.0f} ms)")
        return result
    except Exception as e:  # noqa: BLE001
        print(f"FAIL: {e}")
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    # 1. App imports cleanly
    def imports():
        import app.config
        import app.context
        import app.generator
        import app.llm
        import app.main
        import app.memory
        import app.planner
        import app.rating_model
        import app.reasoner
        import app.recommender
        import app.retrieval
        import app.persona.coldstart
        import app.persona.features
        import app.persona.refine
        import app.persona.store
        return True

    step("imports", imports)

    # 2. Data files present
    def data_present():
        from pathlib import Path
        from app.config import PROCESSED_DIR, MODELS_DIR
        required = [
            PROCESSED_DIR / "reviews.parquet",
            PROCESSED_DIR / "items.parquet",
            PROCESSED_DIR / "users.parquet",
            PROCESSED_DIR / "playstore_reviews.parquet",
            PROCESSED_DIR / "nigerian_voice_samples.json",
            MODELS_DIR / "item_index.pkl",
            MODELS_DIR / "item_sim.pkl",
            MODELS_DIR / "rating_lgbm.pkl",
        ]
        missing = [p for p in required if not p.exists()]
        if missing:
            raise RuntimeError(f"missing artifacts: {missing}")
        return True

    step("data + model artifacts present", data_present)

    # 3. Sample existing user IDs from each domain
    def sample_users():
        import pandas as pd
        from app.config import PROCESSED_DIR
        reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
        amazon_uid = reviews[reviews["domain"] == "food"].groupby("user_id").size().sort_values(ascending=False).index[0]
        gp_users = reviews[reviews["domain"] != "food"]["user_id"].unique()
        gp_uid = gp_users[0] if len(gp_users) else None
        print(f"\n   sample amazon: {amazon_uid}  |  sample play-store: {gp_uid}", end="")
        return {"amazon": amazon_uid, "playstore": gp_uid}

    uids = step("sample real user IDs", sample_users)

    # 4. Retrieval (new CF-favoring defaults)
    def retrieval_amazon():
        from app.retrieval import retrieve
        recs = retrieve(uids["amazon"], top_k=10)
        if not recs:
            raise RuntimeError("no candidates returned")
        sources = [r.get("retrieval_source") for r in recs]
        print(f"\n   top-10 sources: {sources}", end="")
        return recs

    step("retrieval (amazon user, new CF weights)", retrieval_amazon)

    # 5. Cold-start retrieval
    def retrieval_coldstart():
        from app.retrieval import retrieve
        recs = retrieve("definitely_not_in_dataset_xyz", top_k=5)
        # cold-start should fall through to popularity/quality without crash
        if not recs:
            raise RuntimeError("cold-start returned 0 candidates")
        return recs

    step("retrieval cold-start fallback", retrieval_coldstart)

    # 6. Cross-domain retrieval
    def retrieval_cross():
        from app.retrieval import retrieve
        recs = retrieve(uids["amazon"], top_k=5, cross_domain=True)
        if not recs:
            raise RuntimeError("cross-domain returned 0 candidates")
        domains = [r.get("domain") for r in recs]
        print(f"\n   cross-domain domains: {domains}", end="")
        return recs

    step("cross-domain retrieval", retrieval_cross)

    # 7. Persona load (cached -> instant)
    def persona_existing():
        from app.persona.store import get_or_build
        p = get_or_build(uids["amazon"], refine=False)  # refine=False to avoid LLM
        if "behavioral_profile" not in p:
            raise RuntimeError("missing behavioral_profile in persona")
        return p

    step("persona load (existing user, no LLM refine)", persona_existing)

    # 8. Cold-start persona
    def persona_cold():
        from app.persona.coldstart import neutral_persona
        p1 = neutral_persona("brand_new_user_001")
        if "behavioral_profile" not in p1:
            raise RuntimeError("neutral persona missing behavioral_profile")
        p2 = neutral_persona(
            "brand_new_user_002",
            hints={"budget_sensitive": True, "likes": ["spicy", "fast delivery"]},
        )
        if not p2.get("economic_profile", {}).get("budget_sensitive"):
            raise RuntimeError("neutral persona did not pick up hint")
        if p2.get("food_preferences", {}).get("likes_keywords") != ["spicy", "fast delivery"]:
            raise RuntimeError(f"likes hint not stored: {p2.get('food_preferences')}")
        return True

    step("cold-start persona helpers", persona_cold)

    # 9. Memory build for existing + cold-start
    def memory_paths():
        from app.memory import get_or_build
        m1 = get_or_build(uids["amazon"])
        m2 = get_or_build("brand_new_user_003")  # should fall through to neutral_memory
        if "short_term" not in m1 or "short_term" not in m2:
            raise RuntimeError("memory missing short_term")
        return True

    step("memory build (existing + cold-start)", memory_paths)

    # 10. LightGBM cross-check works
    def lightgbm_predict():
        import pandas as pd
        from app.config import PROCESSED_DIR
        from app.persona.store import get_or_build
        from app.rating_model import predict
        items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
        sample_item = items.iloc[0].to_dict()
        persona = get_or_build(uids["amazon"], refine=False)
        pred = predict(persona, sample_item)
        if pred is None or not (1.0 <= pred <= 5.0):
            raise RuntimeError(f"lightgbm prediction out of range: {pred}")
        print(f"\n   sample LightGBM prediction: {pred:.2f}", end="")
        return pred

    step("LightGBM rating predict", lightgbm_predict)

    # 11. FastAPI app boots (without serving)
    def app_boots():
        from fastapi.testclient import TestClient
        from app.main import app
        client = TestClient(app)
        r = client.get("/health")
        if r.status_code != 200:
            raise RuntimeError(f"/health returned {r.status_code}")
        r2 = client.get("/users?limit=3")
        if r2.status_code != 200:
            raise RuntimeError(f"/users returned {r2.status_code}")
        return True

    step("FastAPI boots + /health + /users", app_boots)

    # 12. Voice samples loaded
    def voice_samples():
        import json
        from app.config import PROCESSED_DIR
        data = json.loads((PROCESSED_DIR / "nigerian_voice_samples.json").read_text(encoding="utf-8"))
        if not data.get("best_overall"):
            raise RuntimeError("voice samples missing best_overall")
        print(f"\n   {len(data['best_overall'])} best-overall + {len(data.get('pidgin_strong', []))} pidgin-strong examples", end="")
        return data

    step("Nigerian voice samples", voice_samples)

    print("\n\n[smoke] ALL CHECKS PASSED — non-LLM pipeline is healthy.")


if __name__ == "__main__":
    main()
