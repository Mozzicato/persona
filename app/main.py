"""PersonaFlow AI - FastAPI entrypoint.

Endpoints:
    GET  /health
    GET  /users?limit=20            -> sample user_ids you can query
    POST /persona                    -> full persona for a user
    POST /simulate-review           -> Task A: persona-aware review + rating
    POST /recommend                 -> Task B: behavior-aware recommendations
"""
from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import memory as memory_mod
from app import context as ctx_mod
from app import rating_model
from app.config import PROCESSED_DIR
from app.generator import generate as gen_review
from app.persona.store import get_or_build as persona_for
from app.persona.store import list_user_ids
from app.reasoner import reason
from app.recommender import recommend

app = FastAPI(
    title="PersonaFlow AI",
    description="Behavior-aware review simulation and recommendation for Nigerian consumers.",
    version="0.1.0",
)


class Item(BaseModel):
    name: str
    category: str | None = "food"
    price_range: str | None = None
    item_id: str | None = None


class SimulateReviewRequest(BaseModel):
    user_id: str
    item: Item
    context: dict[str, Any] = Field(default_factory=dict)


class RecommendRequest(BaseModel):
    user_id: str
    context: dict[str, Any] = Field(default_factory=dict)
    top_n: int = 5
    target_domains: list[str] | None = None
    cross_domain: bool = False
    cold_start_hints: dict[str, Any] | None = None


@app.get("/health")
def health():
    return {"status": "ok", "service": "PersonaFlow AI"}


@app.get("/users")
def users(limit: int = 20):
    try:
        return {"user_ids": list_user_ids(limit=limit)}
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@app.post("/persona")
def persona(req: dict):
    user_id = req.get("user_id")
    if not user_id:
        raise HTTPException(400, "user_id required")
    try:
        return persona_for(
            user_id,
            refine=req.get("refine", True),
            force=req.get("force", False),
            cold_start_hints=req.get("cold_start_hints"),
        )
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))


@app.post("/simulate-review")
def simulate_review(req: SimulateReviewRequest):
    try:
        persona = persona_for(req.user_id, refine=True)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))

    memory = memory_mod.get_or_build(req.user_id)
    context = ctx_mod.normalize(req.context)
    item_payload = req.item.model_dump()

    reasoner_out = reason(persona, memory, context, item_payload)

    # Try LightGBM cross-check if a real catalogue item is referenced
    lgbm = None
    if req.item.item_id:
        try:
            items = pd.read_parquet(PROCESSED_DIR / "items.parquet").set_index("item_id")
            if req.item.item_id in items.index:
                lgbm = rating_model.predict(persona, items.loc[req.item.item_id].to_dict())
        except Exception:  # noqa: BLE001
            lgbm = None
    if lgbm is not None:
        reasoner_out["lgbm_rating"] = round(lgbm, 2)
        reasoner_out["blended_rating"] = round(0.6 * reasoner_out["predicted_rating"] + 0.4 * lgbm, 2)

    review_text = gen_review(persona, item_payload, reasoner_out, context)
    final_rating = int(round(reasoner_out.get("blended_rating", reasoner_out["predicted_rating"])))

    return {
        "user_id": req.user_id,
        "item": item_payload,
        "rating": final_rating,
        "review": review_text,
        "reasoning": reasoner_out.get("reasoning"),
        "emotional_state": reasoner_out.get("emotional_state"),
        "key_drivers": reasoner_out.get("key_drivers", []),
        "confidence": reasoner_out.get("confidence"),
        "context_used": context,
        "memory_snapshot": memory.get("short_term"),
    }


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    try:
        return recommend(
            req.user_id,
            req.context,
            top_n=req.top_n,
            target_domains=req.target_domains,
            cross_domain=req.cross_domain,
        )
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
