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
from fastapi.responses import HTMLResponse
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


LANDING_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>PersonaFlow AI</title>
<style>
  body { font-family: -apple-system, system-ui, Segoe UI, Arial, sans-serif;
         max-width: 760px; margin: 40px auto; padding: 0 20px; line-height: 1.55;
         color: #222; background: #fafafa; }
  h1 { margin-bottom: 6px; }
  .tag { color: #666; font-style: italic; margin-top: 0; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-top: 24px; }
  .card { background: #fff; padding: 18px 20px; border-radius: 10px;
          box-shadow: 0 1px 3px rgba(0,0,0,.08); }
  .card h3 { margin-top: 0; }
  code { background: #f0f0f0; padding: 2px 6px; border-radius: 4px;
         font-size: 0.9em; }
  a.btn { display: inline-block; background: #0a7d4b; color: #fff;
          padding: 10px 16px; text-decoration: none; border-radius: 6px;
          margin-top: 8px; margin-right: 8px; font-weight: 600; }
  a.btn:hover { background: #086239; }
  ul { padding-left: 20px; }
  .metric { font-size: 1.4em; font-weight: 700; color: #0a7d4b; }
</style>
</head>
<body>
<h1>PersonaFlow AI</h1>
<p class="tag">Behavior-aware review simulation + recommendation for Nigerian consumers.</p>

<p>
  <a class="btn" href="/docs">Open Swagger UI &rarr;</a>
  <a class="btn" style="background:#444" href="https://github.com/Mozzicato/persona" target="_blank">View Code on GitHub</a>
</p>

<div class="grid">
  <div class="card">
    <h3>Task A</h3>
    <code>POST /simulate-review</code>
    <p>Predict rating + generate persona-consistent Nigerian-flavored review for an item.</p>
  </div>
  <div class="card">
    <h3>Task B</h3>
    <code>POST /recommend</code>
    <p>Behavior-aware recommendations with per-candidate simulation.
       Set <code>cross_domain: true</code> for food&rarr;apps.</p>
  </div>
  <div class="card">
    <h3>Cold-start</h3>
    <code>POST /persona</code>
    <p>Unknown users get a neutral persona; pass <code>cold_start_hints</code>
       to seed preferences.</p>
  </div>
  <div class="card">
    <h3>Browse users</h3>
    <code>GET /users?limit=20</code>
    <p>Sample real Amazon Fine Food user IDs you can plug into the other endpoints.</p>
  </div>
</div>

<h3 style="margin-top:32px">Headline metrics (temporal hold-out, 1,421 users)</h3>
<ul>
  <li>NDCG@10 = <span class="metric">0.649</span></li>
  <li>Hit@10 = <span class="metric">0.675</span></li>
  <li>MRR@10 = <span class="metric">0.642</span></li>
  <li>Rating RMSE (LightGBM) = <span class="metric">0.710</span> &nbsp;(35% better than user-mean baseline 1.090)</li>
</ul>

<p style="margin-top:32px; color:#888; font-size:0.85em">
  Built for the DSN &times; BCT LLM Agent Challenge 2026. Source on
  <a href="https://github.com/Mozzicato/persona">GitHub</a>.
</p>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def landing():
    return LANDING_HTML


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
