"""Task B orchestrator.

Pipeline:
    1. Persona + Memory + Context (normalized, with auto-detected NG flags)
    2. Behavioral Analyzer    -> current decision state
    3. Reasoning Planner      -> high-level objective + ranking weight vector
    4. Hybrid Retriever       -> top-K candidates (semantic + CF + quality + pop)
    5. Behavioral Simulator   -> per-candidate Task-A reasoning + predicted rating
    6. Dynamic Re-ranker      -> blends retrieval + simulation + planner weights
    7. Explainable output
"""
from __future__ import annotations

import json

from app import context as ctx_mod
from app import memory as memory_mod
from app import rating_model
from app.llm import chat_json
from app.persona.store import get_or_build as persona_for
from app.planner import plan as plan_step
from app.reasoner import reason
from app.retrieval import retrieve

ANALYZER_SYSTEM = """You analyze a user persona + recent memory + situational context
to summarize the user's CURRENT DECISION-MAKING STATE. Return strict JSON only."""

ANALYZER_USER = """PERSONA: {persona}
MEMORY: {memory}
CONTEXT: {context}

Return JSON:
{{
  "current_mood": <one of: tired, hungry, stressed, curious, cheerful, frustrated, neutral>,
  "budget_mode": <bool>,
  "wants_fast_delivery": <bool>,
  "exploration_tendency": <float 0-1>,
  "decision_priorities": [<up to 3 of: "price","speed","quality","comfort","novelty","reliability">]
}}"""


def analyze_behavior(persona: dict, memory: dict, context: dict) -> dict:
    user_msg = ANALYZER_USER.format(
        persona=json.dumps(
            {k: persona.get(k) for k in ("behavioral_profile", "economic_profile", "temporal_profile", "llm_traits")},
            default=str,
        ),
        memory=json.dumps(memory.get("short_term", {}), default=str),
        context=json.dumps(context, default=str),
    )
    try:
        return chat_json(ANALYZER_SYSTEM, user_msg, temperature=0.2)
    except Exception:  # noqa: BLE001
        return {
            "current_mood": memory.get("short_term", {}).get("mood", "neutral"),
            "budget_mode": persona.get("economic_profile", {}).get("budget_sensitive", False),
            "wants_fast_delivery": persona.get("behavioral_profile", {}).get("delivery_sensitivity", 0) > 0.3,
            "exploration_tendency": 0.5,
            "decision_priorities": ["quality", "price"],
        }


def _rerank_score(candidate: dict, simulated: dict, planner_out: dict) -> float:
    """Blend retrieval signals + simulation + planner ranking weights."""
    sem = float(candidate.get("sim_semantic", 0.0))
    collab = float(candidate.get("sim_collaborative", 0.0))
    item_q = float(candidate.get("item_quality", 0.6))
    pop = float(candidate.get("popularity", 0.0))
    pred_rating = float(simulated.get("predicted_rating", 3)) / 5.0
    confidence = float(simulated.get("confidence", 0.6))

    w = planner_out.get("ranking_weights", {})
    # Map planner weights to retrieval signals (heuristic mapping):
    # quality   -> item_q + sim_semantic
    # novelty   -> sim_semantic (less seen-similar)
    # reliability -> collab + item_q
    # speed     -> proxy via item_quality (no real speed data in Amazon)
    # price     -> proxy via popularity (cheaper-popular bias; no price data)
    retrieval_blend = (
        w.get("quality", 0.25) * (0.6 * item_q + 0.4 * sem)
        + w.get("novelty", 0.10) * sem
        + w.get("reliability", 0.20) * (0.5 * collab + 0.5 * item_q)
        + w.get("speed", 0.20) * item_q
        + w.get("price", 0.25) * pop
    )

    # Simulation is the trump card — confidence-weighted predicted reaction.
    simulation_signal = pred_rating * confidence

    return 0.45 * retrieval_blend + 0.55 * simulation_signal


def _should_skip(candidate: dict, must_avoid: list[str]) -> bool:
    if not must_avoid:
        return False
    blob = (str(candidate.get("sample_summary") or "")).lower()
    return any(av.lower() in blob for av in must_avoid)


def recommend(
    user_id: str,
    raw_context: dict | None,
    top_n: int = 5,
    candidates_k: int = 12,
    target_domains: list[str] | None = None,
    cross_domain: bool = False,
) -> dict:
    persona = persona_for(user_id, refine=True)
    memory = memory_mod.get_or_build(user_id)
    context = ctx_mod.normalize(raw_context)
    behavior = analyze_behavior(persona, memory, context)
    planner_out = plan_step(persona, memory, behavior, context)

    candidates = retrieve(
        user_id,
        top_k=candidates_k,
        target_domains=target_domains,
        cross_domain=cross_domain,
    )

    enriched = []
    for c in candidates:
        if _should_skip(c, planner_out.get("must_avoid", [])):
            continue
        item_payload = {
            "name": c.get("sample_summary") or c["item_id"],
            "item_id": c["item_id"],
            "category": "food",
            "avg_rating": c.get("avg_rating"),
        }
        sim_out = reason(persona, memory, context, item_payload)
        lgbm = rating_model.predict(persona, c)
        if lgbm is not None:
            sim_out["lgbm_rating"] = round(lgbm, 2)
            sim_out["blended_rating"] = round(0.6 * sim_out["predicted_rating"] + 0.4 * lgbm, 2)
        enriched.append({"candidate": c, "simulation": sim_out})

    for e in enriched:
        e["rerank_score"] = _rerank_score(e["candidate"], e["simulation"], planner_out)
    enriched.sort(key=lambda x: x["rerank_score"], reverse=True)

    recs = []
    for e in enriched[:top_n]:
        c, s = e["candidate"], e["simulation"]
        recs.append({
            "item_id": c["item_id"],
            "item": c.get("sample_summary") or c["item_id"],
            "retrieval_source": c.get("retrieval_source"),
            "predicted_rating": s.get("blended_rating", s["predicted_rating"]),
            "emotional_state": s.get("emotional_state"),
            "reason": s.get("reasoning"),
            "key_drivers": s.get("key_drivers", []),
            "score": round(e["rerank_score"], 4),
        })

    return {
        "user_id": user_id,
        "context": context,
        "behavior_analysis": behavior,
        "plan": planner_out,
        "cross_domain": cross_domain,
        "target_domains": target_domains,
        "recommendations": recs,
    }
