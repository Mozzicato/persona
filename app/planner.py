"""Reasoning Planner.

Runs ONCE per /recommend request, BEFORE per-candidate behavioral simulation.
It produces a high-level decision plan that biases the downstream re-ranker.

Why separate from the per-candidate reasoner?
- Per-candidate reasoning is item-specific ("how would the user react to X?").
- The planner is request-specific ("given everything I know about this user
  RIGHT NOW, what should the recommender prioritise?").

This separation matches the PRD's distinction between "Reasoning Planner" and
"Behavioral Simulator" components in Task B.
"""
from __future__ import annotations

import json

from app.llm import chat_json

SYSTEM = """You are a recommendation strategist. Given a user persona, their
recent emotional state, and the current situational context, you decide what
the recommender should PRIORITISE for this request. Return strict JSON only."""

USER_TEMPLATE = """PERSONA: {persona}
MEMORY: {memory}
BEHAVIOR_ANALYSIS: {behavior}
CONTEXT: {context}

Decide the user's current decision priorities and return JSON with EXACTLY:
{{
  "primary_objective": <one of:
      "comfort_and_speed", "value_for_money", "novelty_and_exploration",
      "reliability_and_safety", "indulgence", "routine_replenishment">,
  "ranking_weights": {{
    "price": <0-1>,
    "speed": <0-1>,
    "quality": <0-1>,
    "novelty": <0-1>,
    "reliability": <0-1>
  }},
  "must_avoid": [<short phrases — categories/qualities to deprioritize>],
  "plan_summary": "<one sentence: the strategy in plain English>"
}}

Hard rules:
- ranking_weights must sum to approximately 1.0 (within 0.1)
- must_avoid should be grounded in the user's open_friction (if any) and persona harshness
"""


def plan(persona: dict, memory: dict, behavior: dict, context: dict) -> dict:
    user_msg = USER_TEMPLATE.format(
        persona=json.dumps(
            {k: persona.get(k) for k in ("behavioral_profile", "economic_profile", "temporal_profile", "llm_traits")},
            default=str,
        ),
        memory=json.dumps(memory.get("short_term", {}), default=str),
        behavior=json.dumps(behavior, default=str),
        context=json.dumps(context, default=str),
    )
    try:
        out = chat_json(SYSTEM, user_msg, temperature=0.2)
    except Exception as e:  # noqa: BLE001
        out = _fallback(persona, behavior)
        out["plan_summary"] = f"(LLM planner unavailable: {e}) — heuristic fallback"
        return out

    # Sanity-clamp weights
    weights = out.get("ranking_weights") or {}
    total = sum(float(weights.get(k, 0.0)) for k in ("price", "speed", "quality", "novelty", "reliability"))
    if total <= 0:
        out["ranking_weights"] = _fallback(persona, behavior)["ranking_weights"]
    else:
        out["ranking_weights"] = {
            k: round(float(weights.get(k, 0.0)) / total, 3)
            for k in ("price", "speed", "quality", "novelty", "reliability")
        }
    out.setdefault("must_avoid", [])
    return out


def _fallback(persona: dict, behavior: dict) -> dict:
    behav = persona.get("behavioral_profile", {})
    econ = persona.get("economic_profile", {})
    weights = {
        "price": 0.30 if econ.get("budget_sensitive") else 0.15,
        "speed": 0.25 if behavior.get("wants_fast_delivery") else 0.15,
        "quality": 0.20 + 0.10 * behav.get("quality_sensitivity", 0.0),
        "novelty": 0.10,
        "reliability": 0.20,
    }
    s = sum(weights.values())
    weights = {k: round(v / s, 3) for k, v in weights.items()}
    return {
        "primary_objective": "value_for_money" if econ.get("budget_sensitive") else "comfort_and_speed",
        "ranking_weights": weights,
        "must_avoid": [],
        "plan_summary": "Heuristic plan derived from persona traits.",
    }
