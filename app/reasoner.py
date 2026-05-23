"""Review Reasoner: produces explicit chain-of-thought + predicted rating
BEFORE the generator writes any user-facing text. This is the differentiator.
"""
from __future__ import annotations

from app.llm import chat_json
from app.config import GROQ_MODEL_REASONING

SYSTEM = """You are a behavioral simulation engine. Given a user's persona,
their recent emotional state, situational context, and an item under
consideration, you reason about how THIS specific user would react RIGHT NOW.

You must:
1. Cite specific persona traits and recent signals that drive the reaction.
2. Acknowledge contextual modifiers (time, weather, Nigerian flags).
3. Predict an integer rating 1-5 grounded in the user's historical distribution.
4. Describe the likely emotional state.

Return STRICT JSON. No prose outside JSON."""

USER_TEMPLATE = """PERSONA:
{persona}

MEMORY (recent state):
{memory}

CONTEXT (this moment):
{context}

ITEM under consideration:
{item}

Return JSON with this exact schema:
{{
  "reasoning": "<3-5 sentences explaining the user's likely reaction, citing traits>",
  "emotional_state": "<one of: delighted, satisfied, neutral, annoyed, frustrated, disappointed>",
  "key_drivers": [<up to 4 short phrases — the dominant signals>],
  "predicted_rating": <integer 1-5>,
  "confidence": <float 0-1>
}}"""


def reason(persona: dict, memory: dict, context: dict, item: dict) -> dict:
    user_msg = USER_TEMPLATE.format(
        persona=_compact(persona),
        memory=_compact(memory),
        context=_compact(context),
        item=_compact(item),
    )
    out = chat_json(SYSTEM, user_msg, model=GROQ_MODEL_REASONING, temperature=0.3)
    out["predicted_rating"] = int(max(1, min(5, int(out.get("predicted_rating", 3)))))
    out.setdefault("confidence", 0.6)
    return out


def _compact(d: dict) -> str:
    """Shrink dicts before sending to LLM — drop noise fields."""
    import json
    keep = {}
    for k, v in d.items():
        if k in {"last_k_summaries"}:
            keep[k] = [str(x)[:80] for x in (v or [])][:5]
        elif k == "llm_traits" and isinstance(v, dict):
            keep[k] = {kk: v[kk] for kk in v if kk != "summary"}
            keep[k]["summary"] = v.get("summary", "")
        else:
            keep[k] = v
    return json.dumps(keep, ensure_ascii=False, default=str)
