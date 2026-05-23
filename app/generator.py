"""Review Generator: writes the final user-facing review text, conditioned on
persona + reasoner output + Nigerian voice few-shot examples from real
Play Store reviews.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from app.config import PROCESSED_DIR
from app.llm import chat
from app.config import GROQ_MODEL_FAST

NIGERIAN_VOICE_GUIDE = """NIGERIAN VOICE GUIDE:
- Direct, blunt complaints when negative. Specific issues - data, charges, support, delivery.
- Light pidgin markers when emotionally intense: "abeg", "sha", "omo", "no wahala", "well well", "stress me die", "vex"
- Match slang_level to persona: minimal=plain English, low=at most one marker, medium=1-2 markers per review, high=natural pidgin throughout
- NEVER force pidgin if persona is analytical/formal
- Real Nigerian reviews are often UNFILTERED - typos, missing punctuation, all caps for emphasis are AUTHENTIC"""


_VOICE_SAMPLES_PATH = PROCESSED_DIR / "nigerian_voice_samples.json"
_VOICE_CACHE: dict | None = None


def _load_voice_samples() -> dict:
    global _VOICE_CACHE
    if _VOICE_CACHE is not None:
        return _VOICE_CACHE
    if not _VOICE_SAMPLES_PATH.exists():
        _VOICE_CACHE = {"by_rating_domain": {}, "pidgin_strong": [], "best_overall": []}
        return _VOICE_CACHE
    _VOICE_CACHE = json.loads(_VOICE_SAMPLES_PATH.read_text(encoding="utf-8"))
    return _VOICE_CACHE


def _few_shot_examples(rating: int, slang_level: str, n: int = 3) -> list[str]:
    """Pick few-shot Nigerian examples that match the target rating's tone."""
    samples = _load_voice_samples()
    rating_bucket = "low" if rating <= 2 else ("mid" if rating == 3 else "high")
    pool: list[dict] = []
    if slang_level in ("medium", "high"):
        pool.extend(samples.get("pidgin_strong", []))
    by_rd = samples.get("by_rating_domain", {}).get(rating_bucket, {})
    for items in by_rd.values():
        pool.extend(items)
    pool.extend(samples.get("best_overall", []))

    bucket_matches = [s for s in pool if s.get("rating") and (
        (rating <= 2 and s["rating"] <= 2)
        or (rating == 3 and s["rating"] == 3)
        or (rating >= 4 and s["rating"] >= 4)
    )]
    chosen_pool = bucket_matches or pool
    random.shuffle(chosen_pool)
    return [s["text"] for s in chosen_pool[:n] if s.get("text")]


SYSTEM = f"""You write authentic Nigerian consumer reviews. You preserve a
user's personality and produce text that feels human - slightly imperfect,
emotionally believable, never over-polished.

{NIGERIAN_VOICE_GUIDE}

Hard rules:
- Output ONLY the review text. No labels, no rating, no preamble.
- 1-4 sentences depending on persona verbosity (high=3-4, medium=2-3, low=1-2).
- Keep it grounded in the reasoning provided - don't invent new facts.
- Match the predicted rating's emotional tone.
- The few-shot examples below are REAL Nigerian app reviews - mirror their cadence and bluntness, do NOT copy their content."""

USER_TEMPLATE = """REAL NIGERIAN VOICE EXAMPLES (rating={rating}, slang_level={slang_level}):
{few_shot}

---

PERSONA SUMMARY:
verbosity={verbosity} | harshness={harshness} | optimism={optimism} | emotional_intensity={emotional_intensity}
slang_level={slang_level} | archetype={archetype}

ITEM: {item}

PREDICTED RATING: {rating}/5
EMOTIONAL STATE: {emotional_state}
REASONING: {reasoning}

CONTEXT FLAGS: {flags}

Write the review now (mirror the cadence of the examples, but speak about THIS item):"""


def generate(persona: dict, item: dict, reasoner_out: dict, context: dict) -> str:
    comm = persona.get("communication_style", {})
    behav = persona.get("behavioral_profile", {})
    llm_traits = persona.get("llm_traits", {}) or {}
    intensity = comm.get("emotional_intensity", 0.3)
    if intensity > 0.5:
        slang_level = "medium"
    elif intensity > 0.2:
        slang_level = "low"
    else:
        slang_level = "minimal"

    rating = reasoner_out["predicted_rating"]
    examples = _few_shot_examples(rating, slang_level, n=3)
    few_shot_block = "\n".join(f"- {e}" for e in examples) if examples else "(no examples available)"

    user_msg = USER_TEMPLATE.format(
        few_shot=few_shot_block,
        verbosity=comm.get("verbosity", "medium"),
        harshness=behav.get("harshness", 0.5),
        optimism=behav.get("optimism", 0.5),
        emotional_intensity=intensity,
        slang_level=slang_level,
        archetype=llm_traits.get("reviewer_archetype", "fair_judge"),
        item=_item_blurb(item),
        rating=rating,
        emotional_state=reasoner_out.get("emotional_state", "neutral"),
        reasoning=reasoner_out.get("reasoning", ""),
        flags=", ".join(context.get("nigerian_flags", [])) or "none",
    )
    text = chat(SYSTEM, user_msg, model=GROQ_MODEL_FAST, temperature=0.85, max_tokens=220)
    return text.strip().strip('"')


def _item_blurb(item: dict) -> str:
    parts = []
    for k in ("name", "category", "domain", "price_range", "price_level"):
        if item.get(k):
            parts.append(f"{k}={item[k]}")
    return " | ".join(parts) or str(item)
