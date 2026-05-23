"""LLM-based persona refinement.

Uses Groq to look at a few sample reviews and extract qualitative traits
(food likes/dislikes, sarcasm tendency, sensitivity hotspots) that are
hard to capture with regex stats alone.
"""
from __future__ import annotations

import pandas as pd

from app.llm import chat_json

SYSTEM = """You analyze consumer review histories to infer behavioral traits.
You return STRICT JSON only - no prose, no markdown. Be concise and grounded
in the actual reviews provided."""

USER_TEMPLATE = """Below are sample reviews written by a single user. Infer their qualitative
behavioral traits. Return JSON with EXACTLY this schema:

{{
  "likes_keywords": [<up to 6 short noun phrases the user clearly likes>],
  "dislikes_keywords": [<up to 6 short noun phrases the user clearly dislikes>],
  "sarcasm": <float 0-1>,
  "sensitivities": [<up to 4 of: "delivery","packaging","price","freshness","portion","taste","service">],
  "reviewer_archetype": <one of: "harsh_critic","fair_judge","enthusiast","analytical","emotional","pragmatic">,
  "summary": <one sentence describing this reviewer>
}}

REVIEWS:
{reviews}
"""


def refine_persona(reviews: pd.DataFrame, max_samples: int = 8) -> dict:
    sample = reviews.sample(n=min(max_samples, len(reviews)), random_state=0)
    blocks = []
    for _, row in sample.iterrows():
        text = str(row["text"])[:400]
        blocks.append(f"- [rating={int(row['rating'])}] {text}")
    user = USER_TEMPLATE.format(reviews="\n".join(blocks))
    try:
        return chat_json(SYSTEM, user, temperature=0.2)
    except Exception as e:  # noqa: BLE001
        return {
            "likes_keywords": [],
            "dislikes_keywords": [],
            "sarcasm": 0.0,
            "sensitivities": [],
            "reviewer_archetype": "fair_judge",
            "summary": f"(LLM refinement unavailable: {e})",
        }
