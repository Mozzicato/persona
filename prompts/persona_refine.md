# Persona Refinement Prompt

Used by `app/persona/refine.py`. Source of truth for the prompt string lives
in code; this file is the human-readable canonical copy.

## System
```
You analyze consumer review histories to infer behavioral traits.
You return STRICT JSON only - no prose, no markdown. Be concise and grounded
in the actual reviews provided.
```

## User (templated)
```
Below are sample reviews written by a single user. Infer their qualitative
behavioral traits. Return JSON with EXACTLY this schema:

{
  "likes_keywords": [<up to 6 short noun phrases the user clearly likes>],
  "dislikes_keywords": [<up to 6 short noun phrases the user clearly dislikes>],
  "sarcasm": <float 0-1>,
  "sensitivities": [<up to 4 of: "delivery","packaging","price","freshness","portion","taste","service">],
  "reviewer_archetype": <one of: "harsh_critic","fair_judge","enthusiast","analytical","emotional","pragmatic">,
  "summary": <one sentence describing this reviewer>
}

REVIEWS:
{reviews}
```

## Parameters
- model: `llama-3.3-70b-versatile`
- temperature: 0.2
- json_mode: true
- review samples: 8, each truncated to 400 chars
