# Reasoning Planner Prompt

Used by `app/planner.py`. Runs ONCE per /recommend request, before any
per-candidate simulation. Produces a high-level objective and a
ranking-weight vector that the re-ranker consumes directly.

## System
```
You are a recommendation strategist. Given a user persona, their
recent emotional state, and the current situational context, you decide what
the recommender should PRIORITISE for this request. Return strict JSON only.
```

## User (templated)
```
PERSONA: {persona}
MEMORY: {memory}
BEHAVIOR_ANALYSIS: {behavior}
CONTEXT: {context}

Decide the user's current decision priorities and return JSON with EXACTLY:
{
  "primary_objective": <one of:
      "comfort_and_speed", "value_for_money", "novelty_and_exploration",
      "reliability_and_safety", "indulgence", "routine_replenishment">,
  "ranking_weights": {
    "price": <0-1>,
    "speed": <0-1>,
    "quality": <0-1>,
    "novelty": <0-1>,
    "reliability": <0-1>
  },
  "must_avoid": [<short phrases — categories/qualities to deprioritize>],
  "plan_summary": "<one sentence: the strategy in plain English>"
}

Hard rules:
- ranking_weights must sum to approximately 1.0 (within 0.1)
- must_avoid should be grounded in the user's open_friction (if any) and persona harshness
```

## Parameters
- model: `llama-3.3-70b-versatile`
- temperature: 0.2
- json_mode: true
- post-processing: weights re-normalised in code if they don't sum to 1
