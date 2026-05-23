# Prompt Engineering

PersonaFlow uses **five** distinct prompt templates, each tuned to a specific
job. They all run through Groq (`llama-3.1-8b-instant` for low-stakes / fast
calls, `llama-3.3-70b-versatile` for reasoning and JSON-structured outputs).

The actual prompt strings live alongside the code that uses them — see
`prompts/` for canonical copies.

## 1. Persona Refinement (`app/persona/refine.py`)

**Role.** Extract qualitative traits (likes, dislikes, archetype) that
deterministic regex/stats can't capture.

**Design choices**
- *JSON-strict mode* (`response_format={"type": "json_object"}`). We never
  parse free text.
- *Few-shot via real data*: we feed 8 actual reviews from the user, including
  the rating, so the model grounds its inferences in the user's voice.
- *Schema in the user message, not the system*: the system message says
  "return JSON only"; the user message specifies field names and value
  ranges. This stops the model from copying the schema literally when the
  reviews fail to support inference.
- *Bounded enumerations* for `sensitivities` and `reviewer_archetype` — the
  downstream code matches on these strings, so we constrain the vocabulary.
- *Sample bounded at 8 reviews × 400 chars each* — keeps context small,
  inference fast, and avoids drowning the model in repetitive text.

## 2. Review Reasoner (`app/reasoner.py`)

**Role.** Produce the chain-of-thought that drives both the rating and the
review *before* any user-facing text is generated. Same prompt is reused
inside Task B for per-candidate simulation.

**Design choices**
- *Compact input via `_compact()`*: we trim `last_k_summaries` to ≤ 5
  entries of ≤ 80 chars; we drop the LLM-refined `summary` blob from
  `llm_traits` since it duplicates other fields. This cuts prompt size by
  ~40% with no measurable quality loss.
- *Hard schema with `predicted_rating: <integer 1-5>`*: the wrapper clamps
  to `[1, 5]` defensively in case the model returns `0` or `6`.
- *Forced citation*: the system prompt says "cite specific persona traits
  and recent signals that drive the reaction" — encourages traceable
  reasoning rather than hand-wavy generalities.
- *Temperature 0.3*: low enough to be consistent across re-runs of the same
  request, high enough for the reasoning text to feel natural.
- *`emotional_state` enumeration* is closed: `delighted / satisfied /
  neutral / annoyed / frustrated / disappointed`. This makes downstream
  conditional logic (e.g., voice selection in the generator) tractable.

## 3. Reasoning Planner (`app/planner.py`)

**Role.** One-shot request strategy: what should the recommender prioritise?

**Design choices**
- *Output is a weight vector, not prose*: the keys are wired directly into
  `_rerank_score()`. The model returns numbers we use.
- *Self-normalising weights*: the post-processing code re-normalises
  `ranking_weights` to sum to 1.0 if the model drifts. We don't trust the
  model to do math reliably.
- *Closed `primary_objective` enum*: six labels, none vague. This is what
  the LLM is allowed to choose between.
- *Heuristic fallback in code*: if the LLM call fails, we synthesise a
  fallback plan from `behavioral_profile` + `economic_profile` so the API
  still returns. The summary string tells the caller it was a fallback.
- *Grounded `must_avoid`*: the prompt says "must_avoid should be grounded
  in the user's `open_friction`" — pulling forward the tagged-experience
  layer from memory.

## 4. Behavioral Analyzer (`app/recommender.py:analyze_behavior`)

**Role.** Decode "what is this user feeling/wanting right now" into 5 fields
that the planner and reasoner can consume.

**Design choices**
- *Three inputs only* — persona slice, short-term memory, context — to
  prevent the model from over-reasoning on irrelevant historical detail.
- *Bool/enum/float typing in the schema*: we ask for booleans where we
  want booleans. The model is good at this when the schema makes it
  explicit; less good when we ask for "describe in a sentence."
- *Heuristic fallback* using `delivery_sensitivity > 0.3` etc., so the
  endpoint never hard-fails on an LLM error.

## 5. Review Generator (`app/generator.py`)

**Role.** Write the final user-facing review text.

**Design choices**
- *System prompt holds the Nigerian voice guide*; user prompt holds the
  reasoner's output. This separates style (stable, system-level) from
  content (per-request).
- *Voice scaling*: the slang_level passed in the user message is computed
  in code (`high → medium`, `medium → low`, `low → minimal`) based on the
  user's measured `emotional_intensity`. We don't ask the model to decide
  whether to use pidgin — we tell it how much to use.
- *Output-only rule*: the system says "Output ONLY the review text. No
  labels, no rating, no preamble." Otherwise the model loves to prepend
  "Rating: 3/5\n\nReview: …".
- *Length cap via verbosity*: `high=3-4 sentences`, `medium=2-3`, `low=1-2`
  — matches the user's actual reviewing length distribution.
- *Higher temperature (0.85)*: this is the only call where we want
  variation. Same persona + same item should still produce slightly
  different reviews on re-run.
- *Rule: do not invent new facts*. "Keep it grounded in the reasoning
  provided — don't invent new facts." Without this, the model fabricates
  prices and delivery times.

## Cross-cutting principles

1. **JSON or text — never both.** Every call is one or the other. We don't
   try to parse JSON out of a free-text response.
2. **Closed enumerations for everything that branches downstream.**
   `emotional_state`, `reviewer_archetype`, `primary_objective`,
   `current_mood` — all closed sets.
3. **Heuristic fallback for every LLM call.** No request should fail
   because a model call failed; it should degrade.
4. **Citation-first reasoning.** Prompts ask the model to cite which
   persona trait / memory tag / context flag drives each conclusion. This
   produces traceable outputs and makes the system easier to debug.
5. **Trim before you send.** `_compact()` in the reasoner is non-trivial
   savings at scale. Drop fields the model can't use.
6. **Two-model split.** `llama-3.1-8b-instant` for the generator (fast,
   creative). `llama-3.3-70b-versatile` for reasoner / planner /
   analyzer / persona refine (slower, more accurate).
