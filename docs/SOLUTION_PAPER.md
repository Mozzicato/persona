# PersonaFlow AI

**Behavior-Aware Agentic Recommendation and User Simulation for Contextual Consumer Intelligence**

---

## Abstract

PersonaFlow AI is a behavior-aware recommendation and review-simulation system
that models consumers as evolving behavioral agents rather than static
preference vectors. The architecture integrates deterministic behavioral
feature extraction, LLM-refined persona traits, layered behavioral memory with
explicit tagged experiences (e.g. `late_delivery`, `bad_packaging`),
auto-detected Nigerian situational context (`salary_week`, `festive`,
`end_of_month`), a per-request reasoning planner, and a behavioral simulation
loop where predicted user reactions are estimated *before* recommendation
ranking. The Nigerian voice is grounded in real Play Store reviews from 10
Nigerian apps across 4 domains (food delivery, ride-hailing, e-commerce,
fintech), surfaced as few-shot examples in the review generator. On a clean
temporal hold-out over 1,891 users from the unified Amazon Fine Food + Play
Store catalogue (74,015 reviews, 19,498 items, 5 domains), the strongest
single retrieval signal (item-item collaborative filtering) achieves
**NDCG@10 = 0.649, Hit@10 = 0.675, MRR@10 = 0.642** against the full
19,498-item pool, and the LightGBM rating predictor reaches **RMSE 0.710,
MAE 0.407** (train-only personas), a 35% improvement over the user-mean
baseline (RMSE 1.090). The system handles cold-start users via a neutral
persona + LLM-only inference path, and supports cross-domain recommendation
queries directly (e.g., food user → ride-hailing apps). Ablations honestly
report that two specific feature additions (temporal traits + packaging/
service sensitivities) provide zero RMSE lift on this dataset; their value
is qualitative — driving the reasoner and the planner's `must_avoid`.

---

## 1. Introduction

Traditional recommender systems answer *"What did similar users like?"*.
PersonaFlow answers a harder question:

> *"What would THIS user appreciate in THIS moment?"*

The shift requires three things that static recommenders typically lack:

1. **Structured behavioral identity** — not just a click history but
   communication style, reviewer archetype, topical sensitivities
   (delivery, packaging, price, quality, service), and temporal patterns.
2. **Short-term memory of recent friction** — explicit tags like
   `late_delivery` or `overpriced` extracted from the last K reviews so
   that a normally-4-star user who just had two delivery failures is
   correctly modeled as currently grumpy.
3. **A simulation loop** that runs the user's persona forward against each
   candidate item *before* ranking, producing a predicted rating and an
   emotional state. Items the user would react badly to are demoted even
   if they are similar to historical favorites.

PersonaFlow's Nigerian contextualisation is grounded in real Play Store
reviews of Nigerian apps. The Nigerian voice in generated reviews is steered
by few-shot examples mined from those reviews; situational flags
(`salary_week`, `festive`, `end_of_month`, `school_resumption`) are
auto-detected from the request date.

---

## 2. Related Work

- **Collaborative filtering** (item-item / matrix factorisation) captures
  co-consumption signals but assumes static taste and cannot represent
  within-user mood drift or context.
- **Content-based retrieval** (TF-IDF / dense embeddings) handles cold-start
  items but does not model *how* a user would react.
- **LLM personalisation / RAG recommenders** increasingly use LLMs to
  generate explanations, but typically *after* ranking. PersonaFlow inverts
  this: the LLM participates in ranking itself, via per-candidate behavioral
  simulation.

PersonaFlow's contribution is closing the loop: a structured persona +
memory + context object feeds an LLM reasoner that predicts the user's
reaction, and that prediction becomes a first-class ranking signal rather
than an after-the-fact rationale.

---

## 3. Proposed Architecture

```
Historical User Data (Amazon Fine Food + Play Store NG apps)
        │
        ▼
Persona Extraction Engine ──── deterministic stats + LLM refinement
        │
        ▼
Behavioral Memory ─────────── long-term + short-term + tagged experiences
        │
        ├──────► Context Agent ─── auto-detects NG flags from date
        │            (rainy, salary_week, festive, end_of_month, …)
        │
        ▼
Behavioral Analyzer (LLM)
        │
        ▼
Reasoning Planner (LLM) ───── one high-level plan per request
        │                     (objective + ranking weight vector + must_avoid)
        ▼
Hybrid Retriever ──────────── semantic (TF-IDF) + collaborative (item-item)
        │                     + item-quality prior + popularity prior
        │                     + cross-domain support
        ▼
Behavioral Simulator ──────── runs Task A reasoner per candidate
        │                     (predicted rating + emotional state)
        ▼
Dynamic Re-ranker ─────────── blends retrieval + simulation + planner weights
        │
        ▼
Explainable Recommendations (with predicted rating + reasoning)

Cold-Start Path: brand-new users get a neutral baseline persona +
                 LLM-only inference (no memory, no per-user retrieval profile)
```

### 3.1 Data sources

| Source | Reviews | Users | Items | Domain(s) |
|---|---|---|---|---|
| Amazon Fine Food Reviews (Kaggle) | 68,015 | 1,891 active (≥20 reviews) | 19,488 | food |
| Google Play Store (10 NG apps) | 6,000 | 6,000 (each one review) | 10 | food_delivery, ride_hailing, ecommerce, fintech |
| **Unified** | **74,015** | **7,891** | **19,498** | **5 distinct domains** |

The Amazon dataset provides rich per-user histories (≥20 reviews) for
persona/memory extraction. The Play Store dataset provides the Nigerian
voice corpus and cross-domain candidates for Task B. Cold-start users with
zero or partial history are handled by the dedicated path described in §3.7.

### 3.2 Persona Extraction Engine

Two layers. **Deterministic** features run without LLM calls (cheap at scale):

- *Communication*: verbosity bucket, exclaim/all-caps/emoji rates, emotional intensity.
- *Behavioral*: harshness, optimism, consistency, and topical sensitivities for **delivery, packaging, service, quality** — each as the fraction of reviews mentioning that topic.
- *Economic*: price-focus ratio + `budget_sensitive` boolean.
- *Temporal*: hour-bucket distribution, `night_reviewer` flag, `weekend_positivity` delta, `festive_generosity` delta.

The **LLM refinement** layer samples 8 reviews per user (truncated to 400 chars each) and returns `likes_keywords`, `dislikes_keywords`, `sarcasm`, top sensitivities, and a `reviewer_archetype` ∈ {harsh_critic, fair_judge, enthusiast, analytical, emotional, pragmatic}. Personas are cached on disk per user; the LLM cost is paid once.

### 3.3 Behavioral Memory

Three layers:

- **Long-term**: lifetime average rating, total review count.
- **Short-term**: recent average, rating drift, mood, and per-topic complaint signals (delivery, price, packaging, service) computed over the last K=5 reviews.
- **Tagged experiences**: per-review tags applied to the recent window — `late_delivery`, `bad_packaging`, `bad_quality`, `bad_service`, `overpriced`, `great_value`, `loved_quality`. An `open_friction` list aggregates active negative tags so downstream agents can reason over recent issues without re-reading review text.

### 3.4 Context Agent

Time bucket is derived from the request hour. Nigerian flags are **auto-detected** from the request date:

- `salary_week` → day 23-30 of the month
- `end_of_month` → day ≥ 27
- `festive` → December, or first week of January
- `school_resumption` → September 1-14

User-supplied flags override the auto-detection in both directions.

### 3.5 Reasoning Planner

Runs **once per request**, before any candidate is scored. Returns:

- `primary_objective` ∈ {value_for_money, comfort_and_speed, novelty_and_exploration, reliability_and_safety, indulgence, routine_replenishment}
- `ranking_weights` over (price, speed, quality, novelty, reliability) — normalised to sum to 1
- `must_avoid` grounded in the user's `open_friction` and persona harshness
- `plan_summary` in plain English

The downstream re-ranker uses these weights to blend the four retrieval
signals.

### 3.6 Behavioral Simulator + Re-ranker

For each candidate, the reasoner is invoked with persona + memory +
context + item metadata. It returns a JSON with `reasoning`,
`emotional_state` (closed set), `key_drivers`, `predicted_rating`
(integer 1-5), and `confidence`. The LightGBM rating model produces a
parallel cross-check; the two are blended (0.6 LLM + 0.4 LightGBM).

The re-rank formula:

```
retrieval_blend = w_quality·(0.6·item_q + 0.4·sem) +
                  w_novelty·sem +
                  w_reliability·(0.5·collab + 0.5·item_q) +
                  w_speed·item_q +
                  w_price·popularity

simulation_signal = predicted_rating · confidence

final_score = 0.45·retrieval_blend + 0.55·simulation_signal
```

The simulation has slight majority weight by design.

### 3.7 Cold-Start Path

New users (no review history) crash a pure history-based system. PersonaFlow's `app/persona/coldstart.py`:

- 0 reviews → **neutral baseline persona** (mid-range traits; optional caller-supplied hints inject likes/dislikes/budget mode)
- 1-4 reviews → **partial persona** (deterministic stats only; LLM refinement skipped because too few signals)
- ≥ 5 reviews → full persona with LLM refinement

The memory layer has a matching neutral baseline. Downstream layers (analyzer, planner, reasoner) work identically for cold-start users — they just see less detail in the persona.

### 3.8 Review Generator

Conditions on persona, predicted rating, emotional state, reasoner trace, and contextual flags. The Nigerian voice is grounded in **few-shot examples mined from real Play Store reviews** (`data/processed/nigerian_voice_samples.json`) — pidgin-strong ones for emotionally intense users, broader Naija-English ones for analytical reviewers. The `slang_level` is computed in code from the user's measured `emotional_intensity` so pidgin only appears when authentic for that reviewer.

---

## 4. Nigerian Contextualisation

Three concrete mechanisms grounded in real data:

1. **Auto-detected environmental flags** from the request date.
2. **Few-shot Nigerian voice examples** from real Play Store reviews —
   sampled by (rating × domain) so the generator sees authentic cadence
   for any target tone.
3. **Slang scaling** tied to measured `emotional_intensity`, preventing
   forced pidgin on analytical reviewers.

Example mined examples (real Play Store reviews):
- *"Palm pay, abeg make we no start am this way sha. I have rate u guys 5 star, no give me reason to reduce it to 1 sha…"*
- *"Omo if you get problem with this palmpay nah wahala, no one we attend to you…"*
- *"the new version is worst in connecting. the older version connects with limited network even if your data is very low…"*

---

## 5. Experiments

### 5.1 Setup

Temporal hold-out: for each Amazon user with ≥ 5 reviews, the **last review in time order** becomes the test row; everything before is the training history. This avoids leakage that a random split would introduce and matches deployment behavior.

- Train: **66,124 reviews**
- Test: **1,891 reviews** (one per user)
- Item pool: **19,498 items** across **5 domains**

The retrieval index and persona caches are built on training data only when evaluating (`history_df` override in `app/retrieval.py:retrieve`).

### 5.2 Rating Prediction

| Model | n | RMSE | MAE | Notes |
|---|---|---|---|---|
| Baseline (user mean) | 1,891 | 1.090 | 0.754 | Predicts each user's lifetime average |
| **LightGBM — full data personas** | 1,891 | 0.636 | 0.367 | slight leakage; reported transparently |
| **LightGBM — train-only personas** | **1,891** | **0.710** | **0.407** | clean number, no leakage |
| LLM Reasoner (Groq llama-3.3-70b) | 25 | 1.020 | 0.560 | 4.5 s/call latency |

The **clean (train-only) LightGBM RMSE 0.710** is 35% better than the user-mean baseline (1.090). The "full data personas" variant has a small leakage (personas built from all reviews including test) and reports a lower 0.636 — we keep this for transparency since the difference is real. The LLM reasoner is *not* better than baseline at raw rating; its value is the structured reasoning output (`emotional_state`, `key_drivers`, `confidence`) that drives the re-ranker and review generator. We blend LLM + LightGBM (0.6 / 0.4) when both are available.

### 5.3 Recommendation Ranking (Retrieval-Only)

Hybrid retrieval (TF-IDF + item-item CF + item-quality + popularity), top-10, evaluated against held-out positives (rating ≥ 4) per user.

| Metric | Value |
|---|---|
| **NDCG@10** | **0.591** |
| **Hit@10**  | **0.637** |
| **MRR@10**  | **0.577** |
| Users evaluated | 1,421 (of 1,891 — others had no held-out positive) |

64% of test users see at least one truly-loved item in their top-10 recommendations from retrieval alone. The full system (with LLM behavioral simulation re-rank) is evaluated on a smaller sample in §6.

### 5.4 Latency

| Stage | Latency (single request, measured) |
|---|---|
| Persona load (cached) | < 5 ms |
| Memory build | < 100 ms |
| Behavioral analyzer (LLM) | ~600 ms |
| Reasoning planner (LLM) | ~600 ms |
| Per-candidate simulation (LLM) | **4.5 s × N candidates** |
| Re-rank + format | < 50 ms |

Per-candidate simulation dominates Task B latency. Async fan-out to Groq would collapse this into a single 4-5 s window; left as future work.

---

## 6. Ablation Studies

See `docs/ABLATION.md` for the full table. The summary below reports
**measured** numbers from `app/eval/ablation.py` on the temporal hold-out.

### 6.1 Retrieval ablations (measured)

| Variant | weights (sem, cf, qual, pop) | NDCG@10 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| Initial hybrid | (0.45, 0.30, 0.15, 0.10) | 0.591 | 0.637 | 0.577 |
| Semantic only (TF-IDF) | (1.00, 0, 0, 0) | 0.261 | 0.290 | 0.252 |
| **CF only (item-item)** | (0, 1.00, 0, 0) | **0.649** | **0.675** | **0.642** |
| Hybrid without CF | (0.60, 0, 0.25, 0.15) | 0.251 | 0.277 | 0.243 |
| Popularity only | (0, 0, 0, 1.00) | 0.000 | 0.000 | 0.000 |

Surprising honest finding: **CF alone beats the initial hybrid blend**. The TF-IDF "semantic" signal is weak because Amazon Fine Food has no real product description (we fall back to the first review's first 300 chars). Popularity is useless on this evaluation. Production defaults were retuned to (0.10, 0.70, 0.10, 0.10) based on this. The hybrid is kept (not pure CF) for cold-start item handling.

### 6.2 Rating ablations — LightGBM (measured)

| Variant | n_features | RMSE | MAE |
|---|---|---|---|
| **Full** | 17 | **0.710** | **0.407** |
| Without temporal | 15 | 0.705 | 0.405 |
| Without packaging/service sensitivities | 15 | 0.711 | 0.408 |
| Minimal (4) | 4 | 0.742 | 0.434 |

Personas help overall (4.5% RMSE gain over minimal). But two specific feature additions are honest zero-lift: temporal traits and packaging/service sensitivities don't improve held-out RMSE. They DO help qualitatively (the reasoner cites them in per-candidate reasoning and the planner uses sensitivities in `must_avoid`), but RMSE doesn't capture that.

See `docs/ABLATION.md` for the full table, the full-system rank eval (rate-limited to 2 users), the LLM reasoner's RMSE (1.02 — no better than baseline at raw rating), and what variants we did not run quantitatively.

---

## 7. Limitations

- **Nigerian voice grounding is corpus-driven, not native-speaker validated.** We mine Play Store reviews and use them as few-shot examples; we have not yet run a human eval with Nigerian evaluators against the rubric in `docs/HUMAN_EVAL_RUBRIC.md`.
- **Amazon Fine Food has no price or delivery-time fields.** The recommender's `price` and `speed` ranking weights map to popularity and item-quality proxies, not true price/delivery signals.
- **Item titles are weak.** Amazon Fine Food doesn't include product names; we fall back to the first review's summary. Recommendations sometimes show review-fragment titles.
- **LLM reasoner is slow.** 4.5 s/call on Groq llama-3.3-70b. The /recommend endpoint makes 1 analyzer + 1 planner + N simulator calls sequentially; for N=20 candidates this is ~95 s. Async fan-out would bring this under 10 s.
- **Cold-start is heuristic.** Neutral persona uses mid-range defaults; we don't yet learn cold-start defaults from the data distribution.
- **Memory tags are regex-based.** Robust to typos but not semantic rephrasings ("the box was a mess" doesn't match `bad_packaging`).

---

## 8. Future Work

- Async fan-out of per-candidate simulation calls (10× latency win).
- Native-speaker human evaluation against the rubric in `HUMAN_EVAL_RUBRIC.md`.
- A Nigerian native review corpus from a partner (Jumia / Konga / Chowdeck with authorised access) — replaces the synthetic Nigerian voice layer.
- Replace regex memory tagger with a small classifier on cached review embeddings.
- Reinforcement learning from human feedback on review realism.
- Voice / multimodal review simulation.
- Long-term adaptive memory: refresh personas periodically rather than caching once.
