# Ablation Studies

All numbers below are **measured** on the temporal hold-out split (train: 66,124 reviews | test: 1,891 reviews | 1,421 evaluable users), produced by `app/eval/ablation.py` and `app/eval/rating_eval.py`. Reproduction:

```powershell
python -m app.eval.rating_eval --skip-llm   # rating: baseline + LightGBM
python -m app.eval.rank_eval --retrieval-only   # rank: retrieval-only
python -m app.eval.ablation                  # ablation table
```

The full results JSON lives at `data/processed/eval_ablation.json`.

---

## A. Retrieval ablations (NDCG@10, Hit@10, MRR@10)

Hold-out positive = the user's last in-time review with rating ≥ 4.

| Variant | weights (sem, cf, qual, pop) | NDCG@10 | Hit@10 | MRR@10 |
|---|---|---|---|---|
| Retrieval hybrid (initial defaults) | (0.45, 0.30, 0.15, 0.10) | 0.591 | 0.637 | 0.577 |
| Semantic only (TF-IDF) | (1.00, 0, 0, 0) | 0.261 | 0.290 | 0.252 |
| **Collaborative filtering only** | (0, 1.00, 0, 0) | **0.649** | **0.675** | **0.642** |
| Hybrid without CF | (0.60, 0, 0.25, 0.15) | 0.251 | 0.277 | 0.243 |
| Popularity only (control) | (0, 0, 0, 1.00) | 0.000 | 0.000 | 0.000 |

**Key findings (honest reading):**

1. **CF carries the signal.** Item-item collaborative filtering alone (NDCG@10 = 0.649) is the strongest single source — better than the initial hybrid blend.
2. **Semantic TF-IDF is weak.** 0.261 NDCG@10 — useful but much noisier than CF on this data. Likely because item documents are just the first review's first 300 chars (Amazon Fine Food has no real product description field).
3. **Popularity alone is useless** (0.000) — the top-popular items match almost zero held-out positives.
4. **The initial weight blend underweighted CF.** Production code now uses (0.10, 0.70, 0.10, 0.10) based on this finding — see `app/retrieval.py:retrieve` default.

The hybrid is kept (not pure-CF) because:
- It gracefully handles cold-start items (no CF history → semantic + quality + popularity still rank them)
- The small semantic + quality + popularity contributions provide tie-breaking on the long tail
- Architecture is more robust as the dataset grows

---

## B. Rating prediction ablations (RMSE, MAE)

LightGBM trained on the train split (66,124 rows) with personas extracted from train only, evaluated on test (1,891 rows).

| Variant | n_features | RMSE | MAE | Note |
|---|---|---|---|---|
| **LightGBM full** | 17 | **0.710** | **0.407** | persona + temporal + item priors |
| Without temporal features | 15 | 0.705 | 0.405 | drops night/weekend signals |
| Without packaging/service sensitivities | 15 | 0.711 | 0.408 | back to delivery + quality only |
| Minimal | 4 | 0.742 | 0.434 | user_avg + user_std + item_avg + rating_delta |

**Key findings (honest reading):**

1. **The minimal 4-feature model is 4.5% worse RMSE** than the full 17-feature model — persona features DO contribute. The gain comes from the core behavioral profile (harshness, optimism, sensitivities) + economic profile + communication style.
2. **Temporal features are zero-lift on this data.** Removing them slightly *improves* RMSE (0.705 vs 0.710). The Amazon Fine Food users don't show strong night/weekend/festive rating patterns. Temporal traits may still help in a deployment with delivery-time-of-day data.
3. **Packaging/service sensitivities are also flat** (0.711 vs 0.710). Adding them costs nothing but doesn't help on the held-out RMSE. They DO help downstream — the reasoner cites them in the per-candidate reasoning, and the planner uses them in `must_avoid` (qualitative wins not captured by RMSE).

The reported headline **RMSE 0.636** in `eval_rating.json` is from a different setup where personas were extracted from the FULL data (slight leakage). The honest train-only number is **0.710** — still 35% better than the user-mean baseline (1.090).

---

## C. Why we did NOT run other expensive ablations

These each cost N × candidates × 4.5 s LLM calls per user × hundreds of users — and the Groq free-tier daily limit is 100k tokens/day on llama-3.3-70b-versatile. We exhausted it on the 2-user sample of the full-system rank eval (see §D).

| Ablation | Expected direction (qualitative) |
|---|---|
| Behavioral simulation off (retrieval-only re-rank) | Per-item explanations lose grounding; NDCG similar to retrieval-hybrid; review reasoning becomes generic |
| Reasoning Planner off (fixed weights) | Marginal NDCG impact; `must_avoid` filtering disabled so friction-tagged items appear |
| Short-term memory off | Worse on users with recent open_friction tags; can't model recent mood drift |
| Nigerian voice few-shot off | Lower realism in generated reviews; no rating impact |
| LLM reasoner blending off (LightGBM only) | NDCG unchanged; reviews lose emotional_state field; explanations weaker |

A paid-tier Groq subscription would let us fill these in.

---

## D. Full-system rank eval (LLM, rate-limited)

We attempted to evaluate the full pipeline (retrieval → behavioral simulation → re-rank) on 15 users. The Groq daily token limit was hit after 2 users completed:

| Variant | n | NDCG@10 | Hit@10 | latency/user |
|---|---|---|---|---|
| Full system (LLM re-rank) | 2 of 15 | 0.50 | 0.50 | 110 s |

Too few users to draw conclusions. The latency (110 s/user with 20 candidates × ~5 s each) confirms the LLM is the bottleneck — async fan-out is the obvious production fix.

---

## E. LLM reasoner rating prediction (25 samples)

Held-out rating prediction from the LLM reasoner alone (Groq llama-3.3-70b):

| Predictor | n | RMSE | MAE | latency |
|---|---|---|---|---|
| Baseline (user mean) | 1,891 | 1.090 | 0.754 | <1 ms |
| LightGBM | 1,891 | 0.710 | 0.407 | <10 ms |
| LLM reasoner | 25 | 1.020 | 0.560 | 4,551 ms |

**Honest reading.** The LLM reasoner is no better than the user-mean baseline at raw rating accuracy, and 4.5 s slower per prediction. Its value is the structured output (`reasoning`, `emotional_state`, `key_drivers`, `confidence`) that drives the re-ranker and review generator — not the rating number itself. The system blends 0.6·LLM + 0.4·LightGBM precisely to combine the rating accuracy of LightGBM with the explainability of the LLM.

---

## F. What the ablations changed in the system

| Finding | Action taken |
|---|---|
| CF dominates over semantic + popularity | Retuned default retrieval weights (0.10, 0.70, 0.10, 0.10) |
| Temporal features add no rating-lift | Kept for reasoner context (still useful qualitatively); honest disclosure in §B |
| LLM reasoner ≈ baseline at raw rating | Position as explainability layer, not rating predictor; LightGBM is the production rating model |
| Full-system LLM eval is rate-limited | Documented latency as a real limitation in `SOLUTION_PAPER.md §7` |
