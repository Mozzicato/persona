# PersonaFlow AI

**Behavior-aware recommendation and review simulation for dynamic Nigerian consumers.**

🚀 **Live demo:** https://huggingface.co/spaces/mozzic/personaflow
📂 **Code:** https://github.com/Mozzicato/persona

PersonaFlow models each user as an *evolving behavioral agent* rather than a
static preference vector. Two tasks, one architecture:

- **Task A — Review Simulation** (`POST /simulate-review`): persona-aware
  rating prediction + Nigerian-voice review generation
- **Task B — Recommendation** (`POST /recommend`): behavior-aware
  recommendations with cross-domain support and cold-start handling. Task A
  is embedded *inside* Task B — every candidate is simulated before ranking.

---

## 🧪 How to use the live demo

The fastest way to evaluate PersonaFlow: open the live demo and use the
interactive UI. No setup needed.

**Open:** [https://mozzic-personaflow.hf.space](https://mozzic-personaflow.hf.space)

You'll see three test cards, each with **editable form fields** and a **Run** button:

### 🟢 Card 1 — Task A: Simulate a Review

What it does: predicts a rating and generates a persona-aware Nigerian-voice
review for any item under any context.

Fields you can edit:
- **User ID** — defaults to a real Amazon Fine Food user (`A100WO06OQR8BQ`). Click *Load sample user IDs* below the heading to see more real IDs. Or enter any random string for a cold-start.
- **Item name + category** — what the user is reviewing (e.g. Chowdeck / food_delivery).
- **Time of day** — morning / afternoon / evening / night.
- **Nigerian context flags** — toggle rainy, heavy traffic, salary week, festive, fuel scarcity, power outage. These shape the reasoner's output.

Click **Run** → in ~5 s you'll see:
- ⭐⭐⭐ Star rating + emotion chip (e.g. "frustrated", "satisfied")
- The review text in a green quote box
- *Why this rating?* — expandable reasoning explaining which persona traits + context flags drove the prediction
- *Raw JSON* — the full API response

### 🔵 Card 2 — Task B: Recommend (with cross-domain option)

What it does: returns top-N personalised recommendations. With
`cross_domain: true` it deliberately recommends items from domains the user
has NOT touched (food → Nigerian apps).

Fields you can edit:
- **User ID, time, mood, top-N**
- **Cross-domain checkbox** — flips between same-domain food recommendations and cross-domain Nigerian apps
- **Salary week / Festive** — context flags

Click **Run** → in ~40-60 s (multiple LLM calls, one per candidate) you'll see:
- 📋 The planner's **strategy** at the top (one-sentence plain English)
- Top-N ranked items, each with:
  - Predicted rating + emotion chip
  - The reasoner's explanation of *why* this item fits the user right now
  - Key drivers (e.g. "high quality sensitivity", "recent price complaints")
  - Retrieval source (collaborative / semantic / popularity / item_quality)

### 🟣 Card 3 — Cold-Start: Brand-new user

What it does: a user with **zero history** in our catalogue still gets a
coherent persona-driven review.

Fields you can edit:
- **New user ID** — any string that doesn't exist in our dataset
- **Item name + category**
- **Likes** (comma-separated) — preference hints to seed the neutral persona
- **Budget-sensitive, salary week, festive, rainy** flags

Click **Run** → see how the system synthesises a persona from hints alone.

### ⚙️ Power user: direct API access

If you prefer Swagger or curl:
- **Swagger UI:** https://mozzic-personaflow.hf.space/docs
- **Sample IDs:** `GET /users?limit=20`
- **Endpoints:** `POST /simulate-review` (Task A), `POST /recommend` (Task B), `POST /persona` (view full persona for a user)

Full sample request/response bodies are in [`SUBMISSION.md`](SUBMISSION.md).

## Headline numbers

On a clean temporal hold-out (last review per user; train: 66,124 / test:
1,891 / 1,421 evaluable users):

| Metric | Value | Note |
|---|---|---|
| **NDCG@10** | **0.649** | item-item CF (strongest single retrieval signal) |
| **Hit@10**  | **0.675** | 67.5% of users see a relevant item in top-10 |
| **MRR@10**  | **0.642** | mean reciprocal rank of first relevant hit |
| **Rating RMSE (LightGBM)** | **0.710** | clean train-only personas |
| Rating MAE (LightGBM) | 0.407 | |
| Rating RMSE (user-mean baseline) | 1.090 | LightGBM is 35% better |

LightGBM is **35% better** than the user-mean baseline on RMSE. NDCG@10 of
0.649 against a 19,498-item pool is strong. The ablation surprisingly found
CF alone beats the initial hybrid blend (which underweighted CF); production
weights were retuned to favour CF (see `docs/ABLATION.md` §A).

## Data

| Source | Reviews | Users | Items | Domain(s) |
|---|---|---|---|---|
| Amazon Fine Food Reviews | 68,015 | 1,891 active | 19,488 | food |
| Google Play (10 NG apps) | 6,000 | 6,000 | 10 | food_delivery, ride_hailing, ecommerce, fintech |
| **Unified** | **74,015** | **7,891** | **19,498** | **5 distinct domains** |

The Amazon dataset gives rich per-user histories; Play Store gives Nigerian
voice corpus + cross-domain candidates. Nigerian voice few-shot examples
are mined from real Naija reviews like *"Omo if you get problem with this
palmpay nah wahala"*.

## Architecture

```
History (Amazon + Play Store)
        │
        ▼
Persona Extraction ───── stats + temporal + LLM refinement
        │
        ▼
Behavioral Memory ────── long-term + short-term + tagged experiences
        │                (late_delivery, bad_packaging, overpriced, …)
        │
        ├── Context Agent ── auto NG flags (salary_week, festive, …)
        │
        ▼
Behavioral Analyzer + Reasoning Planner (one LLM call each per request)
        │
        ▼
Hybrid Retriever ─────── TF-IDF + item-item CF + quality + popularity
        │                (+ cross_domain flag)
        ▼
Behavioral Simulator ─── runs Task A reasoner per candidate
        │                (predicted rating + emotional state + reasoning)
        ▼
Dynamic Re-ranker ────── blends retrieval + simulation + planner weights
        │
        ▼
Explainable Output (rating + review/recommendation + reason + key_drivers)
```

## Quick start

```powershell
# 1. install
python -m pip install -r requirements.txt

# 2. make sure .env has GROQ_API_KEY=...

# 3. run the whole data pipeline (one-time, ~10 min)
.\scripts\reproduce.ps1
# or on Linux/Mac: bash scripts/reproduce.sh

# 4. start the API
$env:PYTHONPATH = "."
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive Swagger UI.

## Endpoints

### `GET /users?limit=20`
Returns sample `user_id`s you can plug into the other endpoints.

### `POST /persona`
```json
{ "user_id": "A3SGXH7AUHU8GW" }
```
Returns the full persona (communication, behavioral, economic, temporal
profiles + LLM-refined traits). For cold-start users, pass
`cold_start_hints`:
```json
{ "user_id": "new_user_42",
  "cold_start_hints": { "budget_sensitive": true, "likes": ["spicy"], "archetype": "pragmatic" } }
```

### `POST /simulate-review` (Task A)
```json
{
  "user_id": "A100WO06OQR8BQ",
  "item": {"name": "Chowdeck", "category": "food_delivery", "item_id": "com.chowdeck.app"},
  "context": {"time": "night", "weather": "rainy", "traffic_heavy": true}
}
```
Returns:
```json
{
  "rating": 2,
  "review": "Omo the food no bad but rider stress me die...",
  "reasoning": "User is delivery-sensitive (0.42), recent late_delivery tag fired…",
  "emotional_state": "frustrated",
  "key_drivers": ["delivery delay", "rainy weather", "open_friction: late_delivery"],
  "confidence": 0.78,
  "context_used": { "nigerian_flags": ["rainy","traffic_heavy"] }
}
```

### `POST /recommend` (Task B)
Same-domain:
```json
{ "user_id": "A3SGXH7AUHU8GW",
  "context": {"time": "night", "mood": "tired"},
  "top_n": 5 }
```
Cross-domain (food user → recommend apps):
```json
{ "user_id": "A3SGXH7AUHU8GW",
  "context": {"time": "evening"},
  "top_n": 5,
  "cross_domain": true }
```
Returns `plan` (planner's strategy + weights), `behavior_analysis`,
`cross_domain` flag, and a ranked list with `predicted_rating`,
`emotional_state`, `reason`, `key_drivers`, `retrieval_source` per item.

## What's measured vs what's not

Measured on the temporal hold-out (`app/eval/`):
- ✅ Rating: RMSE, MAE (baseline, LightGBM, LLM-reasoner)
- ✅ Retrieval ranking: NDCG@10, Hit@10, MRR@10
- ✅ Retrieval ablations (5 weight variants)
- ✅ LightGBM ablations (4 feature subsets)
- 🔄 Full-system rank eval (LLM reasoner re-rank, 15-user sample)

Not measured (limitations §7 in the paper):
- ❌ Human eval of Nigerian voice realism — `docs/HUMAN_EVAL_RUBRIC.md`
  is the rubric we would use; not yet run with Nigerian evaluators.
- ❌ Per-component ablations that require LLM (memory off, planner off):
  cost-prohibitive to run on many users.

## Folder structure

```
/app
  /data          ingest.py, playstore.py, nigerian_voice.py, unify.py
  /persona       features.py, refine.py, store.py, coldstart.py
  /eval          split.py, metrics.py, rating_eval.py, rank_eval.py, ablation.py
  context.py     auto-detected Nigerian flags
  memory.py      long-term + short-term + tagged experiences
  reasoner.py    LLM chain-of-thought (Task A core + Task B simulator)
  generator.py   Nigerian-voice review generator with few-shot grounding
  rating_model.py LightGBM cross-check
  retrieval.py   TF-IDF + item-item CF, with cross-domain support
  planner.py     Reasoning Planner (one call per /recommend)
  recommender.py Task B orchestrator
  main.py        FastAPI app
  llm.py, config.py
/scripts         reproduce.ps1 / reproduce.sh, final_smoke.py, smoke_test.py
/prompts         canonical markdown copies of every prompt
/docs            SOLUTION_PAPER.md, ABLATION.md, SYSTEM_DESIGN.md,
                 PROMPT_ENGINEERING.md, HUMAN_EVAL_RUBRIC.md
/tests           pytest component tests (deterministic only, no LLM)
Dockerfile, docker-compose.yml
```

## Reproducibility

`scripts/reproduce.ps1` (or `.sh`) runs the entire pipeline from scratch:
data download → voice extraction → unification → index build → model
training → smoke test. All measured numbers in the paper come from
`app/eval/*.py` and write JSON results to `data/processed/eval_*.json`.

To run an evaluation yourself:
```powershell
python -m app.eval.rank_eval --retrieval-only   # NDCG/Hit/MRR
python -m app.eval.rating_eval --skip-llm        # baseline + LightGBM RMSE
python -m app.eval.ablation                      # full ablation table
```

## The strategic move

Task A is embedded INSIDE Task B. Every recommendation candidate passes
through the Task A behavioral reasoner before ranking. The predicted
rating becomes a first-class ranking signal, not a post-hoc explanation.
This is what distinguishes "would similar users like this" (collaborative
filtering) from "would this specific user appreciate this right now"
(behavioral simulation).
