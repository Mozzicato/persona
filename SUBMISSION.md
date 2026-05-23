# PersonaFlow AI — Submission Map

**DSN × BCT LLM Agent Challenge, May 2026**

This document maps our submission to the brief's scoring rubric so the
judges can verify each criterion quickly.

---

## Deliverables checklist

| # | Brief deliverable | Where it lives | Status |
|---|---|---|---|
| 1 | **Containerized application** with API endpoint | `Dockerfile`, `docker-compose.yml`, `app/main.py` | ✅ |
| 2 | **Solution paper (4-8 pages)** | [`docs/SOLUTION_PAPER.md`](docs/SOLUTION_PAPER.md) | ✅ |
| 3 | **Code repository** (clean, reproducible) | This repo + [`README.md`](README.md) | ✅ |

Supporting docs:
- [`docs/ABLATION.md`](docs/ABLATION.md) — measured ablation studies (retrieval + LightGBM)
- [`docs/SYSTEM_DESIGN.md`](docs/SYSTEM_DESIGN.md) — architecture + scalability
- [`docs/PROMPT_ENGINEERING.md`](docs/PROMPT_ENGINEERING.md) — design of all 5 LLM prompts
- [`docs/HUMAN_EVAL_RUBRIC.md`](docs/HUMAN_EVAL_RUBRIC.md) — our human-eval criteria
- [`prompts/`](prompts/) — canonical copies of each prompt

---

## Task A scoring

| Criterion | Where evidenced |
|---|---|
| Review Text Quality (ROUGE / BERTScore) | Persona-conditioned generation in [`app/generator.py`](app/generator.py); grounded in real Nigerian Play Store reviews via [`data/processed/nigerian_voice_samples.json`](data/processed/nigerian_voice_samples.json) |
| Rating Accuracy (RMSE) | **0.710** (LightGBM, clean train-only personas) vs **1.090** (baseline) — see [`docs/ABLATION.md §B`](docs/ABLATION.md). LLM reasoner: 1.020 over 25 samples |
| Behavioural Fidelity (human eval) | Rubric in [`docs/HUMAN_EVAL_RUBRIC.md`](docs/HUMAN_EVAL_RUBRIC.md); persona + memory + reasoner + Nigerian voice all coupled |
| Solution Paper | [`docs/SOLUTION_PAPER.md`](docs/SOLUTION_PAPER.md) |
| Code Reproducibility | `docker compose --profile init up` rebuilds everything from raw data; `python -m pytest tests/` + `python scripts/smoke_no_llm.py` + `python scripts/smoke_llm.py` |

---

## Task B scoring (out of 100)

| Pts | Criterion | Where evidenced |
|---|---|---|
| **30** | Ranking Quality (NDCG@10 / Hit Rate) | **NDCG@10 = 0.649, Hit@10 = 0.675, MRR@10 = 0.642** on temporal hold-out (1,421 users). See [`docs/ABLATION.md §A`](docs/ABLATION.md) + raw JSON [`data/processed/eval_ablation.json`](data/processed/eval_ablation.json) |
| **25** | Cold-Start & Cross-Domain | **Cold-start**: [`app/persona/coldstart.py`](app/persona/coldstart.py) — neutral persona with optional caller hints (`budget_sensitive`, `likes`, etc.). End-to-end demo in [`scripts/smoke_llm.py`](scripts/smoke_llm.py). **Cross-domain**: 5 distinct domains in catalogue (`food`, `food_delivery`, `ride_hailing`, `ecommerce`, `fintech`). API accepts `cross_domain: true` — implemented in [`app/retrieval.py:retrieve`](app/retrieval.py) (excludes user's home domain) and [`app/recommender.py:recommend`](app/recommender.py) |
| **20** | Contextual Relevance (human eval) | Reasoning Planner ([`app/planner.py`](app/planner.py)) produces per-request ranking weights; Behavioral Simulator runs Task-A reasoner per candidate; auto-detected Nigerian context flags (`salary_week`, `festive`, `end_of_month`) in [`app/context.py`](app/context.py) |
| **15** | Solution Paper | [`docs/SOLUTION_PAPER.md`](docs/SOLUTION_PAPER.md) |
| **10** | Code Reproducibility | Single `docker compose up` rebuilds data + indices + LightGBM and serves the API. Unit tests + two smoke scripts |

---

## "Additional marks for sounding like Nigerians"

| Mechanism | Evidence |
|---|---|
| Real Nigerian Play Store reviews as voice grounding | [`app/data/playstore.py`](app/data/playstore.py) pulls 6,000 reviews across 10 Nigerian apps (Chowdeck via Glovo, Bolt, Uber, inDrive, Jumia, Temu, Opay, PalmPay, Kuda) — see [`data/processed/playstore_reviews.parquet`](data/processed/playstore_reviews.parquet) |
| Few-shot Nigerian examples in the generator | [`app/data/nigerian_voice.py`](app/data/nigerian_voice.py) curates 36 high-Naija examples (25 best-overall + 11 pidgin-strong); injected into the generator's system prompt at runtime |
| Auto-detected Nigerian context | `salary_week` (day 23-30), `end_of_month` (≥27), `festive` (Dec + early Jan), `school_resumption` (early Sept) — [`app/context.py:_auto_flags`](app/context.py) |
| Slang scaling | Pidgin density in generated reviews scales with the user's measured `emotional_intensity` — [`app/generator.py`](app/generator.py); analytical reviewers don't suddenly start writing pidgin |

Sample real output (cold-start user, Bolt Food, salary week):
> *"I tried Bolt Food for the first time this week because of the rush at home. Ordered my favourite fried chicken from KFC and it got to me in less than 30mins which is okay. But the main issue was with..."*

---

## How to run (one command)

```powershell
# 1. Put your Groq key in .env (use .env.example as template)
copy .env.example .env
# edit .env to add GROQ_API_KEY

# 2. Build + serve
docker compose --profile init up   # one-shot: data + indices + model
docker compose up api               # serve on http://localhost:8000

# 3. Try it
curl http://localhost:8000/health
# Open Swagger UI: http://localhost:8000/docs
```

Or run locally without Docker:

```powershell
python -m pip install -r requirements.txt
python -m app.data.ingest          # ~3-5 min, downloads Amazon Fine Food from Kaggle
python -m app.data.playstore       # ~2 min, pulls 6k Nigerian reviews
python -m app.data.nigerian_voice  # extracts voice samples
python -m app.data.unify           # unifies the catalogue
python -m app.retrieval            # builds TF-IDF + CF indices
python -m app.rating_model         # trains LightGBM
$env:PYTHONPATH = "."
uvicorn app.main:app --port 8000
```

## Reproducibility evidence

```powershell
# All deterministic checks (no LLM tokens used)
python -m pytest tests/ -v          # 6/6 passing
python scripts/smoke_no_llm.py      # 11/11 checks passing

# Full LLM end-to-end (~2-3 min, uses Groq tokens)
python scripts/smoke_llm.py         # 5/5 checks passing

# Reproduce headline numbers
python -m app.eval.rating_eval --skip-llm   # rating: baseline + LightGBM
python -m app.eval.rank_eval --retrieval-only   # rank: NDCG/Hit/MRR
python -m app.eval.ablation                  # full ablation table
```

---

## Honest limitations

These are also called out in the paper, but listed here too because they
matter for fair evaluation:

1. **Behavioral data is American Amazon Fine Food**; Nigerian context is a layer
   on top, not learned from data. Mitigated by real Nigerian Play Store voice
   grounding but not eliminated.
2. **LLM full-system rank eval is rate-limited.** Free Groq tier hits its daily
   token cap after ~2-3 full /recommend runs. Production should use async
   fan-out + paid tier.
3. **Temporal + packaging/service persona features are zero-lift on RMSE.** Kept
   for qualitative use (drives reasoner + planner) but disclosed honestly in
   the ablation.
4. **TF-IDF item docs are first-review excerpts** (Amazon dataset has no
   product-description field). Semantic retrieval is correspondingly weak;
   CF carries the signal.

We chose to disclose all four rather than hide them — judges value
intellectual honesty per the brief.
