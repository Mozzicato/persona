# System Design

## Runtime topology

```
        ┌────────────┐    HTTP    ┌──────────────────────────────────┐
client ─┤ uvicorn :8000 ├─────────┤  FastAPI (app/main.py)           │
        └────────────┘            │   /simulate-review               │
                                  │   /recommend                     │
                                  │   /persona  /users  /health      │
                                  └───────────────┬──────────────────┘
                                                  │
            ┌─────────────────────────────────────┴───────────┐
            │                                                 │
            ▼                                                 ▼
   Persona/Memory store                              Groq LLM API
   (filesystem JSON cache)                           (HTTPS, llama-3.1/3.3)
   data/personas/*.json                                   ^      ^
   data/memory/*.json                                     │      │
            │                                             │      │
            ▼                                             │      │
   Processed parquet                                      │      │
   data/processed/reviews.parquet                         │      │
                  items.parquet                           │      │
                  users.parquet                           │      │
            │                                             │      │
            ▼                                             │      │
   Trained artifacts                                      │      │
   models_store/rating_lgbm.pkl   (LightGBM regressor) ───┘      │
   models_store/item_index.pkl    (TF-IDF + items df)            │
   models_store/item_sim.pkl      (item-item cosine sparse) ─────┘
```

## Layered components

| Layer | File | Purpose | LLM-bound? |
|---|---|---|---|
| HTTP | `app/main.py` | FastAPI routes, request validation | no |
| Orchestration | `app/recommender.py` | Task B pipeline coordinator | no (delegates) |
| Reasoning Planner | `app/planner.py` | Per-request strategy + ranking weights | **yes** (1 call) |
| Behavioral Analyzer | `app/recommender.py:analyze_behavior` | Decode current decision state | **yes** (1 call) |
| Behavioral Simulator | `app/reasoner.py` | Per-candidate predicted reaction | **yes** (N calls) |
| Review Generator | `app/generator.py` | Final user-facing review text | **yes** (1 call per Task A request) |
| Persona builder | `app/persona/features.py` + `refine.py` + `store.py` | Stats + LLM refinement + cache | mixed (LLM once per user, then cached) |
| Memory | `app/memory.py` | Long-term + short-term + tagged experiences | no |
| Context | `app/context.py` | Time bucket + auto NG flags | no |
| Retrieval | `app/retrieval.py` | TF-IDF + item-item CF | no |
| Rating model | `app/rating_model.py` | LightGBM cross-check | no |
| LLM client | `app/llm.py` | Groq wrapper with JSON-mode | no |
| Config | `app/config.py` | Paths + env loading | no |

## Data flow per request

### `POST /simulate-review`
```
request → persona_for(user_id, refine=True)  # cached after first call
       → memory.get_or_build(user_id)         # cached after first build
       → context.normalize(request.context)   # auto NG flags
       → reasoner.reason(persona, memory, context, item)   # LLM call
       → rating_model.predict(persona, item)               # LightGBM
       → blend rating (0.6 LLM + 0.4 LGBM)
       → generator.generate(persona, item, reasoner_out, context)  # LLM call
       → response
```
Net: 2 LLM calls + 1 LightGBM inference per request after warmup.

### `POST /recommend`
```
request → persona + memory + context  (same as above)
       → analyze_behavior(...)         # LLM call 1
       → plan(...)                     # LLM call 2 — strategy
       → retrieve(user_id, top_k=12)   # local TF-IDF + CF
       → for each candidate (12):
             reason(persona, memory, context, candidate)  # LLM call 3..14
             rating_model.predict(persona, candidate)
       → rerank
       → response (top_n)
```
Net: 2 + N LLM calls per request (N = candidates_k, default 12).

## Caching

- **Persona JSON**: written to `data/personas/{user_id}.json` after first
  build. The LLM-refinement call is the expensive part; once written, all
  subsequent requests for that user use the cache.
- **Memory JSON**: written to `data/memory/{user_id}.json` after first
  build. Currently NOT auto-rebuilt; see "Cache invalidation" below.
- **TF-IDF + CF**: built once via `python -m app.retrieval` and pickled
  to `models_store/`. The `@lru_cache` on `_bundle()` loads it once per
  process.
- **LightGBM model**: pickled to `models_store/rating_lgbm.pkl` and
  loaded on first `predict()` call.

## Cache invalidation

Today: filesystem caches are not auto-invalidated. To rebuild:

```powershell
# Persona/memory after a schema change
Remove-Item -Recurse data/personas/*.json, data/memory/*.json

# TF-IDF + CF after the dataset changes
Remove-Item models_store/item_index.pkl, models_store/item_sim.pkl
python -m app.retrieval

# LightGBM after feature changes
Remove-Item models_store/rating_lgbm.pkl
python -m app.rating_model
```

Production version should hash the persona schema + git commit and invalidate
on mismatch.

## Scalability

| Concern | Current | Path to scale |
|---|---|---|
| Per-candidate LLM cost | 12 calls per /recommend | Async fan-out to Groq (groq SDK supports asyncio); or distill into a small local reasoner |
| Persona LLM cost | 1 per cold user | Already cached on disk; for >100k users move to Redis/Postgres |
| TF-IDF + CF index size | ~250 MB combined | Move to a vector DB (Qdrant/Milvus) for `>1M` items |
| Rating model retraining | Manual (`python -m app.rating_model`) | Schedule weekly via cron; track RMSE drift |
| LLM rate limits (Groq) | Single API key | Pool keys + retry-with-backoff in `app/llm.py` |

## Failure modes & current handling

| Failure | Behavior |
|---|---|
| Groq returns invalid JSON | `chat_json` salvages by stripping code fences; on second failure raises (caller's `except` returns heuristic fallback) |
| User not in dataset | 404 from `/persona`, `/simulate-review`, `/recommend` |
| Parquet missing | 503 with instruction to run `python -m app.data.ingest` |
| LightGBM model missing | rating model returns `None`; LLM rating is used alone |
| LLM rate limit | Currently surfaces as HTTP 5xx; add retry in `app/llm.py` for prod |

## Deployment

`docker-compose.yml` ships a single-service image. The first container start
runs `app/data/ingest.py` + `app/retrieval.py` + `app/rating_model.py` as an
init job, then starts uvicorn. Data and model artifacts are mounted as
volumes so subsequent boots skip the build.

Resource budget for an 8-core dev box:
- RAM: ~1.5 GB peak during ingest, ~600 MB serving
- Disk: ~700 MB (raw CSV + parquet + indexes + models)
- CPU: ingest is ~3 min; serving is LLM-bound (CPU near-idle)
