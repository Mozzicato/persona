"""Deploy PersonaFlow to a Hugging Face Docker Space.

What this does:
1. Creates / updates the Space `mozzic/personaflow` with Docker SDK
2. Sets GROQ_API_KEY as an encrypted Space secret (read from local .env)
3. Writes a Space-specific README.md (with required YAML frontmatter)
4. Adjusts the Dockerfile to COPY data + models (HF has no volumes)
5. Uploads code + data + models in a single batched push

Usage (inside the project root):
    python scripts/deploy_hf.py
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parent.parent
HF_TOKEN = os.environ["HF_TOKEN"]
SPACE_ID = "mozzic/personaflow"

api = HfApi(token=HF_TOKEN)


def ensure_space() -> str:
    print(f"[deploy] ensuring Space exists: {SPACE_ID}")
    try:
        create_repo(
            repo_id=SPACE_ID,
            repo_type="space",
            space_sdk="docker",
            token=HF_TOKEN,
            exist_ok=True,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  ! create_repo error (probably already exists): {e}")
    return f"https://huggingface.co/spaces/{SPACE_ID}"


def set_secrets():
    print("[deploy] setting GROQ_API_KEY secret")
    env_path = ROOT / ".env"
    groq_key = None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("GROQ_API_KEY="):
            groq_key = line.split("=", 1)[1].strip()
            break
    if not groq_key:
        raise SystemExit("[deploy] GROQ_API_KEY not found in .env")
    api.add_space_secret(
        repo_id=SPACE_ID,
        key="GROQ_API_KEY",
        value=groq_key,
    )
    print("  -> GROQ_API_KEY secret set")


SPACE_README = """---
title: PersonaFlow AI
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
license: mit
short_description: Behavior-aware NG-context review simulation + recommendation
---

# PersonaFlow AI (live demo)

Behavior-aware review simulation (Task A) and contextual recommendation
(Task B) for Nigerian consumers. Built for the **DSN × BCT LLM Agent
Challenge 2026**.

Source code: https://github.com/Mozzicato/persona

## API endpoints

- `GET  /health` — liveness probe
- `GET  /users?limit=20` — sample user IDs you can plug in
- `POST /simulate-review` — Task A: persona-aware review + rating
- `POST /recommend` — Task B: behavior-aware recommendations (supports `cross_domain: true`)
- `POST /persona` — view full persona for a user
- Swagger UI: append `/docs` to the Space URL

## Quick test (Swagger)

Open `<space-url>/docs` and try:

**Task A** — `POST /simulate-review`:
```json
{
  "user_id": "A100WO06OQR8BQ",
  "item": {"name": "Chowdeck", "category": "food_delivery", "item_id": "com.chowdeck.app"},
  "context": {"time": "night", "weather": "rainy", "traffic_heavy": true}
}
```

**Task B** — `POST /recommend`:
```json
{
  "user_id": "A100WO06OQR8BQ",
  "context": {"time": "evening", "mood": "tired"},
  "top_n": 3,
  "cross_domain": true
}
```

**Cold-start** — `POST /simulate-review`:
```json
{
  "user_id": "totally_new_user_xyz123",
  "item": {"name": "Bolt Food", "category": "food_delivery"},
  "context": {"time": "afternoon", "salary_week": true},
  "cold_start_hints": {"budget_sensitive": true, "likes": ["affordable", "fast"]}
}
```

## Headline metrics (temporal hold-out, 1,421 users)

| Metric | Value |
|---|---|
| NDCG@10 | **0.649** |
| Hit@10  | **0.675** |
| MRR@10  | **0.642** |
| Rating RMSE (LightGBM) | **0.710** |
| Rating RMSE (baseline) | 1.090 |

See `docs/SOLUTION_PAPER.md` and `docs/ABLATION.md` in the GitHub repo for the full write-up.
"""


SPACE_DOCKERFILE = """FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential libgomp1 curl \\
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY app ./app
COPY prompts ./prompts
COPY scripts ./scripts

# Pre-built data + models — shipped via LFS in this Space
COPY data ./data
COPY models_store ./models_store

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
# Hugging Face Spaces sets PORT; honor it but default to 8000.
ENV PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \\
  CMD curl -fsS http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
"""


GITATTRIBUTES = """*.parquet filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
"""


def build_staging() -> Path:
    """Copy what we want to ship into a clean staging dir."""
    staging = Path(tempfile.mkdtemp(prefix="hf-deploy-"))
    print(f"[deploy] staging dir: {staging}")

    # Folders to ship as-is
    for folder in ("app", "prompts", "scripts", "docs", "data/processed", "models_store"):
        src = ROOT / folder
        if not src.exists():
            print(f"  ! missing {folder} — skipping")
            continue
        dst = staging / folder
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)

    # Empty placeholder dirs for runtime cache writes
    for d in ("data/personas", "data/memory", "data/raw"):
        (staging / d).mkdir(parents=True, exist_ok=True)
        (staging / d / ".gitkeep").write_text("")

    # Top-level files
    for f in ("requirements.txt", ".env.example", "SUBMISSION.md"):
        if (ROOT / f).exists():
            shutil.copy(ROOT / f, staging / f)

    # Space-specific README + Dockerfile + .gitattributes
    (staging / "README.md").write_text(SPACE_README, encoding="utf-8")
    (staging / "Dockerfile").write_text(SPACE_DOCKERFILE, encoding="utf-8")
    (staging / ".gitattributes").write_text(GITATTRIBUTES, encoding="utf-8")

    # Strip __pycache__
    for cache in staging.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    # Report size
    total = sum(f.stat().st_size for f in staging.rglob("*") if f.is_file())
    print(f"[deploy] staging size: {total / 1024 / 1024:.1f} MB")
    return staging


def main() -> None:
    url = ensure_space()
    set_secrets()
    staging = build_staging()

    print(f"[deploy] uploading staging -> {SPACE_ID}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=SPACE_ID,
        repo_type="space",
        commit_message="Deploy PersonaFlow AI",
    )
    print(f"\n[deploy] DONE — Space URL:\n  {url}")
    print(f"  Swagger UI:  {url.replace('huggingface.co/spaces', 'huggingface.co/spaces') + '/-/'}docs")
    print(f"  Once build completes (~3-5 min), test:\n    {url}/-/health")


if __name__ == "__main__":
    main()
