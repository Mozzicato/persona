"""Central config — paths, model names, LLM settings."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PERSONA_DIR = DATA_DIR / "personas"
MEMORY_DIR = DATA_DIR / "memory"
MODELS_DIR = ROOT / "models_store"

for _d in (RAW_DIR, PROCESSED_DIR, PERSONA_DIR, MEMORY_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL_FAST = os.getenv("GROQ_MODEL_FAST", "llama-3.1-8b-instant")
GROQ_MODEL_REASONING = os.getenv("GROQ_MODEL_REASONING", "llama-3.3-70b-versatile")

MIN_USER_REVIEWS = 20
MAX_USERS = 2000
RANDOM_SEED = 42
