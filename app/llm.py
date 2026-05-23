"""Thin Groq client wrapper with JSON-mode + plain-text helpers."""
from __future__ import annotations

import json
from typing import Any

from groq import Groq

from app.config import GROQ_API_KEY, GROQ_MODEL_FAST, GROQ_MODEL_REASONING

_client: Groq | None = None


def client() -> Groq:
    global _client
    if _client is None:
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY missing from .env")
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def chat(
    system: str,
    user: str,
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 600,
    json_mode: bool = False,
) -> str:
    kwargs: dict[str, Any] = {
        "model": model or GROQ_MODEL_FAST,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client().chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def chat_json(system: str, user: str, *, model: str | None = None, temperature: float = 0.3) -> dict:
    raw = chat(system, user, model=model or GROQ_MODEL_REASONING, temperature=temperature, json_mode=True, max_tokens=900)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Salvage attempt: strip code fences
        cleaned = raw.strip().strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        return json.loads(cleaned)
