#!/usr/bin/env bash
# Full reproducibility script for PersonaFlow AI (Linux/macOS/WSL).

set -euo pipefail
export PYTHONPATH=.

echo "[1/8] Installing dependencies..."
python -m pip install -r requirements.txt

echo "[2/8] Downloading Amazon Fine Food Reviews from Kaggle..."
python -m app.data.ingest

echo "[3/8] Pulling Nigerian app reviews from Play Store..."
python -m app.data.playstore

echo "[4/8] Extracting Nigerian voice samples..."
python -m app.data.nigerian_voice

echo "[5/8] Unifying datasets into cross-domain catalogue..."
python -m app.data.unify

echo "[6/8] Building TF-IDF + collaborative-filtering item index..."
python -m app.retrieval

echo "[7/8] Training LightGBM rating model..."
python -m app.rating_model

echo "[8/8] Final smoke test..."
python scripts/final_smoke.py

echo
echo "✓ Reproduction complete. Start the API with:"
echo "    uvicorn app.main:app --reload --port 8000"
