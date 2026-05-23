# Full reproducibility script for PersonaFlow AI.
# Run from project root in PowerShell.

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "."

Write-Host "[1/8] Installing dependencies..." -ForegroundColor Cyan
python -m pip install -r requirements.txt | Out-Host

Write-Host "[2/8] Downloading Amazon Fine Food Reviews from Kaggle..." -ForegroundColor Cyan
python -m app.data.ingest

Write-Host "[3/8] Pulling Nigerian app reviews from Play Store..." -ForegroundColor Cyan
python -m app.data.playstore

Write-Host "[4/8] Extracting Nigerian voice samples..." -ForegroundColor Cyan
python -m app.data.nigerian_voice

Write-Host "[5/8] Unifying datasets into cross-domain catalogue..." -ForegroundColor Cyan
python -m app.data.unify

Write-Host "[6/8] Building TF-IDF + collaborative-filtering item index..." -ForegroundColor Cyan
python -m app.retrieval

Write-Host "[7/8] Training LightGBM rating model..." -ForegroundColor Cyan
python -m app.rating_model

Write-Host "[8/8] Final smoke test..." -ForegroundColor Cyan
python scripts/final_smoke.py

Write-Host "`n✓ Reproduction complete. Start the API with:" -ForegroundColor Green
Write-Host "    uvicorn app.main:app --reload --port 8000" -ForegroundColor Yellow
