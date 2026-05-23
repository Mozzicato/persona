"""Download Amazon Fine Food Reviews, filter to active users, persist parquet.

Run once:
    python -m app.data.ingest
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from app.config import MAX_USERS, MIN_USER_REVIEWS, PROCESSED_DIR, RAW_DIR, RANDOM_SEED

KAGGLE_DATASET = "snap/amazon-fine-food-reviews"
RAW_CSV = RAW_DIR / "Reviews.csv"
REVIEWS_PARQUET = PROCESSED_DIR / "reviews.parquet"
USERS_PARQUET = PROCESSED_DIR / "users.parquet"
ITEMS_PARQUET = PROCESSED_DIR / "items.parquet"


def _download_if_missing() -> Path:
    if RAW_CSV.exists():
        print(f"[ingest] raw CSV already present at {RAW_CSV}")
        return RAW_CSV
    print("[ingest] downloading Amazon Fine Food Reviews from Kaggle via kagglehub…")
    import kagglehub

    cache_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    src = cache_dir / "Reviews.csv"
    if not src.exists():
        # Look for it
        candidates = list(cache_dir.rglob("Reviews.csv"))
        if not candidates:
            raise FileNotFoundError(f"Reviews.csv not found under {cache_dir}")
        src = candidates[0]
    shutil.copy2(src, RAW_CSV)
    print(f"[ingest] copied -> {RAW_CSV}")
    return RAW_CSV


def _load_and_filter() -> pd.DataFrame:
    print("[ingest] loading CSV…")
    df = pd.read_csv(
        RAW_CSV,
        usecols=["Id", "ProductId", "UserId", "ProfileName", "Score", "Time", "Summary", "Text"],
    )
    df = df.rename(
        columns={
            "Id": "review_id",
            "ProductId": "item_id",
            "UserId": "user_id",
            "ProfileName": "user_name",
            "Score": "rating",
            "Time": "timestamp",
            "Summary": "summary",
            "Text": "text",
        }
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.dropna(subset=["user_id", "item_id", "rating", "text"])

    counts = df.groupby("user_id").size()
    active = counts[counts >= MIN_USER_REVIEWS].index
    df = df[df["user_id"].isin(active)].copy()
    print(f"[ingest] users with >= {MIN_USER_REVIEWS} reviews: {df['user_id'].nunique()}")

    if df["user_id"].nunique() > MAX_USERS:
        rng = pd.Series(active).sample(n=MAX_USERS, random_state=RANDOM_SEED).tolist()
        df = df[df["user_id"].isin(rng)].copy()
        print(f"[ingest] sampled down to {MAX_USERS} users -> {len(df)} reviews")

    df = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return df


def _build_items(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("item_id").agg(
        avg_rating=("rating", "mean"),
        n_reviews=("rating", "size"),
        sample_summary=("summary", lambda s: s.dropna().iloc[0] if s.dropna().size else ""),
        sample_text=("text", lambda s: s.iloc[0][:300]),
    ).reset_index()
    return grp


def _build_users(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("user_id").agg(
        n_reviews=("rating", "size"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        avg_review_len=("text", lambda s: s.str.len().mean()),
        sample_name=("user_name", lambda s: s.dropna().iloc[0] if s.dropna().size else ""),
    ).reset_index()
    grp["std_rating"] = grp["std_rating"].fillna(0.0)
    return grp


def main() -> None:
    _download_if_missing()
    df = _load_and_filter()
    print(f"[ingest] final reviews: {len(df):,} | users: {df['user_id'].nunique():,} | items: {df['item_id'].nunique():,}")

    df.to_parquet(REVIEWS_PARQUET, index=False)
    _build_items(df).to_parquet(ITEMS_PARQUET, index=False)
    _build_users(df).to_parquet(USERS_PARQUET, index=False)

    print(f"[ingest] wrote {REVIEWS_PARQUET.name}, {ITEMS_PARQUET.name}, {USERS_PARQUET.name}")


if __name__ == "__main__":
    main()
