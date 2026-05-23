"""Temporal hold-out split for both rating prediction and recommendation eval.

For each user, the LAST N (default 1) reviews go to the test set. Everything
before becomes the training history. This avoids leakage (vs random split)
and matches how the system would actually be used in production.
"""
from __future__ import annotations

import pandas as pd

from app.config import PROCESSED_DIR


def temporal_split(min_reviews: int = 5, holdout_per_user: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_reviews, test_reviews) for users with >= min_reviews."""
    df = pd.read_parquet(PROCESSED_DIR / "reviews.parquet").sort_values(["user_id", "timestamp"])
    counts = df.groupby("user_id").size()
    keep_users = counts[counts >= min_reviews].index
    df = df[df["user_id"].isin(keep_users)].copy()

    # last N per user -> test
    df["rank_from_end"] = df.groupby("user_id").cumcount(ascending=False)
    test = df[df["rank_from_end"] < holdout_per_user].drop(columns=["rank_from_end"])
    train = df[df["rank_from_end"] >= holdout_per_user].drop(columns=["rank_from_end"])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def write_splits(train: pd.DataFrame, test: pd.DataFrame) -> tuple:
    train_path = PROCESSED_DIR / "reviews_train.parquet"
    test_path = PROCESSED_DIR / "reviews_test.parquet"
    train.to_parquet(train_path, index=False)
    test.to_parquet(test_path, index=False)
    return train_path, test_path
