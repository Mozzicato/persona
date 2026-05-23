"""Combine Amazon Fine Food + Play Store into a unified cross-domain catalogue.

Outputs (overwrite the canonical processed/*.parquet files used everywhere):
- reviews.parquet — unified review history with a `domain` column
- items.parquet   — unified item catalogue with a `domain` column
- users.parquet   — unified per-user aggregates

After this, retrieval / recommender / persona extraction all "see" both
domains transparently. domain values:
  food            : Amazon Fine Food
  food_delivery   : Chowdeck/Glovo/Bolt Food
  ride_hailing    : Bolt/Uber/inDrive
  ecommerce       : Jumia/Konga/Temu
  fintech         : Opay/PalmPay/Kuda

Run after both ingest.py and playstore.py have completed.
"""
from __future__ import annotations

import pandas as pd

from app.config import PROCESSED_DIR


REVIEWS_AMAZON = PROCESSED_DIR / "reviews.parquet"
ITEMS_AMAZON = PROCESSED_DIR / "items.parquet"
REVIEWS_PS = PROCESSED_DIR / "playstore_reviews.parquet"
ITEMS_PS = PROCESSED_DIR / "playstore_items.parquet"

OUT_REVIEWS = PROCESSED_DIR / "reviews.parquet"
OUT_ITEMS = PROCESSED_DIR / "items.parquet"
OUT_USERS = PROCESSED_DIR / "users.parquet"


def _backup_amazon() -> None:
    """Keep an Amazon-only copy so we can re-split for evaluation later."""
    if (PROCESSED_DIR / "reviews_amazon.parquet").exists():
        return
    if REVIEWS_AMAZON.exists():
        pd.read_parquet(REVIEWS_AMAZON).to_parquet(PROCESSED_DIR / "reviews_amazon.parquet", index=False)
    if ITEMS_AMAZON.exists():
        pd.read_parquet(ITEMS_AMAZON).to_parquet(PROCESSED_DIR / "items_amazon.parquet", index=False)


def main() -> None:
    _backup_amazon()

    amazon_reviews = pd.read_parquet(PROCESSED_DIR / "reviews_amazon.parquet")
    amazon_items = pd.read_parquet(PROCESSED_DIR / "items_amazon.parquet")
    ps_reviews = pd.read_parquet(REVIEWS_PS)
    ps_items = pd.read_parquet(ITEMS_PS)

    amazon_reviews = amazon_reviews.assign(domain="food")
    amazon_items = amazon_items.assign(domain="food")
    # Amazon items don't have a category/title; synthesise sample_summary already exists.
    if "sample_text" not in amazon_items.columns:
        amazon_items["sample_text"] = ""
    amazon_items["title"] = amazon_items.get("sample_summary", amazon_items["item_id"])
    amazon_items["description"] = amazon_items.get("sample_text", "").fillna("").astype(str).str[:500]
    amazon_items["category"] = "food"

    # Play Store reviews: align columns with Amazon schema
    ps_reviews = ps_reviews.rename(columns={"item_id": "item_id", "rating": "rating", "text": "text"})
    # Ensure required columns
    for col in ("review_id", "user_id", "item_id", "rating", "text", "summary", "timestamp", "domain"):
        if col not in ps_reviews.columns:
            ps_reviews[col] = None
    ps_reviews = ps_reviews[["review_id", "user_id", "item_id", "rating", "text", "summary", "timestamp", "domain"]]
    ps_reviews["rating"] = ps_reviews["rating"].astype(float)

    # Play Store items: align with Amazon items schema (add what amazon expected)
    if "avg_rating" not in ps_items.columns:
        ps_items["avg_rating"] = ps_items["rating"]
    ps_items["sample_summary"] = ps_items["title"].fillna(ps_items["item_name"])
    ps_items["sample_text"] = ps_items["description"].fillna("").astype(str).str[:500]
    ps_items["n_reviews"] = ps_items["n_reviews"].fillna(0).astype(int)

    amazon_cols = ["review_id", "user_id", "item_id", "rating", "text", "summary", "timestamp", "domain"]
    for col in amazon_cols:
        if col not in amazon_reviews.columns:
            amazon_reviews[col] = None
    amazon_reviews = amazon_reviews[amazon_cols]
    amazon_reviews["rating"] = amazon_reviews["rating"].astype(float)

    unified_reviews = pd.concat([amazon_reviews, ps_reviews], ignore_index=True)
    unified_reviews = unified_reviews.dropna(subset=["user_id", "item_id", "rating", "text"])
    unified_reviews["review_id"] = unified_reviews["review_id"].astype(str)
    unified_reviews["user_id"] = unified_reviews["user_id"].astype(str)
    unified_reviews["item_id"] = unified_reviews["item_id"].astype(str)
    unified_reviews["text"] = unified_reviews["text"].astype(str)
    unified_reviews["summary"] = unified_reviews["summary"].fillna("").astype(str)
    unified_reviews["timestamp"] = pd.to_datetime(unified_reviews["timestamp"], errors="coerce")

    item_cols = ["item_id", "domain", "category", "title", "description", "sample_summary", "sample_text", "avg_rating", "n_reviews"]
    for col in item_cols:
        if col not in amazon_items.columns:
            amazon_items[col] = None
        if col not in ps_items.columns:
            ps_items[col] = None
    unified_items = pd.concat([amazon_items[item_cols], ps_items[item_cols]], ignore_index=True).drop_duplicates(
        subset=["item_id"], keep="first"
    )

    users = unified_reviews.groupby("user_id").agg(
        n_reviews=("rating", "size"),
        avg_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        domains=("domain", lambda s: list(set(s))),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
    ).reset_index()
    users["std_rating"] = users["std_rating"].fillna(0.0)
    users["n_domains"] = users["domains"].apply(len)

    unified_reviews.to_parquet(OUT_REVIEWS, index=False)
    unified_items.to_parquet(OUT_ITEMS, index=False)
    users.to_parquet(OUT_USERS, index=False)

    print(
        f"[unify] reviews: {len(unified_reviews):,} | items: {len(unified_items):,} | "
        f"users: {len(users):,}"
    )
    print(f"[unify] domain mix (reviews): {unified_reviews['domain'].value_counts().to_dict()}")
    print(f"[unify] domain mix (items):   {unified_items['domain'].value_counts().to_dict()}")
    print(f"[unify] cross-domain users (>1 domain): {(users['n_domains']>1).sum():,}")


if __name__ == "__main__":
    main()
