"""Pull real Nigerian reviews from Google Play Store across 4 domains.

Each app becomes an ITEM in the cross-domain catalogue. Each review becomes a
user review. Reviewers who reviewed 2+ apps become cross-domain users we can
build personas for.

Run once:
    python -m app.data.playstore
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from google_play_scraper import Sort, reviews, app as app_info

from app.config import PROCESSED_DIR, RAW_DIR

# 12 Nigerian apps across 4 domains
APPS = [
    # food delivery
    ("com.chowdeck.app",                    "Chowdeck",       "food_delivery"),
    ("com.glovo",                           "Glovo",          "food_delivery"),
    ("com.bolt.deliveryclient",             "Bolt Food",      "food_delivery"),
    # ride-hailing
    ("ee.mtakso.client",                    "Bolt",           "ride_hailing"),
    ("com.ubercab",                         "Uber",           "ride_hailing"),
    ("sinet.startup.inDriver",              "inDrive",        "ride_hailing"),
    # e-commerce
    ("com.jumia.android",                   "Jumia",          "ecommerce"),
    ("com.kongahq.android",                 "Konga",          "ecommerce"),
    ("com.einnovation.temu",                "Temu",           "ecommerce"),
    # fintech
    ("team.opay.pay",                       "Opay",           "fintech"),
    ("com.transsnet.palmpay",               "PalmPay",        "fintech"),
    ("com.kudabank.app",                    "Kuda",           "fintech"),
]

REVIEWS_PER_APP = 500  # 12 apps * 500 = 6000 reviews target
COUNTRY = "ng"  # Nigeria storefront
LANG = "en"

OUT_REVIEWS = PROCESSED_DIR / "playstore_reviews.parquet"
OUT_ITEMS = PROCESSED_DIR / "playstore_items.parquet"
OUT_RAW = RAW_DIR / "playstore_reviews.parquet"


def pull_app(app_id: str, name: str, domain: str, n: int = REVIEWS_PER_APP) -> tuple[pd.DataFrame, dict]:
    print(f"[playstore] {name} ({app_id}) …")
    # 1. App info (used as the item metadata)
    try:
        info = app_info(app_id, lang=LANG, country=COUNTRY)
    except Exception as e:  # noqa: BLE001
        print(f"  ! app_info failed for {app_id}: {e}")
        info = {"title": name, "description": "", "score": None, "ratings": 0}

    # 2. Reviews — paginate
    all_rows = []
    token = None
    pulled = 0
    while pulled < n:
        try:
            batch, token = reviews(
                app_id,
                lang=LANG,
                country=COUNTRY,
                sort=Sort.MOST_RELEVANT,
                count=min(200, n - pulled),
                continuation_token=token,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ! reviews failed for {app_id}: {e}")
            break
        if not batch:
            break
        for r in batch:
            all_rows.append(
                {
                    "review_id": r.get("reviewId"),
                    "user_id": f"gp_{r.get('userName', 'anon')}_{(r.get('reviewId') or '')[:6]}",
                    "user_name": r.get("userName"),
                    "item_id": app_id,
                    "item_name": name,
                    "domain": domain,
                    "rating": r.get("score"),
                    "text": r.get("content") or "",
                    "summary": "",
                    "timestamp": r.get("at"),
                    "thumbs_up": r.get("thumbsUpCount", 0),
                }
            )
        pulled += len(batch)
        if token is None:
            break
        time.sleep(0.4)  # polite rate-limit

    df = pd.DataFrame(all_rows)
    item_row = {
        "item_id": app_id,
        "item_name": name,
        "domain": domain,
        "category": domain,
        "title": info.get("title") or name,
        "description": (info.get("description") or "")[:1000],
        "avg_rating": info.get("score") or (df["rating"].mean() if len(df) else None),
        "n_reviews": info.get("ratings") or len(df),
        "sample_summary": info.get("title") or name,
        "sample_text": (info.get("description") or name)[:300],
    }
    print(f"  -> {len(df)} reviews pulled")
    return df, item_row


def main() -> None:
    all_reviews = []
    all_items = []
    for app_id, name, domain in APPS:
        df, item = pull_app(app_id, name, domain)
        if len(df):
            all_reviews.append(df)
            all_items.append(item)
        time.sleep(0.8)

    if not all_reviews:
        raise RuntimeError("[playstore] no reviews pulled")

    df_all = pd.concat(all_reviews, ignore_index=True)
    df_all = df_all.dropna(subset=["rating", "text"])
    df_all = df_all[df_all["text"].str.len() >= 3].copy()
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"], errors="coerce")
    df_all = df_all.dropna(subset=["timestamp"])

    items_df = pd.DataFrame(all_items)

    df_all.to_parquet(OUT_REVIEWS, index=False)
    df_all.to_parquet(OUT_RAW, index=False)
    items_df.to_parquet(OUT_ITEMS, index=False)

    print(
        f"[playstore] DONE — {len(df_all):,} reviews | {df_all['user_id'].nunique():,} users | "
        f"{len(items_df)} items | {df_all['domain'].nunique()} domains"
    )
    print(f"  domains: {df_all['domain'].value_counts().to_dict()}")
    print(f"  ratings: {df_all['rating'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
