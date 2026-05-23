"""Extract authentic Nigerian voice few-shot examples from Play Store reviews.

Two tiers of "Naija-ness":
- PIDGIN: explicit pidgin markers ("abeg", "sef", "no work", "stress me die")
- NAIJA_ENGLISH: Nigerian-specific topical signals (data charges, network,
  support fraud, scammers, Naija app names, etc.) — direct, blunt voice with
  Nigerian-specific complaints

We curate examples by (rating bucket x domain) using a combined score so the
generator has a diverse few-shot pool, not just heavy pidgin.
"""
from __future__ import annotations

import json
import re
import sys

import pandas as pd

from app.config import PROCESSED_DIR

sys.stdout.reconfigure(encoding="utf-8")  # PowerShell cp1252 fix

PIDGIN_MARKERS = re.compile(
    r"\b("
    r"abeg|abi|sha|omo|wahala|wallahi|na\b|biko|chai|chei|haba|"
    r"oga|madam|sef|kuku|jare|joor|nawa|"
    r"wetin|"
    r"vex|stress me|no fit|no work|no get|"
    r"well well|die\b|"
    r"like say|make i|make e|"
    r"oga at the top"
    r")\b",
    flags=re.IGNORECASE,
)
NAIJA_TOPICS = re.compile(
    r"\b("
    r"naira|kobo|nepa|phcn|"
    r"data|airtime|network|mtn|glo|airtel|"
    r"scammers?|fraudsters?|"
    r"support|customer service|customer care|"
    r"refund|charge|charged me|charged twice|debit"
    r")\b",
    flags=re.IGNORECASE,
)
EXCLAIM_RE = re.compile(r"!{2,}")


def _score(text: str) -> tuple[int, int]:
    pidgin = len(PIDGIN_MARKERS.findall(text)) * 3
    naija = len(NAIJA_TOPICS.findall(text)) * 2
    energy = len(EXCLAIM_RE.findall(text))
    return pidgin, pidgin + naija + energy


OUT = PROCESSED_DIR / "nigerian_voice_samples.json"


def main() -> None:
    df = pd.read_parquet(PROCESSED_DIR / "playstore_reviews.parquet")
    df = df[df["text"].str.len().between(30, 300)].copy()

    scored = df["text"].apply(_score)
    df["pidgin_score"] = scored.apply(lambda t: t[0])
    df["naija_score"] = scored.apply(lambda t: t[1])

    print(f"[voice] reviews in range:        {len(df):,}")
    print(f"[voice] with any pidgin marker:  {(df['pidgin_score']>0).sum():,}")
    print(f"[voice] with any Naija signal:   {(df['naija_score']>0).sum():,}")
    print(f"[voice] Naija by domain: {df[df['naija_score']>0]['domain'].value_counts().to_dict()}")

    pool = df[df["naija_score"] > 0].copy()

    rating_buckets = {"low": [1, 2], "mid": [3], "high": [4, 5]}
    examples: dict = {"by_rating_domain": {}, "pidgin_strong": [], "best_overall": []}
    for label, ratings in rating_buckets.items():
        for dom in pool["domain"].unique():
            sub = pool[(pool["rating"].isin(ratings)) & (pool["domain"] == dom)]
            sub = sub.sort_values(["naija_score", "thumbs_up"], ascending=False).head(3)
            for _, r in sub.iterrows():
                examples["by_rating_domain"].setdefault(label, {}).setdefault(dom, []).append(
                    {
                        "rating": int(r["rating"]),
                        "text": r["text"],
                        "naija_score": int(r["naija_score"]),
                        "pidgin_score": int(r["pidgin_score"]),
                    }
                )

    pidgin_strong = pool[pool["pidgin_score"] >= 3].sort_values("pidgin_score", ascending=False).head(15)
    examples["pidgin_strong"] = [
        {"rating": int(r["rating"]), "domain": r["domain"], "text": r["text"]}
        for _, r in pidgin_strong.iterrows()
    ]

    best = pool.sort_values("naija_score", ascending=False).head(25)
    examples["best_overall"] = [
        {"rating": int(r["rating"]), "domain": r["domain"], "text": r["text"]}
        for _, r in best.iterrows()
    ]

    OUT.write_text(json.dumps(examples, indent=2, ensure_ascii=False), encoding="utf-8")
    total = (
        sum(len(v) for buckets in examples["by_rating_domain"].values() for v in buckets.values())
        + len(examples["pidgin_strong"])
        + len(examples["best_overall"])
    )
    print(f"[voice] wrote {OUT.name} with {total} examples\n")

    print("Sample pidgin-strong:")
    for s in examples["pidgin_strong"][:6]:
        print(f"  [{s['rating']} {s['domain']}] {s['text']}")
    print("\nSample best-overall (Naija topics):")
    for s in examples["best_overall"][:6]:
        print(f"  [{s['rating']} {s['domain']}] {s['text']}")


if __name__ == "__main__":
    main()
