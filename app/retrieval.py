"""Hybrid candidate retrieval for Task B.

Three signals, all sparse / lightweight to stay friendly to an 8-core PC:

1. Semantic — TF-IDF over item description (summary + sample text), cosine vs
   the user's liked-items profile.
2. Collaborative — item-item similarity from a sparse mean-centered user x item
   rating matrix (cosine on column vectors). Score = sum of similarities to
   items the target user rated positively.
3. Contextual — re-weights candidates by item average rating + a popularity
   prior + optional time-bucket preference (built from items the user's
   "similar" cohort consumes at this time of day).

Final retrieval score is a weighted blend. Re-ranking with simulation happens
later in app/recommender.py.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize

from app.config import MODELS_DIR, PROCESSED_DIR

ITEM_INDEX_PATH = MODELS_DIR / "item_index.pkl"
ITEMSIM_PATH = MODELS_DIR / "item_sim.pkl"


@lru_cache(maxsize=1)
def _bundle() -> dict:
    if ITEM_INDEX_PATH.exists():
        return joblib.load(ITEM_INDEX_PATH)
    return _build_index()


@lru_cache(maxsize=1)
def _cf_bundle() -> dict:
    if ITEMSIM_PATH.exists():
        return joblib.load(ITEMSIM_PATH)
    return _build_item_similarity()


def _build_index() -> dict:
    items = pd.read_parquet(PROCESSED_DIR / "items.parquet")
    items["doc"] = (items["sample_summary"].fillna("") + " " + items["sample_text"].fillna("")).str.lower()
    vec = TfidfVectorizer(min_df=3, max_df=0.85, ngram_range=(1, 2), max_features=30000, stop_words="english")
    X = vec.fit_transform(items["doc"].tolist())

    keep_cols = [c for c in ["item_id", "avg_rating", "n_reviews", "sample_summary", "domain", "title"] if c in items.columns]
    bundle = {
        "vectorizer": vec,
        "matrix": X,
        "items": items[keep_cols].reset_index(drop=True),
    }
    ITEM_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, ITEM_INDEX_PATH)
    return bundle


def _build_item_similarity(top_k_per_item: int = 50) -> dict:
    """Build sparse item-item cosine similarity from co-rating patterns.

    For each item we keep only the top-K most similar items to bound memory.
    """
    reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
    items = _bundle()["items"]
    item_to_idx = {iid: i for i, iid in enumerate(items["item_id"].tolist())}
    user_to_idx: dict[str, int] = {}
    rows, cols, vals = [], [], []
    for _, r in reviews.iterrows():
        item_idx = item_to_idx.get(r["item_id"])
        if item_idx is None:
            continue
        uid = r["user_id"]
        if uid not in user_to_idx:
            user_to_idx[uid] = len(user_to_idx)
        rows.append(user_to_idx[uid])
        cols.append(item_idx)
        vals.append(float(r["rating"]) - 3.0)  # mean-center on midpoint

    n_users = len(user_to_idx)
    n_items = len(item_to_idx)
    M = csr_matrix((vals, (rows, cols)), shape=(n_users, n_items), dtype=np.float32)
    # Cosine over columns: normalize columns then dot.
    M_norm = l2_normalize(M.T, axis=1)  # items x users
    sim = M_norm @ M_norm.T  # items x items (sparse-ish; many zeros for cold items)
    sim = sim.tocsr()
    # Truncate per row to top_k_per_item to keep size bounded.
    truncated = lil_matrix(sim.shape, dtype=np.float32)
    for i in range(sim.shape[0]):
        row = sim.getrow(i)
        if row.nnz == 0:
            continue
        data = row.data
        indices = row.indices
        # exclude self
        mask = indices != i
        data = data[mask]
        indices = indices[mask]
        if len(data) > top_k_per_item:
            top = np.argpartition(-data, top_k_per_item)[:top_k_per_item]
            data = data[top]
            indices = indices[top]
        truncated[i, indices] = data
    truncated_csr = truncated.tocsr()

    bundle = {"item_sim": truncated_csr, "item_to_idx": item_to_idx}
    joblib.dump(bundle, ITEMSIM_PATH)
    return bundle


def build_index() -> None:
    b = _build_index()
    print(
        f"[retrieval] built TF-IDF index over {b['matrix'].shape[0]} items, vocab={b['matrix'].shape[1]}"
    )
    cf = _build_item_similarity()
    print(f"[retrieval] built item-item CF similarity ({cf['item_sim'].shape}, nnz={cf['item_sim'].nnz})")


def _user_profile_vector(user_id: str, history_df: pd.DataFrame | None = None) -> csr_matrix:
    """Mean of TF-IDF rows for items the user rated >= 4.

    If `history_df` is provided, use ONLY those reviews (lets eval pass a
    train-only slice). Otherwise read the full reviews parquet.
    """
    b = _bundle()
    if history_df is None:
        history_df = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
        history_df = history_df[history_df["user_id"] == user_id]
    liked = history_df[history_df["rating"] >= 4]["item_id"].unique()
    if len(liked) == 0:
        liked = history_df["item_id"].unique()
    items = b["items"]
    idx = items[items["item_id"].isin(liked)].index.values
    if len(idx) == 0:
        return csr_matrix((1, b["matrix"].shape[1]))
    profile = b["matrix"][idx].mean(axis=0)
    return csr_matrix(profile)


def _collab_scores(user_id: str, n_items: int, history_df: pd.DataFrame | None = None) -> np.ndarray:
    """Sum of item-item similarities to items the user rated >= 4."""
    cf = _cf_bundle()
    item_sim = cf["item_sim"]
    item_to_idx = cf["item_to_idx"]
    if history_df is None:
        history_df = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
        history_df = history_df[history_df["user_id"] == user_id]
    liked = history_df[history_df["rating"] >= 4]["item_id"].unique()
    idxs = [item_to_idx[i] for i in liked if i in item_to_idx]
    if not idxs:
        return np.zeros(n_items, dtype=np.float32)
    scores = np.asarray(item_sim[idxs].sum(axis=0)).ravel()
    if scores.max() > 0:
        scores = scores / scores.max()
    return scores.astype(np.float32)


def retrieve(
    user_id: str,
    top_k: int = 30,
    exclude_seen: bool = True,
    weights: tuple[float, float, float, float] = (0.10, 0.70, 0.10, 0.10),
    target_domains: list[str] | None = None,
    cross_domain: bool = False,
    history_df: pd.DataFrame | None = None,
) -> list[dict]:
    """Hybrid retrieval.

    weights = (semantic, collaborative, item_quality, popularity).

    target_domains: if given, only items whose `domain` is in this list are
        returned. Useful for cross-domain queries ("recommend me apps even
        though my history is food").
    cross_domain: if True, automatically EXCLUDE the user's home domain(s)
        and return items from other domains. Implements the brief's
        cross-domain criterion directly.
    """
    b = _bundle()
    items = b["items"].copy()

    # 1. semantic similarity
    profile = _user_profile_vector(user_id, history_df=history_df)
    sem = np.asarray((b["matrix"] @ profile.T).todense()).ravel()
    sem = sem / (sem.max() or 1.0)

    # 2. collaborative
    collab = _collab_scores(user_id, n_items=len(items), history_df=history_df)

    # 3. item-quality prior
    quality = (items["avg_rating"].values / 5.0).astype(np.float32)

    # 4. popularity prior
    pop_raw = np.log1p(items["n_reviews"].values)
    pop = (pop_raw / (pop_raw.max() or 1.0)).astype(np.float32)

    w_sem, w_cf, w_qual, w_pop = weights
    items["sim_semantic"] = sem
    items["sim_collaborative"] = collab
    items["item_quality"] = quality
    items["popularity"] = pop
    items["score"] = w_sem * sem + w_cf * collab + w_qual * quality + w_pop * pop

    if history_df is not None:
        user_reviews_df = history_df
    else:
        reviews = pd.read_parquet(PROCESSED_DIR / "reviews.parquet")
        user_reviews_df = reviews[reviews["user_id"] == user_id]
    if exclude_seen:
        seen = set(user_reviews_df["item_id"])
        items = items[~items["item_id"].isin(seen)]

    # Cross-domain handling
    if cross_domain and "domain" in items.columns:
        home_domains = set(user_reviews_df["domain"].unique()) if "domain" in user_reviews_df.columns else set()
        items = items[~items["domain"].isin(home_domains)] if home_domains else items
    elif target_domains and "domain" in items.columns:
        items = items[items["domain"].isin(target_domains)]

    def _why(row) -> str:
        contributions = {
            "collaborative": w_cf * row["sim_collaborative"],
            "semantic": w_sem * row["sim_semantic"],
            "item_quality": w_qual * row["item_quality"],
            "popularity": w_pop * row["popularity"],
        }
        # Return the source with the largest weighted contribution
        return max(contributions, key=contributions.get)

    items["retrieval_source"] = items.apply(_why, axis=1)
    top = items.sort_values("score", ascending=False).head(top_k)
    return top.to_dict("records")


if __name__ == "__main__":
    build_index()
