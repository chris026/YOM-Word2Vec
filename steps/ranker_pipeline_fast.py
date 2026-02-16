import polars as pl
import numpy as np
import lightgbm as lgb

from gensim.models import Word2Vec
from tqdm import tqdm
from zenml import step, pipeline


# ============================================================
# STEP 1 — LOAD WORD2VEC
# ============================================================

@step
def load_w2v(model_path: str) -> Word2Vec:
    return Word2Vec.load(model_path)


# ============================================================
# STEP 2 — BUILD BASKETS
# ============================================================

@step
def build_baskets(orders_path: str) -> pl.DataFrame:

    baskets = (
        pl.scan_parquet(orders_path)
        .group_by("orderid")
        .agg(
            pl.col("productid").alias("basket"),
            pl.first("userid"),
            pl.first("origin")
        )
        .collect()
    )

    return baskets


# ============================================================
# STEP 3 — FAST CANDIDATE GENERATION + SIMILARITY
# ============================================================

@step
def generate_candidates_fast(
    baskets_df: pl.DataFrame,
    w2v_model: Word2Vec,
    topk: int = 20
) -> pl.DataFrame:

    rows = []

    wv = w2v_model.wv

    for row in tqdm(
        baskets_df.iter_rows(named=True),
        total=len(baskets_df),
        desc="Generating candidates (fast)"
    ):
        orderid = row["orderid"]
        basket = row["basket"]
        basket_set = set(basket)

        for anchor in basket:

            if anchor not in wv:
                continue

            anchor_vec = wv[anchor]

            retrieved = wv.most_similar(anchor, topn=topk)

            # Add positives
            candidates = {pid for pid, _ in retrieved} | (basket_set - {anchor})

            for cand in candidates:

                if cand not in wv:
                    continue

                cand_vec = wv[cand]

                # FAST cosine similarity via dot product
                sim = float(np.dot(anchor_vec, cand_vec))

                rows.append({
                    "orderid": orderid,
                    "anchor": anchor,
                    "candidate": cand,
                    "sim_item2vec": sim
                })

    return pl.DataFrame(rows)


# ============================================================
# STEP 4 — FEATURE ENGINEERING (Lazy)
# ============================================================

@step
def build_training_dataset(
    candidates_df: pl.DataFrame,
    baskets_df: pl.DataFrame,
    commerces_path: str,
    products_path: str,
) -> tuple[pl.DataFrame, np.ndarray]:

    candidates_lf = candidates_df.lazy()
    baskets_lf = baskets_df.lazy()

    commerces_lf = pl.scan_parquet(commerces_path)
    products_lf = pl.scan_parquet(products_path)

    # --- Label ---
    ranker_lf = (
        candidates_lf
        .join(
            baskets_lf.select("orderid", "basket"),
            on="orderid",
            how="left"
        )
        .with_columns(
            (
                pl.col("basket")
                .list.contains(pl.col("candidate"))
            )
            .cast(pl.Int8)
            .alias("label")
        )
    )

    # --- Add store info ---
    ranker_lf = ranker_lf.join(
        baskets_lf.select("orderid", "userid", "origin"),
        on="orderid",
        how="left"
    )

    ranker_lf = ranker_lf.join(
        commerces_lf,
        on="userid",
        how="left"
    )

    # --- Candidate product features ---
    ranker_lf = ranker_lf.join(
        products_lf,
        left_on="candidate",
        right_on="productid",
        how="left"
    )

    # --- Anchor category ---
    ranker_lf = ranker_lf.join(
        products_lf.select(
            pl.col("productid").alias("anchor_pid"),
            pl.col("category").alias("anchor_category")
        ),
        left_on="anchor",
        right_on="anchor_pid",
        how="left"
    )

    ranker_lf = ranker_lf.with_columns(
        (
            pl.col("category") == pl.col("anchor_category")
        )
        .cast(pl.Int8)
        .alias("same_category")
    )

    # --- Build groups ---
    groups = (
        ranker_lf
        .group_by(["orderid", "anchor"])
        .len()
        .select("count")
        .collect()
        .to_numpy()
        .flatten()
    )

    return ranker_lf.collect(), groups


# ============================================================
# STEP 5 — TRAIN RANKER
# ============================================================

@step
def train_ranker(
    train_df: pl.DataFrame,
    groups: np.ndarray
) -> lgb.LGBMRanker:

    feature_cols = [
        "sim_item2vec",
        "same_category",
    ]

    X = train_df.select(feature_cols).to_pandas()
    y = train_df["label"].to_pandas()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=63,
    )

    model.fit(X, y, group=groups)

    return model


# ============================================================
# PIPELINE
# ============================================================

@pipeline
def ranker_training_pipeline_fast(
    orders_path: str,
    commerces_path: str,
    products_path: str,
    w2v_path: str,
    topk: int = 20,
):

    w2v_model = load_w2v(w2v_path)

    baskets_df = build_baskets(orders_path)

    candidates_df = generate_candidates_fast(
        baskets_df,
        w2v_model,
        topk
    )

    train_df, groups = build_training_dataset(
        candidates_df,
        baskets_df,
        commerces_path,
        products_path,
    )

    train_ranker(train_df, groups)