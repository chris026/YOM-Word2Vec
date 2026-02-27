import os
import polars as pl
import numpy as np
import lightgbm as lgb
import pandas as pd

from gensim.models import Word2Vec
from tqdm import tqdm
from zenml import step, pipeline


# ==============================
# Helpers
# ==============================

def _artifact_path(base_dir: str, name: str, ext: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    #return os.path.join(base_dir, f"{name}_{uuid.uuid4().hex}.{ext}")
    return os.path.join(base_dir, f"{name}.{ext}")


# ==============================
# STEP 1 — PREPARE DATA (baskets + popularity tables) -> PARQUET PATHS
# ==============================

@step
def prepare_data(
    data_path: str,
    commerces_path: str,
    products_path: str,
    artifacts_dir: str = "artifacts",
) -> tuple[str, str, str, str, str, str]:
    """
    Returns:
      baskets_path,
      pop_global_path,
      pop_store_path,
      pop_origin_path,
      pop_region_path,
      pop_subch_path
    """

    orders = pl.scan_parquet(data_path)
    commerces = pl.scan_parquet(commerces_path)
    _ = pl.scan_parquet(products_path)  # just ensures file exists / schema can be read

    # --- baskets ---
    baskets_lf = (
        orders
        .group_by("orderid")
        .agg(
            pl.col("productid").alias("basket"),
            pl.first("userid").alias("userid"),
            pl.first("origin").alias("origin"),
        )
    )
    baskets_path = _artifact_path(artifacts_dir, "baskets", "parquet")
    baskets_lf.collect().write_parquet(baskets_path)

    # --- popularity tables ---
    pop_global_lf = orders.group_by("productid").agg(pl.len().alias("pop_global"))
    pop_store_lf = orders.group_by(["userid", "productid"]).agg(pl.len().alias("pop_store"))
    pop_origin_lf = orders.group_by(["origin", "productid"]).agg(pl.len().alias("pop_origin"))

    # For region/subchannel, join orders with commerces first
    orders_enriched = orders.join(
        commerces.select(["userid", "region", "subchannel"]),
        on="userid",
        how="left",
    )

    pop_region_lf = orders_enriched.group_by(["region", "productid"]).agg(pl.len().alias("pop_region"))
    pop_subch_lf = orders_enriched.group_by(["subchannel", "productid"]).agg(pl.len().alias("pop_subch"))

    pop_global_path = _artifact_path(artifacts_dir, "pop_global", "parquet")
    pop_store_path = _artifact_path(artifacts_dir, "pop_store", "parquet")
    pop_origin_path = _artifact_path(artifacts_dir, "pop_origin", "parquet")
    pop_region_path = _artifact_path(artifacts_dir, "pop_region", "parquet")
    pop_subch_path = _artifact_path(artifacts_dir, "pop_subch", "parquet")

    pop_global_lf.collect().write_parquet(pop_global_path)
    pop_store_lf.collect().write_parquet(pop_store_path)
    pop_origin_lf.collect().write_parquet(pop_origin_path)
    pop_region_lf.collect().write_parquet(pop_region_path)
    pop_subch_lf.collect().write_parquet(pop_subch_path)

    return (
        baskets_path,
        pop_global_path,
        pop_store_path,
        pop_origin_path,
        pop_region_path,
        pop_subch_path,
    )


# ==============================
# STEP 2 — FAST CANDIDATES + SIMILARITY -> PARQUET
# ==============================

@step(enable_cache=False)
def generate_candidates_fast_to_parquet(
    baskets_path: str,
    w2v_model_path: str,
    topk: int,
    artifacts_dir: str = "artifacts",
) -> str:
    baskets_df = pl.read_parquet(baskets_path)

    w2v_model = Word2Vec.load(w2v_model_path)
    wv = w2v_model.wv
    try:
        wv.fill_norms()
    except Exception:
        print("wv.fill_norms has not worked -> pass")

    anchors = list(wv.key_to_index.keys())
    neigh_rows = []
    for a in anchors:
        for c, s in wv.most_similar(a, topn=len(anchors)):
            if c != a:
                neigh_rows.append((a, c, float(s)))

    neigh_df = pl.DataFrame(
        neigh_rows,
        schema={
            "anchor": pl.String,
            "candidate": pl.String,
            "sim_item2vec": pl.Float32,
    },
        orient="row")

    out_path = _artifact_path(artifacts_dir, "candidates", "parquet")

    """
    baskets_df.schema
    Schema({'orderid': String, 'basket': List(String), 'userid': String, 'origin': String})
    neigh_df.schema
    Schema({'anchor': String, 'candidate': String, 'sim_item2vec': Float32})
    """

    (
        baskets_df
        .select(["orderid", "basket"])
        .explode("basket")
        .rename({"basket": "anchor"})
        .join(neigh_df, on="anchor", how="inner")
        .write_parquet(out_path)
    )

    return out_path


# ==============================
# STEP 2.5 — FAST CANDIDATES + SIMILARITY -> PARQUET
# ==============================

@step(enable_cache=False)
def generate_negatives_to_parquet(
    baskets_path: str,
    w2v_model_path: str,
    topk: int,
    artifacts_dir: str = "artifacts",
) -> str:
    baskets_df = pl.read_parquet(baskets_path)

    w2v_model = Word2Vec.load(w2v_model_path)
    wv = w2v_model.wv
    try:
        wv.fill_norms()
    except Exception:
        print("wv.fill_norms has not worked -> pass")

    anchors = list(wv.key_to_index.keys())
    neigh_rows = []
    for a in anchors:
        for c, s in wv.most_similar(a, topn=topk):
            if c != a:
                neigh_rows.append((a, c, float(s)))

    neigh_df = pl.DataFrame(
        neigh_rows,
        schema={
            "anchor": pl.String,
            "candidate": pl.String,
            "sim_item2vec": pl.Float32,
    },
        orient="row")

    out_path = _artifact_path(artifacts_dir, "negatives", "parquet")

    """
    baskets_df.schema
    Schema({'orderid': String, 'basket': List(String), 'userid': String, 'origin': String})
    neigh_df.schema
    Schema({'anchor': String, 'candidate': String, 'sim_item2vec': Float32})
    """

    (
        baskets_df
        .select(["orderid", "basket"])
        .explode("basket")
        .rename({"basket": "anchor"})
        .join(neigh_df, on="anchor", how="inner")
        .write_parquet(out_path)
    )

    return out_path


@step
def train1(
    candidates_path: str,
    negatives_path: str,
    baskets_path: str,
    commerces_path: str,
    products_path: str,
    artifacts_dir: str = "artifacts",
) -> str:
    candidates_lf = pl.scan_parquet(candidates_path)
    negative_lf = pl.scan_parquet(negatives_path)
    baskets_lf    = pl.scan_parquet(baskets_path)
    commerces_lf  = pl.scan_parquet(commerces_path)
    products_lf   = pl.scan_parquet(products_path)

    # --- baskets: wie vorher "basket" als List-Spalte im Endresult behalten
    baskets_meta_lf = baskets_lf.select(["orderid", "userid", "origin"])
    commerces_lf = commerces_lf.drop("sellerid", "active")
    products_lf = products_lf.select(["category", "productid"])

    # --- positives: (orderid, candidate) aus dem basket, zum Labeln per Join
    positives_lf = (
        baskets_lf
        .select(["orderid", "basket"])
        .explode("basket")
        .rename({"basket": "candidate"})
        .unique(["orderid", "candidate"])
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    )

    # Base joins: candidates + basket + store meta
    ranker_lf = (
        positives_lf
        .join(candidates_lf, on=["orderid", "candidate"], how="left")
    )

    key_cols = ["orderid", "anchor", "candidate"]  # falls du nur orderid+candidate willst, hier anpassen
    base_cols = ["orderid", "candidate", "label", "anchor", "sim_item2vec"]

    negatives_to_add_lf = (
        negative_lf
        .join(
            ranker_lf.select(key_cols),
            on=key_cols,
            how="anti",  # nur Zeilen, die NICHT in ranker_lf sind
        )
        .with_columns(pl.lit(0, dtype=pl.Int8).alias("label"))
        .select(base_cols)
    )

    ranker_lf = (
        pl.concat(
            [
                ranker_lf,
                negatives_to_add_lf,
            ],
            how="vertical",
        )
    )

    path = _artifact_path(artifacts_dir, "train1", "parquet")
    ranker_lf.sink_parquet(path=path)
    return path

@step
def train2(
    candidates_path: str,
    negatives_path: str,
    baskets_path: str,
    commerces_path: str,
    products_path: str,
    pop_global_path: str,
    pop_store_path: str,
    pop_origin_path: str,
    pop_region_path: str,
    pop_subch_path: str,
    ranker_lf_path: str,
    artifacts_dir: str = "artifacts",
) -> tuple[str, str]:
    baskets_lf    = pl.scan_parquet(baskets_path)
    commerces_lf  = pl.scan_parquet(commerces_path)
    products_lf   = pl.scan_parquet(products_path)

    baskets_meta_lf = baskets_lf.select(["orderid", "userid", "origin"])
    commerces_lf = commerces_lf.drop("sellerid", "active")
    products_lf = products_lf.select(["category", "productid"])


    # Candidate product meta
    ranker_lf = pl.scan_parquet(ranker_lf_path)
    ranker_lf = ranker_lf.join(baskets_meta_lf, on="orderid", how="left")
    ranker_lf = ranker_lf.join(commerces_lf, on="userid", how="left")

    # Candidate product meta
    ranker_lf = ranker_lf.join(
        products_lf,
        left_on="candidate",
        right_on="productid",
        how="left",
    )

    # Popularity joins
    pop_global_lf = pl.scan_parquet(pop_global_path)
    pop_store_lf  = pl.scan_parquet(pop_store_path)
    pop_origin_lf = pl.scan_parquet(pop_origin_path)
    pop_region_lf = pl.scan_parquet(pop_region_path)
    pop_subch_lf  = pl.scan_parquet(pop_subch_path)
    
    ranker_lf = (
        ranker_lf
        .join(pop_global_lf, left_on="candidate", right_on="productid", how="left")
        .join(pop_store_lf, left_on=["userid", "candidate"], right_on=["userid", "productid"], how="left")
        .join(pop_origin_lf, left_on=["origin", "candidate"], right_on=["origin", "productid"], how="left")
        .join(pop_region_lf, left_on=["region", "candidate"], right_on=["region", "productid"], how="left")
        .join(pop_subch_lf, left_on=["subchannel", "candidate"], right_on=["subchannel", "productid"], how="left")
        .with_columns(
            pl.col("pop_global").fill_null(0),
            pl.col("pop_store").fill_null(0),
            pl.col("pop_origin").fill_null(0),
            pl.col("pop_region").fill_null(0),
            pl.col("pop_subch").fill_null(0),
        )
    )

    ranker_lf = ranker_lf.with_columns(
        pl.col("channel").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("commune").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("origin").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("region").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("subchannel").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).alias("cand_category"),
    )

    ranker_lf = ranker_lf.select([
        "sim_item2vec",
        "pop_global",
        "pop_subch",
        "pop_origin",
        "pop_region",
        "channel",
        "pop_store",
        "commune",
        "cand_category",
        "origin",
        "region",
        "subchannel",
        "label",
        "orderid",
        "anchor"]
    )

    ranker_lf = ranker_lf.sort(["orderid", "anchor"])

    groups = (
        ranker_lf
        .select(["orderid", "anchor"])
        .group_by(["orderid", "anchor"], maintain_order=True)
        .agg(pl.len().alias("group_size"))
        .select("group_size")
        .collect()
        .to_numpy()
        .astype(np.int32)
        .ravel()
    )

    groups_path = _artifact_path(artifacts_dir, "groups", "npy")
    np.save(groups_path, groups)
    """
    ranker_lf = ranker_lf.select([
        "sim_item2vec",
        "pop_global",
        "pop_subch",
        "pop_origin",
        "pop_region",
        "channel",
        "pop_store",
        "commune",
        "cand_category",
        "origin",
        "region",
        "subchannel",
        "label"]
    )
    """
    train_path = _artifact_path(artifacts_dir, "train", "parquet")
    ranker_lf.sink_parquet(train_path, compression="lz4")

    return train_path, groups_path

# ==============================
# STEP 3 — BUILD TRAINING DATASET (joins + pops) -> PARQUET + GROUPS NPY
# ==============================

@step(enable_cache=False)
def build_training_dataset_to_files(
    candidates_path: str,
    negatives_path: str,
    baskets_path: str,
    commerces_path: str,
    products_path: str,
    pop_global_path: str,
    pop_store_path: str,
    pop_origin_path: str,
    pop_region_path: str,
    pop_subch_path: str,
    artifacts_dir: str = "artifacts",
) -> tuple[str, str]:

    candidates_lf = pl.scan_parquet(candidates_path)
    negative_lf = pl.scan_parquet(negatives_path)
    baskets_lf    = pl.scan_parquet(baskets_path)
    commerces_lf  = pl.scan_parquet(commerces_path)
    products_lf   = pl.scan_parquet(products_path)

    """
    baskets_lf
    Schema({'orderid': String, 'basket': List(String), 'userid': String, 'origin': String})

    candidates
    Schema({'orderid': String, 'anchor': String, 'candidate': String, 'sim_item2vec': Float32})

    commerces_lf
    Schema({'userid': String, 'sellerid': String, 'active': Boolean, 'commune': String, 'channel': String, 'subchannel': String, 'region': String})

    products_lf
    Schema({'productid': String, 'name': String, 'category': String, 'subcategory': String, 'blocked': Boolean, 'packageunit': String, 'amountperpackage': Float64, 'boxunit': String, 'amountperbox': Int64, 'salesunit': String, 'description': String, 'categoricallevel1': String})
    """

    # --- baskets: wie vorher "basket" als List-Spalte im Endresult behalten
    baskets_meta_lf = baskets_lf.select(["orderid", "userid", "origin"])
    commerces_lf = commerces_lf.drop("sellerid", "active")
    products_lf = products_lf.select(["category", "productid"])

    # --- positives: (orderid, candidate) aus dem basket, zum Labeln per Join
    positives_lf = (
        baskets_lf
        .select(["orderid", "basket"])
        .explode("basket")
        .rename({"basket": "candidate"})
        .unique(["orderid", "candidate"])
        .with_columns(pl.lit(1, dtype=pl.Int8).alias("label"))
    )

    # Base joins: candidates + basket + store meta
    ranker_lf = (
        positives_lf
        .join(candidates_lf, on=["orderid", "candidate"], how="left")
    )

    key_cols = ["orderid", "anchor", "candidate"]  # falls du nur orderid+candidate willst, hier anpassen
    base_cols = ["orderid", "candidate", "label", "anchor", "sim_item2vec"]

    negatives_to_add_lf = (
        negative_lf
        .join(
            ranker_lf.select(key_cols),
            on=key_cols,
            how="anti",  # nur Zeilen, die NICHT in ranker_lf sind
        )
        .with_columns(pl.lit(0, dtype=pl.Int8).alias("label"))
        .select(base_cols)
    )

    ranker_lf = (
        pl.concat(
            [
                ranker_lf,
                negatives_to_add_lf,
            ],
            how="vertical",
        )
    )

    ranker_lf = ranker_lf.sort(["orderid", "anchor"])

    # Candidate product meta
    ranker_lf = ranker_lf.join(baskets_meta_lf, on="orderid", how="left")
    ranker_lf = ranker_lf.join(commerces_lf, on="userid", how="left")

    # Candidate product meta
    ranker_lf = ranker_lf.join(
        products_lf,
        left_on="candidate",
        right_on="productid",
        how="left",
    )

    # Popularity joins
    pop_global_lf = pl.scan_parquet(pop_global_path)
    pop_store_lf  = pl.scan_parquet(pop_store_path)
    pop_origin_lf = pl.scan_parquet(pop_origin_path)
    pop_region_lf = pl.scan_parquet(pop_region_path)
    pop_subch_lf  = pl.scan_parquet(pop_subch_path)
    
    ranker_lf = (
        ranker_lf
        .join(pop_global_lf, left_on="candidate", right_on="productid", how="left")
        .join(pop_store_lf, left_on=["userid", "candidate"], right_on=["userid", "productid"], how="left")
        .join(pop_origin_lf, left_on=["origin", "candidate"], right_on=["origin", "productid"], how="left")
        .join(pop_region_lf, left_on=["region", "candidate"], right_on=["region", "productid"], how="left")
        .join(pop_subch_lf, left_on=["subchannel", "candidate"], right_on=["subchannel", "productid"], how="left")
        .with_columns(
            pl.col("pop_global").fill_null(0),
            pl.col("pop_store").fill_null(0),
            pl.col("pop_origin").fill_null(0),
            pl.col("pop_region").fill_null(0),
            pl.col("pop_subch").fill_null(0),
        )
    )

    ranker_lf = ranker_lf.with_columns(
        pl.col("channel").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("commune").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("origin").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("region").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("subchannel").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).alias("cand_category"),
    )

    ranker_lf = ranker_lf.select([
        "sim_item2vec",
        "pop_global",
        "pop_subch",
        "pop_origin",
        "pop_region",
        "channel",
        "pop_store",
        "commune",
        "cand_category",
        "origin",
        "region",
        "subchannel",
        "label",
        "orderid",
        "anchor"]
    )
    
    ranker_lf = ranker_lf.sort(["orderid", "anchor"])

    groups = (
        ranker_lf
        .select(["orderid", "anchor"])
        .group_by(["orderid", "anchor"], maintain_order=True)
        .agg(pl.len().alias("group_size"))
        .select("group_size")
        .collect()
        .to_numpy()
        .astype(np.int32)
        .ravel()
    )

    groups_path = _artifact_path(artifacts_dir, "groups", "npy")
    np.save(groups_path, groups)

    ranker_lf = ranker_lf.select([
        "sim_item2vec",
        "pop_global",
        "pop_subch",
        "pop_origin",
        "pop_region",
        "channel",
        "pop_store",
        "commune",
        "cand_category",
        "origin",
        "region",
        "subchannel",
        "label"]
    )

    train_path = _artifact_path(artifacts_dir, "train", "parquet")
    ranker_lf.sink_parquet(train_path, compression="lz4")

    """
    n_orderid = int(
        candidates_lf
        .select(pl.col("orderid").count())
        .collect()
        .item()
    )
    
    # Achtung: n_orderid muss hier die ANZAHL ZEILEN in candidates sein (nicht unique orderids)
    if n_orderid % topk != 0:
        raise ValueError(f"n_orderid={n_orderid} ist nicht durch topk={topk} teilbar")

    n_groups = n_orderid // topk
    groups = np.full(n_groups, topk, dtype=np.int32)
    groups_path = _artifact_path(artifacts_dir, "groups", "npy")
    np.save(groups_path, groups)
    """

    return train_path, groups_path


# ==============================
# STEP 4 — TRAIN RANKER
# ==============================

@step(enable_cache=False)
def train_ranker_from_files(
    train_parquet_path: str,
    groups_npy_path: str,
) -> str:

    train_df = pl.read_parquet(train_parquet_path)
    groups = np.load(groups_npy_path)

    feature_cols = [
        "sim_item2vec",
        "pop_global",
        "pop_subch",
        "pop_origin",
        "pop_region",
        #"same_category",
        "channel",
        "pop_store",
        "commune",
        "cand_category",
        "origin",
        "region",
        "subchannel",
    ]
    categorical_cols = [
        "channel",
        "commune",
        "cand_category",
        "origin",
        "region",
        "subchannel",
    ]

    train_pd = train_df.select(feature_cols + ["label"]).to_pandas()
    for col in categorical_cols:
        train_pd[col] = train_pd[col].fillna("UNKNOWN").astype("category")
    for col in feature_cols:
        if col not in categorical_cols:
            train_pd[col] = pd.to_numeric(train_pd[col], errors="coerce").fillna(0.0).astype(np.float32)

    X = train_pd[feature_cols]
    y = train_df.select("label").to_numpy().ravel()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10],
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        force_row_wise=True
    )

    model.fit(X, y, group=groups, categorical_feature=categorical_cols)

    model_path = "models/lgbm_ranker.txt"
    model.booster_.save_model(model_path)
    return model_path


# ==============================
# PIPELINE
# ==============================

@pipeline(enable_cache=False)
def ranker_training_pipeline_fast(
    orders_path: str,
    commerces_path: str,
    products_path: str,
    w2v_path: str,
    artifacts_dir: str = "artifacts",
    topk: int = 10,
) -> str:
    (
        baskets_path,
        pop_global_path,
        pop_store_path,
        pop_origin_path,
        pop_region_path,
        pop_subch_path,
    ) = prepare_data(
        data_path=orders_path,
        commerces_path=commerces_path,
        products_path=products_path,
        artifacts_dir=artifacts_dir,
    )

    candidates_path = generate_candidates_fast_to_parquet(
        baskets_path=baskets_path,
        w2v_model_path=w2v_path,
        topk=topk,
        artifacts_dir=artifacts_dir,
    )

    negatives_path = generate_negatives_to_parquet(
        baskets_path=baskets_path,
        w2v_model_path=w2v_path,
        topk=topk,
        artifacts_dir=artifacts_dir,
    )

    train1_path = train1(
        candidates_path=candidates_path,
        negatives_path=negatives_path,
        baskets_path=baskets_path,
        commerces_path=commerces_path,
        products_path=products_path,
        artifacts_dir=artifacts_dir,
    )

    train_path, groups_path = train2(
        candidates_path=candidates_path,
        negatives_path=negatives_path,
        baskets_path=baskets_path,
        commerces_path=commerces_path,
        products_path=products_path,
        pop_global_path=pop_global_path,
        pop_store_path=pop_store_path,
        pop_origin_path=pop_origin_path,
        pop_region_path=pop_region_path,
        pop_subch_path=pop_subch_path,
        ranker_lf_path = train1_path,
        artifacts_dir=artifacts_dir,
    )
    """
    train_path, groups_path = build_training_dataset_to_files(
        candidates_path=candidates_path,
        negatives_path=negatives_path,
        baskets_path=baskets_path,
        commerces_path=commerces_path,
        products_path=products_path,
        pop_global_path=pop_global_path,
        pop_store_path=pop_store_path,
        pop_origin_path=pop_origin_path,
        pop_region_path=pop_region_path,
        pop_subch_path=pop_subch_path,
        artifacts_dir=artifacts_dir,
    )
    """
    return train_ranker_from_files(train_path, groups_path)
