import os
import polars as pl
import numpy as np
import lightgbm as lgb

from gensim.models import Word2Vec
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
) -> tuple[str, str, str, str, str]:
    """Build product baskets and popularity tables from raw order data.

    Creates one basket per order (list of product IDs) and four popularity
    count tables: global, per store, per region, and per subchannel.
    All outputs are written to Parquet files in ``artifacts_dir``.

    Args:
        data_path: Path to the orders Parquet file.
        commerces_path: Path to the commerces Parquet file. Used to join
            ``region`` and ``subchannel`` onto orders.
        products_path: Path to the products Parquet file. Read to validate
            the file exists and the schema is readable.
        artifacts_dir: Output directory for artifact files.
            Defaults to ``"artifacts"``.

    Returns:
        A tuple of five Parquet file paths:
        ``(baskets_path, pop_global_path, pop_store_path,
        pop_region_path, pop_subch_path)``.
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
            #pl.first("origin").alias("origin"),
        )
    )
    baskets_path = _artifact_path(artifacts_dir, "baskets", "parquet")
    baskets_lf.collect().write_parquet(baskets_path)

    # --- popularity tables ---
    pop_global_lf = orders.group_by("productid").agg(pl.len().alias("pop_global"))
    pop_store_lf = orders.group_by(["userid", "productid"]).agg(pl.len().alias("pop_store"))
    #pop_origin_lf = orders.group_by(["origin", "productid"]).agg(pl.len().alias("pop_origin"))

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
    #pop_origin_path = _artifact_path(artifacts_dir, "pop_origin", "parquet")
    pop_region_path = _artifact_path(artifacts_dir, "pop_region", "parquet")
    pop_subch_path = _artifact_path(artifacts_dir, "pop_subch", "parquet")

    pop_global_lf.collect().write_parquet(pop_global_path)
    pop_store_lf.collect().write_parquet(pop_store_path)
    #pop_origin_lf.collect().write_parquet(pop_origin_path)
    pop_region_lf.collect().write_parquet(pop_region_path)
    pop_subch_lf.collect().write_parquet(pop_subch_path)

    return (
        baskets_path,
        pop_global_path,
        pop_store_path,
        #pop_origin_path,
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
    artifacts_dir: str = "artifacts",
) -> str:
    """Generate the full Word2Vec candidate set for every anchor product.

    For each product in the Word2Vec vocabulary, retrieves all other
    products ordered by cosine similarity (``topn = vocab_size``). The
    resulting ``(anchor, candidate, sim_item2vec)`` pairs are joined with
    the baskets so that every ``(orderid, anchor)`` combination gets its
    candidate list.

    Args:
        baskets_path: Path to the baskets Parquet file produced by
            :func:`prepare_data`.
        w2v_model_path: Path to the trained Word2Vec model.
        artifacts_dir: Output directory for artifact files.
            Defaults to ``"artifacts"``.

    Returns:
        Path to the candidates Parquet file with columns
        ``orderid``, ``anchor``, ``candidate``, ``sim_item2vec``.
    """
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
    
    neigh_df_path = _artifact_path(artifacts_dir, "neigh_df", "csv")
    neigh_df.write_csv(neigh_df_path)
    out_path = _artifact_path(artifacts_dir, "candidates", "parquet")

    """
    baskets_df.schema
    Schema({'orderid': String, 'basket': List(String), 'userid': String, 'origin': String})
    neigh_df.schema
    Schema({'anchor': String, 'candidate': String, 'sim_item2vec': Float32})
    """
    
    baskets_df = pl.scan_parquet(baskets_path)
    (
        baskets_df
        .select(["orderid", "basket"])
        .explode("basket")
        .rename({"basket": "anchor"})
        .join(pl.scan_csv(neigh_df_path, schema_overrides={"sim_item2vec": pl.Float32}), on="anchor", how="inner")
        .sink_parquet(out_path)
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
    """Generate a limited candidate set for negative sampling.

    Similar to :func:`generate_candidates_fast_to_parquet` but restricts
    retrieval to the top ``topk`` neighbours per anchor. The smaller set
    is used as the negative pool in :func:`label_candidates`.

    Args:
        baskets_path: Path to the baskets Parquet file.
        w2v_model_path: Path to the trained Word2Vec model.
        topk: Number of nearest neighbours to retrieve per anchor product.
        artifacts_dir: Output directory for artifact files.
            Defaults to ``"artifacts"``.

    Returns:
        Path to the negatives Parquet file with columns
        ``orderid``, ``anchor``, ``candidate``, ``sim_item2vec``.
    """
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
def label_candidates(
    candidates_path: str,
    negatives_path: str,
    baskets_path: str,
    artifacts_dir: str = "artifacts",
) -> str:
    """Assign relevance labels to candidates and negatives.

    Marks every ``(orderid, candidate)`` pair that actually appears in the
    basket as a positive (``label=1``). Negative examples are taken from
    the negatives set, excluding any pair already present as a positive
    (anti-join). The labelled dataset is the input for feature enrichment
    in :func:`build_feature_matrix`.

    Args:
        candidates_path: Path to the full candidates Parquet file from
            :func:`generate_candidates_fast_to_parquet`.
        negatives_path: Path to the negatives Parquet file from
            :func:`generate_negatives_to_parquet`.
        baskets_path: Path to the baskets Parquet file.
        artifacts_dir: Output directory for artifact files.
            Defaults to ``"artifacts"``.

    Returns:
        Path to the labelled training Parquet file with columns
        ``orderid``, ``anchor``, ``candidate``, ``sim_item2vec``, ``label``.
    """
    candidates_lf = pl.scan_parquet(candidates_path)
    negative_lf = pl.scan_parquet(negatives_path)
    baskets_lf    = pl.scan_parquet(baskets_path)

    """
    baskets_lf
    Schema({'orderid': String, 'basket': List(String), 'userid': String, 'origin': String})

    candidates
    Schema({'orderid': String, 'anchor': String, 'candidate': String, 'sim_item2vec': Float32})
    """

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

    key_cols = ["orderid", "anchor", "candidate"]
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
    ranker_lf.sink_parquet(path=path, compression="lz4")
    return path

@step
def build_feature_matrix(
    baskets_path: str,
    commerces_path: str,
    products_path: str,
    pop_global_path: str,
    pop_store_path: str,
    #pop_origin_path: str,
    pop_region_path: str,
    pop_subch_path: str,
    ranker_lf_path: str,
    artifacts_dir: str = "artifacts",
) -> tuple[str, str]:
    """Enrich the labelled dataset with contextual features.

    Joins store metadata (channel, commune, region, subchannel) and product
    category onto each candidate row. Also attaches the four popularity
    scores (global, store, region, subchannel). Computes LightGBM group
    sizes (number of candidates per ``(orderid, anchor)`` pair) and saves
    them as a NumPy array.

    Args:
        baskets_path: Path to the baskets Parquet file.
        commerces_path: Path to the commerces Parquet file.
        products_path: Path to the products Parquet file.
        pop_global_path: Path to the global popularity Parquet file.
        pop_store_path: Path to the per-store popularity Parquet file.
        pop_region_path: Path to the per-region popularity Parquet file.
        pop_subch_path: Path to the per-subchannel popularity Parquet file.
        ranker_lf_path: Path to the labelled training Parquet file from
            :func:`label_candidates`.
        artifacts_dir: Output directory for artifact files.
            Defaults to ``"artifacts"``.

    Returns:
        A tuple ``(train_parquet_path, groups_npy_path)`` where
        ``train_parquet_path`` is the enriched training Parquet file and
        ``groups_npy_path`` is the NumPy file with group sizes for LightGBM.
    """
    baskets_lf    = pl.scan_parquet(baskets_path)
    commerces_lf  = pl.scan_parquet(commerces_path)
    products_lf   = pl.scan_parquet(products_path)
    
    """
    baskets_lf
    Schema({'orderid': String, 'basket': List(String), 'userid': String, 'origin': String})

    commerces_lf
    Schema({'userid': String, 'sellerid': String, 'active': Boolean, 'commune': String, 'channel': String, 'subchannel': String, 'region': String})

    products_lf
    Schema({'productid': String, 'name': String, 'category': String, 'subcategory': String, 'blocked': Boolean, 'packageunit': String, 'amountperpackage': Float64, 'boxunit': String, 'amountperbox': Int64, 'salesunit': String, 'description': String, 'categoricallevel1': String})
    """

    baskets_meta_lf = baskets_lf.select(["orderid", "userid"])
    commerces_lf = commerces_lf.drop("sellerid", "active")
    products_lf = products_lf.select(["category", "productid"])


    ranker_lf = pl.scan_parquet(ranker_lf_path)
    ranker_lf = ranker_lf.join(baskets_meta_lf, on="orderid", how="left")
    ranker_lf = ranker_lf.join(commerces_lf, on="userid", how="left")
    ranker_lf = ranker_lf.join(products_lf, left_on="candidate", right_on="productid", how="left")

    # Popularity joins
    pop_global_lf = pl.scan_parquet(pop_global_path)
    pop_store_lf  = pl.scan_parquet(pop_store_path)
    #pop_origin_lf = pl.scan_parquet(pop_origin_path)
    pop_region_lf = pl.scan_parquet(pop_region_path)
    pop_subch_lf  = pl.scan_parquet(pop_subch_path)
    
    ranker_lf = (
        ranker_lf
        .join(pop_global_lf, left_on="candidate", right_on="productid", how="left")
        .join(pop_store_lf, left_on=["userid", "candidate"], right_on=["userid", "productid"], how="left")
        #.join(pop_origin_lf, left_on=["origin", "candidate"], right_on=["origin", "productid"], how="left")
        .join(pop_region_lf, left_on=["region", "candidate"], right_on=["region", "productid"], how="left")
        .join(pop_subch_lf, left_on=["subchannel", "candidate"], right_on=["subchannel", "productid"], how="left")
        .with_columns(
            pl.col("pop_global").fill_null(0),
            pl.col("pop_store").fill_null(0),
            #pl.col("pop_origin").fill_null(0),
            pl.col("pop_region").fill_null(0),
            pl.col("pop_subch").fill_null(0),
        )
    )

    ranker_lf = ranker_lf.with_columns(
        pl.col("channel").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("commune").fill_null("UNKNOWN").cast(pl.Utf8),
        #pl.col("origin").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("region").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("subchannel").fill_null("UNKNOWN").cast(pl.Utf8),
        pl.col("category").fill_null("UNKNOWN").cast(pl.Utf8).alias("cand_category"),
    )

    ranker_lf = ranker_lf.select([
        "sim_item2vec",
        "pop_global",
        "pop_subch",
        #"pop_origin",
        "pop_region",
        "channel",
        "pop_store",
        "commune",
        "cand_category",
        #"origin",
        "region",
        "subchannel",
        "label",
        "orderid",
        "anchor"]
    )

    ranker_lf = ranker_lf.sort(["orderid", "anchor"])
    train_path = _artifact_path(artifacts_dir, "train", "parquet")
    ranker_lf.sink_parquet(train_path, compression="lz4")

    groups = (
        pl.scan_parquet(train_path)
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
    return train_path, groups_path


# ==============================
# STEP 4 — TRAIN RANKER
# ==============================

@step(enable_cache=False)
def train_ranker_from_files(
    train_parquet_path: str,
    groups_npy_path: str,
) -> str:
    """Train a LightGBM LambdaRank model from the prepared training data.

    Loads the enriched feature matrix, encodes categorical columns, and
    fits an :class:`lgb.LGBMRanker` with ``objective="lambdarank"`` and
    ``metric="ndcg"``. The trained booster is saved as a plain-text model
    file.

    Features used: ``sim_item2vec``, ``pop_global``, ``pop_subch``,
    ``pop_region``, ``pop_store``, ``channel``, ``commune``,
    ``cand_category``, ``region``, ``subchannel``.

    Args:
        train_parquet_path: Path to the enriched training Parquet file
            produced by :func:`build_feature_matrix`.
        groups_npy_path: Path to the NumPy ``.npy`` file with group sizes
            (number of candidates per query) produced by :func:`build_feature_matrix`.

    Returns:
        Path to the saved LightGBM model (``models/lgbm_ranker.txt``).
    """
    feature_cols = [
        "sim_item2vec",
        "pop_global",
        "pop_subch",
        #"pop_origin",
        "pop_region",
        "channel",
        "pop_store",
        "commune",
        "cand_category",
        #"origin",
        "region",
        "subchannel",
    ]
    categorical_cols = [
        "channel",
        "commune",
        "cand_category",
        #"origin",
        "region",
        "subchannel",
    ]

    train_lf = (
        pl.scan_parquet(train_parquet_path)
        .select(feature_cols + ["label"])
        .with_columns(
            pl.col("sim_item2vec").fill_null(0.0).cast(pl.Float32),
            pl.col("pop_global").fill_null(0).cast(pl.UInt32),
            pl.col("pop_subch").fill_null(0).cast(pl.UInt16),
            pl.col("pop_region").fill_null(0).cast(pl.UInt16),
            pl.col("pop_store").fill_null(0).cast(pl.UInt8),
            pl.col("channel").fill_null("UNKNOWN"),
            pl.col("commune").fill_null("UNKNOWN"),
            pl.col("cand_category").fill_null("UNKNOWN"),
            pl.col("region").fill_null("UNKNOWN"),
            pl.col("subchannel").fill_null("UNKNOWN"),
        )
    )

    train_df = train_lf.collect()
    train_pd = train_df.to_pandas()

    for c in categorical_cols:
        train_pd[c] = train_pd[c].astype("category")

    X = train_pd[feature_cols]
    y = train_df.select("label").to_numpy().ravel()

    model = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        ndcg_eval_at=[5, 10],
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        force_row_wise=True,
        n_jobs=max(1, os.cpu_count() - 1)
    )

    groups = np.load(groups_npy_path)
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
    """Orchestrate the full LightGBM ranker training pipeline.

    Runs all sub-steps in order: data preparation, candidate generation,
    negative sampling, label assignment, feature enrichment, and ranker
    training.

    Args:
        orders_path: Path to the orders Parquet file.
        commerces_path: Path to the commerces (store metadata) Parquet file.
        products_path: Path to the products Parquet file.
        w2v_path: Path to the trained Word2Vec model.
        artifacts_dir: Directory for intermediate Parquet and NumPy
            artifacts. Defaults to ``"artifacts"``.
        topk: Number of Word2Vec neighbours used for negative sampling.
            Defaults to ``10``.

    Returns:
        Path to the trained LightGBM ranker model (``models/lgbm_ranker.txt``).
    """
    (
        baskets_path,
        pop_global_path,
        pop_store_path,
        #pop_origin_path,
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
        artifacts_dir=artifacts_dir,
    )

    negatives_path = generate_negatives_to_parquet(
        baskets_path=baskets_path,
        w2v_model_path=w2v_path,
        topk=topk,
        artifacts_dir=artifacts_dir,
    )

    label_candidates_path = label_candidates(
        candidates_path=candidates_path,
        negatives_path=negatives_path,
        baskets_path=baskets_path,
        artifacts_dir=artifacts_dir,
    )

    train_path, groups_path = build_feature_matrix(
        baskets_path=baskets_path,
        commerces_path=commerces_path,
        products_path=products_path,
        pop_global_path=pop_global_path,
        pop_store_path=pop_store_path,
        #pop_origin_path=pop_origin_path,
        pop_region_path=pop_region_path,
        pop_subch_path=pop_subch_path,
        ranker_lf_path = label_candidates_path,
        artifacts_dir=artifacts_dir,
    )

    return train_ranker_from_files(train_path, groups_path)
