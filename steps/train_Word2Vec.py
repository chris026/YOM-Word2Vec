from zenml import step
import polars as pl
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from sklearn.manifold import TSNE
from gensim.models import Word2Vec

@step
def build_baskets(df_path: str) -> str:
    """Group order lines into per-order product baskets and save to Parquet.

    Args:
        df_path: Path to the orders Parquet file. Must contain columns
            ``orderid`` and ``productid``.

    Returns:
        Path to the written baskets Parquet file (``data/baskets.parquet``).
        Each row contains one ``orderid`` and the list of associated
        ``productid`` values. Baskets with fewer than 2 items are dropped.
    """
    df = pl.scan_parquet(df_path)

    baskets = (
        df
        .drop_nulls(["orderid","productid"])
        .select(["orderid", "productid"])
        .group_by("orderid")
        .agg(pl.col("productid"))
        # nur Baskets mit >=2 Items
        .filter(pl.col("productid").list.len() >= 2)
        .collect()
    )

    baskets_path = "data/baskets.parquet"
    baskets.write_parquet(baskets_path)
    return baskets_path

@step
def build_baskets_monthly(df_path: str) -> str:
    """Group order lines into per-order product baskets, retaining the order date.

    Like :func:`build_baskets`, but also preserves the ``orderdt`` column
    so that downstream steps can perform time-based splits. Rows with a
    null ``orderid``, ``productid``, or ``orderdt`` are dropped. Baskets
    with fewer than 2 items are removed.

    Args:
        df_path: Path to the orders Parquet file. Must contain columns
            ``orderid``, ``productid``, and ``orderdt``.

    Returns:
        Path to the written baskets Parquet file (``data/baskets.parquet``).
        Each row contains one ``orderid``, the list of associated
        ``productid`` values, and the first ``orderdt`` of that order.
    """

    df = pl.scan_parquet(df_path)

    baskets = (
        df
        .drop_nulls(["orderid", "productid", "orderdt"])
        .select(["orderid", "productid", "orderdt"])
        .group_by("orderid")
        .agg(
            [
                pl.col("productid"),
                pl.col("orderdt").first(),
            ]
        )
        # nur Baskets mit >=2 Items
        .filter(pl.col("productid").list.len() >= 2)
        .collect()
    )

    baskets_path = "data/baskets.parquet"
    baskets.write_parquet(baskets_path)
    return baskets_path

@step
def data_split(baskets_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split baskets into training and test sets using an 80 / 20 ratio.

    Args:
        baskets_path: Path to the baskets Parquet file produced by
            :func:`build_baskets`.

    Returns:
        A tuple ``(train_df, test_df)`` where the first 80 % of rows form
        the training set and the remaining 20 % form the test set.
    """
    baskets = pl.read_parquet(baskets_path)
    split_idx = int(len(baskets) * 0.8)
    train_df = baskets[:split_idx]
    test_df = baskets[split_idx:]
    return train_df, test_df

def _shift_month(month_start: date, months: int) -> date:
    month_index = month_start.month - 1 + months
    year = month_start.year + (month_index // 12)
    month = (month_index % 12) + 1
    return date(year, month, 1)

@step
def data_split_monthly(baskets_path: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split baskets into training and test sets by calendar month.

    The two most recent months are held out as the test set; all earlier
    months form the training set. Supports ``orderdt`` columns of type
    ``pl.Date``, ``pl.Datetime``, or string (auto-parsed).

    Args:
        baskets_path: Path to the baskets Parquet file. Must contain an
            ``orderdt`` column with parseable date values.

    Returns:
        A tuple ``(train_df, test_df)``. Both DataFrames have the
        ``orderdt``.

    Raises:
        ValueError: If the ``orderdt`` column is missing or contains no
            parseable dates.
    """
    baskets = pl.read_parquet(baskets_path)

    if "orderdt" not in baskets.columns:
        raise ValueError("data_split_monthly requires an 'orderdt' column.")

    orderdt_dtype = baskets.schema["orderdt"]
    orderdt_col = pl.col("orderdt")

    if orderdt_dtype == pl.Date:
        month_expr = orderdt_col.dt.truncate("1mo")
    elif orderdt_dtype == pl.Datetime:
        month_expr = orderdt_col.dt.date().dt.truncate("1mo")
    else:
        parsed_date_expr = pl.coalesce(
            [
                orderdt_col.str.strptime(pl.Datetime, strict=False).dt.date(),
                orderdt_col.str.strptime(pl.Date, strict=False),
            ]
        )
        month_expr = parsed_date_expr.dt.truncate("1mo")

    with_month = baskets.with_columns(month_expr.alias("_order_month"))

    month_values = (
        with_month
        .select(pl.col("_order_month").drop_nulls().unique().sort())
        .to_series()
    )

    if len(month_values) == 0:
        raise ValueError("data_split_monthly found no parseable order dates in 'orderdt'.")

    latest_month = month_values[-1]
    test_from_month = _shift_month(latest_month, -1)

    train_df = (
        with_month
        .filter(pl.col("_order_month").is_null() | (pl.col("_order_month") < test_from_month))
        .drop(["_order_month", "orderdt"])
    )
    test_df = (
        with_month
        .filter(pl.col("_order_month").is_not_null() & (pl.col("_order_month") >= test_from_month))
        .drop(["_order_month", "orderdt"])
    )

    return train_df, test_df

@step(enable_cache=False)
def train_model(train_df_path: str) -> str:
    """Train a Word2Vec skip-gram model on the provided training baskets.

    Hyperparameters: ``vector_size=35``, ``window=100``, ``sg=1``,
    ``min_count=2``, ``shrink_windows=False``. Workers are set to
    ``cpu_count - 1`` (minimum 1).

    Args:
        train_df_path: Path to the training baskets Parquet file. Must
            contain a ``productid`` column with lists of product ID strings.

    Returns:
        Path to the saved Word2Vec model (``models/word2vec.model``).
    """
    train_df = pl.read_parquet(train_df_path)
    sentences = PolarsBasketIterator(train_df)
    model = Word2Vec(
        sentences=sentences,
        vector_size=35,
        window=100,
        sg=1,
        shrink_windows=False,
        workers=max(1, os.cpu_count() - 1),
        min_count=2
    )

    model_path = "models/word2vec.model"
    model.save(model_path)
    return model_path

class PolarsBasketIterator:
    def __init__(self, df):
        self.df = df

    def __iter__(self):
        for row in self.df.iter_rows(named=True):
            yield row["productid"]

@step
def plot_all_items_2d(model_path, max_labels=60, random_state=43):
    model = Word2Vec.load(model_path)
    items = list(model.wv.index_to_key)
    X = np.array([model.wv[pid] for pid in items])

    reducer = TSNE(n_components=2, perplexity=30, init="pca", random_state=random_state)
    X2 = reducer.fit_transform(X)

    plt.figure(figsize=(12, 9))
    plt.scatter(X2[:, 0], X2[:, 1])

    # label only a subset to avoid unreadable mess
    if len(items) <= max_labels:
        label_idx = range(len(items))
    else:
        label_idx = np.random.RandomState(random_state).choice(len(items), size=max_labels, replace=False)

    #for i in label_idx:
    #    name = prod_name.get(items[i], "UNKNOWN")
    #    label = f"{name}"   # oder nur name
    #    plt.text(X2[i, 0], X2[i, 1], label, fontsize=6)

    plt.title("t-SNE 2D projection of all product embeddings")
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.savefig("word2vec.png")

@step
def test_model(model: Word2Vec):
    print(retrieve_candidates(model, "000051-007"))

def retrieve_candidates(model: Word2Vec, anchor_pid: str, topk: int = 5):
    """Return the top-k most similar products for a given anchor product.

    Uses cosine similarity in the Word2Vec embedding space.

    Args:
        model: A trained :class:`gensim.models.Word2Vec` model.
        anchor_pid: Product ID to use as the query vector.
        topk: Number of nearest neighbours to return. Defaults to ``5``.

    Returns:
        A list of ``(product_id, similarity_score)`` tuples ordered by
        descending similarity. Returns an empty list if ``anchor_pid`` is
        not in the model vocabulary.
    """
    anchor_pid = str(anchor_pid)
    if anchor_pid not in model.wv:
        return []
    return [(pid, float(sim)) for pid, sim in model.wv.most_similar(anchor_pid, topn=topk)]
