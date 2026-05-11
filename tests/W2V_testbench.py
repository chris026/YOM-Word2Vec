"""
Offline evaluation testbench for the Word2Vec product-recommendation model.

For every product (anchor) in every evaluation basket the model retrieves the
top-K most-similar products.  The remaining basket items serve as ground-truth
positives.  The script computes the standard information-retrieval metrics
HitRate, Recall, MRR and Precision at each cutoff K and prints a coverage
breakdown (OOV rate, unusable anchors, etc.).

Usage:
    Edit the global constants at the top of the file, then run:
        python tests/W2V_testbench.py
"""

import random
from dataclasses import dataclass

import pandas as pd
import polars as pl
from gensim.models import Word2Vec
from tqdm import tqdm


# Global configuration (edit here, then run the file directly)
MODEL_PATH = "models/word2vec.model"
EVAL_ORDERS_PATH = "data/test_4weeks.csv"
KS = [5, 10, 20, 50]
RETRIEVAL_TOPK = 50
MIN_BASKET_SIZE = 2
EVAL_MAX_ORDERS = 0  # 0 = all orders
NUM_ANCHORS = 0  # 0 = all anchors
SEED = 42
SHOW_PROGRESS = True


@dataclass
class EvalCounts:
    """Accumulates raw metric sums for a single cutoff K across all evaluated anchors.

    All fields are running totals that are later divided by the number of anchors
    to produce the final mean metrics reported in the evaluation table.

    Attributes:
        hit_sum: Number of anchors for which at least one positive appeared in the top-K.
        recall_sum: Sum of per-anchor recall values (hits / number of positives).
        mrr_sum: Sum of per-anchor reciprocal rank values.
        precision_sum: Sum of per-anchor precision values (hits / K).
        positives_sum: Total number of true positives retrieved across all anchors.
    """

    hit_sum: float = 0.0
    recall_sum: float = 0.0
    mrr_sum: float = 0.0
    precision_sum: float = 0.0
    positives_sum: float = 0.0


def validate_config() -> None:
    """Validates the global configuration constants before the evaluation run.

    Raises:
        ValueError: If KS contains non-positive values or if RETRIEVAL_TOPK is
            smaller than the largest cutoff in KS (which would make higher-K
            metrics meaningless).
    """
    if not KS or any(k <= 0 for k in KS):
        raise ValueError("KS must contain positive integers, e.g. [5, 10, 20, 50]")

    if RETRIEVAL_TOPK < max(KS):
        raise ValueError(f"RETRIEVAL_TOPK ({RETRIEVAL_TOPK}) must be >= max(KS) ({max(KS)})")


def read_orders(path: str) -> pl.DataFrame:
    """Loads order–product rows from a CSV or Parquet file into a Polars DataFrame.

    Only the columns ``orderid`` and ``productid`` are read.  Both columns are
    cast to ``Utf8`` (string) and rows with null values in either column are
    dropped.

    Args:
        path: File path to the order data.  Must end with ``.csv`` or ``.parquet``.

    Returns:
        A Polars DataFrame with columns ``orderid`` (Utf8) and ``productid`` (Utf8).

    Raises:
        ValueError: If the file extension is neither ``.csv`` nor ``.parquet``.
    """
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pl.read_parquet(path, columns=["orderid", "productid"])
    elif lower.endswith(".csv"):
        df = pl.read_csv(
            path,
            columns=["orderid", "productid"],
            schema_overrides={"orderid": pl.Utf8, "productid": pl.Utf8},
        )
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .csv or .parquet")

    return (
        df.drop_nulls(["orderid", "productid"])
        .with_columns(
            pl.col("orderid").cast(pl.Utf8),
            pl.col("productid").cast(pl.Utf8),
        )
    )


def unique_preserve_order(items: list[str]) -> list[str]:
    """Removes duplicate entries from a list while preserving the original order.

    Args:
        items: Input list that may contain duplicate strings.

    Returns:
        A new list with duplicates removed, first-occurrence order retained.
    """
    return list(dict.fromkeys(items))


def build_anchor_tasks(
    orders: pl.DataFrame,
    min_basket_size: int,
    eval_max_orders: int,
) -> list[tuple[str, list[str]]]:
    """Constructs the list of (anchor, basket) evaluation tasks from order data.

    Each order is first aggregated into a deduplicated basket of product IDs.
    Orders with fewer than ``min_basket_size`` unique products are discarded.
    Every product in a qualifying basket then becomes an anchor, paired with the
    full basket (including itself) so that the caller can derive the positives
    as ``basket - {anchor}``.

    Args:
        orders: DataFrame with columns ``orderid`` and ``productid``.
        min_basket_size: Minimum number of distinct products an order must contain
            to be included in the evaluation.
        eval_max_orders: If > 0, only the first ``eval_max_orders`` baskets are
            used; 0 means all baskets.

    Returns:
        A list of ``(anchor_product_id, basket_product_ids)`` tuples.
    """
    baskets = (
        orders.group_by("orderid", maintain_order=True)
        .agg(pl.col("productid"))
        .with_columns(pl.col("productid").list.unique())
        .filter(pl.col("productid").list.len() >= min_basket_size)
    )

    if eval_max_orders > 0:
        baskets = baskets.head(eval_max_orders)

    tasks_df = (
        baskets.select([
            pl.col("productid").alias("basket"),
            pl.col("productid").alias("anchor")
        ])
        .explode("anchor")
    )

    tasks = list(tasks_df.select("anchor", "basket").iter_rows())
    return tasks


def first_relevant_rank(pred: list[str], positives: set[str]) -> int | None:
    """Returns the 1-based rank of the first relevant item in a ranked prediction list.

    Used to compute the reciprocal rank (MRR) component for a single query.

    Args:
        pred: Ordered list of predicted product IDs (index 0 = rank 1).
        positives: Set of ground-truth positive product IDs.

    Returns:
        The 1-based position of the first hit, or ``None`` if no positive
        appears in ``pred``.
    """
    for idx, pid in enumerate(pred, start=1):
        if pid in positives:
            return idx
    return None


def evaluate(
    model: Word2Vec,
    tasks: list[tuple[str, list[str]]],
    ks: list[int],
    topk: int,
    num_anchors: int,
    seed: int,
    show_progress: bool,
) -> tuple[dict[int, EvalCounts], dict[str, int]]:
    """Runs the retrieval evaluation loop and accumulates metric counts per cutoff K.

    For each (anchor, basket) task the model retrieves the top ``topk`` most
    similar products.  The other basket items are the ground-truth positives.
    Anchors that are out-of-vocabulary (OOV) or whose basket contains no other
    products are skipped and counted in the coverage statistics.

    The metrics are accumulated as raw sums in ``EvalCounts`` objects; divide by
    ``used_anchors`` (or ``total_anchors`` for a conservative OOV-penalised view)
    to obtain mean values.

    Args:
        model: Loaded Gensim Word2Vec model.
        tasks: List of ``(anchor, basket)`` tuples as produced by
            :func:`build_anchor_tasks`.
        ks: Cutoff values at which metrics are computed (e.g. ``[5, 10, 20, 50]``).
        topk: Number of candidates retrieved from the model per anchor; must be
            >= ``max(ks)``.
        num_anchors: If > 0, a random subsample of this size is drawn from
            ``tasks`` before evaluation; 0 evaluates all tasks.
        seed: Random seed used for the optional subsampling.
        show_progress: If ``True``, a tqdm progress bar is shown.

    Returns:
        A tuple ``(results, coverage)`` where:

        - ``results`` maps each K to an :class:`EvalCounts` instance holding the
          accumulated metric sums.
        - ``coverage`` is a dict with keys ``total_anchors``, ``used_anchors``,
          ``oov_anchors``, ``no_positive_anchors``, ``no_candidate_anchors``.
    """
    rng = random.Random(seed)
    if num_anchors > 0 and len(tasks) > num_anchors:
        tasks = rng.sample(tasks, k=num_anchors)

    results = {k: EvalCounts() for k in ks}

    total_anchors = 0
    used_anchors = 0
    oov_anchors = 0
    no_positive_anchors = 0
    no_candidate_anchors = 0

    iterator = tasks
    if show_progress:
        iterator = tqdm(tasks, desc="Evaluating anchors", unit="anchor")

    for anchor, basket in iterator:
        total_anchors += 1

        if anchor not in model.wv:
            oov_anchors += 1
            continue

        positives = {p for p in basket if p != anchor}
        if not positives:
            no_positive_anchors += 1
            continue

        try:
            ranked = [pid for pid, _ in model.wv.most_similar(anchor, topn=topk)]
        except KeyError:
            oov_anchors += 1
            continue

        if not ranked:
            no_candidate_anchors += 1
            continue

        used_anchors += 1

        for k in ks:
            pred = ranked[:k]
            hits = sum(1 for pid in pred if pid in positives)

            rr_rank = first_relevant_rank(pred, positives)
            rr = 0.0 if rr_rank is None else (1.0 / rr_rank)

            results[k].hit_sum += 1.0 if hits > 0 else 0.0
            results[k].recall_sum += hits / len(positives)
            results[k].mrr_sum += rr
            results[k].precision_sum += hits / k
            results[k].positives_sum += hits

    coverage = {
        "total_anchors": total_anchors,
        "used_anchors": used_anchors,
        "oov_anchors": oov_anchors,
        "no_positive_anchors": no_positive_anchors,
        "no_candidate_anchors": no_candidate_anchors,
    }
    return results, coverage


def to_report_frame(results: dict[int, EvalCounts], all_anchors: int) -> pd.DataFrame:
    """Converts accumulated EvalCounts into a human-readable metrics DataFrame.

    Each row corresponds to one cutoff K.  All metric values are mean values
    computed by dividing the accumulated sums by ``all_anchors`` (which includes
    OOV anchors, giving a conservative estimate that penalises low vocabulary
    coverage).

    Args:
        results: Mapping from cutoff K to its :class:`EvalCounts` instance.
        all_anchors: Total number of anchors processed (including OOV).  Used as
            the denominator for all metrics.

    Returns:
        A pandas DataFrame with columns ``K``, ``HitRate``, ``Recall``, ``MRR``,
        ``Precision``, and ``Positives``, sorted ascending by K.
    """
    rows = []
    denom = max(1, all_anchors)

    for k in sorted(results):
        agg = results[k]
        rows.append(
            {
                "K": k,
                "HitRate": agg.hit_sum / denom,
                "Recall": agg.recall_sum / denom,
                "MRR": agg.mrr_sum / denom,
                "Precision": agg.precision_sum / denom,
                "Positives": agg.positives_sum / denom,
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    """Entry point for the W2V evaluation testbench.

    Orchestrates the full evaluation pipeline:

    1. Validates the global configuration constants.
    2. Loads the Word2Vec model and the evaluation order data.
    3. Builds the anchor tasks from the orders.
    4. Runs the evaluation loop and collects metric counts and coverage stats.
    5. Prints a coverage breakdown and a per-K metrics table to stdout.
    """
    validate_config()

    print("W2V Testbench")
    print(f"model_path       : {MODEL_PATH}")
    print(f"eval_orders_path : {EVAL_ORDERS_PATH}")
    print(f"ks               : {KS}")
    print(f"retrieval_topk   : {RETRIEVAL_TOPK}")

    model = Word2Vec.load(MODEL_PATH)
    orders = read_orders(EVAL_ORDERS_PATH)
    tasks = build_anchor_tasks(
        orders=orders,
        min_basket_size=MIN_BASKET_SIZE,
        eval_max_orders=EVAL_MAX_ORDERS,
    )

    results, coverage = evaluate(
        model=model,
        tasks=tasks,
        ks=KS,
        topk=RETRIEVAL_TOPK,
        num_anchors=NUM_ANCHORS,
        seed=SEED,
        show_progress=SHOW_PROGRESS,
    )

    report_df = to_report_frame(results, all_anchors=coverage["total_anchors"])

    print("\nCoverage")
    print(f"total_anchors       : {coverage['total_anchors']}")
    print(f"used_anchors        : {coverage['used_anchors']}")
    print(f"oov_anchors         : {coverage['oov_anchors']}")
    print(f"no_positive_anchors : {coverage['no_positive_anchors']}")
    print(f"no_candidate_anchors: {coverage['no_candidate_anchors']}")

    print("\nMetrics (mean over all anchors, including OOV)")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
