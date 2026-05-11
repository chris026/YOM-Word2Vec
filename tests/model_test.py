"""
End-to-end evaluation testbench for the bundle recommendation pipeline.

For every qualifying order a single product is randomly chosen as anchor.
The remaining basket items are the ground-truth positives.  The live
``getMultiRec`` inference function is called for all anchors at once, and
the returned recommendations are scored with standard information-retrieval
metrics (HitRate, Recall, MRR, Precision) at multiple cutoffs K.

Unlike W2V_testbench.py this script tests the full serving stack
(``serve_bundle.getMultiRec``) rather than the raw Word2Vec embeddings.

Usage:
    Edit the global constants at the top of the file, then run:
        python tests/model_test.py
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from tqdm.auto import tqdm

import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve_bundle import getMultiRec


# Global configuration
TEST_DATA_PATH = ROOT_DIR / "data" / "test_df_1m.parquet"
KS = [5, 10, 20]
RANDOM_SEED = 42
MIN_BASKET_SIZE = 2
MAX_ORDERS = 0  # 0 = all orders


@dataclass
class EvalTask:
    """Holds all data needed to evaluate a single anchor within one order.

    Attributes:
        orderid: Unique identifier of the source order.
        userid: User who placed the order; forwarded to ``getMultiRec`` for
            personalised recommendations.
        anchor: The product ID used as the query for the recommendation call.
        positives: Set of all other product IDs in the same basket; these are
            the ground-truth items the model should retrieve.
    """

    orderid: str
    userid: str
    anchor: str
    positives: set[str]


@dataclass
class MetricAgg:
    """Accumulates raw metric sums for a single cutoff K across all evaluated tasks.

    All fields are running totals that are divided by the number of evaluated
    anchors in ``print_report`` to produce the final mean metric values.

    Attributes:
        hit_sum: Number of tasks for which at least one positive appeared in
            the top-K recommendations.
        recall_sum: Sum of per-task recall values (hits / number of positives).
        mrr_sum: Sum of per-task reciprocal rank values.
        precision_sum: Sum of per-task precision values (hits / K).
        positives_sum: Total number of true positives retrieved across all tasks.
    """

    hit_sum: float = 0.0
    recall_sum: float = 0.0
    mrr_sum: float = 0.0
    precision_sum: float = 0.0
    positives_sum: float = 0.0


def read_orders(path: Path) -> pl.DataFrame:
    """Loads order–product–user rows from a Parquet file into a Polars DataFrame.

    Only the columns ``orderid``, ``productid``, and ``userid`` are read.
    Rows with null values in any of those columns are dropped.

    Args:
        path: Filesystem path to the Parquet test-data file.

    Returns:
        A collected Polars DataFrame with columns ``orderid``, ``productid``,
        and ``userid``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Testdatei nicht gefunden: {path}")

    df = pl.scan_parquet(
        str(path),
    ).select(["orderid", "productid", "userid"])

    return (
        df.drop_nulls(["orderid", "productid", "userid"])
        .collect()
    )


def build_eval_tasks(
    orders: pl.DataFrame,
    seed: int,
    min_basket_size: int,
    max_orders: int,
) -> list[EvalTask]:
    """Constructs the list of :class:`EvalTask` objects from order data.

    Each order is aggregated into a deduplicated basket of products.  Orders
    with fewer than ``min_basket_size`` distinct products are discarded.  For
    each qualifying order a single product is drawn at random as the anchor;
    the remaining products become the positives.

    Args:
        orders: DataFrame with columns ``orderid``, ``productid``, and
            ``userid``.
        seed: Random seed used for reproducible anchor selection.
        min_basket_size: Minimum number of distinct products an order must
            contain to be included.
        max_orders: If > 0, only the first ``max_orders`` qualifying baskets
            are processed; 0 means all baskets.

    Returns:
        A list of :class:`EvalTask` instances, one per qualifying order.
    """
    baskets = (
        orders.group_by("orderid")
        .agg(
            pl.col("productid").unique().alias("products"),
            pl.col("userid").first()
        )
        .filter(pl.col("products").list.len() >= min_basket_size)
    )

    if max_orders > 0:
        baskets = baskets.head(max_orders)

    rng = random.Random(seed)
    tasks: list[EvalTask] = []
    for orderid, products, userid in tqdm(
        baskets.iter_rows(),
        total=baskets.height,
        desc="Build eval tasks",
    ):
        product_list = [str(p) for p in products if p is not None]

        anchor = rng.choice(product_list)
        positives = {pid for pid in product_list if pid != anchor}

        tasks.append(
            EvalTask(
                orderid=orderid,
                userid=userid,
                anchor=anchor,
                positives=positives
            )
        )

    return tasks


def infer_bundles(tasks: list[EvalTask]) -> list[list[str]]:
    """Calls the live bundle recommendation service for all evaluation tasks.

    Builds a single batched DataFrame of ``(anchor_pid, userid)`` pairs and
    passes it to ``getMultiRec``.  The returned DataFrame must contain exactly
    one row per task; a ``RuntimeError`` is raised otherwise.

    Args:
        tasks: List of :class:`EvalTask` instances whose ``anchor`` and
            ``userid`` fields are forwarded to ``getMultiRec``.

    Returns:
        A list of recommendation lists, one per task, in the same order as
        ``tasks``.  Each inner list contains string product IDs ordered by
        relevance (most relevant first).

    Raises:
        RuntimeError: If the number of result rows returned by ``getMultiRec``
            does not match the number of tasks.
    """
    anchors_df = pl.DataFrame(
        {
            "anchor_pid": [t.anchor for t in tasks],
            "userid": [t.userid for t in tasks]
        }
    )

    bundle_df = getMultiRec(anchors_df)
    if bundle_df.height != len(tasks):
        raise RuntimeError(
            f"Anzahl Bundle-Ergebnisse passt nicht: expected={len(tasks)} got={bundle_df.height}"
        )

    recs_col = bundle_df.get_column("recs").to_list()
    return [[str(pid) for pid in (recs or [])] for recs in recs_col]


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
    tasks: list[EvalTask],
    recs_per_task: list[list[str]],
    ks: list[int],
) -> tuple[dict[int, MetricAgg], dict[str, int]]:
    """Scores the bundle recommendations against the ground-truth positives.

    Iterates over every (task, recommendation list) pair and computes
    HitRate, Recall, MRR and Precision at each cutoff K.  Metric values are
    accumulated as raw sums in :class:`MetricAgg` objects; dividing by
    ``used_anchors`` yields the final mean values.

    Args:
        tasks: Evaluation tasks as produced by :func:`build_eval_tasks`.
        recs_per_task: Recommendation lists in the same order as ``tasks``,
            as returned by :func:`infer_bundles`.
        ks: Cutoff values at which metrics are computed (e.g. ``[5, 10, 20]``).

    Returns:
        A tuple ``(metrics, coverage)`` where:

        - ``metrics`` maps each K to a :class:`MetricAgg` holding the
          accumulated sums.
        - ``coverage`` is a dict with keys ``total_anchors``,
          ``used_anchors``, and ``empty_recommendations``.
    """
    metrics = {k: MetricAgg() for k in ks}
    coverage = {
        "total_anchors": len(tasks),
        "used_anchors": 0,
        "empty_recommendations": 0,
    }

    for task, recs in zip(tasks, recs_per_task):
        coverage["used_anchors"] += 1
        if not recs:
            coverage["empty_recommendations"] += 1

        for k in ks:
            pred = recs[:k]
            hits = sum(1 for pid in pred if pid in task.positives)
            rr_rank = first_relevant_rank(pred, task.positives)

            metrics[k].hit_sum += 1.0 if hits > 0 else 0.0
            metrics[k].recall_sum += hits / len(task.positives)
            metrics[k].mrr_sum += 0.0 if rr_rank is None else (1.0 / rr_rank)
            metrics[k].precision_sum += hits / k
            metrics[k].positives_sum += hits

    return metrics, coverage


def print_report(
    metrics: dict[int, MetricAgg],
    coverage: dict[str, int],
    recs_per_task: list[list[str]],
) -> None:
    """Prints a human-readable evaluation report to stdout.

    Outputs two sections:

    1. **Configuration & coverage** – global settings, anchor counts, empty
       recommendation rate, and the largest bundle size observed.
    2. **Metrics table** – mean HitRate, Recall, MRR, Precision and Positives
       at each cutoff K, averaged over ``used_anchors``.

    A warning is printed when ``getMultiRec`` returns fewer items than the
    largest K, since metrics for those cutoffs are based on truncated lists.

    Args:
        metrics: Accumulated metric sums per K as returned by :func:`evaluate`.
        coverage: Coverage counters as returned by :func:`evaluate`.
        recs_per_task: Raw recommendation lists, used to determine the maximum
            bundle size actually returned by the serving stack.
    """
    denom = max(1, coverage["used_anchors"])
    max_recs = max((len(r) for r in recs_per_task), default=0)

    print("Model Test")
    print(f"test_data_path         : {TEST_DATA_PATH}")
    print(f"random_seed            : {RANDOM_SEED}")
    print(f"ks                     : {KS}")
    print(f"total_anchors          : {coverage['total_anchors']}")
    print(f"used_anchors           : {coverage['used_anchors']}")
    print(f"empty_recommendations  : {coverage['empty_recommendations']}")
    print(f"max_bundle_size_found  : {max_recs}")

    if max_recs < max(KS):
        print(
            f"WARNING: getMultiRecs liefert aktuell max {max_recs} Empfehlungen; "
            f"Metriken fuer k>{max_recs} basieren auf kuerzeren Listen."
        )

    print("\nMetrics (mean over used anchors)")
    header = f"{'K':>4} {'HitRate':>10} {'Recall':>10} {'MRR':>10} {'Precision':>10} {'Positives':>10}"
    print(header)
    print("-" * len(header))
    for k in sorted(metrics):
        m = metrics[k]
        print(
            f"{k:>4} "
            f"{(m.hit_sum / denom):>10.6f} "
            f"{(m.recall_sum / denom):>10.6f} "
            f"{(m.mrr_sum / denom):>10.6f} "
            f"{(m.precision_sum / denom):>10.6f} "
            f"{(m.positives_sum / denom):>10.6f}"
        )


def main() -> None:
    """Entry point for the end-to-end model evaluation testbench.

    Orchestrates the full evaluation pipeline:

    1. Loads order data from the configured Parquet file.
    2. Builds one :class:`EvalTask` per qualifying order (random anchor selection).
    3. Calls ``getMultiRec`` in batch to obtain bundle recommendations.
    4. Scores the recommendations and prints the coverage and metrics report.

    Raises:
        RuntimeError: If no valid evaluation tasks could be built from the
            test data, or if the number of recommendation rows does not match
            the number of tasks.
    """
    orders = read_orders(TEST_DATA_PATH)
    tasks = build_eval_tasks(
        orders,
        seed=RANDOM_SEED,
        min_basket_size=MIN_BASKET_SIZE,
        max_orders=MAX_ORDERS,
    )

    if not tasks:
        raise RuntimeError("Keine gueltigen Anchor-Tasks aus den Testdaten erstellt.")

    recs_per_task = infer_bundles(tasks)
    metrics, coverage = evaluate(tasks, recs_per_task, KS)
    print_report(metrics, coverage, recs_per_task)


if __name__ == "__main__":
    main()
