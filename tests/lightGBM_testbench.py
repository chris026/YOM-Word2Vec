"""
Offline evaluation of the full recommendation pipeline (Word2Vec + LightGBM ranker).

For each order in the test dataset every contained product is used as an anchor.
The ranker generates recommendations that are then evaluated against the remaining
products of the same order (positives). Reported metrics are Precision, Recall, F1,
HitRate, MRR, MAP, and NDCG at K = 5, 10, 20.

Configuration: adjust the constants at the top of this file, then run directly.
"""

import math
import random
import sys
import time
from pathlib import Path

import pandas as pd
import polars as pl
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from serve_bundle import build_lookup_dicts, load_models, recommend_candidates


# Global configuration (edit here, then run this file directly)
LGBM_MODEL_PATH = "models/lgbm_ranker.txt"
W2V_MODEL_PATH = "models/word2vec.model"
EVAL_ORDERS_PATH = "data/test_4weeks_short.csv"
COMMERCES_PATH = "data/commerces.parquet"
PRODUCTS_PATH = "data/products_v2.parquet"
POP_GLOBAL_PATH = "artifacts/pop_global.parquet"
POP_STORE_PATH = "artifacts/pop_store.parquet"
POP_REGION_PATH = "artifacts/pop_region.parquet"
POP_SUBCH_PATH = "artifacts/pop_subch.parquet"

KS = [5, 10, 20]
TOPN = 50
MIN_BASKET_SIZE = 2
EVAL_MAX_ORDERS = 0  # 0 = all orders
NUM_ANCHORS = 0  # 0 = all anchors
SEED = 42
SHOW_PROGRESS = True


def validate_config() -> None:
    """Verify that the global configuration constants are consistent.

    Raises:
        ValueError: If KS is empty or contains non-positive values.
        ValueError: If TOPN is smaller than the largest K, meaning the ranker
            would return fewer items than the deepest evaluation cutoff requires.
    """
    if not KS or any(k <= 0 for k in KS):
        raise ValueError("KS must contain positive integers, e.g. [5, 10, 20]")

    if TOPN < max(KS):
        raise ValueError(f"TOPN ({TOPN}) must be >= max(KS) ({max(KS)})")


def read_orders(path: str) -> pl.DataFrame:
    """Load order data from a Parquet or CSV file.

    Only the columns ``orderid``, ``productid``, and ``userid`` are read.
    Rows with null values in any of these columns are dropped, and all
    columns are cast to ``Utf8`` to ensure a consistent string type.

    Args:
        path: Path to the input file. Must end with ``.parquet`` or ``.csv``.

    Returns:
        A Polars DataFrame with columns ``orderid``, ``productid``, ``userid``.

    Raises:
        ValueError: If the file extension is neither ``.parquet`` nor ``.csv``.
    """
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pl.read_parquet(path, columns=["orderid", "productid", "userid"])
    elif lower.endswith(".csv"):
        df = pl.read_csv(
            path,
            columns=["orderid", "productid", "userid"],
            schema_overrides={
                "orderid": pl.Utf8,
                "productid": pl.Utf8,
                "userid": pl.Utf8,
            },
        )
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .csv or .parquet")

    return (
        df.drop_nulls(["orderid", "productid", "userid"])
        .with_columns(
            pl.col("orderid").cast(pl.Utf8),
            pl.col("productid").cast(pl.Utf8),
            pl.col("userid").cast(pl.Utf8),
        )
    )


def build_anchor_tasks(
    orders: pl.DataFrame,
    min_basket_size: int,
    eval_max_orders: int,
    num_anchors: int,
    seed: int,
) -> list[tuple[str, list[str], str]]:
    """Build evaluation tasks from order data using an exhaustive anchor strategy.

    Groups orders into baskets of unique products and filters out baskets that are
    smaller than ``min_basket_size``. For each basket every product is used as an
    anchor once, producing one task per (anchor, basket, user) combination. This
    means larger baskets contribute proportionally more tasks.

    Args:
        orders: DataFrame with columns ``orderid``, ``productid``, ``userid``.
        min_basket_size: Minimum number of unique products a basket must contain
            to be included. Must be at least 2 so that at least one positive
            exists for each anchor.
        eval_max_orders: Maximum number of baskets to use. ``0`` means no limit.
        num_anchors: Maximum number of tasks to sample. ``0`` means use all tasks.
            Sampling is reproducible via ``seed``.
        seed: Random seed for the optional task subsampling.

    Returns:
        A list of ``(anchor, basket, userid)`` tuples, where ``anchor`` is a
        single product ID, ``basket`` is the full list of unique product IDs in
        the order, and ``userid`` identifies the store.
    """
    baskets = (
        orders.group_by("orderid", maintain_order=True)
        .agg(
            pl.col("productid").alias("basket"),
            pl.first("userid"),
        )
        .with_columns(pl.col("basket").list.unique())
        .filter(pl.col("basket").list.len() >= min_basket_size)
    )

    if eval_max_orders > 0:
        baskets = baskets.head(eval_max_orders)

    tasks: list[tuple[str, list[str], str]] = []
    for row in baskets.iter_rows(named=True):
        basket = [str(pid) for pid in row["basket"] if pid is not None]
        basket = list(dict.fromkeys(basket))
        userid = str(row["userid"])

        for anchor in basket:
            tasks.append((anchor, basket, userid))

    if num_anchors > 0 and len(tasks) > num_anchors:
        rng = random.Random(seed)
        tasks = rng.sample(tasks, k=num_anchors)

    return tasks


def dcg_at_k(rel: list[int], k: int) -> float:
    """Compute Discounted Cumulative Gain (DCG) at cutoff K.

    Uses the standard logarithmic discount ``1 / log2(rank + 1)``.
    Only the first ``k`` entries of ``rel`` are considered.

    Args:
        rel: List of binary relevance labels (0 or 1) in ranked order.
        k: Cutoff depth. Returns ``0.0`` if ``k <= 0``.

    Returns:
        DCG score as a float.
    """
    if k <= 0:
        return 0.0
    score = 0.0
    for i, r in enumerate(rel[:k], start=1):
        score += float(r) / math.log2(i + 1)
    return score


def evaluate_ranker(
    w2v,
    ranker,
    store_meta,
    prod_cat,
    pop_global,
    pop_store,
    pop_region,
    pop_subch,
    tasks: list[tuple[str, list[str], str]],
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    """Run the ranker on all tasks and compute ranking metrics at every K in KS.

    For each task the anchor product is looked up in the Word2Vec vocabulary.
    OOV anchors are skipped. Positives are defined as the other basket products
    that are also in the W2V vocabulary — items outside the vocabulary cannot be
    returned by the ranker and are therefore excluded from the positive set.

    Metrics are accumulated per K and averaged over all evaluated anchors
    (i.e. anchors that are in-vocabulary and produced at least one recommendation).
    OOV anchors, anchors with no valid positives, and anchors that yielded no
    candidates are excluded from both numerator and denominator.

    Inference time is measured per ``recommend_candidates`` call and averaged
    over all evaluated anchors.

    Args:
        w2v: Loaded Word2Vec model.
        ranker: Loaded LightGBM Booster.
        store_meta: Store context lookup dict (userid → context).
        prod_cat: Product category lookup dict (productid → category).
        pop_global: Global popularity lookup dict (productid → count).
        pop_store: Per-store popularity lookup dict ((userid, productid) → count).
        pop_region: Per-region popularity lookup dict ((region, productid) → count).
        pop_subch: Per-subchannel popularity lookup dict ((subchannel, productid) → count).
        tasks: List of ``(anchor, basket, userid)`` tuples from :func:`build_anchor_tasks`.

    Returns:
        A tuple ``(metrics_at_k, coverage)`` where:

        - ``metrics_at_k`` maps each K to a dict of mean metric values:
          ``precision``, ``recall``, ``f1``, ``hitrate``, ``mrr``, ``map``,
          ``ndcg``, ``avg_true_positives``.
        - ``coverage`` is a flat dict with counts and rates for
          ``total_anchors``, ``evaluated_anchors``, ``oov_anchors``,
          ``no_positive_anchors``, ``no_candidate_anchors``,
          ``eval_anchor_rate``, ``oov_anchor_rate``, ``avg_inference_ms``.
    """
    sums: dict[int, dict[str, float]] = {
        k: {
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hitrate": 0.0,
            "mrr": 0.0,
            "map": 0.0,
            "ndcg": 0.0,
            "tp_avg": 0.0,
        }
        for k in KS
    }

    total_anchors = 0
    evaluated_anchors = 0
    oov_anchors = 0
    no_positive_anchors = 0
    no_candidate_anchors = 0
    inference_time_sum_sec = 0.0

    iterator = tasks
    if SHOW_PROGRESS:
        iterator = tqdm(tasks, desc="Evaluating LGBM ranker", unit="anchor")

    for anchor, basket, userid in iterator:
        total_anchors += 1

        if anchor not in w2v.wv:
            oov_anchors += 1
            continue

        positives = {pid for pid in basket if pid != anchor and pid in w2v.wv}
        if not positives:
            no_positive_anchors += 1
            continue

        t0 = time.perf_counter()
        recs = recommend_candidates(
            anchor=anchor,
            userid=userid,
            w2v=w2v,
            ranker=ranker,
            store_meta=store_meta,
            prod_cat=prod_cat,
            pop_global=pop_global,
            pop_store=pop_store,
            pop_region=pop_region,
            pop_subch=pop_subch,
            topn=TOPN,
            basket=set(),
        )
        inference_time_sum_sec += (time.perf_counter() - t0)

        if not recs:
            no_candidate_anchors += 1
            continue

        evaluated_anchors += 1
        preds = [str(pid) for pid, _ in recs]

        for k in KS:
            topk = preds[:k]
            pred_set = set(topk)

            tp = len(pred_set & positives)
            fp = len(pred_set - positives)
            fn = len(positives - pred_set)

            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
            hitrate = 1.0 if tp > 0 else 0.0

            rr = 0.0
            hits = 0
            ap_sum = 0.0
            rel: list[int] = []
            for rank, pid in enumerate(topk, start=1):
                is_rel = 1 if pid in positives else 0
                rel.append(is_rel)
                if is_rel:
                    hits += 1
                    ap_sum += hits / rank
                    if rr == 0.0:
                        rr = 1.0 / rank

            denom_ap = min(len(positives), k) if positives else 1
            map_k = ap_sum / denom_ap if positives else 0.0
            idcg = dcg_at_k([1] * min(len(positives), k), k)
            ndcg = (dcg_at_k(rel, k) / idcg) if idcg > 0 else 0.0

            sums[k]["precision"] += precision
            sums[k]["recall"] += recall
            sums[k]["f1"] += f1
            sums[k]["hitrate"] += hitrate
            sums[k]["mrr"] += rr
            sums[k]["map"] += map_k
            sums[k]["ndcg"] += ndcg
            sums[k]["tp_avg"] += float(tp)

    denominator = max(evaluated_anchors, 1)
    metrics_at_k: dict[int, dict[str, float]] = {}
    for k in KS:
        metrics_at_k[k] = {
            "precision": sums[k]["precision"] / denominator,
            "recall": sums[k]["recall"] / denominator,
            "f1": sums[k]["f1"] / denominator,
            "hitrate": sums[k]["hitrate"] / denominator,
            "mrr": sums[k]["mrr"] / denominator,
            "map": sums[k]["map"] / denominator,
            "ndcg": sums[k]["ndcg"] / denominator,
            "avg_true_positives": sums[k]["tp_avg"] / denominator,
        }

    coverage = {
        "total_anchors": float(total_anchors),
        "evaluated_anchors": float(evaluated_anchors),
        "oov_anchors": float(oov_anchors),
        "no_positive_anchors": float(no_positive_anchors),
        "no_candidate_anchors": float(no_candidate_anchors),
        "eval_anchor_rate": (evaluated_anchors / total_anchors) if total_anchors else 0.0,
        "oov_anchor_rate": (oov_anchors / total_anchors) if total_anchors else 0.0,
        "avg_inference_ms": ((inference_time_sum_sec / evaluated_anchors) * 1000.0) if evaluated_anchors else 0.0,
    }

    return metrics_at_k, coverage


def metrics_to_frame(metrics_at_k: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Convert the metrics dict returned by :func:`evaluate_ranker` into a DataFrame.

    Rows are sorted by K in ascending order. Each row contains the cutoff depth K
    and the corresponding mean metric values.

    Args:
        metrics_at_k: Dict mapping each K to a dict of metric name → float value,
            as returned by :func:`evaluate_ranker`.

    Returns:
        A pandas DataFrame with columns ``K``, ``Precision``, ``Recall``, ``F1``,
        ``HitRate``, ``MRR``, ``MAP``, ``NDCG``, ``AvgTruePositives``.
    """
    rows = []
    for k in sorted(metrics_at_k):
        m = metrics_at_k[k]
        rows.append(
            {
                "K": k,
                "Precision": m["precision"],
                "Recall": m["recall"],
                "F1": m["f1"],
                "HitRate": m["hitrate"],
                "MRR": m["mrr"],
                "MAP": m["map"],
                "NDCG": m["ndcg"],
                "AvgTruePositives": m["avg_true_positives"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Entry point: load models and data, run evaluation, print results.

    Executes the full evaluation pipeline in sequence:

    1. Validate configuration constants.
    2. Load the Word2Vec and LightGBM models from disk.
    3. Build in-memory lookup dicts for store context and popularity signals.
    4. Load the test orders and build anchor tasks.
    5. Run :func:`evaluate_ranker` over all tasks.
    6. Print coverage statistics and the metrics table to stdout.
    """
    validate_config()

    print("LightGBM Ranker Testbench")
    print(f"lgbm_model_path  : {LGBM_MODEL_PATH}")
    print(f"w2v_model_path   : {W2V_MODEL_PATH}")
    print(f"eval_orders_path : {EVAL_ORDERS_PATH}")
    print(f"ks               : {KS}")
    print(f"topn             : {TOPN}")

    w2v, ranker = load_models(W2V_MODEL_PATH, LGBM_MODEL_PATH)
    (
        store_meta,
        prod_cat,
        pop_global,
        pop_store,
        pop_region,
        pop_subch,
    ) = build_lookup_dicts(
        commerces_path=COMMERCES_PATH,
        products_path=PRODUCTS_PATH,
        pop_global_path=POP_GLOBAL_PATH,
        pop_store_path=POP_STORE_PATH,
        pop_region_path=POP_REGION_PATH,
        pop_subch_path=POP_SUBCH_PATH,
    )

    orders = read_orders(EVAL_ORDERS_PATH)
    tasks = build_anchor_tasks(
        orders=orders,
        min_basket_size=MIN_BASKET_SIZE,
        eval_max_orders=EVAL_MAX_ORDERS,
        num_anchors=NUM_ANCHORS,
        seed=SEED,
    )

    metrics_at_k, coverage = evaluate_ranker(
        w2v=w2v,
        ranker=ranker,
        store_meta=store_meta,
        prod_cat=prod_cat,
        pop_global=pop_global,
        pop_store=pop_store,
        pop_region=pop_region,
        pop_subch=pop_subch,
        tasks=tasks,
    )

    report_df = metrics_to_frame(metrics_at_k)

    print("\nCoverage")
    print(f"total_anchors      : {int(coverage['total_anchors'])}")
    print(f"evaluated_anchors  : {int(coverage['evaluated_anchors'])}")
    print(f"oov_anchors        : {int(coverage['oov_anchors'])}")
    print(f"no_positive_anchors: {int(coverage['no_positive_anchors'])}")
    print(f"no_candidate       : {int(coverage['no_candidate_anchors'])}")
    print(f"eval_anchor_rate   : {coverage['eval_anchor_rate']:.6f}")
    print(f"oov_anchor_rate    : {coverage['oov_anchor_rate']:.6f}")
    print(f"avg_inference_ms   : {coverage['avg_inference_ms']:.4f}")

    print("\nMetrics (mean over evaluated anchors)")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
