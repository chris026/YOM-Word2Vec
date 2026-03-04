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
POP_ORIGIN_PATH = "artifacts/pop_origin.parquet"

KS = [5, 10, 20]
TOPK_RETRIEVAL = 100
TOPN = 50
MIN_BASKET_SIZE = 2
EVAL_MAX_ORDERS = 0  # 0 = all orders
NUM_ANCHORS = 0  # 0 = all anchors
SEED = 42
SHOW_PROGRESS = True


def validate_config() -> None:
    if not KS or any(k <= 0 for k in KS):
        raise ValueError("KS must contain positive integers, e.g. [5, 10, 20]")

    if TOPK_RETRIEVAL < max(KS):
        raise ValueError(f"TOPK_RETRIEVAL ({TOPK_RETRIEVAL}) must be >= max(KS) ({max(KS)})")

    if TOPN < max(KS):
        raise ValueError(f"TOPN ({TOPN}) must be >= max(KS) ({max(KS)})")


def read_orders(path: str) -> pl.DataFrame:
    lower = path.lower()
    if lower.endswith(".parquet"):
        df = pl.read_parquet(path, columns=["orderid", "productid", "userid", "origin"])
    elif lower.endswith(".csv"):
        df = pl.read_csv(
            path,
            columns=["orderid", "productid", "userid", "origin"],
            schema_overrides={
                "orderid": pl.Utf8,
                "productid": pl.Utf8,
                "userid": pl.Utf8,
                "origin": pl.Utf8,
            },
        )
    else:
        raise ValueError(f"Unsupported file format: {path}. Use .csv or .parquet")

    return (
        df.drop_nulls(["orderid", "productid", "userid", "origin"])
        .with_columns(
            pl.col("orderid").cast(pl.Utf8),
            pl.col("productid").cast(pl.Utf8),
            pl.col("userid").cast(pl.Utf8),
            pl.col("origin").cast(pl.Utf8),
        )
    )


def build_anchor_tasks(
    orders: pl.DataFrame,
    min_basket_size: int,
    eval_max_orders: int,
    num_anchors: int,
    seed: int,
) -> list[tuple[str, list[str], str, str]]:
    baskets = (
        orders.group_by("orderid", maintain_order=True)
        .agg(
            pl.col("productid").alias("basket"),
            pl.first("userid"),
            pl.first("origin"),
        )
        .with_columns(pl.col("basket").list.unique())
        .filter(pl.col("basket").list.len() >= min_basket_size)
    )

    if eval_max_orders > 0:
        baskets = baskets.head(eval_max_orders)

    tasks: list[tuple[str, list[str], str, str]] = []
    for row in baskets.iter_rows(named=True):
        basket = [str(pid) for pid in row["basket"] if pid is not None]
        basket = list(dict.fromkeys(basket))
        userid = str(row["userid"])
        origin = str(row["origin"])

        for anchor in basket:
            tasks.append((anchor, basket, userid, origin))

    if num_anchors > 0 and len(tasks) > num_anchors:
        rng = random.Random(seed)
        tasks = rng.sample(tasks, k=num_anchors)

    return tasks


def dcg_at_k(rel: list[int], k: int) -> float:
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
    pop_origin,
    tasks: list[tuple[str, list[str], str, str]],
) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
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

    for anchor, basket, userid, origin in iterator:
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
            origin=origin,
            w2v=w2v,
            ranker=ranker,
            store_meta=store_meta,
            prod_cat=prod_cat,
            pop_global=pop_global,
            pop_store=pop_store,
            pop_region=pop_region,
            pop_subch=pop_subch,
            pop_origin=pop_origin,
            topk_retrieval=TOPK_RETRIEVAL,
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
    validate_config()

    print("LightGBM Ranker Testbench")
    print(f"lgbm_model_path  : {LGBM_MODEL_PATH}")
    print(f"w2v_model_path   : {W2V_MODEL_PATH}")
    print(f"eval_orders_path : {EVAL_ORDERS_PATH}")
    print(f"ks               : {KS}")
    print(f"topk_retrieval   : {TOPK_RETRIEVAL}")
    print(f"topn             : {TOPN}")

    w2v, ranker = load_models(W2V_MODEL_PATH, LGBM_MODEL_PATH)
    (
        store_meta,
        prod_cat,
        pop_global,
        pop_store,
        pop_region,
        pop_subch,
        pop_origin,
    ) = build_lookup_dicts(
        commerces_path=COMMERCES_PATH,
        products_path=PRODUCTS_PATH,
        pop_global_path=POP_GLOBAL_PATH,
        pop_store_path=POP_STORE_PATH,
        pop_region_path=POP_REGION_PATH,
        pop_subch_path=POP_SUBCH_PATH,
        pop_origin_path=POP_ORIGIN_PATH,
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
        pop_origin=pop_origin,
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
