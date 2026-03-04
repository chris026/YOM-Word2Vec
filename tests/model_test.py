from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from serve_bundle import getMultiRecs


# Global configuration
TEST_DATA_PATH = ROOT_DIR / "data" / "test_df_1m.csv"
KS = [5, 10, 20, 50]
RANDOM_SEED = 42
MIN_BASKET_SIZE = 2
MAX_ORDERS = 0  # 0 = all orders


@dataclass
class EvalTask:
    orderid: str
    userid: str
    origin: str
    anchor: str
    positives: set[str]


@dataclass
class MetricAgg:
    hit_sum: float = 0.0
    recall_sum: float = 0.0
    mrr_sum: float = 0.0
    precision_sum: float = 0.0
    positives_sum: float = 0.0


def read_orders(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Testdatei nicht gefunden: {path}")

    df = pl.scan_csv(
        str(path),
        schema_overrides={
            "orderid": pl.Utf8,
            "productid": pl.Utf8,
            "userid": pl.Utf8,
            "origin": pl.Utf8,
        }
    ).select(["orderid", "productid", "userid", "origin"])

    return (
        df.drop_nulls(["orderid", "productid", "userid", "origin"])
        .with_columns(
            pl.col("orderid").cast(pl.Utf8),
            pl.col("productid").cast(pl.Utf8),
            pl.col("userid").cast(pl.Utf8),
            pl.col("origin").cast(pl.Utf8),
        )
        .head(100)
        .collect()
    )


def build_eval_tasks(
    orders: pl.DataFrame,
    seed: int,
    min_basket_size: int,
    max_orders: int,
) -> list[EvalTask]:
    baskets = (
        orders.group_by("orderid")
        .agg(
            pl.col("productid").unique().alias("products"),
            pl.col("userid").first(),
            pl.col("origin").first(),
        )
        .filter(pl.col("products").list.len() >= min_basket_size)
    )

    if max_orders > 0:
        baskets = baskets.head(max_orders)

    rng = random.Random(seed)
    tasks: list[EvalTask] = []
    for orderid, products, userid, origin in baskets.iter_rows():
        product_list = [str(p) for p in products if p is not None]

        anchor = rng.choice(product_list)
        positives = {pid for pid in product_list if pid != anchor}

        tasks.append(
            EvalTask(
                orderid=orderid,
                userid=userid,
                origin=origin,
                anchor=anchor,
                positives=positives,
            )
        )

    return tasks


def infer_bundles(tasks: list[EvalTask]) -> list[list[str]]:
    anchors_df = pl.DataFrame(
        {
            "anchor_pid": [t.anchor for t in tasks],
            "userid": [t.userid for t in tasks],
            "origin": [t.origin for t in tasks],
        }
    )

    bundle_df = getMultiRecs(anchors_df)
    if bundle_df.height != len(tasks):
        raise RuntimeError(
            f"Anzahl Bundle-Ergebnisse passt nicht: expected={len(tasks)} got={bundle_df.height}"
        )

    recs_col = bundle_df.get_column("recs").to_list()
    return [[str(pid) for pid in (recs or [])] for recs in recs_col]


def first_relevant_rank(pred: list[str], positives: set[str]) -> int | None:
    for idx, pid in enumerate(pred, start=1):
        if pid in positives:
            return idx
    return None


def evaluate(
    tasks: list[EvalTask],
    recs_per_task: list[list[str]],
    ks: list[int],
) -> tuple[dict[int, MetricAgg], dict[str, int]]:
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
