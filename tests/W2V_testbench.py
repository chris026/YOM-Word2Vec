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
EVAL_MAX_ORDERS = 10000  # 0 = all orders
NUM_ANCHORS = 0  # 0 = all anchors
SEED = 42
SHOW_PROGRESS = True


@dataclass
class EvalCounts:
    hit_sum: float = 0.0
    recall_sum: float = 0.0
    mrr_sum: float = 0.0
    precision_sum: float = 0.0
    positives_sum: float = 0.0


def validate_config() -> None:
    if not KS or any(k <= 0 for k in KS):
        raise ValueError("KS must contain positive integers, e.g. [5, 10, 20, 50]")

    if RETRIEVAL_TOPK < max(KS):
        raise ValueError(f"RETRIEVAL_TOPK ({RETRIEVAL_TOPK}) must be >= max(KS) ({max(KS)})")


def read_orders(path: str) -> pl.DataFrame:
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
    return list(dict.fromkeys(items))


def build_anchor_tasks(
    orders: pl.DataFrame,
    min_basket_size: int,
    eval_max_orders: int,
) -> list[tuple[str, list[str]]]:
    baskets = (
        orders.group_by("orderid", maintain_order=True)
        .agg(pl.col("productid"))
        .with_columns(pl.col("productid").list.unique(maintain_order=True))
        .filter(pl.col("productid").list.len() >= min_basket_size)
    )

    if eval_max_orders > 0:
        baskets = baskets.head(eval_max_orders)

    tasks: list[tuple[str, list[str]]] = []
    for row in baskets.iter_rows(named=True):
        basket = unique_preserve_order([str(x) for x in row["productid"]])
        if len(basket) < min_basket_size:
            continue
        for anchor in basket:
            tasks.append((anchor, basket))
    return tasks


def first_relevant_rank(pred: list[str], positives: set[str]) -> int | None:
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

        positives = {p for p in basket if p != anchor and p in model.wv}
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


def to_report_frame(results: dict[int, EvalCounts], used_anchors: int) -> pd.DataFrame:
    rows = []
    denom = max(1, used_anchors)

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

    report_df = to_report_frame(results, used_anchors=coverage["used_anchors"])

    print("\nCoverage")
    print(f"total_anchors       : {coverage['total_anchors']}")
    print(f"used_anchors        : {coverage['used_anchors']}")
    print(f"oov_anchors         : {coverage['oov_anchors']}")
    print(f"no_positive_anchors : {coverage['no_positive_anchors']}")
    print(f"no_candidate_anchors: {coverage['no_candidate_anchors']}")

    print("\nMetrics (mean over used anchors)")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 120)
    print(report_df.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
