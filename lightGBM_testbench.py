import argparse
import random
import time
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from gensim.models import Word2Vec


FEATURE_COLS = [
    "sim_item2vec",
    "pop_global",
    "pop_subch",
    "pop_origin",
    "pop_region",
    "same_category",
    "channel",
    "pop_store",
    "commune",
    "cand_category",
    "origin",
    "region",
    "subchannel",
]

CATEGORICAL_COLS = [
    "channel",
    "commune",
    "cand_category",
    "origin",
    "region",
    "subchannel",
]


def summarize(values: np.ndarray) -> dict:
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)) if values.size else 0.0,
        "median": float(np.median(values)) if values.size else 0.0,
        "std": float(np.std(values)) if values.size else 0.0,
        "min": float(np.min(values)) if values.size else 0.0,
        "p05": float(np.percentile(values, 5)) if values.size else 0.0,
        "p95": float(np.percentile(values, 95)) if values.size else 0.0,
        "max": float(np.max(values)) if values.size else 0.0,
    }


def print_summary(title: str, stats: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"{k:>24}: {v:.6f}")
        else:
            print(f"{k:>24}: {v}")


def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_orders(path: str) -> pd.DataFrame:
    df = read_table(path)
    needed = {"orderid", "productid", "userid", "origin"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"orders file missing columns: {sorted(missing)}")
    return df[["orderid", "productid", "userid", "origin"]].copy()


def load_models(w2v_path: str, lgbm_path: str) -> tuple[Word2Vec, lgb.Booster]:
    w2v = Word2Vec.load(w2v_path)
    ranker = lgb.Booster(model_file=lgbm_path)
    return w2v, ranker


def to_key(v) -> str:
    return str(v)


def _to_feature_frame(rows: list[dict]) -> pd.DataFrame:
    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    for col in CATEGORICAL_COLS:
        X[col] = X[col].fillna("UNKNOWN").astype("category")
    for col in FEATURE_COLS:
        if col not in CATEGORICAL_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(np.float32)
    return X


def build_lookup_dicts(
    commerces_path: str,
    products_path: str,
    pop_global_path: str,
    pop_store_path: str,
    pop_region_path: str,
    pop_subch_path: str,
    pop_origin_path: str,
):
    commerces = read_table(commerces_path)[["userid", "region", "subchannel", "channel", "commune"]]
    products = read_table(products_path)[["productid", "category", "blocked"]]

    store_meta = {
        to_key(r.userid): {
            "region": r.region,
            "subchannel": r.subchannel,
            "channel": r.channel,
            "commune": r.commune,
        }
        for r in commerces.itertuples(index=False)
    }

    prod_cat = {to_key(r.productid): r.category for r in products.itertuples(index=False)}
    prod_blocked = {to_key(r.productid): bool(r.blocked) for r in products.itertuples(index=False)}

    pop_global_df = read_table(pop_global_path)
    pop_store_df = read_table(pop_store_path)
    pop_region_df = read_table(pop_region_path)
    pop_subch_df = read_table(pop_subch_path)
    pop_origin_df = read_table(pop_origin_path)

    pop_global = {to_key(r.productid): int(r.pop_global) for r in pop_global_df.itertuples(index=False)}
    pop_store = {(to_key(r.userid), to_key(r.productid)): int(r.pop_store) for r in pop_store_df.itertuples(index=False)}
    pop_region = {(to_key(r.region), to_key(r.productid)): int(r.pop_region) for r in pop_region_df.itertuples(index=False)}
    pop_subch = {(to_key(r.subchannel), to_key(r.productid)): int(r.pop_subch) for r in pop_subch_df.itertuples(index=False)}
    pop_origin = {(to_key(r.origin), to_key(r.productid)): int(r.pop_origin) for r in pop_origin_df.itertuples(index=False)}

    return store_meta, prod_cat, prod_blocked, pop_global, pop_store, pop_region, pop_subch, pop_origin


def model_integrity_test(ranker: lgb.Booster) -> dict:
    num_feature = int(ranker.num_feature())
    num_trees = int(ranker.num_trees())
    feature_names = ranker.feature_name()

    return {
        "num_features": num_feature,
        "expected_features": len(FEATURE_COLS),
        "num_trees": num_trees,
        "feature_names": ", ".join(feature_names) if feature_names else "<none>",
        "feature_count_ok": num_feature == len(FEATURE_COLS),
    }


def prediction_sanity_test(ranker: lgb.Booster, seed: int, num_rows: int = 5000) -> dict:
    rng = np.random.default_rng(seed)

    cat_values = {
        "channel": ["TRAD", "MOD", "UNKNOWN"],
        "commune": ["SCL", "VALP", "UNKNOWN"],
        "cand_category": ["BEBIDAS", "ABARROTES", "UNKNOWN"],
        "origin": ["RUTA", "WEB", "UNKNOWN"],
        "region": ["RM", "V", "UNKNOWN"],
        "subchannel": ["MINIMARKET", "MAYORISTA", "UNKNOWN"],
    }

    rows = []
    for _ in range(num_rows):
        rows.append(
            {
                "sim_item2vec": float(rng.uniform(-0.2, 1.0)),
                "pop_global": float(rng.uniform(0.0, 8.0)),
                "pop_subch": float(rng.uniform(0.0, 8.0)),
                "pop_origin": float(rng.uniform(0.0, 8.0)),
                "pop_region": float(rng.uniform(0.0, 8.0)),
                "same_category": int(rng.integers(0, 2)),
                "channel": rng.choice(cat_values["channel"]),
                "pop_store": float(rng.uniform(0.0, 8.0)),
                "commune": rng.choice(cat_values["commune"]),
                "cand_category": rng.choice(cat_values["cand_category"]),
                "origin": rng.choice(cat_values["origin"]),
                "region": rng.choice(cat_values["region"]),
                "subchannel": rng.choice(cat_values["subchannel"]),
            }
        )

    X = _to_feature_frame(rows)
    scores = np.asarray(ranker.predict(X), dtype=np.float64)

    finite_ratio = float(np.mean(np.isfinite(scores))) if scores.size else 0.0
    stats = summarize(scores)
    stats.update(
        {
            "finite_ratio": finite_ratio,
            "score_variance": float(np.var(scores)) if scores.size else 0.0,
        }
    )
    return stats


def _counterfactual_rows(base_rows: np.ndarray, idx: int, low: float, high: float) -> tuple[np.ndarray, np.ndarray]:
    lo = base_rows.copy()
    hi = base_rows.copy()
    lo[:, idx] = low
    hi[:, idx] = high
    return lo, hi


def counterfactual_sensitivity_test(ranker: lgb.Booster, seed: int, num_rows: int = 2000) -> dict:
    rng = np.random.default_rng(seed)

    base_rows = []
    for _ in range(num_rows):
        base_rows.append(
            {
                "sim_item2vec": float(rng.uniform(0.0, 0.8)),
                "pop_global": float(rng.uniform(0.0, 6.0)),
                "pop_subch": float(rng.uniform(0.0, 6.0)),
                "pop_origin": float(rng.uniform(0.0, 6.0)),
                "pop_region": float(rng.uniform(0.0, 6.0)),
                "same_category": int(rng.integers(0, 2)),
                "channel": "TRAD",
                "pop_store": float(rng.uniform(0.0, 6.0)),
                "commune": "SCL",
                "cand_category": "ABARROTES",
                "origin": "RUTA",
                "region": "RM",
                "subchannel": "MINIMARKET",
            }
        )

    sim_lo_rows = [dict(r, sim_item2vec=0.1) for r in base_rows]
    sim_hi_rows = [dict(r, sim_item2vec=0.9) for r in base_rows]
    cat_lo_rows = [dict(r, same_category=0) for r in base_rows]
    cat_hi_rows = [dict(r, same_category=1) for r in base_rows]
    pop_lo_rows = [
        dict(r, pop_global=0.0, pop_subch=0.0, pop_origin=0.0, pop_region=0.0, pop_store=0.0)
        for r in base_rows
    ]
    pop_hi_rows = [
        dict(r, pop_global=6.0, pop_subch=6.0, pop_origin=6.0, pop_region=6.0, pop_store=6.0)
        for r in base_rows
    ]

    s_sim_lo = np.asarray(ranker.predict(_to_feature_frame(sim_lo_rows)), dtype=np.float64)
    s_sim_hi = np.asarray(ranker.predict(_to_feature_frame(sim_hi_rows)), dtype=np.float64)
    s_cat_lo = np.asarray(ranker.predict(_to_feature_frame(cat_lo_rows)), dtype=np.float64)
    s_cat_hi = np.asarray(ranker.predict(_to_feature_frame(cat_hi_rows)), dtype=np.float64)
    s_pop_lo = np.asarray(ranker.predict(_to_feature_frame(pop_lo_rows)), dtype=np.float64)
    s_pop_hi = np.asarray(ranker.predict(_to_feature_frame(pop_hi_rows)), dtype=np.float64)

    return {
        "sim_high_gt_low_rate": float(np.mean(s_sim_hi > s_sim_lo)),
        "sim_avg_delta": float(np.mean(s_sim_hi - s_sim_lo)),
        "samecat_1_gt_0_rate": float(np.mean(s_cat_hi > s_cat_lo)),
        "samecat_avg_delta": float(np.mean(s_cat_hi - s_cat_lo)),
        "pop_high_gt_low_rate": float(np.mean(s_pop_hi > s_pop_lo)),
        "pop_avg_delta": float(np.mean(s_pop_hi - s_pop_lo)),
    }


def build_eval_baskets(orders: pd.DataFrame, max_orders: int | None = None) -> list[dict]:
    grouped = (
        orders.groupby("orderid", sort=False)
        .agg(
            basket=("productid", list),
            userid=("userid", "first"),
            origin=("origin", "first"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["basket"].apply(len) >= 2]

    if max_orders is not None and len(grouped) > max_orders:
        grouped = grouped.head(max_orders)

    return grouped.to_dict("records")


def score_candidates(
    anchor: str,
    userid: str,
    origin: str,
    basket: set[str],
    w2v: Word2Vec,
    ranker: lgb.Booster,
    store_meta: dict,
    prod_cat: dict,
    prod_blocked: dict,
    pop_global: dict,
    pop_store: dict,
    pop_region: dict,
    pop_subch: dict,
    pop_origin: dict,
    topk_retrieval: int,
    exclude_basket: bool,
) -> list[tuple[str, float]]:
    if anchor not in w2v.wv:
        return []

    ctx = store_meta.get(userid, {})
    region = to_key(ctx.get("region", "UNKNOWN"))
    subch = to_key(ctx.get("subchannel", "UNKNOWN"))

    retrieved = w2v.wv.most_similar(anchor, topn=topk_retrieval)

    candidates = []
    for cand, sim in retrieved:
        cand = to_key(cand)
        if cand == anchor:
            continue
        if exclude_basket and cand in basket:
            continue
        if prod_blocked.get(cand, False):
            continue
        candidates.append((cand, float(sim)))

    if not candidates:
        return []

    anchor_cat = to_key(prod_cat.get(anchor, "UNKNOWN"))
    rows = []
    ids = []
    for cand, sim in candidates:
        cand_cat = to_key(prod_cat.get(cand, "UNKNOWN"))
        same_cat = 1 if cand_cat == anchor_cat else 0

        channel = to_key(ctx.get("channel", "UNKNOWN"))
        commune = to_key(ctx.get("commune", "UNKNOWN"))
        pg = np.log1p(pop_global.get(cand, 0))
        ps = np.log1p(pop_store.get((userid, cand), 0))
        pr = np.log1p(pop_region.get((region, cand), 0))
        psub = np.log1p(pop_subch.get((subch, cand), 0))
        po = np.log1p(pop_origin.get((origin, cand), 0))

        rows.append(
            {
                "sim_item2vec": sim,
                "pop_global": pg,
                "pop_subch": psub,
                "pop_origin": po,
                "pop_region": pr,
                "same_category": same_cat,
                "channel": channel,
                "pop_store": ps,
                "commune": commune,
                "cand_category": cand_cat,
                "origin": origin,
                "region": region,
                "subchannel": subch,
            }
        )
        ids.append(cand)

    X = _to_feature_frame(rows)
    scores = np.asarray(ranker.predict(X), dtype=np.float64)

    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)
    return ranked


def offline_ranking_eval(
    orders: pd.DataFrame,
    w2v: Word2Vec,
    ranker: lgb.Booster,
    store_meta: dict,
    prod_cat: dict,
    prod_blocked: dict,
    pop_global: dict,
    pop_store: dict,
    pop_region: dict,
    pop_subch: dict,
    pop_origin: dict,
    topk_retrieval: int,
    eval_k: int,
    max_orders: int,
) -> dict:
    baskets = build_eval_baskets(orders, max_orders=max_orders)

    total_anchors = 0
    used_anchors = 0
    oov_anchors = 0
    empty_candidate_anchors = 0

    hit = 0
    rr_sum = 0.0

    for row in baskets:
        basket = [to_key(x) for x in row["basket"]]
        basket_set = set(basket)
        userid = to_key(row["userid"])
        origin = to_key(row["origin"])

        for anchor in basket:
            total_anchors += 1

            if anchor not in w2v.wv:
                oov_anchors += 1
                continue

            positives = basket_set - {anchor}
            positives = {p for p in positives if p in w2v.wv}
            if not positives:
                continue

            ranked = score_candidates(
                anchor=anchor,
                userid=userid,
                origin=origin,
                basket=basket_set,
                w2v=w2v,
                ranker=ranker,
                store_meta=store_meta,
                prod_cat=prod_cat,
                prod_blocked=prod_blocked,
                pop_global=pop_global,
                pop_store=pop_store,
                pop_region=pop_region,
                pop_subch=pop_subch,
                pop_origin=pop_origin,
                topk_retrieval=topk_retrieval,
                exclude_basket=False,
            )

            if not ranked:
                empty_candidate_anchors += 1
                continue

            used_anchors += 1
            top_ids = [pid for pid, _ in ranked[:eval_k]]

            if any(pid in positives for pid in top_ids):
                hit += 1

            rr = 0.0
            for i, pid in enumerate(top_ids, start=1):
                if pid in positives:
                    rr = 1.0 / i
                    break
            rr_sum += rr

    hit_rate = (hit / used_anchors) if used_anchors else 0.0
    mrr = (rr_sum / used_anchors) if used_anchors else 0.0
    oov_rate = (oov_anchors / total_anchors) if total_anchors else 0.0

    return {
        "num_baskets": len(baskets),
        "total_anchors": total_anchors,
        "used_anchors": used_anchors,
        "oov_anchors": oov_anchors,
        "oov_rate": oov_rate,
        "empty_candidate_anchors": empty_candidate_anchors,
        "hit_rate_at_k": hit_rate,
        "mrr_at_k": mrr,
        "k": eval_k,
        "retrieval_topk": topk_retrieval,
    }


def quick_warnings(
    integrity: dict,
    sanity: dict,
    sensitivity: dict,
    offline: dict,
) -> Iterable[str]:
    warnings = []

    if not integrity.get("feature_count_ok", False):
        warnings.append("Feature count mismatch between trained ranker and expected serving features.")

    if sanity.get("finite_ratio", 0.0) < 1.0:
        warnings.append("Ranker produced non-finite predictions.")

    if sanity.get("score_variance", 0.0) < 1e-8:
        warnings.append("Prediction variance near zero: model may be degenerate.")

    if sensitivity.get("sim_high_gt_low_rate", 0.0) < 0.55:
        warnings.append("Low sensitivity to sim_item2vec in counterfactual test.")

    if offline.get("used_anchors", 0) == 0:
        warnings.append("No usable anchors in offline eval; check IDs and vocab alignment.")

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="LightGBM Ranker Testbench")
    parser.add_argument("--model-path", default="models/lgbm_ranker.txt", help="Path to LightGBM ranker model")
    parser.add_argument("--w2v-path", default="models/word2vec.model", help="Path to Word2Vec model")

    parser.add_argument("--orders-path", default="data/2024-20250001_part_00-001.parquet", help="Orders file (.parquet or .csv)")
    parser.add_argument("--commerces-path", default="data/commerces.parquet", help="Commerces parquet")
    parser.add_argument("--products-path", default="data/products_v2.parquet", help="Products parquet")

    parser.add_argument("--pop-global-path", default="artifacts/pop_global.parquet", help="Global popularity parquet")
    parser.add_argument("--pop-store-path", default="artifacts/pop_store.parquet", help="Store popularity parquet")
    parser.add_argument("--pop-region-path", default="artifacts/pop_region.parquet", help="Region popularity parquet")
    parser.add_argument("--pop-subch-path", default="artifacts/pop_subch.parquet", help="Subchannel popularity parquet")
    parser.add_argument("--pop-origin-path", default="artifacts/pop_origin.parquet", help="Origin popularity parquet")

    parser.add_argument("--topk-retrieval", type=int, default=50, help="Top-K from Word2Vec before ranking")
    parser.add_argument("--eval-k", type=int, default=10, help="K for HitRate/MRR")
    parser.add_argument("--eval-max-orders", type=int, default=10000, help="Max number of orders for offline eval")
    parser.add_argument("--sanity-rows", type=int, default=5000, help="Rows for random sanity prediction test")
    parser.add_argument("--cf-rows", type=int, default=2000, help="Rows for counterfactual test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    t0 = time.time()
    w2v, ranker = load_models(args.w2v_path, args.model_path)
    load_sec = time.time() - t0

    orders = load_orders(args.orders_path)
    (
        store_meta,
        prod_cat,
        prod_blocked,
        pop_global,
        pop_store,
        pop_region,
        pop_subch,
        pop_origin,
    ) = build_lookup_dicts(
        commerces_path=args.commerces_path,
        products_path=args.products_path,
        pop_global_path=args.pop_global_path,
        pop_store_path=args.pop_store_path,
        pop_region_path=args.pop_region_path,
        pop_subch_path=args.pop_subch_path,
        pop_origin_path=args.pop_origin_path,
    )

    print("LightGBM Ranker Testbench")
    print(f"model_path               : {args.model_path}")
    print(f"w2v_path                 : {args.w2v_path}")
    print(f"orders_path              : {args.orders_path}")
    print(f"load_time_sec            : {load_sec:.3f}")

    integrity = model_integrity_test(ranker)
    print_summary("Model Integrity", integrity)

    sanity = prediction_sanity_test(ranker=ranker, seed=args.seed, num_rows=args.sanity_rows)
    print_summary("Prediction Sanity", sanity)

    sensitivity = counterfactual_sensitivity_test(ranker=ranker, seed=args.seed, num_rows=args.cf_rows)
    print_summary("Counterfactual Sensitivity", sensitivity)

    offline = offline_ranking_eval(
        orders=orders,
        w2v=w2v,
        ranker=ranker,
        store_meta=store_meta,
        prod_cat=prod_cat,
        prod_blocked=prod_blocked,
        pop_global=pop_global,
        pop_store=pop_store,
        pop_region=pop_region,
        pop_subch=pop_subch,
        pop_origin=pop_origin,
        topk_retrieval=args.topk_retrieval,
        eval_k=args.eval_k,
        max_orders=args.eval_max_orders,
    )
    print_summary("Offline Ranking Quality", offline)

    warnings = list(quick_warnings(integrity, sanity, sensitivity, offline))
    print("\n=== Quick Warnings ===")
    if warnings:
        for w in warnings:
            print(f"- {w}")
    else:
        print("- none")


if __name__ == "__main__":
    main()
    #python lightGBM_testbench.py --eval-max-orders 100 --sanity-rows 200 --cf-rows 200 --eval-k 5 --topk-retrieval 20
