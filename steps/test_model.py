import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from gensim.models import Word2Vec
from zenml import step


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


def _to_key(v: Any) -> str:
    return str(v)


def _safe_read_parquet(path: str, columns: list[str] | None = None) -> pl.DataFrame:
    if not os.path.exists(path):
        return pl.DataFrame()
    df = pl.read_parquet(path)
    if columns is None:
        return df
    selected = [c for c in columns if c in df.columns]
    return df.select(selected) if selected else pl.DataFrame()


def _read_lookup_dict(path: str, key_cols: list[str], val_col: str) -> dict:
    df = _safe_read_parquet(path, key_cols + [val_col])
    if df.is_empty() or val_col not in df.columns:
        return {}

    out = {}
    for row in df.iter_rows(named=True):
        if len(key_cols) == 1:
            key = _to_key(row[key_cols[0]])
        else:
            key = tuple(_to_key(row[k]) for k in key_cols)
        out[key] = int(row[val_col])
    return out


def _build_context_lookup(
    orders_path: str = "data/2024-20250001_part_00-001.parquet",
    commerces_path: str = "data/commerces.parquet",
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    order_meta = {}
    store_meta = {}

    orders = _safe_read_parquet(orders_path, ["orderid", "userid", "origin"])
    if not orders.is_empty() and {"orderid", "userid", "origin"}.issubset(set(orders.columns)):
        order_meta = {
            _to_key(r["orderid"]): {
                "userid": _to_key(r["userid"]),
                "origin": _to_key(r["origin"]),
            }
            for r in (
                orders
                .unique(subset=["orderid"], keep="first")
                .iter_rows(named=True)
            )
        }

    commerces = _safe_read_parquet(
        commerces_path,
        ["userid", "region", "subchannel", "channel", "commune"],
    )
    if not commerces.is_empty() and "userid" in commerces.columns:
        store_meta = {
            _to_key(r["userid"]): {
                "region": _to_key(r.get("region", "UNKNOWN")),
                "subchannel": _to_key(r.get("subchannel", "UNKNOWN")),
                "channel": _to_key(r.get("channel", "UNKNOWN")),
                "commune": _to_key(r.get("commune", "UNKNOWN")),
            }
            for r in commerces.iter_rows(named=True)
        }

    return order_meta, store_meta


def _build_product_lookup(products_path: str = "data/products_v2.parquet") -> tuple[dict[str, str], dict[str, bool]]:
    products = _safe_read_parquet(products_path, ["productid", "category", "blocked"])
    if products.is_empty() or "productid" not in products.columns:
        return {}, {}

    prod_cat = {_to_key(r["productid"]): _to_key(r.get("category", "UNKNOWN")) for r in products.iter_rows(named=True)}
    prod_blocked = {_to_key(r["productid"]): bool(r.get("blocked", False)) for r in products.iter_rows(named=True)}
    return prod_cat, prod_blocked


def _prepare_features_for_candidates(
    anchor: str,
    candidates: list[tuple[str, float]],
    userid: str,
    origin: str,
    store_meta: dict[str, dict[str, str]],
    prod_cat: dict[str, str],
    pop_global: dict,
    pop_store: dict,
    pop_region: dict,
    pop_subch: dict,
    pop_origin: dict,
) -> pd.DataFrame:
    ctx = store_meta.get(userid, {})
    region = _to_key(ctx.get("region", "UNKNOWN"))
    subch = _to_key(ctx.get("subchannel", "UNKNOWN"))
    channel = _to_key(ctx.get("channel", "UNKNOWN"))
    commune = _to_key(ctx.get("commune", "UNKNOWN"))

    anchor_cat = _to_key(prod_cat.get(anchor, "UNKNOWN"))

    rows = []
    for cand, sim in candidates:
        cand_cat = _to_key(prod_cat.get(cand, "UNKNOWN"))
        same_cat = 1 if cand_cat == anchor_cat else 0

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

    X = pd.DataFrame(rows, columns=FEATURE_COLS)
    for col in CATEGORICAL_COLS:
        X[col] = X[col].fillna("UNKNOWN").astype("category")
    for col in FEATURE_COLS:
        if col not in CATEGORICAL_COLS:
            X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0.0).astype(np.float32)
    return X


def _minmax(scores: np.ndarray) -> np.ndarray:
    if len(scores) == 0:
        return scores
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if hi - lo < 1e-12:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - lo) / (hi - lo)).astype(np.float32)


def _dcg_at_k(rel: list[int], k: int) -> float:
    if not rel or k <= 0:
        return 0.0
    rel_k = rel[:k]
    return float(sum((r / np.log2(i + 2) for i, r in enumerate(rel_k))))


def _metrics_for_single_ranking(
    ranked: list[str],
    positives: set[str],
    candidate_universe: set[str],
    k_eval: int,
) -> dict[str, float]:
    pred_topk = ranked[:k_eval]
    pred_set = set(pred_topk)

    tp = len(pred_set & positives)
    fp = len(pred_set - positives)
    fn = len(positives - pred_set)
    tn = max(0, len(candidate_universe) - tp - fp - fn)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    hitrate = 1.0 if tp > 0 else 0.0

    hits = 0
    ap_sum = 0.0
    rr = 0.0
    rel = []
    for i, pid in enumerate(pred_topk, start=1):
        is_rel = 1 if pid in positives else 0
        rel.append(is_rel)
        if is_rel:
            hits += 1
            ap_sum += hits / i
            if rr == 0.0:
                rr = 1.0 / i

    denom_ap = min(len(positives), k_eval) if positives else 1
    map_k = ap_sum / denom_ap if positives else 0.0
    idcg = _dcg_at_k([1] * min(len(positives), k_eval), k_eval)
    ndcg = (_dcg_at_k(rel, k_eval) / idcg) if idcg > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "hitrate": float(hitrate),
        "map_at_k": float(map_k),
        "mrr_at_k": float(rr),
        "ndcg_at_k": float(ndcg),
    }


def _aggregate_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "hitrate": 0.0,
            "map_at_k": 0.0,
            "mrr_at_k": 0.0,
            "ndcg_at_k": 0.0,
            "num_samples": 0.0,
        }

    out = {}
    keys = [k for k in rows[0].keys()]
    for key in keys:
        out[key] = float(np.mean([r[key] for r in rows]))
    out["num_samples"] = float(len(rows))
    return out


def _evaluate_models(
    test_df: pl.DataFrame,
    w2v: Word2Vec,
    ranker: lgb.Booster,
    order_meta: dict[str, dict[str, str]],
    store_meta: dict[str, dict[str, str]],
    prod_cat: dict[str, str],
    prod_blocked: dict[str, bool],
    pop_global: dict,
    pop_store: dict,
    pop_region: dict,
    pop_subch: dict,
    pop_origin: dict,
    topk_retrieval: int = 50,
    topk_eval: int = 10,
    blend_alpha: float = 0.5,
) -> dict[str, dict[str, float]]:
    w2v_metrics = []
    lgbm_metrics = []
    blend_metrics = []

    for row in test_df.iter_rows(named=True):
        basket = [_to_key(x) for x in row.get("basket", []) if x is not None]
        if len(basket) < 2:
            continue

        orderid = _to_key(row.get("orderid", "UNKNOWN"))
        ctx = order_meta.get(orderid, {})
        userid = _to_key(ctx.get("userid", "UNKNOWN"))
        origin = _to_key(ctx.get("origin", "UNKNOWN"))

        for anchor in basket:
            if anchor not in w2v.wv:
                continue

            positives = {x for x in basket if x != anchor}
            if not positives:
                continue

            retrieved = w2v.wv.most_similar(anchor, topn=topk_retrieval)
            candidates = []
            for cand, sim in retrieved:
                c = _to_key(cand)
                if c == anchor:
                    continue
                if c in prod_blocked and prod_blocked[c]:
                    continue
                candidates.append((c, float(sim)))

            if not candidates:
                continue

            candidate_universe = {c for c, _ in candidates}

            ranked_w2v = [c for c, _ in sorted(candidates, key=lambda x: x[1], reverse=True)]
            w2v_metrics.append(
                _metrics_for_single_ranking(ranked_w2v, positives, candidate_universe, topk_eval)
            )

            X = _prepare_features_for_candidates(
                anchor=anchor,
                candidates=candidates,
                userid=userid,
                origin=origin,
                store_meta=store_meta,
                prod_cat=prod_cat,
                pop_global=pop_global,
                pop_store=pop_store,
                pop_region=pop_region,
                pop_subch=pop_subch,
                pop_origin=pop_origin,
            )

            lgb_scores = ranker.predict(X)
            ranked_lgbm = [
                c for c, _ in sorted(zip([c for c, _ in candidates], lgb_scores), key=lambda x: x[1], reverse=True)
            ]
            lgbm_metrics.append(
                _metrics_for_single_ranking(ranked_lgbm, positives, candidate_universe, topk_eval)
            )

            sim_scores = np.array([s for _, s in candidates], dtype=np.float32)
            lgb_scores_arr = np.array(lgb_scores, dtype=np.float32)
            blend_scores = blend_alpha * _minmax(sim_scores) + (1.0 - blend_alpha) * _minmax(lgb_scores_arr)
            ranked_blend = [
                c for c, _ in sorted(zip([c for c, _ in candidates], blend_scores), key=lambda x: x[1], reverse=True)
            ]
            blend_metrics.append(
                _metrics_for_single_ranking(ranked_blend, positives, candidate_universe, topk_eval)
            )

    return {
        "word2vec": _aggregate_metrics(w2v_metrics),
        "lightgbm": _aggregate_metrics(lgbm_metrics),
        "combined": _aggregate_metrics(blend_metrics),
    }


def _print_report(metrics: dict[str, dict[str, float]], topk_eval: int, topk_retrieval: int, blend_alpha: float) -> None:
    print("\n=== MODEL EVALUATION REPORT ===")
    print(f"retrieval@{topk_retrieval}, eval@{topk_eval}, blend_alpha={blend_alpha:.2f}")
    for model_name, vals in metrics.items():
        print(f"\n[{model_name}]")
        print(f"  samples     : {int(vals['num_samples'])}")
        print(f"  accuracy    : {vals['accuracy']:.4f}")
        print(f"  precision   : {vals['precision']:.4f}")
        print(f"  recall      : {vals['recall']:.4f}")
        print(f"  f1          : {vals['f1']:.4f}")
        print(f"  hitrate@{topk_eval} : {vals['hitrate']:.4f}")
        print(f"  map@{topk_eval}     : {vals['map_at_k']:.4f}")
        print(f"  mrr@{topk_eval}     : {vals['mrr_at_k']:.4f}")
        print(f"  ndcg@{topk_eval}    : {vals['ndcg_at_k']:.4f}")


@step(enable_cache=False)
def test_model(test_df_path: str, W2Vmodel_path: str, LGM_model_path: str):
    test_df = pl.read_parquet(test_df_path)
    w2v = Word2Vec.load(W2Vmodel_path)
    ranker = lgb.Booster(model_file=LGM_model_path)

    order_meta, store_meta = _build_context_lookup()
    prod_cat, prod_blocked = _build_product_lookup()

    pop_global = _read_lookup_dict("artifacts/pop_global.parquet", ["productid"], "pop_global")
    pop_store = _read_lookup_dict("artifacts/pop_store.parquet", ["userid", "productid"], "pop_store")
    pop_region = _read_lookup_dict("artifacts/pop_region.parquet", ["region", "productid"], "pop_region")
    pop_subch = _read_lookup_dict("artifacts/pop_subch.parquet", ["subchannel", "productid"], "pop_subch")
    pop_origin = _read_lookup_dict("artifacts/pop_origin.parquet", ["origin", "productid"], "pop_origin")

    topk_retrieval = 50
    topk_eval = 10
    blend_alpha = 0.5

    metrics = _evaluate_models(
        test_df=test_df,
        w2v=w2v,
        ranker=ranker,
        order_meta=order_meta,
        store_meta=store_meta,
        prod_cat=prod_cat,
        prod_blocked=prod_blocked,
        pop_global=pop_global,
        pop_store=pop_store,
        pop_region=pop_region,
        pop_subch=pop_subch,
        pop_origin=pop_origin,
        topk_retrieval=topk_retrieval,
        topk_eval=topk_eval,
        blend_alpha=blend_alpha,
    )

    _print_report(metrics, topk_eval=topk_eval, topk_retrieval=topk_retrieval, blend_alpha=blend_alpha)
    return metrics
