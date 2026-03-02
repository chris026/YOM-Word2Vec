import numpy as np
import pandas as pd
import polars as pl
import lightgbm as lgb
from gensim.models import Word2Vec
import time


# -----------------------
# Load artifacts
# -----------------------

FEATURE_COLS = [
    "sim_item2vec",
    "pop_global",
    "pop_subch",
    "pop_origin",
    "pop_region",
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


def _to_key(v) -> str:
    return str(v)

def load_models(w2v_path: str, lgbm_path: str):
    w2v = Word2Vec.load(w2v_path)
    ranker = lgb.Booster(model_file=lgbm_path)
    return w2v, ranker


def build_lookup_dicts(
    commerces_path: str,
    products_path: str,
    pop_global_path: str,
    pop_store_path: str,
    pop_region_path: str,
    pop_subch_path: str,
    pop_origin_path: str,
):
    commerces = pl.read_parquet(commerces_path).select(
        ["userid", "region", "subchannel", "channel", "commune"]
    )
    products = pl.read_parquet(products_path).select(
        ["productid", "category", "blocked"]
    )

    # store / product meta
    store_meta = {
        _to_key(r["userid"]): {
            "region": r["region"],
            "subchannel": r["subchannel"],
            "channel": r["channel"],
            "commune": r["commune"],
        }
        for r in commerces.iter_rows(named=True)
    }

    prod_cat = {_to_key(r["productid"]): r["category"] for r in products.iter_rows(named=True)}
    #prod_blocked = {_to_key(r["productid"]): bool(r["blocked"]) for r in products.iter_rows(named=True)}

    # popularity lookup dicts
    pop_global = {
        _to_key(r["productid"]): int(r["pop_global"])
        for r in pl.read_parquet(pop_global_path).iter_rows(named=True)
    }

    pop_store = {
        (_to_key(r["userid"]), _to_key(r["productid"])): int(r["pop_store"])
        for r in pl.read_parquet(pop_store_path).iter_rows(named=True)
    }

    pop_region = {
        (_to_key(r["region"]), _to_key(r["productid"])): int(r["pop_region"])
        for r in pl.read_parquet(pop_region_path).iter_rows(named=True)
    }

    pop_subch = {
        (_to_key(r["subchannel"]), _to_key(r["productid"])): int(r["pop_subch"])
        for r in pl.read_parquet(pop_subch_path).iter_rows(named=True)
    }

    pop_origin = {
        (_to_key(r["origin"]), _to_key(r["productid"])): int(r["pop_origin"])
        for r in pl.read_parquet(pop_origin_path).iter_rows(named=True)
    }

    #return store_meta, prod_cat, prod_blocked, pop_global, pop_store, pop_region, pop_subch, pop_origin
    return store_meta, prod_cat, pop_global, pop_store, pop_region, pop_subch, pop_origin


# -----------------------
# Inference
# -----------------------

def recommend_candidates(
    anchor: str,
    userid: str,
    origin: str,
    w2v: Word2Vec,
    ranker: lgb.Booster,
    store_meta: dict,
    prod_cat: dict,
    #prod_blocked: dict,
    pop_global: dict,
    pop_store: dict,
    pop_region: dict,
    pop_subch: dict,
    pop_origin: dict,
    topk_retrieval: int = 50,
    topn: int = 10,
    basket: set[str] | None = None,
):
    basket = {_to_key(x) for x in (basket or set())}
    anchor = _to_key(anchor)
    userid = _to_key(userid)
    origin = _to_key(origin)

    if anchor not in w2v.wv:
        return []

    ctx = store_meta.get(userid, {})
    region = _to_key(ctx.get("region", "UNKNOWN"))
    subch = _to_key(ctx.get("subchannel", "UNKNOWN"))
    channel = _to_key(ctx.get("channel", "UNKNOWN"))
    commune = _to_key(ctx.get("commune", "UNKNOWN"))

    # 1) retrieve
    retrieved = w2v.wv.most_similar(anchor, topn=topk_retrieval)

    # 2) build candidate list with rank
    candidates = []
    for rank, (cand, sim) in enumerate(retrieved, start=1):
        cand = _to_key(cand)
        if cand == anchor:
            continue
        if cand in basket:
            continue
        #if prod_blocked.get(cand, False):
        #    continue
        if cand not in w2v.wv:
            continue

        candidates.append((cand, rank, float(sim)))

    if not candidates:
        return []

    # 3) feature matrix
    anchor_cat = _to_key(prod_cat.get(anchor, "UNKNOWN"))

    rows = []
    for cand, r_rank, w2v_sim in candidates:
        cand_cat = _to_key(prod_cat.get(cand, "UNKNOWN"))

        pg = pop_global.get(cand, 0)
        ps = pop_store.get((userid, cand), 0)
        pr = pop_region.get((region, cand), 0)
        psub = pop_subch.get((subch, cand), 0)
        po = pop_origin.get((origin, cand), 0)

        pg = np.log1p(pg)
        ps = np.log1p(ps)
        pr = np.log1p(pr)
        psub = np.log1p(psub)
        po = np.log1p(po)

        rows.append(
            {
                "sim_item2vec": w2v_sim,
                "pop_global": pg,
                "pop_subch": psub,
                "pop_origin": po,
                "pop_region": pr,
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

    # 4) predict scores
    scores = ranker.predict(X)

    # 5) sort
    ranked = sorted(
        zip([c[0] for c in candidates], scores),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:topn]


# Example usage
if __name__ == "__main__":
    w2v_path = "models/word2vec.model"
    lgbm_path = "models/lgbm_ranker.txt"
    anchor_pid = "000480-013"
    userid = "b0a75e15a8fe900abbcbe66d11494954"
    origin = "ZHH1"

    w2v, ranker = load_models(w2v_path, lgbm_path)

    #store_meta, prod_cat, prod_blocked, pop_global, pop_store, pop_region, pop_subch, pop_origin = build_lookup_dicts(
    store_meta, prod_cat, pop_global, pop_store, pop_region, pop_subch, pop_origin = build_lookup_dicts(
        commerces_path="data/commerces.parquet",
        products_path="data/products_v2.parquet",
        pop_global_path="artifacts/pop_global.parquet",
        pop_store_path="artifacts/pop_store.parquet",
        pop_region_path="artifacts/pop_region.parquet",
        pop_subch_path="artifacts/pop_subch.parquet",
        pop_origin_path="artifacts/pop_origin.parquet",
    )
    prod_name = {
        _to_key(r["productid"]): _to_key(r["name"])
        for r in pl.read_parquet("data/products_v2.parquet").select(["productid", "name"]).iter_rows(named=True)
    }

    all_Products = ["000120-001", "000295-003", "000295-008", "000120-001", "000295-003", "000295-008", "000120-001", "000295-003", "000295-008", "000120-001"]

    time_start = time.time()

    for i in all_Products:
        anchor_pid = i
        recs = recommend_candidates(
        anchor=anchor_pid,
        userid=userid,
        origin=origin,
        w2v=w2v,
        ranker=ranker,
        store_meta=store_meta,
        prod_cat=prod_cat,
        #prod_blocked=prod_blocked,
        pop_global=pop_global,
        pop_store=pop_store,
        pop_region=pop_region,
        pop_subch=pop_subch,
        pop_origin=pop_origin,
        topk_retrieval=50,
        topn=10,
        basket=set(),  # falls gerade einen Warenkorb besteht, hier rein
        )

        anchor_pid = _to_key(anchor_pid)
        anchor_name = prod_name.get(anchor_pid, "UNKNOWN")
        print(f"Eingabeprodukt: {anchor_pid} | {anchor_name}")

        recs_with_name = [
            {
                "productid": pid,
                "name": prod_name.get(_to_key(pid), "UNKNOWN"),
                "score": float(score),
            }
            for pid, score in recs
        ]
        recs_df = pd.DataFrame(recs_with_name, columns=["productid", "name", "score"])
        if recs_df.empty:
            print("Keine Empfehlungen gefunden.")
        else:
            print(recs_df.to_string(index=False))
            
    time_end = time.time()

    print("Zeit: ", time_end - time_start)